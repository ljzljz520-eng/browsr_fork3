"""
Archive Path Adapter

Expose the entries inside local ZIP / TAR archives as virtual path nodes
(``archive.zip::dir/file``) that behave like ``pathlib`` / ``UPath`` objects
for the directory tree and the content windows.

Design goals
------------

* Directory expansion lists entries lazily from the archive *index*
  (ZIP central directory / TAR member headers) - archive *data* is never
  read just to populate the tree.
* Reading an entry always opens a fresh, bounded stream over a single
  member. The whole archive is never read into memory.
* Unsafe entries (path traversal, absolute paths, symlinks / hardlinks /
  device nodes) are rejected and surfaced as security error nodes instead
  of being extracted or previewed.
* Remote and corrupt archives produce recoverable errors in the tree and
  in the preview windows - they never bubble up as unhandled tracebacks.
"""

from __future__ import annotations

import calendar
import contextlib
import io
import os
import posixpath
import re
import stat
import tarfile
import tempfile
import threading
import zipfile
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Any, BinaryIO, ClassVar, cast

from textual_universal_directorytree import UPath, is_remote_path

from browsr.exceptions import FileSizeError

ARCHIVE_MARKER = "::"
"""
Separator between the parent archive path and the in-archive entry path.
"""

ZIP_SUFFIXES: tuple[str, ...] = (".zip",)
TAR_SUFFIXES: tuple[str, ...] = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tbz",
    ".tar.xz",
    ".txz",
)

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class ArchiveError(OSError):
    """
    Base class for recoverable archive errors.

    Subclasses :class:`OSError` so that generic filesystem error handling
    treats archive failures as ordinary, recoverable IO failures.
    """


class ArchiveSecurityError(ArchiveError):
    """
    Raised when an archive entry is unsafe (traversal, absolute path,
    symlink / hardlink / device node).
    """


class ArchiveUnavailableError(ArchiveError):
    """
    Raised when an archive cannot be opened - e.g. it is remote, missing,
    or corrupt.
    """


@dataclass(frozen=True)
class EntryInfo:
    """
    Metadata for a single, validated archive entry.
    """

    name: str
    """
    Normalized, root-relative, posix-style entry path (no leading slash).
    """

    raw_name: str
    """
    The entry name exactly as stored in the archive, used for reads.
    """

    size: int
    """
    Uncompressed size of the entry in bytes.
    """

    mtime: float
    """
    Last modified time, seconds since the epoch (UTC).
    """

    is_dir: bool = False


def _normalize_entry_name(raw_name: str) -> str | None:
    """
    Normalize an archive member name to a safe, root-relative posix path.

    Returns ``None`` when the name escapes the archive root (``..``
    traversal) or is absolute.
    """
    name = raw_name.replace("\\", "/")
    is_dir = name.endswith("/")
    if _WINDOWS_DRIVE_RE.match(name):
        return None
    if name.startswith("/"):
        # Absolute paths are unsafe - do not silently turn them into
        # root-relative entries.
        return None
    name = name.lstrip("/")
    parts = [part for part in name.split("/") if part not in ("", ".")]
    depth = 0
    for part in parts:
        if part == "..":
            if depth == 0:
                return None
            depth -= 1
        else:
            depth += 1
    normalized = "/".join(parts)
    if is_dir and normalized:
        normalized += "/"
    return normalized


def _split_dir_key(name: str) -> tuple[str, str]:
    """
    Split a normalized entry path into ``(parent_dir, base_name)``.
    """
    stripped = name.rstrip("/")
    parent, _, base = stripped.rpartition("/")
    return parent, base


def _parent_dirs(name: str) -> set[str]:
    """
    Return all implicit parent directory keys for a normalized entry.
    """
    parent, _ = _split_dir_key(name)
    parents: set[str] = set()
    while parent:
        parents.add(parent)
        parent, _, _ = parent.rpartition("/")
    return parents


def archive_kind(path: str | os.PathLike[str] | UPath) -> str | None:
    """
    Return ``"zip"`` / ``"tar"`` when ``path`` is a supported archive.
    """
    name = str(path).lower()
    if name.endswith(ZIP_SUFFIXES):
        return "zip"
    for suffix in TAR_SUFFIXES:
        if name.endswith(suffix):
            return "tar"
    return None


def parse_archive_uri(
    value: str,
) -> tuple[UPath, str] | None:
    """
    Parse an ``archive.zip::dir/file`` URI into ``(archive_path, entry)``.

    Returns ``None`` when the value is not a supported archive URI.
    """
    marker_index = value.find(ARCHIVE_MARKER)
    if marker_index <= 0:
        return None
    archive_part = value[:marker_index]
    if archive_kind(archive_part) is None:
        return None
    entry_part = value[marker_index + len(ARCHIVE_MARKER) :]
    entry_part = entry_part.replace("\\", "/").lstrip("/")
    if entry_part.endswith("/") and entry_part != "":
        entry_part = entry_part.rstrip("/")
    return UPath(archive_part), entry_part


class _BoundedReader(io.RawIOBase):
    """
    Read-through wrapper that enforces a maximum number of bytes.

    Implements the raw binary IO protocol so it can be wrapped in an
    :class:`io.TextIOWrapper` for tabular / text decoders.
    """

    def __init__(self, stream: BinaryIO, max_bytes: int | None) -> None:
        super().__init__()
        self._stream = stream
        self._max_bytes = max_bytes
        self._read = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            return b""
        if self._max_bytes is None:
            return self._stream.read(size)
        remaining = self._max_bytes - self._read
        if remaining <= 0:
            msg = "Archive member exceeds the maximum allowed file size"
            raise FileSizeError(msg)
        if size is None or size < 0 or size > remaining:
            size = remaining
        chunk = self._stream.read(size)
        self._read += len(chunk)
        if len(chunk) == size and size == remaining:
            # Peek for one more byte to detect an over-budget member.
            extra = self._stream.read(1)
            if extra:
                self._read += 1
                msg = "Archive member exceeds the maximum allowed file size"
                raise FileSizeError(msg)
        return chunk

    def readall(self) -> bytes:
        chunks: list[bytes] = []
        while True:
            chunk = self.read(io.DEFAULT_BUFFER_SIZE)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if not self.closed:
            super().close()
            self._stream.close()


class ArchiveIndex:
    """
    Lazy, in-memory *metadata* index over a single archive file.

    The index only stores member headers - member data is streamed on
    demand through fresh file handles.
    """

    def __init__(self, archive: UPath, kind: str) -> None:
        self.archive = archive
        self.kind = kind
        self._files: dict[str, EntryInfo] = {}
        self._dirs: set[str] = set()
        self.problems: list[tuple[str, str]] = []
        """
        ``(raw_entry_name, reason)`` pairs for rejected unsafe entries.
        """

        self._load()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _add_entry(
        self,
        raw_name: str,
        size: int,
        mtime: float,
        is_dir: bool,
        is_link: bool,
        is_device: bool = False,
    ) -> None:
        """
        Validate and index a single archive member.
        """
        if is_link:
            self.problems.append((raw_name, "symlink or hardlink entry is unsafe"))
            return
        if is_device:
            self.problems.append((raw_name, "device entry is unsafe"))
            return
        normalized = _normalize_entry_name(raw_name)
        if normalized is None:
            self.problems.append(
                (raw_name, "absolute path or parent traversal is unsafe")
            )
            return
        if is_dir:
            normalized = normalized.rstrip("/")
            if normalized:
                self._dirs.add(normalized)
            return
        if not normalized:
            self.problems.append((raw_name, "empty file name is unsafe"))
            return
        self._files[normalized] = EntryInfo(
            name=normalized,
            raw_name=raw_name,
            size=max(int(size), 0),
            mtime=int(mtime or 0),
            is_dir=False,
        )
        self._dirs.update(_parent_dirs(normalized))

    def _load(self) -> None:
        """
        Build the index from the archive headers.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    @property
    def root_entry(self) -> ArchiveMemberPath:
        """
        Virtual path for the archive's root directory.
        """
        return ArchiveMemberPath(archive=self.archive, entry_name="")

    def info(self, name: str) -> EntryInfo:
        normalized = name.rstrip("/")
        try:
            return self._files[normalized]
        except KeyError:
            if normalized in self._dirs or normalized == "":
                return EntryInfo(
                    name=normalized,
                    raw_name=normalized + "/" if normalized else "",
                    size=0,
                    mtime=0,
                    is_dir=True,
                )
            msg = f"No such archive entry: {name}"
            raise FileNotFoundError(msg) from None

    def is_dir(self, name: str) -> bool:
        normalized = name.rstrip("/")
        return normalized == "" or normalized in self._dirs

    def is_file(self, name: str) -> bool:
        return name.rstrip("/") in self._files

    def list_dir(self, prefix: str) -> tuple[list[EntryInfo], list[EntryInfo]]:
        """
        List the direct children of ``prefix``.

        Returns a tuple of ``(directory_entries, file_entries)``.
        Rejected entries are surfaced separately (at the archive root).
        """
        prefix = prefix.rstrip("/")
        files: dict[str, EntryInfo] = {}
        dirs: dict[str, str] = {}
        for info in self._files.values():
            parent, base = _split_dir_key(info.name)
            if parent == prefix:
                files[base] = info
        for dirname in self._dirs:
            parent, base = _split_dir_key(dirname + "/")
            if parent == prefix and base not in files:
                dirs[base] = dirname
        file_entries = sorted(files.values(), key=lambda item: item.name)
        dir_entries = [
            EntryInfo(
                name=dirname,
                raw_name=dirname,
                size=0,
                mtime=0,
                is_dir=True,
            )
            for dirname in sorted(dirs.values())
        ]
        return dir_entries, file_entries

    def open_member(self, name: str) -> BinaryIO:
        """
        Open a *single* archive member as a fresh binary stream.
        """
        raise NotImplementedError


class ZipArchiveIndex(ArchiveIndex):
    """
    Archive index backed by :mod:`zipfile` (reads the central directory).
    """

    def _load(self) -> None:
        fileobj = self.archive.open("rb")
        try:
            try:
                zf = zipfile.ZipFile(fileobj)
            except (zipfile.BadZipFile, OSError) as exc:
                msg = f"Corrupt or unreadable ZIP archive {self.archive}: {exc}"
                raise ArchiveUnavailableError(msg) from exc
            with zf:
                for zip_info in zf.infolist():
                    mode = zip_info.external_attr >> 16
                    mtime = (
                        calendar.timegm(zip_info.date_time)
                        if any(zip_info.date_time)
                        else 0
                    )
                    self._add_entry(
                        raw_name=zip_info.filename,
                        size=zip_info.file_size,
                        mtime=mtime,
                        is_dir=zip_info.is_dir() or stat.S_ISDIR(mode),
                        is_link=stat.S_ISLNK(mode),
                    )
        finally:
            fileobj.close()

    def open_member(self, name: str) -> BinaryIO:
        info = self.info(name)
        if info.is_dir:
            msg = f"{name} is a directory inside {self.archive}"
            raise IsADirectoryError(msg)
        fileobj = self.archive.open("rb")
        try:
            try:
                zf = zipfile.ZipFile(fileobj)
                member = zf.open(info.raw_name)
            except (zipfile.BadZipFile, KeyError, OSError) as exc:
                fileobj.close()
                msg = f"Unable to read {name} from {self.archive}: {exc}"
                raise ArchiveUnavailableError(msg) from exc
        except BaseException:
            fileobj.close()
            raise

        def _close() -> None:
            with contextlib.suppress(Exception):
                member.close()
            with contextlib.suppress(Exception):
                zf.close()
            fileobj.close()

        return cast(BinaryIO, _ClosingStream(member, on_close=_close))


class TarArchiveIndex(ArchiveIndex):
    """
    Archive index backed by :mod:`tarfile` (random-access member headers).
    """

    def _load(self) -> None:
        fileobj = self.archive.open("rb")
        try:
            try:
                tf = tarfile.open(fileobj=fileobj, mode="r:*")
            except (tarfile.TarError, OSError, EOFError) as exc:
                msg = f"Corrupt or unreadable TAR archive {self.archive}: {exc}"
                raise ArchiveUnavailableError(msg) from exc
            with tf:
                for member in tf.getmembers():
                    is_link = member.issym() or member.islnk()
                    is_device = (
                        member.ischr()
                        or member.isblk()
                        or member.isfifo()
                        or member.isdev()
                    )
                    self._add_entry(
                        raw_name=member.name,
                        size=member.size,
                        mtime=member.mtime,
                        is_dir=member.isdir(),
                        is_link=is_link,
                        is_device=is_device,
                    )
        finally:
            fileobj.close()

    def open_member(self, name: str) -> BinaryIO:
        info = self.info(name)
        if info.is_dir:
            msg = f"{name} is a directory inside {self.archive}"
            raise IsADirectoryError(msg)
        fileobj = self.archive.open("rb")
        try:
            try:
                tf = tarfile.open(fileobj=fileobj, mode="r:*")
                tar_info = tf.getmember(info.raw_name)
                member = tf.extractfile(tar_info)
            except (tarfile.TarError, KeyError, OSError) as exc:
                fileobj.close()
                msg = f"Unable to read {name} from {self.archive}: {exc}"
                raise ArchiveUnavailableError(msg) from exc
        except BaseException:
            fileobj.close()
            raise
        if member is None:
            fileobj.close()
            msg = f"{name} inside {self.archive} is not a regular file"
            raise ArchiveUnavailableError(msg)

        def _close() -> None:
            with contextlib.suppress(Exception):
                member.close()
            with contextlib.suppress(Exception):
                tf.close()
            fileobj.close()

        return cast(BinaryIO, _ClosingStream(member, on_close=_close))


class _ClosingStream:
    """
    Wrap a member stream so closing it also releases the archive handles.
    """

    def __init__(self, stream: Any, on_close: Any) -> None:
        self._stream = stream
        self._on_close = on_close
        self._closed = False

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            with contextlib.suppress(Exception):
                self._on_close()

    def seekable(self) -> bool:
        return bool(getattr(self._stream, "seekable", lambda: False)())

    def seek(self, *args: Any, **kwargs: Any) -> int:
        return self._stream.seek(*args, **kwargs)

    def tell(self) -> int:
        return self._stream.tell()

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._stream)

    def __enter__(self) -> _ClosingStream:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


@dataclass(frozen=True)
class _CachedIndex:
    index: ArchiveIndex
    signature: tuple[int, int]


_INDEX_CACHE: dict[str, _CachedIndex] = {}
_INDEX_LOCK = threading.Lock()


def invalidate_archive_caches() -> None:
    """
    Drop all cached archive indexes (used by the directory tree reload).
    """
    with _INDEX_LOCK:
        _INDEX_CACHE.clear()


def _archive_signature(archive: UPath) -> tuple[int, int]:
    stat_result = archive.stat()
    return int(stat_result.st_mtime), int(stat_result.st_size)


def get_archive_index(archive: UPath | str | os.PathLike[str]) -> ArchiveIndex:
    """
    Get a cached :class:`ArchiveIndex` for a local archive path.

    Remote archives are rejected with :class:`ArchiveUnavailableError`.
    """
    archive_path = archive if isinstance(archive, UPath) else UPath(archive)
    kind = archive_kind(archive_path)
    if kind is None:
        msg = f"Not a supported archive: {archive_path}"
        raise ArchiveUnavailableError(msg)
    if is_remote_path(archive_path):
        msg = (
            f"Remote archives cannot be browsed in place: {archive_path}. "
            "Download the archive locally first."
        )
        raise ArchiveUnavailableError(msg)
    try:
        signature = _archive_signature(archive_path)
    except (OSError, PermissionError) as exc:
        msg = f"Cannot open archive {archive_path}: {exc}"
        raise ArchiveUnavailableError(msg) from exc
    cache_key = str(archive_path)
    with _INDEX_LOCK:
        cached = _INDEX_CACHE.get(cache_key)
        if cached is not None and cached.signature == signature:
            return cached.index
        index_cls = ZipArchiveIndex if kind == "zip" else TarArchiveIndex
        try:
            index = index_cls(archive=archive_path, kind=kind)
        except ArchiveError:
            raise
        except Exception as exc:
            msg = f"Unable to read archive {archive_path}: {exc}"
            raise ArchiveUnavailableError(msg) from exc
        _INDEX_CACHE[cache_key] = _CachedIndex(index=index, signature=signature)
        return index


class _ArchiveVirtualPath(os.PathLike[str]):
    """
    Shared machinery for virtual ``archive::entry`` path nodes.
    """

    def __init__(self, archive: UPath, entry_name: str) -> None:
        self._archive = archive
        self._entry = entry_name.strip("/")

    def __fspath__(self) -> str:
        return str(self)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def parent_archive(self) -> UPath:
        """
        The archive file that contains this entry.
        """
        return self._archive

    @property
    def entry_name(self) -> str:
        """
        The normalized, root-relative path of the entry within the archive.
        """
        return self._entry

    @property
    def protocol(self) -> str:
        """
        Protocol of the parent archive (for compatibility with UPath).
        """
        return self._archive.protocol

    # ------------------------------------------------------------------
    # PurePosixPath-style semantics over the in-archive entry name
    # ------------------------------------------------------------------

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(self._entry.split("/")) if self._entry else ()

    @property
    def name(self) -> str:
        if not self._entry:
            return self._archive.name
        return posixpath.basename(self._entry.rstrip("/"))

    @property
    def suffix(self) -> str:
        return posixpath.splitext(self.name)[1]

    @property
    def suffixes(self) -> list[str]:
        stem = self.name
        found: list[str] = []
        while True:
            stem, suffix = posixpath.splitext(stem)
            if not suffix:
                break
            found.append(suffix)
        found.reverse()
        return found

    @property
    def stem(self) -> str:
        return posixpath.splitext(self.name)[0]

    @property
    def parent(self) -> Any:
        raise NotImplementedError

    def with_name(self, new_name: str) -> _ArchiveVirtualPath:
        parent, _, _ = self._entry.rpartition("/")
        entry = f"{parent}/{new_name}" if parent else new_name
        return ArchiveMemberPath(self._archive, entry)

    def joinpath(self, *segments: str) -> _ArchiveVirtualPath:
        combined = "/".join(
            segment.strip("/") for segment in (self._entry, *segments) if segment
        )
        return ArchiveMemberPath(self._archive, combined)

    def __truediv__(self, segment: str) -> _ArchiveVirtualPath:
        return self.joinpath(segment)

    def expanduser(self) -> _ArchiveVirtualPath:
        return self

    def resolve(self, strict: bool = False) -> _ArchiveVirtualPath:  # noqa: ARG002
        return self

    def absolute(self) -> _ArchiveVirtualPath:
        return self

    # ------------------------------------------------------------------
    # Concrete filesystem operations (implemented by subclasses)
    # ------------------------------------------------------------------

    def exists(self, *, follow_symlinks: bool = True) -> bool:
        raise NotImplementedError

    def is_dir(self) -> bool:
        raise NotImplementedError

    def is_file(self) -> bool:
        raise NotImplementedError

    def stat(self) -> os.stat_result:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _ArchiveVirtualPath):
            return NotImplemented
        return (
            type(self) is type(other)
            and str(self._archive) == str(other._archive)
            and self._entry == other._entry
        )

    def __hash__(self) -> int:
        return hash((type(self).__name__, str(self._archive), self._entry))


class ArchiveMemberPath(_ArchiveVirtualPath):
    """
    A virtual path to a file or directory inside an archive.

    String form: ``/path/to/archive.zip::dir/file.txt``.
    """

    def __str__(self) -> str:
        if self._entry:
            return f"{self._archive}{ARCHIVE_MARKER}{self._entry}"
        return f"{self._archive}{ARCHIVE_MARKER}"

    @property
    def parent(self) -> Any:
        if not self._entry:
            # The archive root lives inside the archive's containing folder.
            return self._archive.parent
        parent, _, _ = self._entry.rpartition("/")
        return ArchiveMemberPath(self._archive, parent)

    @property
    def parents(self) -> tuple[Any, ...]:
        found: list[Any] = []
        current: Any = self
        while isinstance(current, ArchiveMemberPath):
            current = current.parent
            found.append(current)
        return tuple(found)

    def _index(self) -> ArchiveIndex:
        return get_archive_index(self._archive)

    def exists(self, *, follow_symlinks: bool = True) -> bool:  # noqa: ARG002
        try:
            index = self._index()
        except ArchiveError:
            return False
        return index.is_dir(self._entry) or index.is_file(self._entry)

    def is_dir(self) -> bool:
        if self._entry == "":
            # Opening the index validates that the archive is readable.
            self._index()
            return True
        return self._index().is_dir(self._entry)

    def is_file(self) -> bool:
        return self._index().is_file(self._entry)

    def stat(self) -> os.stat_result:
        info = self._index().info(self._entry)
        mode = stat.S_IFDIR | 0o555 if info.is_dir else stat.S_IFREG | 0o444
        return os.stat_result(
            (
                mode,
                0,
                0,
                1,
                0,
                0,
                info.size,
                int(info.mtime),
                int(info.mtime),
                int(info.mtime),
            )
        )

    def owner(self) -> str:
        raise NotImplementedError

    def group(self) -> str:
        raise NotImplementedError

    def iterdir(self) -> Generator[ArchiveMemberPath | ArchiveProblemPath, None, None]:
        index = self._index()
        if not index.is_dir(self._entry):
            msg = f"Not a directory inside archive: {self}"
            raise NotADirectoryError(msg)
        prefix = self._entry
        dir_entries, file_entries = index.list_dir(prefix)
        for entry in list(dir_entries) + list(file_entries):
            # ``EntryInfo.name`` already carries the full normalized key.
            yield ArchiveMemberPath(self._archive, entry.name)
        if prefix == "":
            for raw_name, reason in index.problems:
                yield ArchiveProblemPath(
                    archive=self._archive,
                    label=problem_label(raw_name),
                    message=f"Rejected unsafe entry {raw_name!r}: {reason}",
                    entry_name=raw_name,
                    security=True,
                )

    def open(
        self,
        mode: str = "rb",
        max_bytes: int | None = None,
    ) -> BinaryIO:
        """
        Open the entry for reading through a fresh, bounded member stream.
        """
        if "b" not in mode:
            msg = "Archive members only support binary mode ('rb')"
            raise ValueError(msg)
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            msg = "Archive members are read-only"
            raise ValueError(msg)
        index = self._index()
        stream = index.open_member(self._entry)
        return cast(BinaryIO, _BoundedReader(stream, max_bytes))

    def read_bytes(self, max_bytes: int | None = None) -> bytes:
        """
        Read the entry's bytes, enforcing ``max_bytes``.
        """
        chunks: list[bytes] = []
        with self.open("rb", max_bytes=max_bytes) as stream:
            while True:
                chunk = stream.read(io.DEFAULT_BUFFER_SIZE)
                if not chunk:
                    break
                chunks.append(chunk)
        return b"".join(chunks)

    def read_text(
        self,
        encoding: str | None = "utf-8",
        errors: str | None = None,
        max_bytes: int | None = None,
    ) -> str:
        """
        Read the entry as text, enforcing ``max_bytes``.
        """
        data = self.read_bytes(max_bytes=max_bytes)
        return data.decode(encoding or "utf-8", errors or "strict")


class ArchiveProblemPath(_ArchiveVirtualPath):
    """
    A virtual, non-expandable tree node that surfaces an archive error
    (security rejection / corrupt / remote archive) in the tree and in
    the preview windows.
    """

    security_prefix: ClassVar[str] = "⚠UNSAFE:"
    error_prefix: ClassVar[str] = "⚠ARCHIVE-ERROR:"

    def __init__(
        self,
        archive: UPath,
        label: str,
        message: str,
        entry_name: str = "",
        security: bool = False,
    ) -> None:
        super().__init__(archive=archive, entry_name=entry_name or label)
        # Preserve the raw (unsafe) entry identity verbatim - the base
        # constructor strips leading slashes.
        self._entry = entry_name
        self._label = label
        self.message = message
        self.security = security

    @classmethod
    def from_exception(cls, archive: UPath, exc: ArchiveError) -> ArchiveProblemPath:
        """
        Build a problem node describing an archive-level exception.
        """
        security = isinstance(exc, ArchiveSecurityError)
        return cls(
            archive=archive,
            label=cls.error_prefix,
            message=str(exc),
            entry_name="",
            security=security,
        )

    @property
    def name(self) -> str:
        return self._label

    @property
    def suffix(self) -> str:
        return ""

    @property
    def suffixes(self) -> list[str]:
        return []

    @property
    def stem(self) -> str:
        return self._label

    @property
    def parent(self) -> Any:
        return self._archive

    @property
    def parts(self) -> tuple[str, ...]:
        return (self._label,)

    def joinpath(self, *segments: str) -> ArchiveProblemPath:  # noqa: ARG002
        return self

    def __truediv__(self, segment: str) -> ArchiveProblemPath:
        return self

    def __str__(self) -> str:
        return f"{self._archive}{ARCHIVE_MARKER}{self._label}"

    def exists(self, *, follow_symlinks: bool = True) -> bool:  # noqa: ARG002
        return True

    def is_dir(self) -> bool:
        return False

    def is_file(self) -> bool:
        return True

    def stat(self) -> os.stat_result:
        return os.stat_result((stat.S_IFREG | 0o444, 0, 0, 1, 0, 0, 0, 0, 0, 0))

    def owner(self) -> str:
        raise NotImplementedError

    def group(self) -> str:
        raise NotImplementedError

    def raise_error(self) -> None:
        """
        Raise the error represented by this node.
        """
        exc_type = ArchiveSecurityError if self.security else ArchiveUnavailableError
        raise exc_type(self.message)

    def open(
        self,
        mode: str = "rb",  # noqa: ARG002
        max_bytes: int | None = None,  # noqa: ARG002
    ) -> BinaryIO:
        self.raise_error()
        raise AssertionError("unreachable")

    def read_bytes(self, max_bytes: int | None = None) -> bytes:  # noqa: ARG002
        self.raise_error()
        raise AssertionError("unreachable")

    def read_text(
        self,
        encoding: str | None = "utf-8",  # noqa: ARG002
        errors: str | None = None,  # noqa: ARG002
        max_bytes: int | None = None,  # noqa: ARG002
    ) -> str:
        self.raise_error()
        raise AssertionError("unreachable")


def problem_label(raw_name: str) -> str:
    """
    Build a single-token tree label for a rejected unsafe entry.
    """
    sanitized = re.sub(r"\s+", "_", raw_name).strip("/")
    return f"{ArchiveProblemPath.security_prefix}{sanitized}"[:120]


def is_archive_path(path: Any) -> bool:
    """
    Return ``True`` when ``path`` is any kind of virtual archive node.
    """
    return isinstance(path, _ArchiveVirtualPath)


def is_archive_member(path: Any) -> bool:
    """
    Return ``True`` when ``path`` points at an entry inside an archive.
    """
    return isinstance(path, ArchiveMemberPath)


def virtual_archive_root(
    path: os.PathLike[str],
) -> ArchiveMemberPath | None:
    """
    Coerce ``path`` into an archive root if it references an archive.

    * :class:`ArchiveMemberPath` is returned as-is.
    * A real ``UPath`` pointing at a supported archive file becomes the
    archive's virtual root (``entry_name=""``).
    """
    if isinstance(path, ArchiveMemberPath):
        return path
    if isinstance(path, ArchiveProblemPath):
        return None
    if isinstance(path, UPath) and archive_kind(path) is not None:
        try:
            if path.is_file():
                return ArchiveMemberPath(archive=path, entry_name="")
        except OSError:
            return ArchiveMemberPath(archive=path, entry_name="")
    return None


def parse_browsr_path(value: str | os.PathLike[str]) -> UPath | _ArchiveVirtualPath:
    """
    ``DirectoryTree.PATH`` factory that preserves virtual archive nodes.

    Plain values fall back to :class:`UPath`.
    """
    if isinstance(value, ArchiveMemberPath):
        return ArchiveMemberPath(
            archive=value.parent_archive, entry_name=value.entry_name
        )
    if isinstance(value, ArchiveProblemPath):
        return value
    if isinstance(value, UPath) and archive_kind(value) is not None:
        with contextlib.suppress(OSError):
            if value.is_file():
                return ArchiveMemberPath(archive=value, entry_name="")
        return value
    text = os.fspath(value)
    parsed = parse_archive_uri(text)
    if parsed is not None:
        archive, entry = parsed
        return ArchiveMemberPath(archive=archive, entry_name=entry)
    return UPath(value)


def safe_download_name(entry_name: str) -> str:
    """
    Return the only file-name component allowed inside ``Downloads``.

    Raises :class:`ArchiveSecurityError` if the entry provides no safe
    basename - callers must never create nested or traversal paths.
    """
    normalized = _normalize_entry_name(entry_name)
    if normalized is None:
        msg = f"Refusing to download entry with unsafe file name: {entry_name!r}"
        raise ArchiveSecurityError(msg)
    base = posixpath.basename(normalized.rstrip("/"))
    if not base or base in {".", ".."} or base != os.path.basename(base):
        msg = f"Refusing to download entry with unsafe file name: {entry_name!r}"
        raise ArchiveSecurityError(msg)
    return base


@contextlib.contextmanager
def materialize_member(
    path: ArchiveMemberPath,
    suffix: str | None = None,
    max_bytes: int | None = None,
) -> Generator[str, None, None]:
    """
    Extract a single archive member to a temporary file (bounded stream),
    yielding its path. The temp file is removed on exit.
    """
    suffix = suffix if suffix is not None else path.suffix
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_name = tmp.name
    tmp.close()
    try:
        with path.open("rb", max_bytes=max_bytes) as src, open(tmp_name, "wb") as dst:
            while True:
                chunk = src.read(io.DEFAULT_BUFFER_SIZE)
                if not chunk:
                    break
                dst.write(chunk)
        yield tmp_name
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
