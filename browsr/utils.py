"""
Code Browsr Utility Functions
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import pypdfium2 as pdfium
from PIL import Image
from rich_pixels import Pixels
from textual_universal_directorytree import UPath, is_remote_path

from browsr.archive import (
    ArchiveError,
    ArchiveMemberPath,
    ArchiveProblemPath,
    is_archive_path,
    materialize_member,
)


def _open_pdf_as_image(buf: BinaryIO) -> Image.Image:
    """
    Open a PDF file and return a PIL.Image object
    """
    doc = pdfium.PdfDocument(buf)
    page = doc[0]
    bitmap = page.render()
    image = bitmap.to_pil()
    return image


def _resize_image(image: Image.Image, screen_width: float) -> Pixels:
    """
    Resize a PIL image and convert it to Pixels
    """
    image_width = image.width
    image_height = image.height
    size_ratio = image_width / screen_width
    new_width = int(image_width / size_ratio)
    new_height = int(image_height / size_ratio)
    resized = image.resize((new_width, new_height))
    return Pixels.from_image(resized)


def _image_from_buffer(buf: BinaryIO, suffix: str, screen_width: float) -> Pixels:
    """
    Render an image (or the first PDF page) from an open binary buffer
    """
    if suffix.lower() == ".pdf":
        image = _open_pdf_as_image(buf=buf)
    else:
        image = Image.open(buf)
    return _resize_image(image=image, screen_width=screen_width)


def open_image(
    document: UPath | Path,
    screen_width: float,
    max_bytes: int | None = None,
) -> Pixels:
    """
    Open an image file and return a rich_pixels.Pixels object

    Archive members are first extracted (single-entry, bounded stream) to
    a temporary file because image / PDF decoders require a seekable
    buffer. The whole archive is never read into memory.
    """
    if is_archive_path(document):
        with materialize_member(
            document,  # type: ignore[arg-type]
            suffix=document.suffix,
            max_bytes=max_bytes,
        ) as temp_name:
            with open(temp_name, "rb") as buf:
                return _image_from_buffer(
                    buf=buf,
                    suffix=document.suffix,
                    screen_width=screen_width,
                )
    with document.open("rb") as buf:
        return _image_from_buffer(
            buf=buf,
            suffix=document.suffix,
            screen_width=screen_width,
        )


def is_remote(file_path: UPath | Path | Any) -> bool:
    """
    Whether a path lives on a remote filesystem.

    Entries inside a *local* archive are considered local even though
    they are virtual path objects.
    """
    if is_archive_path(file_path):
        return False
    return is_remote_path(file_path)  # type: ignore[arg-type]


@dataclass
class FileInfo:
    """
    File Information Object
    """

    file: UPath | Path | ArchiveMemberPath | ArchiveProblemPath
    size: int
    last_modified: datetime.datetime | None
    stat: dict[str, Any] | os.stat_result
    is_local: bool
    is_file: bool
    owner: str
    group: str
    is_cloudpath: bool
    parent_archive: UPath | Path | None = None
    archive_entry: str | None = None
    archive_error: str | None = None

    @property
    def is_archive_member(self) -> bool:
        """
        Whether the file lives inside an archive
        """
        return is_archive_path(self.file)


def _archive_file_info(
    file_path: ArchiveMemberPath | ArchiveProblemPath,
) -> FileInfo:
    """
    Build FileInfo for a virtual archive path node.
    """
    archive_error = None
    if isinstance(file_path, ArchiveProblemPath):
        archive_error = file_path.message
        try:
            stat_result = file_path.stat()
            is_file = True
        except ArchiveError as exc:
            archive_error = str(exc)
            stat_result = os.stat_result((0o100444, 0, 0, 1, 0, 0, 0, 0, 0, 0))
            is_file = True
    else:
        try:
            stat_result = file_path.stat()
            is_file = file_path.is_file()
        except ArchiveError as exc:
            archive_error = str(exc)
            stat_result = os.stat_result((0o100444, 0, 0, 1, 0, 0, 0, 0, 0, 0))
            is_file = not file_path.entry_name.endswith("/")
    mtime = stat_result.st_mtime
    last_modified = (
        datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
        if mtime
        else None
    )
    return FileInfo(
        file=file_path,
        size=stat_result.st_size,
        last_modified=last_modified,
        stat=stat_result,
        is_local=True,
        is_file=is_file,
        owner="",
        group="",
        is_cloudpath=False,
        parent_archive=file_path.parent_archive,
        archive_entry=file_path.entry_name,
        archive_error=archive_error,
    )


def get_file_info(file_path: UPath | Path | ArchiveMemberPath) -> FileInfo:
    """
    Get File Information, Regardless of the FileSystem
    """
    if is_archive_path(file_path):
        return _archive_file_info(file_path)  # type: ignore[arg-type]
    try:
        stat: dict[str, Any] | os.stat_result = file_path.stat()  # type: ignore[assignment]
        is_file = file_path.is_file()
    except PermissionError:
        stat = {"size": 0}
        is_file = True
    except FileNotFoundError:
        stat = {"size": 0}
        is_file = True
    is_cloudpath = is_remote(file_path)
    if isinstance(stat, dict):
        lower_dict = {key.lower(): value for key, value in stat.items()}
        file_size = lower_dict["size"]
        modified_keys = ["lastmodified", "updated", "mtime"]
        last_modified = None
        for modified_key in modified_keys:
            if modified_key in lower_dict:
                last_modified = lower_dict[modified_key]
                break
        if isinstance(last_modified, str):
            last_modified = datetime.datetime.fromisoformat(last_modified[:-1])
        return FileInfo(
            file=file_path,
            size=file_size,
            last_modified=last_modified,
            stat=stat,
            is_local=False,
            is_file=is_file,
            owner="",
            group="",
            is_cloudpath=is_cloudpath,
        )
    else:
        last_modified = datetime.datetime.fromtimestamp(
            stat.st_mtime, tz=datetime.timezone.utc
        )
        try:
            owner = file_path.owner()
            group = file_path.group()
        except NotImplementedError:
            owner = ""
            group = ""
        return FileInfo(
            file=file_path,
            size=stat.st_size,
            last_modified=last_modified,
            stat=stat,
            is_local=True,
            is_file=is_file,
            owner=owner,
            group=group,
            is_cloudpath=is_cloudpath,
        )


def handle_duplicate_filenames(file_path: UPath | Path) -> UPath | Path:
    """
    Handle Duplicate Filenames

    Duplicate filenames are handled by appending a number to the filename
    in the form of "filename (1).ext", "filename (2).ext", etc.
    """
    if not file_path.exists():
        return file_path
    else:
        i = 1
        while True:
            new_file_stem = f"{file_path.stem} ({i})"
            new_file_path = file_path.with_stem(new_file_stem)
            if not new_file_path.exists():
                return new_file_path
            i += 1


def handle_github_url(url: str) -> str:
    """
    Handle GitHub URLs

    GitHub URLs are handled by converting them to the raw URL.
    """
    try:
        import requests  # type: ignore[import-untyped] # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "The requests library is required to browse GitHub files. "
            "Install browsr with the `remote` extra to install requests."
        ) from e

    gitub_prefix = "github://"
    if gitub_prefix in url and "@" not in url:
        _, user_password = url.split("github://")
        org, repo_str = user_password.split(":")
        repo, *args = repo_str.split("/")
    elif gitub_prefix in url and "@" in url:
        return url
    elif "github.com" in url.lower():
        _, org, repo, *args = url.split("/")
    else:
        msg = f"Invalid GitHub URL: {url}"
        raise ValueError(msg)
    token = os.getenv("GITHUB_TOKEN")
    auth = {"auth": ("Bearer", token)} if token is not None else {}
    resp = requests.get(
        f"https://api.github.com/repos/{org}/{repo}",
        headers={"Accept": "application/vnd.github.v3+json"},
        timeout=10,
        **auth,
    )
    resp.raise_for_status()
    default_branch = resp.json()["default_branch"]
    arg_str = "/".join(args)
    github_uri = f"{gitub_prefix}{org}:{repo}@{default_branch}/{arg_str}".rstrip("/")
    return github_uri


class ArchiveFileError(Exception):
    """
    Archive File Error
    """
