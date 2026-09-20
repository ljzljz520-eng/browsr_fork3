"""
A universal directory tree widget for Textual.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar, cast

from textual import work
from textual.binding import BindingType
from textual.widgets._directory_tree import DirEntry
from textual.widgets._tree import TreeNode
from textual.worker import get_current_worker
from textual_universal_directorytree import UniversalDirectoryTree, UPath

from browsr.archive import (
    ArchiveError,
    ArchiveMemberPath,
    ArchiveProblemPath,
    invalidate_archive_caches,
    parse_browsr_path,
    virtual_archive_root,
)
from browsr.widgets.double_click_directory_tree import DoubleClickDirectoryTree
from browsr.widgets.vim import vim_cursor_bindings


class BrowsrDirectoryTree(DoubleClickDirectoryTree, UniversalDirectoryTree):
    """
    A DirectoryTree that can handle any filesystem.

    Archive files (ZIP / TAR) and their entries are exposed as virtual
    ``archive.zip::entry`` nodes. Expanding an archive lazily lists its
    entries from the archive index without reading archive data.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        *UniversalDirectoryTree.BINDINGS,
        *vim_cursor_bindings,
    ]

    PATH = staticmethod(parse_browsr_path)  # type: ignore[assignment]

    @classmethod
    def _handle_top_level_bucket(cls, dir_path: UPath | Path) -> Iterable[UPath] | None:
        """
        Handle scenarios when someone wants to browse all of s3

        This is because S3FS handles the root directory differently
        than other filesystems
        """
        if str(dir_path) == "s3:/":
            sub_buckets = sorted(
                UPath(f"s3://{bucket.name}") for bucket in dir_path.iterdir()
            )
            return sub_buckets
        return None

    @classmethod
    def _safe_is_dir(
        cls, path: UPath | Path | ArchiveMemberPath | ArchiveProblemPath
    ) -> bool:
        """
        Safely check if a path is a directory.

        Supported archive files are treated as expandable virtual
        directories. Unreadable archives are still expandable so the
        failure surfaces as a recoverable error node.
        """
        if isinstance(path, ArchiveMemberPath):
            if path.entry_name == "":
                return True
            try:
                return path.is_dir()
            except ArchiveError:
                return True
        if isinstance(path, ArchiveProblemPath):
            return False
        archive_path = virtual_archive_root(path)
        if archive_path is not None:
            return True
        return super()._safe_is_dir(path)

    def _virtual_directory_content(
        self, path: ArchiveMemberPath
    ) -> list[ArchiveMemberPath | ArchiveProblemPath]:
        """
        List archive entries for a virtual directory node.

        Any archive failure (corrupt, remote, unsafe) is turned into a
        single :class:`ArchiveProblemPath` node instead of bubbling up.
        """
        try:
            content = list(path.iterdir())
        except ArchiveError as exc:
            return [
                ArchiveProblemPath.from_exception(archive=path.parent_archive, exc=exc)
            ]
        return content

    @work(thread=True, exit_on_error=False)
    def _load_directory(self, node: TreeNode[DirEntry]) -> list[Path]:
        """
        Load the directory contents for a given node.

        Behaves like textual's loader, but virtual ``archive::entry``
        nodes (and archive files) are expanded from the archive index.
        """
        if node.data is None:
            return []
        path = node.data.path
        virtual_root = virtual_archive_root(path)
        if virtual_root is not None:
            if isinstance(path, ArchiveMemberPath):
                virtual_root = path
            content = self._virtual_directory_content(virtual_root)
            virtual_sorted = sorted(
                content,
                key=lambda item: (
                    not self._safe_is_dir(item),
                    item.name.lower(),
                ),
            )
            return cast("list[Path]", virtual_sorted)
        path = path.expanduser().resolve()
        return sorted(
            self.filter_paths(self._directory_content(path, get_current_worker())),
            key=lambda item: (not self._safe_is_dir(item), item.name.lower()),
        )

    def reload(self) -> Any:
        """
        Reload the tree after dropping cached archive indexes.
        """
        invalidate_archive_caches()
        return super().reload()

    def _populate_node(
        self, node: TreeNode[DirEntry], content: Iterable[UPath | Path]
    ) -> None:
        """
        Populate the given tree node with the given directory content.

        This function overrides the original textual method to handle root level
        cloud buckets.
        """
        top_level_buckets = self._handle_top_level_bucket(dir_path=node.data.path)  # type: ignore[union-attr]
        if top_level_buckets is not None:
            content = top_level_buckets
        node.remove_children()
        for path in content:
            if top_level_buckets is not None:
                path_name = str(path).replace("s3://", "").rstrip("/")
            else:
                path_name = path.name
            node.add(
                path_name,
                data=DirEntry(path),
                allow_expand=self._safe_is_dir(path),
            )
        node.expand()
