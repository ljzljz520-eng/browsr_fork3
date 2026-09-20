from __future__ import annotations

import math
from typing import cast

from rich.console import RenderableType
from rich.text import Text
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual_universal_directorytree import (
    GitHubTextualPath,
    S3TextualPath,
    SFTPTextualPath,
    UPath,
)

from browsr.utils import FileInfo, is_remote


class CurrentFileInfoBar(Widget):
    """
    A Widget that displays information about the currently selected file

    Thanks, Kupo. https://github.com/darrenburns/kupo
    """

    file_info: FileInfo | None = reactive(None)  # type: ignore[assignment]

    class FileInfoUpdate(Message):
        """
        File Info Bar Update
        """

        def __init__(self, new_file: FileInfo | None) -> None:
            self.new_file = new_file
            super().__init__()

    def watch_file_info(self, new_file: FileInfo | None) -> None:
        """
        Watch the file_info property for changes
        """
        if new_file is None:
            self.display = False
        else:
            self.display = True

    @classmethod
    def _convert_size(cls, size_bytes: int) -> str:
        """
        Convert Bytes to Human Readable String
        """
        if size_bytes == 0:
            return " 0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        index = math.floor(math.log(size_bytes, 1024))
        p = math.pow(1024, index)
        number = round(size_bytes / p, 2)
        unit = size_name[index]
        return f"{number:.0f}{unit}"

    def render(self) -> RenderableType:
        """
        Render the Current File Info Bar
        """
        status_string = self.render_file_protocol()
        if self.file_info is None:
            return Text(status_string)
        elif self.file_info.is_file:
            file_options = self.render_file_options()
            status_string += file_options
        if (
            self.file_info.last_modified is not None
            and self.file_info.last_modified.timestamp() != 0
        ):
            modify_time = self.file_info.last_modified.strftime("%b %d, %Y %I:%M %p")
            status_string += f"  📅  {modify_time}"
        directory_options = self.render_directory_options()
        status_string += directory_options
        archive_options = self.render_archive_options()
        status_string += archive_options
        return Text(status_string.strip(), style="dim")

    def render_archive_options(self) -> str:
        """
        Render the parent-archive context (and any archive error)
        """
        status_string = ""
        if not self.file_info:
            return status_string
        if (
            self.file_info.is_archive_member
            and self.file_info.parent_archive is not None
        ):
            archive_name = self.file_info.parent_archive.name
            entry = self.file_info.archive_entry or ""
            status_string += f"  📦  {archive_name}"
            if entry:
                status_string += f"::{entry}"
        if self.file_info.archive_error:
            status_string += f"  ⚠️  {self.file_info.archive_error}"
        return status_string

    def render_file_options(self) -> str:
        """
        Render the file options
        """
        status_string = ""
        if not self.file_info:
            return status_string
        if self.file_info.is_file:
            file_size = self._convert_size(self.file_info.size)
            status_string += f"  🗄️️  {file_size}"
        if self.file_info.owner not in ["", None]:
            status_string += f"  👤  {self.file_info.owner}"
        if self.file_info.group.strip() not in ["", None]:
            status_string += f"  🏠  {self.file_info.group}"
        return status_string

    def render_directory_options(self) -> str:
        """
        Render the directory options
        """
        status_string = ""
        if not self.file_info:
            return status_string
        if self.file_info.is_archive_member:
            # Archive members carry their in-archive directory identity.
            if self.file_info.is_file:
                archive_entry = self.file_info.archive_entry or ""
                parent_name = archive_entry.rpartition("/")[0]
                if parent_name:
                    status_string += f"  📂  {parent_name}"
                status_string += f"  💾  {self.file_info.file.name}"
            else:
                status_string += f"  📂  {self.file_info.file.name}"
            return status_string
        file_path = cast(UPath, self.file_info.file)
        if self.file_info.is_file:
            directory_name = file_path.parent.name
            if not directory_name or (
                file_path.protocol and f"{file_path.protocol}://" in directory_name
            ):
                directory_name = str(file_path.parent)
                directory_name = directory_name.lstrip(f"{file_path.protocol}://")
                directory_name = directory_name.rstrip("/")
            status_string += f"  📂  {directory_name}"
            status_string += f"  💾  {file_path.name}"
        else:
            status_string += f"  📂  {file_path.name}"
        return status_string

    def render_file_protocol(self) -> str:
        """
        Render the file protocol
        """
        status_string = ""
        if not self.file_info:
            return status_string
        if is_remote(self.file_info.file):
            file_path = cast(UPath, self.file_info.file)
            if isinstance(file_path, GitHubTextualPath):
                protocol = "GitHub"
            elif isinstance(file_path, S3TextualPath):
                protocol = "S3"
            elif isinstance(file_path, SFTPTextualPath):
                protocol = "SFTP"
            else:
                protocol = file_path.protocol
            status_string += f"🗂️  {protocol}"
        return status_string
