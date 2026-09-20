"""
Tests for the ZIP / TAR archive adapter.

These tests cover lazy indexing, safe member reads, security rejections,
recoverable error handling, the directory-tree integration, the content
windows and the download workflow.
"""

from __future__ import annotations

import io
import pathlib
import stat
import tarfile
import time
import zipfile

import pytest
from PIL import Image
from rich_pixels import Pixels
from textual_universal_directorytree import UPath

from browsr.archive import (
    ArchiveMemberPath,
    ArchiveProblemPath,
    ArchiveSecurityError,
    ArchiveUnavailableError,
    _BoundedReader,
    get_archive_index,
    invalidate_archive_caches,
    is_archive_path,
    parse_archive_uri,
    parse_browsr_path,
    safe_download_name,
)
from browsr.base import TextualAppContext
from browsr.browsr import Browsr
from browsr.exceptions import FileSizeError
from browsr.widgets.universal_directory_tree import BrowsrDirectoryTree

# ---------------------------------------------------------------------------
# Fixture archive builders
# ---------------------------------------------------------------------------


def _png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buf, format="PNG")
    return buf.getvalue()


SAFE_ENTRIES: dict[str, bytes] = {
    "README.md": b"# hello archive",
    "data/config.json": b'{"a": 1, "b": [2, 3]}',
    "data/nested/deep.txt": b"deep text content",
    "data/nested/notes.txt": b"x" * 5000,
    "img/pixel.png": _png_bytes(),
    "table/data.csv": b"a,b\n1,2\n3,4\n",
    "plain.txt": b"plain text",
}

UNSAFE_ENTRIES_ZIP: list[str] = [
    "../evil.txt",
    "/abs/evil2.txt",
    "a/../../evil3.txt",
    "evil-link",
]

UNSAFE_ENTRIES_TAR: list[str] = [
    "../evil.txt",
    "/abs/evil2.txt",
    "a/../../evil3.txt",
    "evil-link",
    "hard-link",
]


def _build_zip(path: str | pathlib.Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in SAFE_ENTRIES.items():
            archive.writestr(name, data)
        archive.writestr("../evil.txt", b"bad")
        archive.writestr("/abs/evil2.txt", b"bad")
        archive.writestr("a/../../evil3.txt", b"bad")
        symlink_info = zipfile.ZipInfo("evil-link")
        symlink_info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(symlink_info, "/etc/passwd")


def _build_tar(path: str | pathlib.Path) -> None:
    with tarfile.open(path, "w:gz") as archive:

        def add_file(name: str, data: bytes) -> None:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))

        for name, data in SAFE_ENTRIES.items():
            add_file(name, data)
        add_file("../evil.txt", b"bad")
        add_file("/abs/evil2.txt", b"bad")
        add_file("a/../../evil3.txt", b"bad")
        symlink = tarfile.TarInfo("evil-link")
        symlink.type = tarfile.SYMTYPE
        symlink.linkname = "/etc/passwd"
        archive.addfile(symlink)
        hardlink = tarfile.TarInfo("hard-link")
        hardlink.type = tarfile.LNKTYPE
        hardlink.linkname = "data/config.json"
        archive.addfile(hardlink)


@pytest.fixture
def archive_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """
    A directory containing a ZIP, a TAR.GZ and a corrupt archive.
    """
    _build_zip(tmp_path / "bundle.zip")
    _build_tar(tmp_path / "bundle.tar.gz")
    (tmp_path / "plain-local.txt").write_text("local file")
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"this is definitely not a zip file")
    return tmp_path


@pytest.fixture
def zip_path(archive_dir: pathlib.Path) -> UPath:
    return UPath(archive_dir / "bundle.zip")


@pytest.fixture
def tar_path(archive_dir: pathlib.Path) -> UPath:
    return UPath(archive_dir / "bundle.tar.gz")


# ---------------------------------------------------------------------------
# Indexing / security / bounded reads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", ["zip_path", "tar_path"])
def test_index_lists_nested_entries(request: pytest.FixtureRequest, fixture_name: str):
    archive = request.getfixturevalue(fixture_name)
    index = get_archive_index(archive)
    root_names = {
        member.entry_name
        for member in ArchiveMemberPath(archive, "").iterdir()
        if isinstance(member, ArchiveMemberPath)
    }
    assert "README.md" in root_names
    assert "data" in root_names
    assert "img" in root_names
    data_children = {
        member.entry_name
        for member in ArchiveMemberPath(archive, "data").iterdir()
        if isinstance(member, ArchiveMemberPath)
    }
    assert data_children == {"data/config.json", "data/nested"}
    nested_children = {
        member.entry_name
        for member in ArchiveMemberPath(archive, "data/nested").iterdir()
    }
    assert "data/nested/deep.txt" in nested_children
    assert index.is_file("data/config.json")
    assert index.is_dir("data/nested")


@pytest.mark.parametrize("fixture_name", ["zip_path", "tar_path"])
def test_unsafe_entries_are_rejected(request: pytest.FixtureRequest, fixture_name: str):
    archive = request.getfixturevalue(fixture_name)
    expected = UNSAFE_ENTRIES_ZIP if fixture_name == "zip_path" else UNSAFE_ENTRIES_TAR
    index = get_archive_index(archive)
    rejected = {raw_name for raw_name, _ in index.problems}
    for raw_name in expected:
        assert raw_name in rejected, f"{raw_name} not rejected in {fixture_name}"
    # The rejected names never appear as ordinary members.
    member_names = set(index._files)
    assert not {
        "evil.txt",
        "evil2.txt",
        "evil3.txt",
        "evil-link",
        "hard-link",
    } & {name.rsplit("/", 1)[-1] for name in member_names if "evil" in name}
    # Problem nodes are surfaced at the archive root.
    root_problems = [
        member
        for member in ArchiveMemberPath(archive, "").iterdir()
        if isinstance(member, ArchiveProblemPath)
    ]
    assert root_problems
    assert all(problem.security for problem in root_problems)
    with pytest.raises(ArchiveSecurityError):
        root_problems[0].read_text()


@pytest.mark.parametrize("fixture_name", ["zip_path", "tar_path"])
def test_member_reads_are_bounded(request: pytest.FixtureRequest, fixture_name: str):
    archive = request.getfixturevalue(fixture_name)
    notes = ArchiveMemberPath(archive, "data/nested/notes.txt")
    with pytest.raises(FileSizeError):
        notes.read_text(max_bytes=1000)
    # A smaller member can still be read within budget.
    deep = ArchiveMemberPath(archive, "data/nested/deep.txt")
    assert deep.read_text(max_bytes=1000) == "deep text content"


def test_bounded_reader_raises_after_budget():
    source = io.BytesIO(b"abcdef")
    reader = _BoundedReader(source, max_bytes=3)
    assert reader.read(2) == b"ab"
    # Reading up to the budget peeks for one more byte and raises when
    # the member is larger than the budget.
    with pytest.raises(FileSizeError):
        reader.read(2)


def test_bounded_reader_allows_exact_budget():
    source = io.BytesIO(b"abc")
    reader = _BoundedReader(source, max_bytes=3)
    assert reader.read() == b"abc"


@pytest.mark.parametrize("fixture_name", ["zip_path", "tar_path"])
def test_member_identity_and_path_semantics(
    request: pytest.FixtureRequest, fixture_name: str
):
    archive = request.getfixturevalue(fixture_name)
    member = ArchiveMemberPath(archive, "data/config.json")
    assert member.name == "config.json"
    assert member.suffix == ".json"
    assert member.suffixes == [".json"]
    assert member.stem == "config"
    assert member.parent.entry_name == "data"
    assert member.parent.parent.entry_name == ""
    assert str(member) == f"{archive}::data/config.json"
    assert str(ArchiveMemberPath(archive, "")) == f"{archive}::"
    # The virtual root's parent is the archive's real containing folder.
    assert ArchiveMemberPath(archive, "").parent == archive.parent
    stat_result = member.stat()
    assert stat_result.st_size == len(b'{"a": 1, "b": [2, 3]}')
    assert member.is_file()
    assert not member.is_dir()


@pytest.mark.parametrize("fixture_name", ["zip_path", "tar_path"])
def test_member_text_json_and_image_bytes(
    request: pytest.FixtureRequest, fixture_name: str
):
    archive = request.getfixturevalue(fixture_name)
    assert ArchiveMemberPath(archive, "README.md").read_text() == "# hello archive"
    assert ArchiveMemberPath(archive, "data/config.json").read_bytes().startswith(b"{")
    png = ArchiveMemberPath(archive, "img/pixel.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    Image.open(io.BytesIO(png)).load()


def test_corrupt_archive_is_recoverable(archive_dir: pathlib.Path):
    corrupt = UPath(archive_dir / "corrupt.zip")
    with pytest.raises(ArchiveUnavailableError):
        get_archive_index(corrupt)
    # The virtual root reports the failure; the tree turns it into a
    # problem node instead of bubbling the exception up.
    root = ArchiveMemberPath(corrupt, "")
    with pytest.raises(ArchiveUnavailableError):
        list(root.iterdir())
    # Emulate the tree's virtual listing without mounting the widget.
    tree = BrowsrDirectoryTree.__new__(BrowsrDirectoryTree)
    content = BrowsrDirectoryTree._virtual_directory_content(tree, root)
    assert len(content) == 1
    problem = content[0]
    assert isinstance(problem, ArchiveProblemPath)
    assert problem.security is False
    with pytest.raises(ArchiveUnavailableError):
        problem.read_bytes()


def test_remote_archive_is_rejected():
    remote = UPath("s3://a-bucket/bundle.zip")
    with pytest.raises(ArchiveUnavailableError):
        get_archive_index(remote)


def test_index_cache_invalidates_on_change(zip_path: UPath):
    index = get_archive_index(zip_path)
    assert get_archive_index(zip_path) is index
    # Rewrite the archive with different content.
    time.sleep(0.01)
    _build_zip(zip_path)
    invalidate_archive_caches()
    assert get_archive_index(zip_path) is not index


def test_uri_parsing_round_trip(zip_path: UPath, tar_path: UPath):
    for archive, entry in ((zip_path, "data/config.json"), (tar_path, "img/pixel.png")):
        parsed = parse_archive_uri(f"{archive}::{entry}")
        assert parsed is not None
        parsed_archive, parsed_entry = parsed
        assert str(parsed_archive) == str(archive)
        assert parsed_entry == entry
        virtual = parse_browsr_path(f"{archive}::{entry}")
        assert isinstance(virtual, ArchiveMemberPath)
        # The directory-tree PATH factory round-trips virtual nodes.
        assert parse_browsr_path(virtual) == virtual
        assert is_archive_path(virtual)
    # A real archive file path becomes a virtual root.
    root = parse_browsr_path(zip_path)
    assert isinstance(root, ArchiveMemberPath) and root.entry_name == ""
    # Ordinary files are untouched.
    plain = parse_browsr_path(str(zip_path.parent / "plain-local.txt"))
    assert not is_archive_path(plain)
    assert parse_archive_uri("/no/marker/file.txt") is None


@pytest.mark.parametrize(
    ("entry_name", "expected"),
    [
        ("dir/file.txt", "file.txt"),
        ("file.txt", "file.txt"),
        ("a/b/c.tar.gz", "c.tar.gz"),
    ],
)
def test_safe_download_name_accepts(entry_name: str, expected: str):
    assert safe_download_name(entry_name) == expected


@pytest.mark.parametrize(
    "entry_name",
    ["../evil.txt", "/etc/passwd", "a/../../evil.txt", "..", "", "../"],
)
def test_safe_download_name_rejects(entry_name: str):
    with pytest.raises(ArchiveSecurityError):
        safe_download_name(entry_name)


# ---------------------------------------------------------------------------
# App / directory-tree / content-window integration
# ---------------------------------------------------------------------------


def _node_by_label(node, label: str):
    for child in node.children:
        if child.label.plain == label:
            return child
    return None


async def _load_node(tree, node) -> None:
    await tree._add_to_load_queue(node)


async def _expand_and_find(app, node, label: str):
    tree = app.code_browser_screen.code_browser.directory_tree
    await _load_node(tree, node)
    found = _node_by_label(node, label)
    if found is None:
        labels = [child.label.plain for child in node.children]
        msg = f"{label} not among {labels}"
        raise AssertionError(msg)
    return found


async def _select(app, pilot, node) -> None:
    tree = app.code_browser_screen.code_browser.directory_tree
    tree.select_node(node)
    await pilot.pause()
    await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("archive_name", ["bundle.zip", "bundle.tar.gz"])
async def test_tree_expands_and_previews_members(archive_dir, archive_name):
    app = Browsr(config_object=TextualAppContext(file_path=str(archive_dir)))
    async with app.run_test() as pilot:
        code_browser = app.code_browser_screen.code_browser
        tree = code_browser.directory_tree
        await pilot.pause()
        archive_node = _node_by_label(tree.root, archive_name)
        assert archive_node is not None
        # The archive file is exposed as an expandable directory node.
        assert archive_node.allow_expand is True
        data_node = await _expand_and_find(app, archive_node, "data")
        await _load_node(tree, data_node)
        json_node = _node_by_label(data_node, "config.json")
        assert json_node is not None

        # Plain-text member
        plain_node = _node_by_label(archive_node, "plain.txt")
        await _select(app, pilot, plain_node)
        switcher = code_browser.window_switcher
        assert switcher.text_window.display is True
        assert "plain text" in switcher.text_window.text

        # Markdown member renders through the static window
        readme_node = _node_by_label(archive_node, "README.md")
        await _select(app, pilot, readme_node)
        assert switcher.vim_scroll.display is True
        markdown = switcher.static_window.content
        assert "# hello archive" in str(markdown.markup)

        # JSON member (rendered through the text window)
        await _select(app, pilot, json_node)
        assert switcher.text_window.display is True
        assert '"a": 1' in switcher.text_window.text

        # Nested text member
        nested_node = _node_by_label(data_node, "nested")
        await _load_node(tree, nested_node)
        deep_node = _node_by_label(nested_node, "deep.txt")
        await _select(app, pilot, deep_node)
        assert switcher.text_window.display is True
        assert "deep text content" in switcher.text_window.text

        # Image member
        img_dir_node = _node_by_label(archive_node, "img")
        await _load_node(tree, img_dir_node)
        png_node = _node_by_label(img_dir_node, "pixel.png")
        await _select(app, pilot, png_node)
        assert switcher.vim_scroll.display is True
        assert isinstance(switcher.static_window.content, Pixels)

        # CSV member
        table_dir_node = _node_by_label(archive_node, "table")
        await _load_node(tree, table_dir_node)
        csv_node = _node_by_label(table_dir_node, "data.csv")
        await _select(app, pilot, csv_node)
        assert switcher.datatable_window.display is True
        assert switcher.datatable_window.row_count == 2

        # FileInfo carries the parent archive context.
        file_info = app.code_browser_screen.file_information.file_info
        assert file_info is not None
        assert file_info.is_archive_member
        assert file_info.parent_archive is not None
        assert file_info.parent_archive.name == archive_name
        assert file_info.archive_entry == "table/data.csv"
        assert str(code_browser.selected_file_path) == (
            f"{archive_dir / archive_name}::table/data.csv"
        )
        # The subtitle keeps the parent-archive identity visible.
        assert f"{archive_name}::table/data.csv" in str(app.sub_title)


@pytest.mark.asyncio
async def test_unsafe_node_preview_is_recoverable(archive_dir):
    app = Browsr(config_object=TextualAppContext(file_path=str(archive_dir)))
    async with app.run_test() as pilot:
        code_browser = app.code_browser_screen.code_browser
        tree = code_browser.directory_tree
        await pilot.pause()
        archive_node = _node_by_label(tree.root, "bundle.zip")
        await _load_node(tree, archive_node)
        unsafe_nodes = [
            child for child in archive_node.children if "UNSAFE" in child.label.plain
        ]
        assert unsafe_nodes
        await _select(app, pilot, unsafe_nodes[0])
        switcher = code_browser.window_switcher
        # The error is rendered in-window rather than bubbling up.
        assert switcher.vim_scroll.display is True
        assert isinstance(code_browser.selected_file_path, ArchiveProblemPath)


@pytest.mark.asyncio
async def test_corrupt_archive_expands_to_error_node(archive_dir):
    app = Browsr(config_object=TextualAppContext(file_path=str(archive_dir)))
    async with app.run_test() as pilot:
        code_browser = app.code_browser_screen.code_browser
        tree = code_browser.directory_tree
        await pilot.pause()
        corrupt_node = _node_by_label(tree.root, "corrupt.zip")
        assert corrupt_node is not None
        await _load_node(tree, corrupt_node)
        assert len(corrupt_node.children) == 1
        problem = corrupt_node.children[0]
        assert "ARCHIVE-ERROR" in problem.label.plain
        tree.select_node(problem)
        await pilot.pause()
        await pilot.pause()
        assert code_browser.window_switcher.vim_scroll.display is True


@pytest.mark.asyncio
async def test_archive_member_download_workflow(
    archive_dir, monkeypatch: pytest.MonkeyPatch
):
    downloads = archive_dir / "Downloads"
    downloads.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", lambda: archive_dir)
    app = Browsr(config_object=TextualAppContext(file_path=str(archive_dir)))
    async with app.run_test() as pilot:
        code_browser = app.code_browser_screen.code_browser
        tree = code_browser.directory_tree
        await pilot.pause()
        archive_node = _node_by_label(tree.root, "bundle.zip")
        await _load_node(tree, archive_node)
        data_node = _node_by_label(archive_node, "data")
        await _load_node(tree, data_node)
        json_node = _node_by_label(data_node, "config.json")
        await _select(app, pilot, json_node)

        # The x / download workflow prompts for archive members.
        code_browser.download_file_workflow()
        await pilot.pause()
        assert code_browser.confirmation_window.display is True

        worker = code_browser.download_selected_file()
        await worker.wait()
        downloaded = downloads / "config.json"
        assert downloaded.exists()
        assert downloaded.read_bytes() == b'{"a": 1, "b": [2, 3]}'
        # No nested in-archive directories are created in Downloads.
        assert not (downloads / "data").exists()

        # A second download handles the name collision safely.
        worker = code_browser.download_selected_file()
        await worker.wait()
        assert (downloads / "config (1).json").exists()


@pytest.mark.asyncio
async def test_archive_member_download_uses_basename_for_nested(
    archive_dir, monkeypatch: pytest.MonkeyPatch
):
    downloads = archive_dir / "Downloads"
    downloads.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", lambda: archive_dir)
    app = Browsr(config_object=TextualAppContext(file_path=str(archive_dir)))
    async with app.run_test() as pilot:
        code_browser = app.code_browser_screen.code_browser
        tree = code_browser.directory_tree
        await pilot.pause()
        archive_node = _node_by_label(tree.root, "bundle.tar.gz")
        await _load_node(tree, archive_node)
        data_node = _node_by_label(archive_node, "data")
        await _load_node(tree, data_node)
        nested_node = _node_by_label(data_node, "nested")
        await _load_node(tree, nested_node)
        deep_node = _node_by_label(nested_node, "deep.txt")
        await _select(app, pilot, deep_node)
        worker = code_browser.download_selected_file()
        await worker.wait()
        assert (downloads / "deep.txt").read_bytes() == b"deep text content"
        # Nothing escaped the Downloads directory.
        entries = {path.name for path in downloads.iterdir()}
        assert entries == {"deep.txt"}


@pytest.mark.asyncio
async def test_opening_member_uri_opens_virtual_path(archive_dir):
    uri = f"{archive_dir / 'bundle.zip'}::plain.txt"
    context = TextualAppContext(file_path=uri)
    member = context.path
    assert isinstance(member, ArchiveMemberPath)
    assert member.entry_name == "plain.txt"
    app = Browsr(config_object=context)
    async with app.run_test() as pilot:
        code_browser = app.code_browser_screen.code_browser
        await pilot.pause()
        assert isinstance(code_browser.selected_file_path, ArchiveMemberPath)
        switcher = code_browser.window_switcher
        assert switcher.text_window.display is True
        assert "plain text" in switcher.text_window.text


@pytest.mark.asyncio
async def test_parent_dir_navigation_inside_archive(archive_dir):
    archive = archive_dir / "bundle.zip"
    app = Browsr(config_object=TextualAppContext(file_path=str(archive_dir)))
    async with app.run_test() as pilot:
        code_browser = app.code_browser_screen.code_browser
        tree = code_browser.directory_tree
        await pilot.pause()
        archive_node = _node_by_label(tree.root, "bundle.zip")
        await _load_node(tree, archive_node)
        data_node = _node_by_label(archive_node, "data")
        data_path = data_node.data.path
        # Navigate into the in-archive directory.
        tree.path = data_path
        await pilot.pause()
        assert tree.path == ArchiveMemberPath(UPath(archive), "data")
        # One step up lands on the archive virtual root.
        app.code_browser_screen.action_parent_dir()
        await pilot.pause()
        assert tree.path == ArchiveMemberPath(UPath(archive), "")
        # Another step leaves the archive for its containing folder.
        app.code_browser_screen.action_parent_dir()
        await pilot.pause()
        assert tree.path == UPath(archive).parent


@pytest.mark.asyncio
async def test_reload_rebuilds_archive_nodes(archive_dir):
    app = Browsr(config_object=TextualAppContext(file_path=str(archive_dir)))
    async with app.run_test() as pilot:
        code_browser = app.code_browser_screen.code_browser
        tree = code_browser.directory_tree
        await pilot.pause()
        archive_node = _node_by_label(tree.root, "bundle.tar.gz")
        await _load_node(tree, archive_node)
        assert _node_by_label(archive_node, "data") is not None
        # ``r`` reloads after dropping cached archive indexes.
        app.code_browser_screen.action_reload()
        await pilot.pause()
        await pilot.pause()
        reloaded_archive = _node_by_label(tree.root, "bundle.tar.gz")
        assert reloaded_archive is not None
        await _load_node(tree, reloaded_archive)
        assert _node_by_label(reloaded_archive, "data") is not None
        assert _node_by_label(reloaded_archive, "img") is not None


@pytest.mark.asyncio
async def test_plain_local_file_unaffected(archive_dir):
    plain = archive_dir / "plain-local.txt"
    app = Browsr(config_object=TextualAppContext(file_path=str(plain)))
    async with app.run_test() as pilot:
        code_browser = app.code_browser_screen.code_browser
        await pilot.pause()
        switcher = code_browser.window_switcher
        assert switcher.text_window.display is True
        assert "local file" in switcher.text_window.text
        file_info = app.code_browser_screen.file_information.file_info
        assert file_info is not None
        assert file_info.is_archive_member is False
        assert file_info.parent_archive is None
