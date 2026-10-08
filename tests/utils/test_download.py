"""Tests for the dataset downloader (adapt_decomp.utils.download) and scripts/pack_data.py."""

import zipfile
from pathlib import Path

import pytest

from adapt_decomp.utils import download
from adapt_decomp.utils.download import ARCHIVES, download_data, select_archives

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_select_archives_resolves_names_and_prefixes():
    assert select_archives([]) == list(ARCHIVES)
    assert select_archives(["fdsi_benchmark"]) == [
        "fdsi_benchmark-data",
        "fdsi_benchmark-outputs-v1.1.0",
    ]
    assert select_archives(["neuromotion-data"]) == ["neuromotion-data"]


def test_select_archives_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown archive"):
        select_archives(["nope"])


def test_download_of_an_unpublished_archive_raises_before_any_request(tmp_path, monkeypatch):
    monkeypatch.setitem(ARCHIVES, "test-data", ("", "test/data", 0.01))
    with pytest.raises(ValueError, match="not published yet"):
        download_data(["test-data"], dest=tmp_path)


def test_download_skips_an_archive_that_is_already_unpacked(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(ARCHIVES, "test-data", ("", "test/data", 0.01))
    (tmp_path / "test" / "data").mkdir(parents=True)
    (tmp_path / "test" / "data" / "file.txt").write_text("x")
    download_data(["test-data"], dest=tmp_path)  # no DOI, so it would raise if it fetched
    assert "skipping" in capsys.readouterr().out


def test_size_is_printed_in_mb_below_one_gb():
    assert download._format_size(0.07) == "70 MB"
    assert download._format_size(1.64) == "1.6 GB"


def test_pack_data_prefixes_every_entry_with_its_dataset(tmp_path, monkeypatch):
    import runpy

    pack_data = runpy.run_path(str(REPO_ROOT / "scripts" / "pack_data.py"))
    monkeypatch.setitem(ARCHIVES, "toy-data", ("", "toy/data", 0.01))
    (tmp_path / "toy" / "data" / "sub").mkdir(parents=True)
    (tmp_path / "toy" / "data" / "sub" / "a.npz").write_bytes(b"a")
    (tmp_path / "toy" / "README.md").write_text("readme")

    zip_path = pack_data["pack"]("toy-data", tmp_path, tmp_path / "dist")
    names = zipfile.ZipFile(zip_path).namelist()
    assert names == ["toy/README.md", "toy/data/sub/a.npz"]


def test_pack_data_keeps_only_results_and_searches_of_a_version_outputs(tmp_path, monkeypatch):
    import runpy

    pack_data = runpy.run_path(str(REPO_ROOT / "scripts" / "pack_data.py"))
    monkeypatch.setitem(ARCHIVES, "toy-outputs-v1", ("", "toy/outputs/v1", 0.01))
    for folder in ("results/fixed", "searches/sv", "calibration/sub"):
        (tmp_path / "toy" / "outputs" / "v1" / folder).mkdir(parents=True)
        (tmp_path / "toy" / "outputs" / "v1" / folder / "a.npz").write_bytes(b"a")

    zip_path = pack_data["pack"]("toy-outputs-v1", tmp_path, tmp_path / "dist")
    names = zipfile.ZipFile(zip_path).namelist()
    assert names == ["toy/outputs/v1/results/fixed/a.npz", "toy/outputs/v1/searches/sv/a.npz"]
