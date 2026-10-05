"""Download the adapt_decomp datasets from Zenodo and unpack them into data/.

Also the adapt-decomp-data command: adapt-decomp-data list, adapt-decomp-data get NAME.
"""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import typer
from tqdm import tqdm

ZENODO_API = "https://zenodo.org/api/records"
CHUNK_BYTES = 1 << 20

# Archive name -> (Zenodo version DOI, folder it unpacks to under data/, size in GB).
# An empty DOI marks an archive that is packed but not published yet.
ARCHIVES: Dict[str, Tuple[str, str, float]] = {
    "fdsi_example-data": ("", "fdsi_example/data", 0.07),
    "neuromotion-data": ("10.5281/zenodo.22880910", "neuromotion/data", 1.64),
    "fdsi_benchmark-data": ("10.5281/zenodo.22882346", "fdsi_benchmark/data", 10.35),
    "fdsi_benchmark-outputs": ("10.5281/zenodo.22882323", "fdsi_benchmark/outputs", 10.75),
}

app = typer.Typer(help="Download the adapt_decomp datasets from Zenodo.")


def select_archives(patterns: List[str]) -> List[str]:
    """Resolve archive names or name prefixes to archive names.

    Args:
        patterns (List[str]): Names or prefixes; empty selects every archive.

    Returns:
        List[str]: The matching archive names.

    Raises:
        ValueError: If a pattern matches no archive.
    """
    if not patterns:
        return list(ARCHIVES)
    selected: List[str] = []
    for pattern in patterns:
        matches = [name for name in ARCHIVES if name.startswith(pattern)]
        if not matches:
            raise ValueError(
                f"Unknown archive: {pattern!r}. Expected a prefix of {list(ARCHIVES)}."
            )
        selected += matches
    return sorted(set(selected))


def _files(doi: str, name: str) -> List[Dict]:
    """List a Zenodo record's downloadable files.

    Args:
        doi (str): The archive's version DOI.
        name (str): The archive name, for error messages.

    Returns:
        List[Dict]: Zenodo's file entries, each with key, size, checksum and links.

    Raises:
        ValueError: If the archive has no DOI yet or the record cannot be read.
    """
    if not doi:
        raise ValueError(
            f"{name!r} is not published yet. Maintainers can build it with "
            f"python scripts/pack_data.py {name} (see .github/RELEASING.md)."
        )
    record_id = doi.rsplit(".", 1)[-1]
    try:
        with urllib.request.urlopen(f"{ZENODO_API}/{record_id}") as response:
            return json.load(response)["files"]
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"Could not read Zenodo record {record_id} for {name!r}: {exc}") from exc


def _fetch(entry: Dict, target: Path) -> None:
    """Download one file, hashing it as it streams, and check Zenodo's checksum.

    Args:
        entry (Dict): One of Zenodo's file entries.
        target (Path): Where to write the file.

    Returns:
        None

    Raises:
        ValueError: If the downloaded file does not match its published checksum.
    """
    digest = hashlib.md5()
    with urllib.request.urlopen(entry["links"]["self"]) as response, open(target, "wb") as handle:
        with tqdm(
            total=entry["size"],
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"  {entry['key']}",
        ) as bar:
            for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
                handle.write(chunk)
                digest.update(chunk)
                bar.update(len(chunk))

    if entry["checksum"] != f"md5:{digest.hexdigest()}":
        target.unlink()
        raise ValueError(f"Checksum mismatch for {entry['key']}: the download was corrupt.")


def _format_size(size_gb: float) -> str:
    """Format an archive size for printing.

    Args:
        size_gb (float): Size in GB.

    Returns:
        str: The size in GB, or in MB below 1 GB.
    """
    return f"{size_gb:.1f} GB" if size_gb >= 1 else f"{size_gb * 1000:.0f} MB"


def download_data(
    archives: Optional[List[str]] = None,
    dest: Union[str, Path] = "data",
    force: bool = False,
) -> None:
    """Download archives from Zenodo, check their checksums and unpack them.

    Each archive unpacks to dest/<folder> (see ARCHIVES). An archive whose folder already
    holds files is skipped unless force is True.

    Args:
        archives (Optional[List[str]]): Archive names or name prefixes. None or empty
            downloads every archive.
        dest (Union[str, Path]): Directory to unpack into.
        force (bool): Whether to download archives that are already unpacked.

    Returns:
        None

    Raises:
        ValueError: If an archive name is unknown, has no DOI yet, or fails its checksum.
    """
    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for name in select_archives(archives or []):
        doi, unpacks_to, size_gb = ARCHIVES[name]
        unpacked = dest_dir / unpacks_to
        if unpacked.is_dir() and any(unpacked.iterdir()) and not force:
            print(f"{name}: already at {unpacked}, skipping (--force to download again)")
            continue

        print(f"{name} (~{_format_size(size_gb)})")
        for entry in _files(doi, name):
            zip_path = dest_dir / entry["key"]
            _fetch(entry, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest_dir)
            zip_path.unlink()
        print(f"  unpacked into {unpacked}")


@app.command(name="list")
def list_archives() -> None:
    """Print each archive's size, unpacked location and DOI.

    Returns:
        None
    """
    for name, (doi, unpacks_to, size_gb) in ARCHIVES.items():
        print(
            f"{name:<24}{_format_size(size_gb):>8}  ->  data/{unpacks_to:<24}"
            f"{doi or '(not published yet)'}"
        )


@app.command(name="get")
def get(
    archives: Optional[List[str]] = typer.Argument(
        None, help="Archive names or prefixes; omit for all"
    ),
    dest: str = typer.Option("data", "--dest", help="Directory to unpack into"),
    force: bool = typer.Option(False, "--force", help="Download archives already unpacked"),
) -> None:
    """Download the selected archives and unpack them into the destination.

    Args:
        archives (Optional[List[str]]): Archive names or prefixes; omit for all.
        dest (str): Directory to unpack into.
        force (bool): Whether to download archives that are already unpacked.

    Returns:
        None
    """
    download_data(archives, dest, force)
