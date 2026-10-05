"""Pack the data archives into dist/ for upload to Zenodo (maintainers, see .github/RELEASING.md).

Each archive is written as dist/adapt_decomp-<archive>.zip, with every entry under the folder
it unpacks to in data/ (e.g. fdsi_example/data/...), so any subset unpacks with
unzip '*.zip' -d data/. Entries are stored uncompressed: the payload (.npz, .hdf5, .pkl,
.mat) is already compressed or barely compressible.
"""

import zipfile
from pathlib import Path
from typing import List, Tuple

import typer

from adapt_decomp.utils.download import ARCHIVES, select_archives

app = typer.Typer(help="Pack the data archives into dist/ for upload to Zenodo.")

# The example recording: one FDSI recording, taken from the fdsi_benchmark data
EXAMPLE_SOURCE = Path("fdsi_benchmark", "data", "sub-01")
EXAMPLE_FILES = (
    "noisy/sub-01_FDSI_triangular-ramp40s_snr30dB_emg.npz",
    "noisy/sub-01_FDSI_triangular-ramp40s_snr30dB_noise_metadata.json",
    "clean/sub-01_FDSI_triangular-ramp40s_spikes.npz",
    "clean/sub-01_FDSI_triangular-ramp40s_angle.npz",
    "clean/sub-01_FDSI_triangular-ramp40s_effort.npz",
    "clean/sub-01_FDSI_triangular-ramp40s_metadata.json",
)
EXAMPLE_README = """# FDSI example recording

One recording of the FDSI benchmark dataset, used by the adapt_decomp quickstart and how-to
guides: subject 1, triangular-ramp contraction with 40 s ramps, noise at 30 dB SNR. It is
synthetic HD-EMG (100 channels, 2048 Hz) simulated with NeuroMotion via MUniverse, with the
ground-truth spike trains of every simulated motor unit.

The full dataset (5 subjects x 5 contractions x 4 SNR levels), with a description of every
file, is the fdsi_benchmark-data archive ({source_doi}).

## Files

```
data/sub-01/noisy/sub-01_FDSI_triangular-ramp40s_snr30dB_emg.npz            key: emg     (samples, 100) float32
data/sub-01/noisy/sub-01_FDSI_triangular-ramp40s_snr30dB_noise_metadata.json the noise added
data/sub-01/clean/sub-01_FDSI_triangular-ramp40s_spikes.npz                  key: spikes  sample indices of each simulated motor unit
data/sub-01/clean/sub-01_FDSI_triangular-ramp40s_angle.npz                   key: angle   (samples,) wrist angle in degrees
data/sub-01/clean/sub-01_FDSI_triangular-ramp40s_effort.npz                  key: effort  (samples,) fraction of MVC
data/sub-01/clean/sub-01_FDSI_triangular-ramp40s_metadata.json               simulation settings
```

Load them with adapt_decomp.utils.load_emg and load_gt.
"""


def archive_members(name: str, data_root: Path) -> List[Tuple[Path, str]]:
    """List an archive's files and their names inside the zip.

    Args:
        name (str): Archive name, a key of ARCHIVES.
        data_root (Path): The data/ directory the archives are packed from.

    Returns:
        List[Tuple[Path, str]]: (source file, name in the zip) pairs. The example
        archive's README has no source file and is written by pack().

    Raises:
        ValueError: If a file the archive needs is missing.
    """
    unpacks_to = ARCHIVES[name][1]
    dataset, kind = unpacks_to.split("/")

    if name == "fdsi_example-data":
        members = [
            (data_root / EXAMPLE_SOURCE / file, f"{unpacks_to}/sub-01/{file}")
            for file in EXAMPLE_FILES
        ]
    else:
        source_dir = data_root / unpacks_to
        members = [
            (path, path.relative_to(data_root).as_posix())
            for path in sorted(source_dir.rglob("*"))
            if path.is_file()
        ]
        # The dataset README rides in its data archive only, so no two archives share a path
        readme = data_root / dataset / "README.md"
        if kind == "data" and readme.exists():
            members.insert(0, (readme, f"{dataset}/README.md"))

    missing = [str(path) for path, _ in members if not path.is_file()]
    if missing or not members:
        raise ValueError(
            f"Cannot pack {name!r}: missing {missing or [str(data_root / unpacks_to)]}"
        )
    return members


def pack(name: str, data_root: Path, out_dir: Path) -> Path:
    """Write one archive to out_dir.

    Args:
        name (str): Archive name, a key of ARCHIVES.
        data_root (Path): The data/ directory the archives are packed from.
        out_dir (Path): Directory the zip is written to.

    Returns:
        Path: The written zip.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"adapt_decomp-{name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        if name == "fdsi_example-data":
            source_doi = ARCHIVES["fdsi_benchmark-data"][0]
            zf.writestr(
                "fdsi_example/README.md",
                EXAMPLE_README.format(source_doi=f"https://doi.org/{source_doi}"),
            )
        for source, arcname in archive_members(name, data_root):
            zf.write(source, arcname)
    return zip_path


@app.command()
def main(
    archives: List[str] = typer.Argument(None, help="Archive names or prefixes; omit for all"),
    data_root: str = typer.Option("data", "--data-root", help="Directory to pack from"),
    out_dir: str = typer.Option("dist", "--out-dir", help="Directory to write the zips to"),
) -> None:
    """Pack the selected archives and print each zip's size and entry count.

    Args:
        archives (List[str]): Archive names or prefixes; omit for all.
        data_root (str): Directory to pack from.
        out_dir (str): Directory to write the zips to.

    Returns:
        None
    """
    for name in select_archives(archives or []):
        zip_path = pack(name, Path(data_root), Path(out_dir))
        with zipfile.ZipFile(zip_path) as zf:
            n_entries = len(zf.namelist())
        print(f"{zip_path}: {zip_path.stat().st_size / 1e6:.0f} MB, {n_entries} entries")


if __name__ == "__main__":
    app()
