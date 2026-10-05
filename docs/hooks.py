"""MkDocs hook: add the notebooks that live outside docs/ to the site.

They stay where they are run (benchmarks/fdsi/, notebooks/), and are rendered from their
stored outputs by mkdocs-jupyter, which converts every notebook in the files collection;
this hook runs first so that collection already holds them.
"""

from pathlib import Path
from typing import Any

from mkdocs.plugins import event_priority
from mkdocs.structure.files import File, Files

REPO_ROOT = Path(__file__).resolve().parents[1]

# Notebooks in the repository, each published at its own repository path
EXTERNAL_NOTEBOOKS = (
    "benchmarks/fdsi/dataset.ipynb",
    "benchmarks/fdsi/report.ipynb",
    "notebooks/original_tutorial/adaptive_emg_decomp_dyn_example.ipynb",
)


@event_priority(100)
def on_files(files: Files, config: Any) -> Files:
    """Add EXTERNAL_NOTEBOOKS to the site's files before the plugins see them.

    Args:
        files (Files): The files MkDocs found in docs/.
        config (Any): The MkDocs config.

    Returns:
        Files: files, plus one File per external notebook, read from the repository root.
    """
    for path in EXTERNAL_NOTEBOOKS:
        files.append(
            File(
                path,
                src_dir=str(REPO_ROOT),
                dest_dir=config["site_dir"],
                use_directory_urls=config["use_directory_urls"],
            )
        )
    return files
