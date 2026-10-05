"""Execute the benchmark's notebooks: the dataset tour (raw data) and the report (on the smoke
spec's tables, from python -m benchmarks.fdsi ... --spec benchmarks/fdsi/benchmark_smoke.yaml)."""

import pytest

from benchmarks.fdsi.spec import REPO_ROOT, load_spec

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

NOTEBOOK_DIR = REPO_ROOT / "benchmarks" / "fdsi"
SMOKE_SPEC = "benchmarks/fdsi/benchmark_smoke.yaml"

pytestmark = pytest.mark.slow


def _execute(name: str) -> "nbformat.NotebookNode":
    """Run a notebook of NOTEBOOK_DIR from its own directory, as Jupyter would."""
    notebook = nbformat.read(NOTEBOOK_DIR / name, as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
    )
    return client.execute()


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "fdsi_benchmark" / "data").exists(), reason="FDSI data not downloaded"
)
def test_dataset_tour_runs_on_the_raw_data():
    notebook = _execute("dataset.ipynb")
    images = [
        o for c in notebook.cells for o in c.get("outputs", []) if "image/png" in o.get("data", {})
    ]
    assert len(images) == 2


def test_report_runs_on_the_smoke_tables(monkeypatch):
    if not (load_spec(SMOKE_SPEC).tables_dir / "units.csv").exists():
        pytest.skip("no smoke tables: run the smoke spec's stages and collect first")
    monkeypatch.setenv("FDSI_SPEC", SMOKE_SPEC)

    notebook = _execute("report.ipynb")

    outputs = [o for c in notebook.cells for o in c.get("outputs", [])]
    assert not [o for o in outputs if o.get("output_type") == "error"]
    assert sum("image/png" in o.get("data", {}) for o in outputs) >= 8
