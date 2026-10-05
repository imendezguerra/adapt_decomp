"""Run the scripts the how-to guides include (docs/snippets/), so their code keeps working.

workflow.py calibrates, adapts and scores one FDSI recording, saving its calibration;
tune.py then runs tiny searches on it. Both need the FDSI data.
"""

import runpy
import shutil
from pathlib import Path

import matplotlib
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SNIPPETS = REPO_ROOT / "docs" / "snippets"
OUTPUTS = REPO_ROOT / "data" / "fdsi_benchmark" / "outputs" / "docs-example"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (REPO_ROOT / "data" / "fdsi_benchmark" / "data").exists(),
        reason="FDSI data not downloaded",
    ),
]


def test_workflow_and_tune_snippets_run_from_the_repository_root(monkeypatch):
    matplotlib.use("Agg")
    monkeypatch.chdir(REPO_ROOT)
    shutil.rmtree(OUTPUTS, ignore_errors=True)

    workflow = runpy.run_path(str(SNIPPETS / "workflow.py"), run_name="__main__")
    assert workflow["calibration"].gt_matched_indices is not None
    assert workflow["adapted"].spikes.shape == workflow["fixed"].spikes.shape
    assert (OUTPUTS / "results.meta.yaml").exists()

    tune = runpy.run_path(str(SNIPPETS / "tune.py"), run_name="__main__")
    assert tune["result"].best_config is not None
    assert (OUTPUTS / "tuned_config.yaml").exists()
