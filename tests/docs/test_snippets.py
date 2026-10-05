"""Run the scripts the how-to guides include (docs/snippets/), so their code keeps working.

workflow.py calibrates, adapts and scores the example recording, saving its calibration;
tune.py then runs tiny searches on it. Both need the fdsi_example-data archive.
"""

import runpy
import shutil
from pathlib import Path

import matplotlib
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SNIPPETS = REPO_ROOT / "docs" / "snippets"
OUTPUTS = REPO_ROOT / "data" / "fdsi_example" / "outputs" / "docs-example"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (REPO_ROOT / "data" / "fdsi_example" / "data").exists(),
        reason="Example data not downloaded (adapt-decomp-data get fdsi_example-data)",
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
    assert (OUTPUTS / "sources.png").exists() and (OUTPUTS / "spikes.png").exists()

    # The process_batch loop reproduces the online mode, batch for batch
    n_loop = (workflow["batch_idx"] + 1) * workflow["batch_size"]
    if workflow["batch"].shape[0] < workflow["batch_size"]:
        n_loop -= workflow["batch_size"]  # the partial last batch was not processed
    last = slice(n_loop - workflow["batch_size"], n_loop)
    assert (
        workflow["batch_spikes"].cpu() == workflow["streamed"].spikes[last].cpu()
    ).all()

    tune = runpy.run_path(str(SNIPPETS / "tune.py"), run_name="__main__")
    assert tune["result"].best_config is not None
    assert (OUTPUTS / "tuned_config.yaml").exists()
