"""The FDSI benchmark's published results (benchmarks/fdsi/results/<version>/) and config:
every version the notebooks read is complete and labelled."""

import pandas as pd
import pytest

from benchmarks.fdsi import pipeline, report

VERSIONS = sorted(p.name for p in report.RESULTS_ROOT.iterdir() if p.is_dir())
UNIT_COLUMNS = {
    "branch",
    "recording",
    "sub",
    "condition",
    "snr",
    "unit",
    "gt_unit",
    "roa_full",
    "sil",
}


@pytest.mark.parametrize("version", VERSIONS)
def test_every_version_has_labelled_per_unit_results(version):
    results = report.load_results(version)

    units = results["units"]
    assert UNIT_COLUMNS <= set(units.columns)
    assert set(units["branch"]) <= set(results["labels"])  # every config has a label
    assert units["config"].notna().all()
    assert units["roa_full"].between(0, 1).all()
    assert len(units.drop_duplicates(["branch", "recording", "unit"])) == len(units)
    configs = report.RESULTS_ROOT / version / "configs"
    assert {p.stem for p in configs.glob("*.yaml")} >= set(units["branch"]) - {"fixed"}


def test_the_config_labels_its_branches_and_names_existing_results():
    cfg = pipeline.load_config()
    assert cfg["version"] in VERSIONS and cfg["previous"] in VERSIONS
    labels = report.load_results(cfg["version"])["labels"]
    assert labels == {
        branch: pipeline.config_label(cfg, branch) for branch in pipeline.branches(cfg)
    }
    trials = pd.read_csv(report.RESULTS_ROOT / cfg["version"] / "trials.csv")
    assert (trials.groupby("search")["chosen"].sum() == 1).all()  # one chosen trial per search
    assert set(trials["search"]) == set(cfg["searches"])
