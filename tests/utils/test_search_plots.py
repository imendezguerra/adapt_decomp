"""Tests for the static, table-driven search plots of adapt_decomp.utils.plots."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from adapt_decomp.adaptation.optimize import front_mask
from adapt_decomp.utils.plots import (
    plot_search_front,
    plot_search_landscape,
    plot_search_parameters,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _trials(n: int = 12, search: str = "pareto_sum", seed: int = 0) -> pd.DataFrame:
    """A trials table in study.trials_dataframe() format, as in the benchmark's searches.csv."""
    rng = np.random.default_rng(seed)
    wh, sv = rng.uniform(1e3, 1e5, n), rng.uniform(10, 100, n)
    return pd.DataFrame(
        {
            "search": search,
            "number": np.arange(n),
            "state": ["COMPLETE"] * (n - 1) + ["FAIL"],
            "values_wh_loss": wh,
            "values_sv_loss": sv,
            "params_wh_learning_rate": rng.uniform(1e-4, 5e-2, n),
            "params_sv_learning_rate": rng.uniform(1e-4, 1e-1, n),
            "params_centroid_momentum": rng.uniform(0, 0.95, n),
            "user_attrs_wh_loss": wh,
            "user_attrs_sv_loss": sv,
            "user_attrs_total_loss": wh + sv,
            "user_attrs_roa_mean_pooled": rng.uniform(20, 80, n),
        }
    )


def test_landscape_draws_one_panel_per_loss_with_the_chosen_and_best_trials():
    trials = _trials()
    axs = plot_search_landscape(trials, chosen_number=3, title="pareto_sum")

    assert len(axs) == 3
    assert [ax.get_xlabel() for ax in axs] == ["wh_loss", "sv_loss", "total_loss"]
    labels = [text.get_text() for text in axs[0].get_legend().get_texts()]
    assert labels == ["Trial", "Highest RoA", "Chosen"]
    assert len(axs[0].collections[0].get_offsets()) == 11  # the FAIL trial is left out
    assert axs[0].get_xscale() == "log"


def test_front_draws_the_front_line_through_the_front_trials():
    trials = _trials()
    complete = trials[trials["state"] == "COMPLETE"]
    mask = front_mask(complete[["values_wh_loss", "values_sv_loss"]].to_numpy())
    front = list(complete["number"][mask])

    ax = plot_search_front(trials, front, chosen_number=front[0], title="pareto_sum")

    line = ax.get_lines()[0]
    assert len(line.get_xdata()) == len(front)
    assert list(line.get_xdata()) == sorted(line.get_xdata())
    assert ax.get_xlabel() == "wh_loss" and ax.get_ylabel() == "sv_loss"


def test_parameters_draw_one_panel_per_param_with_log_learning_rates():
    trials = pd.concat([_trials(search="sv_mean"), _trials(search="roa", seed=1)])
    axs = plot_search_parameters(trials, hue="search")

    assert [ax.get_xlabel() for ax in axs] == [
        "wh_learning_rate",
        "sv_learning_rate",
        "centroid_momentum",
    ]
    assert [ax.get_xscale() for ax in axs] == ["log", "log", "linear"]
    assert axs[-1].get_legend() is not None


def test_parameters_take_an_explicit_subset():
    axs = plot_search_parameters(_trials(), params=["centroid_momentum"])
    assert len(axs) == 1
