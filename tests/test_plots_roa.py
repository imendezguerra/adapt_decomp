"""Smoke tests for adapt_decomp.utils.plots' new RoA/optimisation plotting functions.

These call matplotlib/seaborn/plotly directly (not pure functions), so per CLAUDE.md's
testing conventions they get a smoke test each -- a tiny synthetic input, assert no raise
and the expected return shape -- not deeper rendering assertions.
"""

import matplotlib
matplotlib.use('Agg')  # headless backend, no display needed for a smoke test

import numpy as np
import pandas as pd
import optuna
import pytest

from adapt_decomp.utils.plots import (
    plot_spikes,
    plot_sources,
    plot_roa_heatmap,
    plot_roa_boxplot,
    plot_phase_bar,
    plot_roa_summary_figure,
    plot_optimisation_landscape,
    plot_pareto_scatter,
    plot_optimisation_landscape_grid,
    plot_pareto_scatter_grid,
    _restrict_time_range,
)

CONDITIONS = ['cond_a', 'cond_b']
SNR_LEVELS = [30, 20]
CONFIGS = ['fixed', 'adapted']


def test_plot_spikes_smoke():
    """One row per named signal per unit; RoA/SIL lines appended to the ytick label."""
    n_samples, n_units = 100, 3
    spikes_adapted = np.zeros((n_samples, n_units), dtype=np.float32)
    spikes_gt = np.zeros((n_samples, n_units), dtype=np.float32)
    for unit in range(n_units):
        spikes_adapted[unit::10, unit] = 1.0
        spikes_gt[unit::10, unit] = 1.0
    timestamps = np.arange(n_samples) / 2048
    roa = np.array([0.9, 0.8, 0.7])
    sil = np.array([0.6, 0.5, 0.4])

    ax = plot_spikes({'Adapted': spikes_adapted, 'GT': spikes_gt}, timestamps,
                      roa={'Adapted': roa}, sil={'Adapted': sil})

    assert [line.get_label() for line in ax.get_lines() if line.get_label() in ('Adapted', 'GT')] == ['Adapted', 'GT']
    assert len(ax.get_yticks()) == n_units
    ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert ytick_labels[0] == 'MU 0\nAdapted RoA = 90.0%\nAdapted SIL = 0.60'


def test_plot_spikes_raises_on_unknown_metric_key():
    """roa/sil keys must be a subset of spikes' keys."""
    n_samples, n_units = 50, 2
    spikes = np.zeros((n_samples, n_units), dtype=np.float32)
    timestamps = np.arange(n_samples) / 2048

    with pytest.raises(ValueError):
        plot_spikes({'Adapted': spikes}, timestamps, roa={'Unknown': np.zeros(n_units)})


def test_plot_sources_smoke():
    """One row per unit; named sources overlaid, spikes marked, RoA line appended to the ylabel."""
    rng = np.random.default_rng(0)
    n_samples, n_units = 100, 2
    sources_adapted = rng.normal(size=(n_samples, n_units))
    sources_no_adapt = rng.normal(size=(n_samples, n_units))
    spikes_adapted = np.zeros((n_samples, n_units), dtype=np.float32)
    spikes_adapted[::10] = 1.0
    timestamps = np.arange(n_samples) / 2048
    roa = np.array([0.9, 0.8])

    axs = plot_sources(
        {'Adapted': sources_adapted, 'No adaptation': sources_no_adapt}, timestamps,
        spikes={'Adapted': spikes_adapted}, roa={'Adapted': roa},
    )

    assert len(axs) == n_units
    assert axs[0].get_ylabel() == 'MU 0\nAdapted RoA = 90.0%'


def test_plot_sources_raises_on_spikes_shape_mismatch():
    """spikes[name] must match sources[name]'s shape."""
    n_samples, n_units = 50, 2
    sources = np.zeros((n_samples, n_units))
    spikes = np.zeros((n_samples, n_units - 1))
    timestamps = np.arange(n_samples) / 2048

    with pytest.raises(ValueError):
        plot_sources({'A': sources}, timestamps, spikes={'A': spikes})


def test_restrict_time_range_slices_dicts_and_passes_through_none():
    """Slices timestamps and every signal dict by the same boolean mask; None dicts pass through."""
    timestamps = np.arange(10) / 10.0  # 0.0, 0.1, ..., 0.9
    a = {'x': np.arange(10)}

    ts, sliced_a, sliced_none = _restrict_time_range(timestamps, (0.3, 0.6), a, None)

    np.testing.assert_array_equal(ts, np.array([0.3, 0.4, 0.5, 0.6]))
    np.testing.assert_array_equal(sliced_a['x'], np.array([3, 4, 5, 6]))
    assert sliced_none is None


def test_restrict_time_range_no_op_when_time_range_is_none():
    timestamps = np.arange(10) / 10.0
    a = {'x': np.arange(10)}

    ts, sliced_a = _restrict_time_range(timestamps, None, a)

    assert ts is timestamps
    assert sliced_a is a


def test_plot_spikes_time_range_restricts_xaxis_and_data():
    """time_range slices the plotted spikes and pins the x-axis to that exact window."""
    n_samples, n_units = 200, 2
    spikes = np.zeros((n_samples, n_units), dtype=np.float32)
    spikes[::5, :] = 1.0
    timestamps = np.arange(n_samples) / 100.0  # 0.00 .. 1.99 s

    ax = plot_spikes({'Adapted': spikes}, timestamps, time_range=(0.5, 1.0))

    assert ax.get_xlim() == pytest.approx((0.5, 1.0))
    for line in ax.get_lines():
        xdata = line.get_xdata()
        if len(xdata):
            assert xdata.min() >= 0.5 and xdata.max() <= 1.0


def test_plot_sources_time_range_restricts_xaxis_and_data():
    """time_range slices sources/spikes/timestamps together and pins the x-axis."""
    rng = np.random.default_rng(0)
    n_samples, n_units = 200, 2
    sources = rng.normal(size=(n_samples, n_units))
    timestamps = np.arange(n_samples) / 100.0

    axs = plot_sources({'A': sources}, timestamps, time_range=(0.5, 1.0))

    assert axs[0].get_xlim() == pytest.approx((0.5, 1.0))
    for line in axs[0].get_lines():
        xdata = line.get_xdata()
        if len(xdata):
            assert xdata.min() >= 0.5 and xdata.max() <= 1.0


def _make_df_roa() -> pd.DataFrame:
    rows = []
    for cfg in CONFIGS:
        for cond in CONDITIONS:
            for snr in SNR_LEVELS:
                for unit in range(3):
                    rows.append({'config': cfg, 'condition': cond, 'snr': snr,
                                 'unit': unit, 'roa_pct': 50.0 + unit})
    return pd.DataFrame(rows)


def _make_df_phase() -> pd.DataFrame:
    rows = []
    for cfg in CONFIGS:
        for cond in CONDITIONS:
            for snr in SNR_LEVELS:
                for phase in ('first_iso', 'ramp', 'last_iso'):
                    rows.append({'config': cfg, 'condition': cond, 'snr': snr,
                                 'phase': phase, 'roa_pct': 60.0})
    return pd.DataFrame(rows)


def test_plot_roa_heatmap_smoke():
    """Renders one panel per config, all three agg modes, without raising."""
    df_roa = _make_df_roa()
    for agg, kwargs in [('mean', {}), ('median', {}), ('pct_ge_threshold', {'threshold': 50.0})]:
        axs = plot_roa_heatmap(df_roa, CONFIGS, CONDITIONS, SNR_LEVELS, agg=agg, **kwargs)
        assert len(axs) == len(CONFIGS)


def test_plot_roa_boxplot_smoke():
    axs = plot_roa_boxplot(_make_df_roa(), CONFIGS, CONDITIONS, SNR_LEVELS)
    assert len(axs) == len(CONFIGS)


def test_plot_phase_bar_smoke():
    phase_order = ['first_iso', 'ramp', 'last_iso']
    phase_labels = {'first_iso': 'First iso', 'ramp': 'Ramp', 'last_iso': 'Last iso'}
    axs = plot_phase_bar(_make_df_phase(), CONFIGS, CONDITIONS, SNR_LEVELS, phase_order, phase_labels)
    assert axs.shape == (len(CONFIGS), len(CONDITIONS))


def test_plot_roa_summary_figure_smoke():
    phase_order = ['first_iso', 'ramp', 'last_iso']
    phase_labels = {'first_iso': 'First iso', 'ramp': 'Ramp', 'last_iso': 'Last iso'}
    axs = plot_roa_summary_figure(_make_df_roa(), _make_df_phase(), ['adapted'], CONDITIONS,
                                   holdout_conditions=[], snr_levels=SNR_LEVELS,
                                   phase_order=phase_order, phase_labels=phase_labels)
    assert axs.shape == (1, 2)


def _make_single_objective_study() -> optuna.Study:
    study = optuna.create_study(direction='minimize')

    def objective(trial):
        wh = trial.suggest_float('wh_learning_rate', 1e-4, 1e-1, log=True)
        sv = trial.suggest_float('sv_learning_rate', 1e-4, 1e-1, log=True)
        trial.set_user_attr('wh_loss', wh)
        trial.set_user_attr('sv_loss', sv)
        trial.set_user_attr('total_loss', wh + sv)
        trial.set_user_attr('roa_mean_pooled', 100.0 * (1.0 - wh - sv))
        return wh + sv

    study.optimize(objective, n_trials=5)
    return study


def test_plot_optimisation_landscape_smoke():
    study = _make_single_objective_study()
    columns = plot_optimisation_landscape(study, 'lr_fixed', best_trial=study.best_trial)
    assert len(columns) == 3  # wh_loss, sv_loss, total_loss all present
    for traces in columns:
        assert len(traces) >= 1


def test_plot_optimisation_landscape_falls_back_to_value_when_no_loss_columns():
    """A study with no wh_loss/sv_loss/total_loss user_attrs falls back to the 'value' column."""
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: trial.suggest_float('x', 0, 1), n_trials=3)
    columns = plot_optimisation_landscape(study, 'lr_fixed')
    assert len(columns) == 1


def _make_pareto_study():
    study = optuna.create_study(directions=['minimize', 'minimize'])
    study.set_metric_names(['wh_loss', 'sv_loss'])  # trials_dataframe() -> values_wh_loss/values_sv_loss

    def objective(trial):
        wh = trial.suggest_float('wh_learning_rate', 1e-4, 1e-1, log=True)
        sv = trial.suggest_float('sv_learning_rate', 1e-4, 1e-1, log=True)
        trial.set_user_attr('roa_mean_pooled', 100.0 * (1.0 - wh - sv))
        return wh, sv

    study.optimize(objective, n_trials=8)
    return study


def test_plot_pareto_scatter_smoke():
    study = _make_pareto_study()
    front = study.best_trials
    selected = min(front, key=lambda t: t.values[1])
    traces = plot_pareto_scatter(study, front, selected, 'lr_fixed')
    assert len(traces) == 4  # trials, front line, max-RoA marker, selected marker


def test_plot_optimisation_landscape_grid_dedupes_marker_legend_across_rows():
    """Composing 2 rows (lr_fixed/lr_relerror) must not duplicate the 'max RoA'/'selected'
    legend entries across rows -- each should appear in the legend exactly once, even though
    every row's own trial-scatter series still gets its own entry."""
    studies = {'lr_fixed': _make_single_objective_study(), 'lr_relerror': _make_single_objective_study()}
    best_trials = {t: s.best_trial for t, s in studies.items()}

    fig = plot_optimisation_landscape_grid(studies, {t: t for t in studies}, best_trials)

    legend_names = [tr.name for tr in fig.data if tr.showlegend]
    assert legend_names.count('max RoA') == 1
    assert legend_names.count('selected') == 1
    # Each row's own trial series still gets its own legend entry.
    assert 'lr_fixed' in legend_names and 'lr_relerror' in legend_names


def test_plot_pareto_scatter_grid_smoke():
    studies = {'lr_fixed': _make_pareto_study(), 'lr_relerror': _make_pareto_study()}
    fronts = {t: s.best_trials for t, s in studies.items()}
    selected = {t: min(f, key=lambda tr: tr.values[1]) for t, f in fronts.items()}

    fig = plot_pareto_scatter_grid(studies, fronts, selected)

    assert len(fig.data) == 4 * len(studies)  # 4 traces per column (trials/front/max RoA/selected)
