#!/usr/bin/env python3
"""Generate experiment + comparison notebooks for the simulated dataset in data/example.

Usage (from repo root):
    python tutorials/_generate_sim_opt_notebooks.py

Produces in tutorials/:
    sim_01_kl_to_identity_iter1.ipynb
    sim_02_kl_to_identity_iterN.ipynb
    sim_03_kl_to_cal_iter1.ipynb
    sim_04_kl_to_cal_iterN.ipynb
    sim_05_kl_to_identity_iter1_coupling.ipynb
    sim_06_kl_to_identity_iterN_coupling.ipynb
    sim_07_comparison.ipynb

Each experiment notebook runs:
  1. No-adaptation baseline
  2. Single-objective Optuna optimisation (CMA-ES, minimises scalar wh+sv loss)
  3. Multi-objective Optuna optimisation (NSGA-II, 3-objective Pareto front)

All outputs are saved under data/example/opt_results/<exp_name>/ and can be
reloaded without re-running by setting RUN_OPTIMISATION = False in each notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import nbformat
except ImportError:
    raise SystemExit("nbformat is required: pip install nbformat")

TUTORIALS_DIR = Path(__file__).parent
ITER_N = 5  # max_iter_b for the iterN variants

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    dict(
        idx=1,
        name="kl_to_identity_iter1",
        wh_mode="kl_to_identity",
        max_iter_b=1,
        coupling=False,
        title="KL-to-identity whitening, single B iteration",
        desc=(
            "Whitening mode `kl_to_identity` drives online KL divergence towards the "
            "calibration value.  B is updated with a single fixed-point step per batch "
            "(`max_iter_b=1`).  V→B coupling correction is **disabled**."
        ),
    ),
    dict(
        idx=2,
        name="kl_to_identity_iterN",
        wh_mode="kl_to_identity",
        max_iter_b=ITER_N,
        coupling=False,
        title=f"KL-to-identity whitening, {ITER_N} B iterations",
        desc=(
            f"Same whitening mode as experiment 1 but B is updated with up to "
            f"{ITER_N} fixed-point iterations per batch (`max_iter_b={ITER_N}`), "
            f"giving the separation vectors more time to converge each batch."
        ),
    ),
    dict(
        idx=3,
        name="kl_to_cal_iter1",
        wh_mode="kl_to_cal",
        max_iter_b=1,
        coupling=False,
        title="KL-to-calibration whitening, single B iteration",
        desc=(
            "Whitening mode `kl_to_cal` drives online KL(Rz ‖ Rz_cal) to zero, "
            "targeting the exact calibration covariance rather than identity.  "
            "B uses a single fixed-point step."
        ),
    ),
    dict(
        idx=4,
        name="kl_to_cal_iterN",
        wh_mode="kl_to_cal",
        max_iter_b=ITER_N,
        coupling=False,
        title=f"KL-to-calibration whitening, {ITER_N} B iterations",
        desc=(
            f"Combines `kl_to_cal` whitening with {ITER_N} fixed-point B iterations per batch."
        ),
    ),
    dict(
        idx=5,
        name="kl_to_identity_iter1_coupling",
        wh_mode="kl_to_identity",
        max_iter_b=1,
        coupling=True,
        title="KL-to-identity whitening + V→B coupling, single B iteration",
        desc=(
            "Same as experiment 1 but with the V→B frame coupling correction enabled "
            "(`wh_b_coupling=True`).  The first-order frame change implied by each V step "
            "is propagated to B before spike detection."
        ),
    ),
    dict(
        idx=6,
        name="kl_to_identity_iterN_coupling",
        wh_mode="kl_to_identity",
        max_iter_b=ITER_N,
        coupling=True,
        title=f"KL-to-identity whitening + V→B coupling, {ITER_N} B iterations",
        desc=(
            f"Combines `kl_to_identity` whitening with V→B frame coupling correction "
            f"and {ITER_N} fixed-point B iterations per batch."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Cell builders
# ---------------------------------------------------------------------------

def md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source)


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source)


def _imports_cell() -> nbformat.NotebookNode:
    return code("""\
import sys
sys.path.insert(0, '..')

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from matplotlib.patches import Rectangle

from adapt_decomp.loaders import load_example
from adapt_decomp.adaptation import AdaptDecomp
from adapt_decomp.config import Config
from adapt_decomp.optimize import optimize_adapt_decomp, run_with_optimization, DEFAULT_PARAM_SPACE
from adapt_decomp.utils import (
    rate_of_agreement_paired, rate_of_agreement,
    get_coefficient_of_variation, get_discharge_rate,
    get_pulse_to_noise_ratio, get_silhouette_measure,
    find_reliable_units,
)

%load_ext autoreload
%autoreload 2

sns.set_theme(style='whitegrid', font_scale=1.1)
""")


def _run_control_cell(exp_name: str) -> nbformat.NotebookNode:
    return code(f"""\
# ── Run control ──────────────────────────────────────────────────────────────
# Set RUN_OPTIMISATION = False to reload saved results without re-running Optuna.
RUN_OPTIMISATION = True

EXP_NAME    = {exp_name!r}
RESULTS_DIR = Path('..', 'data', 'example', 'opt_results', EXP_NAME)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _to_json_safe(v):
    if isinstance(v, torch.Tensor): return v.item()
    if hasattr(v, 'item'): return v.item()
    return v
""")


def _load_data_cell() -> nbformat.NotebookNode:
    return code("""\
path_emg    = Path('..', 'data', 'example', 'data_sim.hdf5')
path_decomp = Path('..', 'data', 'example', 'decomp_sim.mat')

data = load_example(path_emg, path_decomp, False)

n_units      = data['sep_vectors'].shape[0]
fs           = data['fs']
timestamps   = data['timestamps'].numpy()
ext_fact     = data['ext_fact']
n_samples    = data['emg'].shape[0]
duration_s   = n_samples / fs

print(f'Units: {n_units}  |  Samples: {n_samples}  |  fs: {fs} Hz  |  Duration: {duration_s:.1f} s')
""")


def _plot_data_cell() -> nbformat.NotebookNode:
    return code("""\
fig, ax = plt.subplots(figsize=(12, 3.5), layout='constrained')
ax.add_patch(Rectangle((0, -1), 30, 3, facecolor='grey', alpha=0.3))
ax.text(15, 5, 'calibration\\nphase', dict(ha='center', va='center', fontsize=10))
ax.plot(data['timestamps'], data['angle_profile'], color='tab:blue')
ax.set(ylabel='Wrist angle (deg)', xlabel='Time (s)')
ax.tick_params(axis='y', labelcolor='tab:blue')
ax1 = ax.twinx()
ax1.plot(data['timestamps'], data['force_profile'], color='tab:orange')
ax1.set(ylabel='Force (% MVC)', yticks=[15])
ax1.tick_params(axis='y', labelcolor='tab:orange')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
    ax1.spines[spine].set_visible(False)
ax1.set(xticks=range(0, int(duration_s) + 1, 30))
plt.suptitle('Dataset overview: angle and force profiles')
plt.show()
""")


def _no_adapt_cell(wh_mode: str, coupling: bool) -> nbformat.NotebookNode:
    coupling_line = "cfg_no.wh_b_coupling  = True" if coupling else "cfg_no.wh_b_coupling  = False"
    return code(f"""\
_out_no_path = RESULTS_DIR / 'out_no.pt'

if RUN_OPTIMISATION:
    cfg_no = Config()
    cfg_no.ext_fact       = data['ext_fact']
    cfg_no.adapt_wh       = False
    cfg_no.adapt_sv       = False
    cfg_no.adapt_sd       = False
    cfg_no.log_loss       = True
    cfg_no.device         = 'cpu'
    cfg_no.wh_mode        = {wh_mode!r}
    cfg_no.contrast_scope = 'spike_based'
    {coupling_line}

    dec_no = AdaptDecomp(
        emg             = data['emg'].clone(),
        whitening       = data['whitening'].clone(),
        sep_vectors     = data['sep_vectors'].clone(),
        base_centroids  = data['base_centroids'].clone(),
        spike_centroids = data['spike_centroids'].clone(),
        emg_calib       = data['emg_calib'].clone(),
        ipts_calib      = data['ipts_calib'].clone(),
        spikes_calib    = data['spikes_calib'].clone(),
        preprocess      = data['preprocess'],
        config          = cfg_no,
    )
    out_no = dec_no.run()
    torch.save(out_no, _out_no_path)
    print(f'Saved → {{_out_no_path}}')
else:
    out_no = torch.load(_out_no_path, weights_only=False)
    print(f'Loaded from {{_out_no_path}}')
print('No-adaptation run complete.')
""")


def _base_config_cell(wh_mode: str, max_iter_b: int, coupling: bool) -> nbformat.NotebookNode:
    coupling_val = "True" if coupling else "False"
    return code(f"""\
# Shared configuration for both optimisation runs
base_config = {{
    'ext_fact':       data['ext_fact'],
    'adapt_wh':       True,
    'adapt_sv':       True,
    'adapt_sd':       True,
    'wh_mode':        {wh_mode!r},
    'contrast_scope': 'spike_based',
    'wh_b_coupling':  {coupling_val},
    'max_iter_b':     {max_iter_b},
    'log_loss':       True,
    'device':         'cpu',
    'trace_check':    True,
    'optim_loss':     'single_obj',
}}

# DEFAULT_PARAM_SPACE: narrowed bounds from root-cause analysis; centroid_momentum
# added as a third parameter to compensate for slow B adaptation in iter1 configs.
# batch_ms is excluded from the default search — use 100 ms (Config default).
# To also search batch_ms: {{'batch_ms': ('int', 50, 200), **DEFAULT_PARAM_SPACE}}
param_space = DEFAULT_PARAM_SPACE
""")


def _single_obj_cell(n_trials_so: int = 50) -> nbformat.NotebookNode:
    return code(f"""\
_cfg_so_path   = RESULTS_DIR / 'best_config_so.json'
_study_so_path = RESULTS_DIR / 'study_so_trials.csv'

if RUN_OPTIMISATION:
    best_config_so, study_so = optimize_adapt_decomp(
        emg             = data['emg'],
        whitening       = data['whitening'],
        sep_vectors     = data['sep_vectors'],
        base_centroids  = data['base_centroids'],
        spike_centroids = data['spike_centroids'],
        emg_calib       = data['emg_calib'],
        ipts_calib      = data['ipts_calib'],
        spikes_calib    = data['spikes_calib'],
        param_space     = param_space,
        base_config     = base_config,
        n_trials        = {n_trials_so},
        preprocess      = data['preprocess'],
    )
    with open(_cfg_so_path, 'w') as f:
        json.dump({{k: _to_json_safe(v) for k, v in best_config_so.items()}}, f, indent=2)
    pd.DataFrame([{{'trial': t.number, 'value': t.value, **t.params}}
                  for t in study_so.trials if t.value is not None]).to_csv(_study_so_path, index=False)
    print(f'Saved → {{_cfg_so_path}}')
else:
    with open(_cfg_so_path) as f:
        best_config_so = json.load(f)
    study_so = None
    print(f'Loaded from {{_cfg_so_path}}')

print('Best hyperparameters (single-obj):')
for k in param_space:
    print(f'  {{k}}: {{best_config_so[k]:.4e}}')
""")


def _so_history_cell() -> nbformat.NotebookNode:
    return code("""\
if study_so is None:
    print('Optimisation history not available (loaded from file).')
else:
    values      = [t.value for t in study_so.trials if t.value is not None]
    best_so_far = np.minimum.accumulate(values)

    # Separate log-scale params (delta_v, delta_b) from linear-scale (centroid_momentum)
    log_params    = [k for k, v in param_space.items() if v[0] == 'log_float']
    linear_params = [k for k, v in param_space.items() if v[0] != 'log_float']
    n_panels = 1 + len(log_params) + len(linear_params)
    fig, axs = plt.subplots(1, n_panels, figsize=(5 * n_panels, 3.5), layout='constrained')

    axs[0].scatter(range(len(values)), values, s=20, alpha=0.6, color='tab:grey', label='Trial loss')
    axs[0].plot(range(len(values)), best_so_far, color='tab:green', linewidth=2, label='Best so far')
    axs[0].set(xlabel='Trial', ylabel='wh + sv loss',
               title='Single-objective optimisation history')
    axs[0].legend()

    for ax, name in zip(axs[1:], log_params + linear_params):
        is_log = param_space[name][0] == 'log_float'
        vals = [t.params[name] for t in study_so.trials if name in t.params and t.value is not None]
        cols = [t.value        for t in study_so.trials if name in t.params and t.value is not None]
        sc = ax.scatter(vals, cols, c=range(len(vals)), cmap='viridis', s=25, alpha=0.8)
        if is_log:
            ax.set_xscale('log')
        ax.set(xlabel=name, ylabel='Trial loss',
               title=f'Loss vs {name}\\n(colour = trial order)')
        plt.colorbar(sc, ax=ax, label='Trial index')

    plt.show()
""")


def _so_run_cell() -> nbformat.NotebookNode:
    return code("""\
_out_so_path = RESULTS_DIR / 'out_so.pt'

if RUN_OPTIMISATION:
    cfg_so = Config()
    for k, v in best_config_so.items():
        setattr(cfg_so, k, v)

    dec_so = AdaptDecomp(
        emg             = data['emg'].clone(),
        whitening       = data['whitening'].clone(),
        sep_vectors     = data['sep_vectors'].clone(),
        base_centroids  = data['base_centroids'].clone(),
        spike_centroids = data['spike_centroids'].clone(),
        emg_calib       = data['emg_calib'].clone(),
        ipts_calib      = data['ipts_calib'].clone(),
        spikes_calib    = data['spikes_calib'].clone(),
        preprocess      = data['preprocess'],
        config          = cfg_so,
    )
    out_so = dec_so.run()
    torch.save(out_so, _out_so_path)
    print(f'Saved → {_out_so_path}')
else:
    out_so = torch.load(_out_so_path, weights_only=False)
    print(f'Loaded from {_out_so_path}')
print('Single-obj adaptation run complete.')
""")


def _multi_obj_cell(n_trials_mo: int = 50) -> nbformat.NotebookNode:
    return code(f"""\
_cfg_mo_path = RESULTS_DIR / 'best_config_mo.json'
_out_mo_path = RESULTS_DIR / 'out_mo.pt'

base_config_mo = {{**base_config, 'optim_loss': 'multi_obj'}}

if RUN_OPTIMISATION:
    out_mo, best_config_mo, study_mo = run_with_optimization(
        emg             = data['emg'],
        whitening       = data['whitening'],
        sep_vectors     = data['sep_vectors'],
        base_centroids  = data['base_centroids'],
        spike_centroids = data['spike_centroids'],
        emg_calib       = data['emg_calib'],
        ipts_calib      = data['ipts_calib'],
        spikes_calib    = data['spikes_calib'],
        param_space     = param_space,
        base_config     = base_config_mo,
        n_trials        = {n_trials_mo},
        preprocess      = data['preprocess'],
        optim_mode      = 'multiobjective',
    )
    with open(_cfg_mo_path, 'w') as f:
        json.dump({{k: _to_json_safe(v) for k, v in best_config_mo.items()}}, f, indent=2)
    torch.save(out_mo, _out_mo_path)
    print(f'Saved → {{_cfg_mo_path}}, {{_out_mo_path}}')
    print(f'Pareto front: {{len(study_mo.best_trials)}} trials')
else:
    with open(_cfg_mo_path) as f:
        best_config_mo = json.load(f)
    out_mo   = torch.load(_out_mo_path, weights_only=False)
    study_mo = None
    print(f'Loaded from {{_cfg_mo_path}}, {{_out_mo_path}}')

print('Selected config (multi-objective):')
for k in param_space:
    print(f'  {{k}}: {{best_config_mo[k]:.4e}}')
""")


def _pareto_cell() -> nbformat.NotebookNode:
    return code("""\
if study_mo is None:
    print('Pareto front not available (loaded from file).')
else:
    pareto = study_mo.best_trials
    wh_v   = [t.values[0] for t in pareto]
    sv_v   = [t.values[1] for t in pareto]
    ct_v   = [t.values[2] for t in pareto]
    dv_v   = [t.params.get('max_rel_delta_v', float('nan')) for t in pareto]
    db_v   = [t.params.get('max_rel_delta_b', float('nan')) for t in pareto]
    sel    = int(np.argmin([w + s + c for w, s, c in zip(wh_v, sv_v, ct_v)]))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    for ax, (xa, ya, c_vals, clbl, lbl) in zip(axs, [
        (wh_v, sv_v, dv_v, 'log₁₀(Δv)', 'wh_loss vs sv_loss'),
        (wh_v, ct_v, dv_v, 'log₁₀(Δv)', 'wh_loss vs centroid_loss'),
        (sv_v, ct_v, db_v, 'log₁₀(Δb)', 'sv_loss vs centroid_loss'),
    ]):
        sc = ax.scatter(xa, ya, c=np.log10(c_vals), cmap='viridis', s=60, zorder=3)
        ax.scatter(xa[sel], ya[sel], marker='*', s=250, color='red', zorder=4, label='selected')
        plt.colorbar(sc, ax=ax, label=clbl)
        ax.set(xlabel=lbl.split(' vs ')[0], ylabel=lbl.split(' vs ')[1], title=lbl)
        ax.legend(fontsize=8)
    plt.suptitle('Pareto front — 3-objective optimisation')
    plt.show()
    print(f'Pareto front: {len(pareto)} non-dominated trials')
    print(f'Selected: wh={wh_v[sel]:.4f}  sv={sv_v[sel]:.4f}  centroid={ct_v[sel]:.4f}')
""")


def _cond_setup_cell() -> nbformat.NotebookNode:
    return code("""\
cond_labels  = ['No adaptation', 'Single-obj', 'Multi-objective']
cond_colors  = {
    'No adaptation': 'tab:orange',
    'Single-obj':    'tab:blue',
    'Multi-objective': 'tab:green',
}
cond_outputs = {
    'No adaptation': out_no,
    'Single-obj':    out_so,
    'Multi-objective': out_mo,
}
""")


def _timing_cell() -> nbformat.NotebookNode:
    return code("""\
for label in cond_labels:
    out = cond_outputs[label]
    print(label)
    print('-' * len(label))
    for key in ['wh_time_ms', 'sv_time_ms', 'sd_time_ms', 'total_time_ms']:
        print(f'  {key:>15}: {out[key].mean():.3f} ± {out[key].std():.3f} ms')
    print()
""")


def _loss_cells() -> list[nbformat.NotebookNode]:
    return [
        code("""\
# Whitening loss
fig, ax = plt.subplots(figsize=(12, 3), layout='constrained')
for label in cond_labels:
    ax.plot(cond_outputs[label]['wh_loss'][10:-1], label=label, color=cond_colors[label])
ax.set(xlabel='Batch', ylabel='KL error²', title='Whitening loss')
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
plt.show()
"""),
        code("""\
# Contrast (sv) loss per unit
fig, axs = plt.subplots(n_units, 1, figsize=(12, n_units), layout='constrained', sharex=True)
for u in range(n_units):
    for label in cond_labels:
        axs[u].plot(cond_outputs[label]['sv_loss'][:, u], label=label, color=cond_colors[label])
    axs[u].set(ylabel=f'U{u}')
    if u == 0:
        axs[u].legend(loc='upper right', bbox_to_anchor=(1.25, 1))
axs[-1].set(xlabel='Batch')
plt.suptitle('Contrast (sv) loss per unit')
plt.show()
"""),
        code("""\
# RMS centroid deviation per unit
fig, axs = plt.subplots(n_units, 1, figsize=(12, n_units), layout='constrained', sharex=True)
for u in range(n_units):
    for label in cond_labels:
        out = cond_outputs[label]
        if 'centroid_loss' in out:
            axs[u].plot(
                np.sqrt(out['centroid_loss'][:, u].numpy()),
                label=label, color=cond_colors[label], alpha=0.8,
            )
    axs[u].set(ylabel=f'U{u}')
    if u == 0:
        axs[u].legend(loc='upper right', bbox_to_anchor=(1.25, 1))
axs[-1].set(xlabel='Batch')
plt.suptitle('RMS centroid deviation  √(centroid_loss)  per unit')
plt.show()
"""),
    ]


def _ipts_cell() -> nbformat.NotebookNode:
    return code("""\
t0, t1 = 0, int(duration_s)
idxs   = np.arange(int(t0 * fs), int(t1 * fs))

fig, axs = plt.subplots(n_units, 1, figsize=(12, n_units), layout='constrained', sharex=True)
for u in range(n_units):
    for label in cond_labels:
        out     = cond_outputs[label]
        ipts_sq = out['ipts'][idxs, u].numpy() ** 2
        axs[u].plot(data['timestamps'][idxs], ipts_sq, color=cond_colors[label], alpha=0.5)
        smask = out['spikes'][idxs, u].to(bool)
        axs[u].plot(
            data['timestamps'][idxs][smask], ipts_sq[smask],
            linestyle='None', marker='.', color=cond_colors[label],
            label=label if u == 0 else None,
        )
    axs[u].set(ylabel=f'U{u + 1}')
axs[0].legend(loc='upper right', bbox_to_anchor=(1.25, 1))
axs[-1].set(xlabel='Time (s)')
plt.suptitle(f'Squared IPTs with spike times  [{t0}–{t1} s]')
plt.show()
""")


def _metrics_cell() -> nbformat.NotebookNode:
    return code("""\
rows = []
for label in cond_labels:
    out       = cond_outputs[label]
    spikes_np = out['spikes'].numpy().astype(float)
    ipts_np   = out['ipts'].numpy()

    sil  = get_silhouette_measure(spikes_np, ipts_np, ext_fact)
    pnr  = get_pulse_to_noise_ratio(spikes_np, ipts_np, ext_fact)
    dr   = get_discharge_rate(spikes_np, timestamps, discard_isi=None)
    cov  = get_coefficient_of_variation(spikes_np, timestamps, discard_isi=None)
    good = find_reliable_units(dr, cov, sil, pnr,
                               dr_low_thr=5, dr_upp_thr=35, cov_thr=0.35,
                               sil_thr=0.9, pnr_thr=30)
    for u in range(spikes_np.shape[1]):
        rows.append({'unit': u, 'condition': label, 'good': bool(good[u]),
                     'SIL': sil[u], 'PNR (dB)': pnr[u], 'DR (pps)': dr[u], 'CoV (%)': cov[u]})

df_all  = pd.DataFrame(rows)
n_total = df_all['unit'].nunique()

print(df_all.groupby('condition')[['SIL', 'PNR (dB)', 'DR (pps)', 'CoV (%)']]
      .agg(['median', 'mean']).round(2))
print()
for cond, grp in df_all.groupby('condition'):
    print(f'{cond}: {grp["good"].sum()} / {n_total} good units')
""")


def _metrics_plot_cell() -> nbformat.NotebookNode:
    return code("""\
order   = cond_labels
palette = cond_colors

df_all['condition'] = pd.Categorical(df_all['condition'], categories=order, ordered=True)

fig = plt.figure(figsize=(16, 7), layout='constrained')
gs  = fig.add_gridspec(2, 4)
ax_sil = fig.add_subplot(gs[0, 0])
ax_pnr = fig.add_subplot(gs[0, 1])
ax_dr  = fig.add_subplot(gs[0, 2])
ax_cov = fig.add_subplot(gs[0, 3])
ax_bar = fig.add_subplot(gs[1, 1:3])

for ax, col, ylabel in [
    (ax_sil, 'SIL',      'Silhouette measure'),
    (ax_pnr, 'PNR (dB)', 'PNR (dB)'),
    (ax_dr,  'DR (pps)', 'Discharge rate (pps)'),
    (ax_cov, 'CoV (%)',  'CoV (%)'),
]:
    sns.boxplot(data=df_all, x='condition', y=col, order=order, palette=palette,
                width=0.5, flierprops=dict(marker='o', markersize=4, alpha=0.5), ax=ax)
    ax.set(xlabel='', ylabel=ylabel, title=ylabel)
    ax.tick_params(axis='x', rotation=15)

good_counts = df_all.groupby('condition', observed=True)['good'].sum().reindex(order)
bars = ax_bar.bar(order, good_counts.values,
                  color=[palette[c] for c in order], width=0.4, zorder=3)
ax_bar.axhline(n_total, color='k', linestyle='--', linewidth=1,
               label=f'Total units (n = {n_total})')
ax_bar.set(ylabel='Good units', ylim=(0, n_total + 2),
           title='Good units  (DR ∈ [5,35] | CoV ≤ 35% | SIL ≥ 0.9 | PNR ≥ 30 dB)')
ax_bar.legend(fontsize=9)
ax_bar.tick_params(axis='x', rotation=15)
for bar, count in zip(bars, good_counts.values):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, count + 0.2,
                f'{int(count)} / {n_total}', ha='center', va='bottom', fontsize=10)
plt.suptitle('Decomposition quality metrics', fontsize=13)
plt.show()
""")


def _roa_cells() -> list[nbformat.NotebookNode]:
    return [
        code("""\
cal_idx = np.arange(0, int(30 * fs))

roa_no_cal,  pair_no,  _ = rate_of_agreement(
    data['spikes_gt'].numpy()[cal_idx],
    out_no['spikes'].numpy()[cal_idx],
    fs=fs,
)
roa_so_cal, pair_so, _ = rate_of_agreement(
    data['spikes_gt'].numpy()[cal_idx],
    out_so['spikes'].numpy()[cal_idx],
    fs=fs,
)
roa_mo_cal, pair_mo, _ = rate_of_agreement(
    data['spikes_gt'].numpy()[cal_idx],
    out_mo['spikes'].numpy()[cal_idx],
    fs=fs,
)

print(f'RoA calibration | No adaptation:  {roa_no_cal.mean()*100:.2f} ± {roa_no_cal.std()*100:.2f}%')
print(f'RoA calibration | Single-obj:     {roa_so_cal.mean()*100:.2f} ± {roa_so_cal.std()*100:.2f}%')
print(f'RoA calibration | Multi-obj:      {roa_mo_cal.mean()*100:.2f} ± {roa_mo_cal.std()*100:.2f}%')

# Use no-adapt pairing as common reference
pair_arr   = np.array(pair_no)
sim_spikes = data['spikes_gt'].numpy()[:, pair_arr[:, 0]]
"""),
        code("""\
roa_no_full, _, _ = rate_of_agreement_paired(
    sim_spikes, out_no['spikes'].numpy(), fs=fs, tol_spike_ms=2)
roa_so_full, _, _ = rate_of_agreement_paired(
    sim_spikes, out_so['spikes'].numpy(), fs=fs, tol_spike_ms=2)
roa_mo_full, _, _ = rate_of_agreement_paired(
    sim_spikes, out_mo['spikes'].numpy(), fs=fs, tol_spike_ms=2)

unit_ids = pair_arr[:, 1]
n_pairs  = len(unit_ids)
x        = np.arange(n_pairs)

print(f'RoA full | No adaptation:  {roa_no_full.mean()*100:.2f} ± {roa_no_full.std()*100:.2f}%  '
      f'(med: {np.median(roa_no_full)*100:.2f}%)')
print(f'RoA full | Single-obj:     {roa_so_full.mean()*100:.2f} ± {roa_so_full.std()*100:.2f}%  '
      f'(med: {np.median(roa_so_full)*100:.2f}%)')
print(f'RoA full | Multi-obj:      {roa_mo_full.mean()*100:.2f} ± {roa_mo_full.std()*100:.2f}%  '
      f'(med: {np.median(roa_mo_full)*100:.2f}%)')

good_lookup = {
    label: df_all[df_all['condition'] == label].set_index('unit')['good']
    for label in cond_labels
}

fig, (ax_roa, ax_good) = plt.subplots(
    2, 1, figsize=(12, 5.5), layout='constrained',
    gridspec_kw={'height_ratios': [3, 1]},
)
ax_roa.plot(x, roa_no_full * 100, marker='o', label='No adaptation',  color=cond_colors['No adaptation'])
ax_roa.plot(x, roa_so_full * 100, marker='o', label='Single-obj',     color=cond_colors['Single-obj'])
ax_roa.plot(x, roa_mo_full * 100, marker='o', label='Multi-obj',      color=cond_colors['Multi-objective'])
ax_roa.set(ylabel='Rate of agreement (%)', xlim=(-0.5, n_pairs - 0.5))
ax_roa.legend(loc='upper right', bbox_to_anchor=(1.22, 1))

y_pos = {'No adaptation': 2, 'Single-obj': 1, 'Multi-objective': 0}
for label in cond_labels:
    for i, uid in enumerate(unit_ids):
        is_good = bool(good_lookup[label].get(uid, False))
        ax_good.scatter(i, y_pos[label],
                        color=cond_colors[label],
                        marker='o' if is_good else 'X',
                        s=55, zorder=3)

ax_good.set_yticks(list(y_pos.values()))
ax_good.set_yticklabels(list(y_pos.keys()), fontsize=8)
ax_good.set(xlabel='Unit (matched)', xlim=(-0.5, n_pairs - 0.5),
            title='Good (●) / Bad (✗) unit classification',
            ylim=(-0.5, len(y_pos) - 0.5))
ax_good.grid(axis='x', alpha=0.3)
plt.suptitle('Rate of agreement with ground truth (full recording)')
plt.show()
"""),
    ]


def _save_roa_cell() -> nbformat.NotebookNode:
    return code("""\
# Save per-unit RoA for the comparison notebook
roa_summary = {
    'unit_ids':    unit_ids.tolist(),
    'roa_no':      roa_no_full.tolist(),
    'roa_so':      roa_so_full.tolist(),
    'roa_mo':      roa_mo_full.tolist(),
}
with open(RESULTS_DIR / 'roa_summary.json', 'w') as f:
    json.dump(roa_summary, f, indent=2)

# Save quality metrics DataFrame
df_all.to_csv(RESULTS_DIR / 'quality_metrics.csv', index=False)
print(f'Saved RoA summary and quality metrics → {RESULTS_DIR}')
""")


# ---------------------------------------------------------------------------
# Assemble experiment notebook
# ---------------------------------------------------------------------------

def make_experiment_nb(exp: dict, n_trials_so: int = 30, n_trials_mo: int = 50) -> nbformat.NotebookNode:
    name      = exp['name']
    wh_mode   = exp['wh_mode']
    max_iter_b = exp['max_iter_b']
    coupling  = exp['coupling']
    title     = exp['title']
    desc      = exp['desc']

    coupling_str = "enabled" if coupling else "disabled"

    cells = [
        md(
            f"# Experiment {exp['idx']:02d}: {title}\n\n"
            f"{desc}\n\n"
            f"**Configuration summary**\n"
            f"- `wh_mode = '{wh_mode}'`\n"
            f"- `max_iter_b = {max_iter_b}`\n"
            f"- `wh_b_coupling = {coupling}` ({coupling_str})\n\n"
            "**Conditions compared**\n"
            "| Condition | Optimiser | Objective |\n"
            "|---|---|---|\n"
            "| No adaptation | — | — |\n"
            "| Single-obj | Optuna CMA-ES | wh_loss + sv_loss (scalar) |\n"
            "| Multi-objective | NSGA-II | wh_loss, sv_loss, centroid_loss (Pareto) |\n"
        ),
        _imports_cell(),
        _run_control_cell(name),
        md("## 1. Load data"),
        _load_data_cell(),
        _plot_data_cell(),
        md("## 2. No-adaptation baseline\n\n"
           "All adaptation flags off. The calibration V and B are applied unchanged."),
        _no_adapt_cell(wh_mode, coupling),
        md("## 3. Single-objective optimisation\n\n"
           "Optuna CMA-ES minimises `wh_loss + sv_loss` (`optim_loss='single_obj'`).  "
           "centroid_loss is excluded from the objective (0–2% of total, mild anti-signal).  "
           "Hyperparameters searched: `max_rel_delta_v`, `max_rel_delta_b`, "
           "`centroid_momentum` (see `DEFAULT_PARAM_SPACE`)."),
        _base_config_cell(wh_mode, max_iter_b, coupling),
        _single_obj_cell(n_trials_so),
        _so_history_cell(),
        md("Apply the best single-obj config and run the full decomposition."),
        _so_run_cell(),
        md("## 4. Multi-objective optimisation (NSGA-II)\n\n"
           "NSGA-II treats wh_loss, sv_loss, and centroid_loss as three independent objectives "
           "and returns a Pareto front.  The operating point is the Pareto trial with minimum "
           "sum of objectives."),
        _multi_obj_cell(n_trials_mo),
        _pareto_cell(),
        md("## 5. Execution time per batch"),
        _cond_setup_cell(),
        _timing_cell(),
        md("## 6. Loss trajectories\n\n"
           "- **wh_loss**: KL divergence between online and calibration whitened covariance\n"
           "- **sv_loss**: squared contrast error per source\n"
           "- **centroid_loss**: squared centroid deviation per source (logged for monitoring, "
           "not included in the single-obj optimisation objective)\n\n"
           "Values near zero mean the online statistics match calibration."),
        *_loss_cells(),
        md("## 7. IPTs and spike trains\n\nSquared IPTs with detected spike times overlaid."),
        _ipts_cell(),
        md("## 8. Decomposition quality metrics\n\n"
           "A unit is **good** if SIL ≥ 0.9, DR ∈ [5, 35] pps, CoV ≤ 35%, PNR ≥ 30 dB."),
        _metrics_cell(),
        _metrics_plot_cell(),
        md("## 9. Rate of agreement with ground truth\n\n"
           "Ground truth spike trains (`spikes_gt`) are available in the simulated dataset.  "
           "Unit matching is performed on the calibration segment (first 30 s) using the "
           "no-adaptation pairing as the common reference."),
        *_roa_cells(),
        _save_roa_cell(),
    ]

    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    }
    nb.metadata['language_info'] = {
        'name': 'python',
        'version': '3.12.0',
    }
    return nb


# ---------------------------------------------------------------------------
# Comparison notebook
# ---------------------------------------------------------------------------

def make_comparison_nb(experiments: list[dict]) -> nbformat.NotebookNode:
    exp_names   = [e['name'] for e in experiments]
    exp_indices = [e['idx']  for e in experiments]

    # Build a multi-line string listing all experiments
    exp_table_rows = "\n".join(
        f"| {e['idx']:02d} | `{e['name']}` | `{e['wh_mode']}` | {e['max_iter_b']} | {e['coupling']} |"
        for e in experiments
    )

    cells = [
        md(
            "# Cross-configuration comparison: simulated dataset\n\n"
            "Aggregates results from all 6 single-condition experiments and compares:\n"
            "- Decomposition quality metrics (SIL, PNR, DR, CoV, good-unit count)\n"
            "- Rate of agreement with ground truth (full recording)\n"
            "- Whitening loss trajectories\n\n"
            "**Experiments**\n\n"
            "| # | Name | wh_mode | max_iter_b | coupling |\n"
            "|---|------|---------|------------|----------|\n"
            f"{exp_table_rows}\n\n"
            "> Run each experiment notebook first (or set `RUN_OPTIMISATION=True`) "
            "to generate the saved results this notebook loads."
        ),
        code("""\
import sys
sys.path.insert(0, '..')

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

from adapt_decomp.utils import (
    rate_of_agreement_paired, rate_of_agreement,
    get_coefficient_of_variation, get_discharge_rate,
    get_pulse_to_noise_ratio, get_silhouette_measure,
    find_reliable_units,
)
from adapt_decomp.loaders import load_example

%load_ext autoreload
%autoreload 2

sns.set_theme(style='whitegrid', font_scale=1.1)
"""),
        code("""\
RESULTS_ROOT = Path('..', 'data', 'example', 'opt_results')

EXPERIMENTS = [
""" + "".join(
            f"    dict(name={e['name']!r}, idx={e['idx']}, wh_mode={e['wh_mode']!r}, "
            f"max_iter_b={e['max_iter_b']}, coupling={e['coupling']}),\n"
            for e in experiments
        ) + """]
"""),
        md("## 1. Load data (needed for spikes_gt and metadata)"),
        code("""\
path_emg    = Path('..', 'data', 'example', 'data_sim.hdf5')
path_decomp = Path('..', 'data', 'example', 'decomp_sim.mat')

data     = load_example(path_emg, path_decomp, False)
n_units  = data['sep_vectors'].shape[0]
fs       = data['fs']
ext_fact = data['ext_fact']
timestamps = data['timestamps'].numpy()
print(f'Units: {n_units}  |  fs: {fs} Hz')
"""),
        md("## 2. Load saved results"),
        code("""\
def _short(name, suffix):
    \"\"\"Create a human-readable condition label.\"\"\"
    return f'{name}/{suffix}'

all_conds   = []  # list of (exp_name, suffix, out_dict)
roa_records = []  # per-unit RoA records

for exp in EXPERIMENTS:
    name      = exp['name']
    res_dir   = RESULTS_ROOT / name

    # ── Load outputs ─────────────────────────────────────────────────────────
    out_no = torch.load(res_dir / 'out_no.pt', weights_only=False)
    out_so = torch.load(res_dir / 'out_so.pt', weights_only=False)
    out_mo = torch.load(res_dir / 'out_mo.pt', weights_only=False)

    for suffix, out in [('no-adapt', out_no), ('single-obj', out_so), ('multi-obj', out_mo)]:
        all_conds.append((_short(name, suffix), out, name, suffix))

    # ── Load RoA summary ────────────────────────────────────────────────────
    with open(res_dir / 'roa_summary.json') as f:
        roa = json.load(f)
    for i, uid in enumerate(roa['unit_ids']):
        roa_records.append({'exp': name, 'unit': uid,
                            'roa_no': roa['roa_no'][i],
                            'roa_so': roa['roa_so'][i],
                            'roa_mo': roa['roa_mo'][i]})

df_roa = pd.DataFrame(roa_records)
print(f'Loaded {len(all_conds)} condition outputs across {len(EXPERIMENTS)} experiments.')
"""),
        md("## 3. Quality metrics per experiment + condition"),
        code("""\
rows = []
for label, out, exp_name, suffix in all_conds:
    spikes_np = out['spikes'].numpy().astype(float)
    ipts_np   = out['ipts'].numpy()

    sil  = get_silhouette_measure(spikes_np, ipts_np, ext_fact)
    pnr  = get_pulse_to_noise_ratio(spikes_np, ipts_np, ext_fact)
    dr   = get_discharge_rate(spikes_np, timestamps, discard_isi=None)
    cov  = get_coefficient_of_variation(spikes_np, timestamps, discard_isi=None)
    good = find_reliable_units(dr, cov, sil, pnr,
                               dr_low_thr=5, dr_upp_thr=35, cov_thr=0.35,
                               sil_thr=0.9, pnr_thr=30)
    for u in range(spikes_np.shape[1]):
        rows.append({'exp': exp_name, 'condition': suffix, 'label': label,
                     'unit': u, 'good': bool(good[u]),
                     'SIL': sil[u], 'PNR (dB)': pnr[u],
                     'DR (pps)': dr[u], 'CoV (%)': cov[u]})

df_metrics = pd.DataFrame(rows)
n_total    = df_metrics['unit'].nunique()

summary = (df_metrics.groupby(['exp', 'condition'])[['SIL', 'PNR (dB)', 'DR (pps)', 'CoV (%)']]
           .median().round(3))
print(summary.to_string())
"""),
        code("""\
# Good-unit count per experiment × condition
good_counts = (df_metrics.groupby(['exp', 'condition'], sort=False)['good']
               .sum().unstack('condition')[['no-adapt', 'single-obj', 'multi-obj']])
print('\\nGood units per experiment (out of', n_total, '):')
print(good_counts.to_string())
"""),
        code("""\
# Bar chart: good units across experiments
fig, ax = plt.subplots(figsize=(14, 4.5), layout='constrained')

x        = np.arange(len(EXPERIMENTS))
w        = 0.25
suffixes = ['no-adapt', 'single-obj', 'multi-obj']
colors   = {'no-adapt': 'tab:orange', 'single-obj': 'tab:blue', 'multi-obj': 'tab:green'}

for i, suf in enumerate(suffixes):
    vals = [
        df_metrics[(df_metrics['exp'] == e['name']) & (df_metrics['condition'] == suf)]['good'].sum()
        for e in EXPERIMENTS
    ]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=suf, color=colors[suf], zorder=3)

ax.axhline(n_total, color='k', linestyle='--', linewidth=1, label=f'Total (n={n_total})')
ax.set(xticks=x,
       xticklabels=[e['name'].replace('_', '\\n') for e in EXPERIMENTS],
       ylabel='Good units', ylim=(0, n_total + 2),
       title='Good units per experiment  (DR ∈ [5,35] | CoV ≤ 35% | SIL ≥ 0.9 | PNR ≥ 30 dB)')
ax.legend(loc='upper right')
ax.tick_params(axis='x', labelsize=8)
plt.show()
"""),
        code("""\
# SIL and PNR boxplots: single-obj and multi-obj across experiments
fig, axs = plt.subplots(1, 2, figsize=(14, 4), layout='constrained')

for ax, col, ylabel in [(axs[0], 'SIL', 'Silhouette measure'),
                         (axs[1], 'PNR (dB)', 'PNR (dB)')]:
    df_adapt = df_metrics[df_metrics['condition'].isin(['single-obj', 'multi-obj'])]
    sns.boxplot(data=df_adapt, x='exp', y=col, hue='condition',
                palette={'single-obj': 'tab:blue', 'multi-obj': 'tab:green'},
                width=0.5, ax=ax)
    ax.set(xlabel='', ylabel=ylabel, title=ylabel)
    ax.tick_params(axis='x', rotation=25, labelsize=8)

plt.suptitle('Quality metrics: adapted conditions across experiments', fontsize=13)
plt.show()
"""),
        md("## 4. Rate of agreement summary"),
        code("""\
# Mean RoA per experiment × condition
roa_summary = df_roa.groupby('exp').agg(
    roa_no_mean=('roa_no', lambda x: x.mean() * 100),
    roa_so_mean=('roa_so', lambda x: x.mean() * 100),
    roa_mo_mean=('roa_mo', lambda x: x.mean() * 100),
    roa_no_std =('roa_no', lambda x: x.std()  * 100),
    roa_so_std =('roa_so', lambda x: x.std()  * 100),
    roa_mo_std =('roa_mo', lambda x: x.std()  * 100),
).round(2)
print(roa_summary[['roa_no_mean', 'roa_so_mean', 'roa_mo_mean']].to_string())
"""),
        code("""\
# Per-unit RoA: single-obj and multi-obj
fig, axs = plt.subplots(len(EXPERIMENTS), 1,
                         figsize=(12, 2.5 * len(EXPERIMENTS)),
                         layout='constrained', sharex=True)

for ax, exp in zip(axs, EXPERIMENTS):
    sub = df_roa[df_roa['exp'] == exp['name']].sort_values('unit')
    x_u = np.arange(len(sub))
    ax.plot(x_u, sub['roa_no'].values * 100, 'o-', color='tab:orange', label='no-adapt', alpha=0.7)
    ax.plot(x_u, sub['roa_so'].values * 100, 'o-', color='tab:blue',   label='single-obj', alpha=0.7)
    ax.plot(x_u, sub['roa_mo'].values * 100, 'o-', color='tab:green',  label='multi-obj', alpha=0.7)
    ax.set(ylabel='RoA (%)', ylim=(0, 105),
           title=exp['name'].replace('_', ' '))
    ax.axhline(100, color='k', linestyle=':', linewidth=0.8)
    if ax is axs[0]:
        ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1), fontsize=9)

axs[-1].set(xlabel='Unit (matched)')
plt.suptitle('Rate of agreement with ground truth (full recording)', fontsize=13)
plt.show()
"""),
        md("## 5. Whitening loss across experiments"),
        code("""\
fig, axs = plt.subplots(len(EXPERIMENTS), 1,
                         figsize=(12, 2.5 * len(EXPERIMENTS)),
                         layout='constrained', sharex=True)

for ax, exp in zip(axs, EXPERIMENTS):
    name = exp['name']
    for suffix, color in [('no-adapt', 'tab:orange'),
                           ('single-obj', 'tab:blue'),
                           ('multi-obj', 'tab:green')]:
        match = [(l, o) for l, o, en, s in all_conds if en == name and s == suffix]
        if match:
            _, out = match[0]
            ax.plot(out['wh_loss'][10:-1], color=color, label=suffix, alpha=0.85)
    ax.set(ylabel='KL error²', title=name.replace('_', ' '))
    if ax is axs[0]:
        ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1), fontsize=9)

axs[-1].set(xlabel='Batch')
plt.suptitle('Whitening loss across experiments', fontsize=13)
plt.show()
"""),
    ]

    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    }
    nb.metadata['language_info'] = {
        'name': 'python',
        'version': '3.12.0',
    }
    return nb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = TUTORIALS_DIR

    for exp in EXPERIMENTS:
        nb_name = f"sim_{exp['idx']:02d}_{exp['name']}.ipynb"
        nb_path = out_dir / nb_name
        nb = make_experiment_nb(exp)
        nbformat.write(nb, nb_path)
        print(f"  wrote {nb_path.name}")

    comp_nb = make_comparison_nb(EXPERIMENTS)
    comp_path = out_dir / "sim_07_comparison.ipynb"
    nbformat.write(comp_nb, comp_path)
    print(f"  wrote {comp_path.name}")

    print(f"\nDone — {len(EXPERIMENTS) + 1} notebooks written to {out_dir}")


if __name__ == "__main__":
    main()
