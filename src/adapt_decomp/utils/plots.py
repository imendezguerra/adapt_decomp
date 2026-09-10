"""Plots to compare adaptation changes"""

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors

import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_whitening_comp(
    wh1: np.ndarray,
    wh2: np.ndarray,
    palette: Optional[str] = 'magma',
    ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
    """Compare two whitening matrices.

    Args:
        wh1 (np.ndarray): Whitening matrix 1 with shape (channels, channels).
        wh2 (np.ndarray): Whitening matrix 2 with shape (channels, channels).
        palette (Optional[str], optional): Colour palette. Defaults to 'magma'.
        ax (Optional[plt.Axes], optional): Axes to plot. Defaults to None.

    Returns:
        plt.Axes: Axes with the plots.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(12, 5), layout='tight')

    vmin = np.min([wh1, wh2])
    vmax = np.max([wh1, wh2])

    im0 = ax[0].imshow(wh1, cmap=palette, vmin=vmin, vmax=vmax)
    ax[0].set(title='Whitening 1', xticks=[], yticks=[])
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)

    im1 = ax[1].imshow(wh2, cmap=palette, vmin=vmin, vmax=vmax)
    ax[1].set(title='Whitening 2', xticks=[], yticks=[])
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    im2 = ax[2].imshow(wh1 - wh2, cmap='coolwarm', norm=colors.CenteredNorm())
    ax[2].set(title='Difference', xticks=[], yticks=[])
    divider2 = make_axes_locatable(ax[2])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im0, cax=cax0, orientation='vertical')
    plt.colorbar(im1, cax=cax1, orientation='vertical')
    plt.colorbar(im2, cax=cax2, orientation='vertical')
    
    return ax

def plot_sep_vectors_comp(
    sv1: np.ndarray,
    sv2: np.ndarray,
    palette: Optional[str] = 'magma',
    ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
    """Compare two separation vectors.

    Args:
        sv1 (np.ndarray): Separation vectors 1 with shape (units, channels).
        sv2 (np.ndarray): Separation vectors 2 with shape (units, channels).
        palette (Optional[str], optional): Colour palette. Defaults to 'magma'.
        ax (Optional[plt.Axes], optional): Axes to plot. Defaults to None.
    Returns:
        plt.Axes: Axes with the plots.
    """

    if ax is None:
        fig, ax = plt.subplots(3, 1, figsize=(12, 5), layout='tight')

    vmin = np.amin([sv1, sv2])
    vmax = np.amax([sv1, sv2])

    im0 = ax[0].imshow(sv1, cmap=palette, vmin=vmin, vmax=vmax, aspect='auto')
    ax[0].set(title='Separation vectors 1', xticks=[], yticks=[])
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)

    im1 = ax[1].imshow(sv2, cmap=palette, vmin=vmin, vmax=vmax, aspect='auto')
    ax[1].set(title='Separation vectors 2', xticks=[], yticks=[])
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    im2 = ax[2].imshow(sv1 - sv2, cmap='coolwarm', aspect='auto', norm=colors.CenteredNorm())
    ax[2].set(title='Difference', xticks=[], yticks=[])
    divider2 = make_axes_locatable(ax[2])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im0, cax=cax0, orientation='vertical')
    plt.colorbar(im1, cax=cax1, orientation='vertical')
    plt.colorbar(im2, cax=cax2, orientation='vertical')
    
    return ax

def plot_sep_vectors_diff(
    sv: np.ndarray,
    ch_map: Optional[np.ndarray],
    palette: Optional[str] = 'coolwarm',
    ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
    """Plot the separation vectors difference.
    
    Args:
        sv (np.ndarray): Separation vectors with shape (units, channels).
        ch_map (Optional[np.ndarray]): Channel map. Defaults to None.
        palette (Optional[str], optional): Colour palette. Defaults to 'coolwarm'.
        ax (Optional[plt.Axes], optional): Axes to plot. Defaults to None.
    Returns:
        plt.Axes: Axes with the plots.
    """ 
    
    units = sv.shape[0]

    if ax is None:
        cols = 3
        rows = -(-units // cols)
        fig, ax = plt.subplots(rows, cols, figsize=(12, 2 * rows), layout='tight')
        ax = np.ravel(ax)

    if ch_map is None:
        ch_map = np.arange(sv.shape[1])

    v = np.amax(np.abs(sv))

    for unit in range(units):

        im = ax[unit].imshow(sv[unit, ch_map], cmap=palette, aspect='auto', vmin=-v, vmax=v)
        ax[unit].set(title=f'Unit {unit}', xticks=[], yticks=[])
        divider = make_axes_locatable(ax[unit])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')

    return ax


def _check_signal_dicts(
    primary: Dict[str, np.ndarray],
    primary_name: str,
    **aux: Optional[Dict[str, np.ndarray]]
    ) -> None:
    """Validate a primary named-array dict and any auxiliary dicts keyed against it.

    Args:
        primary (Dict[str, np.ndarray]): Named arrays that must all share one shape,
            e.g. sources or spikes, each with shape (samples, n_units).
        primary_name (str): primary's argument name, used in raised messages.
        **aux (Optional[Dict[str, np.ndarray]]): Named per-unit metric dicts (e.g.
            roa, sil) whose keys must be a subset of primary's keys, each value
            with shape (n_units,).

    Returns:
        None

    Raises:
        ValueError: If primary is empty, its arrays don't share one shape, or an
            aux dict has a key not present in primary.
    """
    if not primary:
        raise ValueError(f"{primary_name} must not be empty.")

    shapes = {name: arr.shape for name, arr in primary.items()}
    first_shape = next(iter(shapes.values()))
    mismatched = {name: shape for name, shape in shapes.items() if shape != first_shape}
    if mismatched:
        raise ValueError(
            f"All {primary_name} arrays must share one shape; got {shapes}."
        )

    for aux_name, aux_dict in aux.items():
        if aux_dict is None:
            continue
        unknown = [name for name in aux_dict if name not in primary]
        if unknown:
            raise ValueError(
                f"{aux_name} has keys {unknown} not present in {primary_name} "
                f"(known: {list(primary)})."
            )


def _restrict_time_range(
    timestamps: np.ndarray,
    time_range: Optional[Tuple[float, float]],
    *signal_dicts: Optional[Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, ...]:
    """Restrict timestamps and any number of named-array dicts to one time window.

    Args:
        timestamps (np.ndarray): Time axis, shape (samples,).
        time_range (Optional[Tuple[float, float]]): (start, end) in the same units
            as timestamps, inclusive. None leaves everything unchanged.
        *signal_dicts (Optional[Dict[str, np.ndarray]]): Named arrays sharing
            timestamps' sample axis (axis 0), each with shape (samples, ...). A
            None entry passes through unchanged.

    Returns:
        Tuple[np.ndarray, ...]: (timestamps, *signal_dicts), each restricted to
        time_range if given.
    """
    if time_range is None:
        return (timestamps, *signal_dicts)

    mask = (timestamps >= time_range[0]) & (timestamps <= time_range[1])
    sliced = tuple(
        None if d is None else {name: arr[mask] for name, arr in d.items()}
        for d in signal_dicts
    )
    return (timestamps[mask], *sliced)


def plot_spikes(
    spikes: Dict[str, np.ndarray],
    timestamps: np.ndarray,
    roa: Optional[Dict[str, np.ndarray]] = None,
    sil: Optional[Dict[str, np.ndarray]] = None,
    pair_gap: float = 1.0,
    pair_step: float = 3.0,
    palette: Optional[str] = 'tab10',
    time_range: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Grouped spike raster: one row per named spike train, stacked per unit.

    All arrays in spikes must already be aligned to the same unit axis (e.g. a
    ground-truth matrix pre-selected to gt_matched_indices) -- this function only
    displays them, it doesn't pair or match units.

    Args:
        spikes (Dict[str, np.ndarray]): Name -> binary spike matrix, each with
            shape (samples, n_units), all sharing the same shape.
        timestamps (np.ndarray): Time axis, shape (samples,).
        roa (Optional[Dict[str, np.ndarray]], optional): Name -> rate of agreement
            per unit, shape (n_units,), keys a subset of spikes. When present for
            a name, adds a 'name RoA = X%' line to that unit's ytick label.
            Defaults to None.
        sil (Optional[Dict[str, np.ndarray]], optional): Name -> silhouette score
            per unit, shape (n_units,), in normalised units, keys a subset of
            spikes. When present for a name, adds a 'name SIL = Y' line to that
            unit's ytick label. Defaults to None.
        pair_gap (float, optional): Rows between consecutive named signals within
            a unit's group. Defaults to 1.0.
        pair_step (float, optional): Rows between consecutive unit groups (> the
            group's own span, so groups stay visually separated). Defaults to 3.0.
        palette (Optional[str], optional): Qualitative colour palette, one colour
            per named signal. Defaults to 'tab10'.
        time_range (Optional[Tuple[float, float]], optional): (start, end) in
            timestamps' own units -- restricts both the plotted spikes and the
            x-axis to this window. Defaults to None (the full timestamps range).
        ax (Optional[plt.Axes], optional): Axes to plot on. Defaults to None.

    Returns:
        plt.Axes: Axes with the plot.
    """
    _check_signal_dicts(spikes, 'spikes', roa=roa, sil=sil)
    timestamps, spikes = _restrict_time_range(timestamps, time_range, spikes)
    names = list(spikes.keys())
    n_units = next(iter(spikes.values())).shape[1]
    n_signals = len(names)
    palette_colors = sns.color_palette(palette, n_colors=n_signals)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(3, 0.6 * n_units)), layout='constrained')

    ytick_pos, ytick_labels = [], []
    for unit in range(n_units):
        group_y0 = unit * pair_step
        for i, name in enumerate(names):
            idxs = np.flatnonzero(spikes[name][:, unit])
            y = group_y0 + i * pair_gap
            ax.plot(timestamps[idxs], np.full_like(idxs, y), '|', markersize=8,
                    color=palette_colors[i], label=name if unit == 0 else None)

        label_lines = [f'MU {unit}']
        for name in names:
            if roa is not None and name in roa:
                label_lines.append(f'{name} RoA = {roa[name][unit] * 100:.1f}%')
        for name in names:
            if sil is not None and name in sil:
                label_lines.append(f'{name} SIL = {sil[name][unit]:.2f}')

        ytick_pos.append(group_y0 + (n_signals - 1) * pair_gap / 2)
        ytick_labels.append('\n'.join(label_lines))

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels)
    ax.set(xlabel='Time (s)')
    if time_range is not None:
        ax.set_xlim(time_range)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    return ax


def plot_sources(
    sources: Dict[str, np.ndarray],
    timestamps: np.ndarray,
    spikes: Optional[Dict[str, np.ndarray]] = None,
    roa: Optional[Dict[str, np.ndarray]] = None,
    sil: Optional[Dict[str, np.ndarray]] = None,
    palette: Optional[str] = 'tab10',
    time_range: Optional[Tuple[float, float]] = None,
    axs: Optional[np.ndarray] = None,
    square_sources: bool = True,
) -> np.ndarray:
    """Source traces: one row per unit, named signals overlaid.

    Args:
        sources (Dict[str, np.ndarray]): Name -> source signal, each with shape
            (samples, n_units), all sharing the same shape.
        timestamps (np.ndarray): Time axis, shape (samples,).
        spikes (Optional[Dict[str, np.ndarray]], optional): Name -> binary spike
            matrix, each with shape matching sources[name], keys a subset of
            sources. When present for a name, marks that name's detected spikes
            on top of its trace. Defaults to None.
        roa (Optional[Dict[str, np.ndarray]], optional): Name -> rate of agreement
            per unit, shape (n_units,), keys a subset of sources. When present for
            a name, adds a 'name RoA = X%' line to that unit's ylabel.
            Defaults to None.
        sil (Optional[Dict[str, np.ndarray]], optional): Name -> silhouette score
            per unit, shape (n_units,), in normalised units, keys a subset of
            sources. When present for a name, adds a 'name SIL = Y' line to that
            unit's ylabel. Defaults to None.
        palette (Optional[str], optional): Qualitative colour palette, one colour
            per named signal. Defaults to 'tab10'.
        time_range (Optional[Tuple[float, float]], optional): (start, end) in
            timestamps' own units -- restricts both the plotted traces/spikes and
            the x-axis to this window. Defaults to None (the full timestamps
            range).
        axs (Optional[np.ndarray], optional): Axes array, one per unit. Defaults
            to None.
        square_sources (bool, optional): Plot each source squared rather than raw.
            Defaults to True.

    Returns:
        np.ndarray: Axes array used for the plot.
    """
    _check_signal_dicts(sources, 'sources', spikes=spikes, roa=roa, sil=sil)
    if spikes is not None:
        for name, spikes_arr in spikes.items():
            if spikes_arr.shape != sources[name].shape:
                raise ValueError(
                    f"spikes[{name!r}] has shape {spikes_arr.shape}, expected "
                    f"{sources[name].shape} to match sources[{name!r}]."
                )
    if square_sources:
        sources = {name: arr ** 2 for name, arr in sources.items()}
    timestamps, sources, spikes = _restrict_time_range(timestamps, time_range, sources, spikes)

    names = list(sources.keys())
    n_units = next(iter(sources.values())).shape[1]
    n_signals = len(names)
    palette_colors = sns.color_palette(palette, n_colors=n_signals)

    if axs is None:
        fig, axs = plt.subplots(n_units, 1, figsize=(12, n_units), layout='constrained', sharex=True)
    axs = np.atleast_1d(axs)

    for unit in range(n_units):
        ax = axs[unit]
        for i, name in enumerate(names):
            ax.plot(timestamps, sources[name][:, unit], label=name, color=palette_colors[i])
            if spikes is not None and name in spikes:
                mask = spikes[name][:, unit].astype(bool)
                ax.plot(timestamps[mask], sources[name][:, unit][mask],
                        linestyle='None', marker='.', color=palette_colors[i])

        label_lines = [f'MU {unit}']
        for name in names:
            if roa is not None and name in roa:
                label_lines.append(f'{name} RoA = {roa[name][unit] * 100:.1f}%')
        for name in names:
            if sil is not None and name in sil:
                label_lines.append(f'{name} SIL = {sil[name][unit]:.2f}')

        ax.set(ylabel='\n'.join(label_lines))
        if time_range is not None:
            ax.set_xlim(time_range)
        if n_signals > 1:
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        if unit == n_units - 1:
            ax.set(xlabel='Time (s)')

    return axs


def plot_metric_heatmap(
    df_metric: pd.DataFrame,
    value_col: str,
    configs: List[str],
    conditions: List[str],
    snr_levels: List[int],
    agg: Literal['mean', 'median', 'pct_ge_threshold'] = 'mean',
    threshold: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    axs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Heatmap of a per-unit value aggregated by condition x SNR, one panel per config.

    Args:
        df_metric (pd.DataFrame): Long-format table with columns 'config', 'condition',
            'snr', value_col.
        value_col (str): Column name of the metric to aggregate.
        configs (List[str]): Config labels to plot, one panel each, in order.
        conditions (List[str]): Row order for the heatmap index.
        snr_levels (List[int]): Column order for the heatmap columns.
        agg (Literal['mean', 'median', 'pct_ge_threshold'], optional): Aggregation
            function. Defaults to 'mean'.
        threshold (float, optional): Threshold for agg='pct_ge_threshold', in the
            same units as value_col. Defaults to None.
        vmin (float, optional): Minimum value for the heatmap color scale. Defaults
            to None (auto).
        vmax (float, optional): Maximum value for the heatmap color scale. Defaults 
            to None (auto).
        axs (Optional[np.ndarray], optional): Axes array, one per config. Defaults to None.

    Returns:
        np.ndarray: Axes array used for the plot.
    """
    if axs is None:
        fig, axs = plt.subplots(1, len(configs), figsize=(5 * len(configs), 4), layout='constrained')
    axs = np.ravel(axs)

    if agg == 'pct_ge_threshold':
        if threshold is None:
            raise ValueError("threshold must be provided when agg='pct_ge_threshold'.")
        agg_fn = lambda x: (x >= threshold).sum() / len(x) * 100
        label = f'% units >= {threshold:g}'
    else:
        agg_fn = agg
        label = f'{agg.capitalize()} {value_col}'

    if vmax is not None and vmax <= 1:
        fmt_str = '.2f'
    else:
        fmt_str = '.1f' 

    for ax, cfg in zip(axs, configs):
        pivot = (df_metric[df_metric['config'] == cfg]
                 .groupby(['condition', 'snr'])[value_col].agg(agg_fn).unstack('snr')
                 .reindex(index=conditions, columns=snr_levels))
        sns.heatmap(pivot, annot=True, fmt=fmt_str, cmap='YlGn', vmin=vmin, vmax=vmax,
                    ax=ax, cbar_kws={'label': label})
        ax.set(title=cfg, xlabel='SNR (dB)', ylabel='Condition')
    return axs


def plot_metric_boxplot(
    df_metric: pd.DataFrame,
    value_col: str,
    configs: List[str],
    conditions: List[str],
    snr_levels: List[int],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    axs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-unit metric boxplot by condition, one panel per config, hue=SNR.

    Args:
        df_metric (pd.DataFrame): Long-format table with columns 'config', 'condition',
            'snr', value_col.
        value_col (str): Column name of the metric to aggregate.
        configs (List[str]): Config labels to plot, one panel each, in order.
        conditions (List[str]): x-axis category order.
        snr_levels (List[int]): Hue order (rendered as '<snr> dB').
        vmin (float, optional): Minimum value for the y-axis. Defaults to None (auto).
        vmax (float, optional): Maximum value for the y-axis. Defaults to None (auto).
        axs (Optional[np.ndarray], optional): Axes array, one per config. Defaults to None.

    Returns:
        np.ndarray: Axes array used for the plot.
    """
    if axs is None:
        fig, axs = plt.subplots(1, len(configs), figsize=(6 * len(configs), 4.5),
                                 layout='constrained', sharey=True)
    axs = np.ravel(axs)

    snr_order = [f'{s} dB' for s in snr_levels]
    df_plot = df_metric.copy()
    df_plot['SNR'] = df_plot['snr'].astype(str) + ' dB'

    for ax, cfg in zip(axs, configs):
        sub_df = df_plot[df_plot['config'] == cfg]
        sns.boxplot(data=sub_df, x='condition', y=value_col, hue='SNR', hue_order=snr_order,
                    order=conditions, palette='Blues_d',
                    flierprops=dict(marker='.', markersize=3, alpha=0.4), ax=ax)
        ax.set(ylabel=value_col, xlabel='Condition', title=cfg, ylim=(vmin, vmax))
        ax.tick_params(axis='x', rotation=15)
        if ax is not axs[0]:
            ax.get_legend().remove()
    axs[0].legend(title='SNR', fontsize=8)
    return axs


def plot_phase_bar(
    df_phase: pd.DataFrame,
    configs: List[str],
    triangular_conditions: List[str],
    snr_levels: List[int],
    phase_order: List[str],
    phase_labels: Dict[str, str],
    axs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Bar chart of mean +/- SD RoA per triangular phase, rows=config, cols=condition.

    Args:
        df_phase (pd.DataFrame): Long-format phase-RoA table with columns
            'config', 'condition', 'snr', 'phase', 'roa_pct'.
        configs (List[str]): Row order (one row per config).
        triangular_conditions (List[str]): Column order (one column per condition).
        snr_levels (List[int]): Hue order (rendered as '<snr> dB').
        phase_order (List[str]): x-axis category order within each panel.
        phase_labels (Dict[str, str]): Display label per phase key.
        axs (Optional[np.ndarray], optional): Axes array, shape (configs, conditions).
            Defaults to None.

    Returns:
        np.ndarray: Axes array used for the plot.
    """
    n_tri = len(triangular_conditions)
    if axs is None:
        fig, axs = plt.subplots(len(configs), n_tri, figsize=(5 * n_tri, 4 * len(configs)),
                                 layout='constrained', sharey=True)
    axs = np.atleast_2d(axs)

    for row, cfg in enumerate(configs):
        for col, cond in enumerate(triangular_conditions):
            ax = axs[row, col]
            df_c = df_phase[(df_phase['condition'] == cond) & (df_phase['config'] == cfg)].copy()
            df_c['SNR'] = df_c['snr'].astype(str) + ' dB'
            sns.barplot(data=df_c, x='phase', y='roa_pct', hue='SNR',
                        hue_order=[f'{s} dB' for s in snr_levels], order=phase_order,
                        estimator='mean', errorbar='sd', capsize=0.1, palette='Blues_d', ax=ax)
            ax.set(xlabel='', ylabel='Mean RoA (%)' if col == 0 else '',
                   title=f'{cfg} -- {cond}', ylim=(0, 105))
            ax.set_xticks(range(len(phase_order)))
            ax.set_xticklabels([phase_labels[p] for p in phase_order], rotation=20, ha='right', fontsize=9)
            if col != 0:
                ax.get_legend().remove()
    return axs


def plot_roa_summary_figure(
    df_roa: pd.DataFrame,
    df_phase: pd.DataFrame,
    adapted_configs: List[str],
    conditions: List[str],
    holdout_conditions: List[str],
    snr_levels: List[int],
    phase_order: List[str],
    phase_labels: Dict[str, str],
    axs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Combined figure: RoA line trend per condition (left) + phase bar chart (right), one row per config.

    Args:
        df_roa (pd.DataFrame): Long-format RoA table, columns 'config', 'condition', 'snr', 'roa_pct'.
        df_phase (pd.DataFrame): Long-format phase-RoA table, see plot_phase_bar.
        adapted_configs (List[str]): Row order (adapted configs only, excludes 'fixed').
        conditions (List[str]): x-axis order for the left column.
        holdout_conditions (List[str]): Conditions rendered bold/red on the left column's
            x-ticks; pass [] to opt out of this styling.
        snr_levels (List[int]): Line/hue colour order, left and right columns.
        phase_order (List[str]): x-axis order for the right column.
        phase_labels (Dict[str, str]): Display label per phase key.
        axs (Optional[np.ndarray], optional): Axes array, shape (len(adapted_configs), 2).
            Defaults to None.

    Returns:
        np.ndarray: Axes array used for the plot.
    """
    if axs is None:
        fig, axs = plt.subplots(len(adapted_configs), 2, figsize=(15, 4.5 * len(adapted_configs)),
                                 layout='constrained')
    axs = np.atleast_2d(axs)

    palette = sns.color_palette('Blues', n_colors=len(snr_levels))
    colors_snr = dict(zip(sorted(snr_levels, reverse=True), palette[::-1]))

    for row, cfg in enumerate(adapted_configs):
        # -- Left: overall mean +/- SD RoA per condition, one line per SNR --
        cfg_df = df_roa[df_roa['config'] == cfg]
        stats_by = cfg_df.groupby(['condition', 'snr'])['roa_pct'].agg(['mean', 'std']).reset_index()
        for snr in snr_levels:
            sub_df = stats_by[stats_by['snr'] == snr].set_index('condition').reindex(conditions)
            axs[row, 0].errorbar(conditions, sub_df['mean'], yerr=sub_df['std'], marker='o',
                                  capsize=3, label=f'{snr} dB', color=colors_snr[snr])
        axs[row, 0].set(ylabel='Mean RoA (%)', title=f'Overall RoA -- {cfg}', ylim=(0, 105), xlabel='Condition')
        axs[row, 0].set_xticks(range(len(conditions)))
        axs[row, 0].set_xticklabels(
            [f'{c}\n(held out)' if c in holdout_conditions else c for c in conditions], rotation=20, ha='right')
        for tick_label, cond in zip(axs[row, 0].get_xticklabels(), conditions):
            if cond in holdout_conditions:
                tick_label.set_fontweight('bold')
                tick_label.set_color('firebrick')
        axs[row, 0].legend(title='SNR', fontsize=9)

        # -- Right: three-phase RoA (mean +/- SD) over all triangular conditions --
        phase_cfg_df = df_phase[df_phase['config'] == cfg].copy()
        if len(phase_cfg_df):
            phase_cfg_df['SNR'] = phase_cfg_df['snr'].astype(str) + ' dB'
            sns.barplot(data=phase_cfg_df, x='phase', y='roa_pct', hue='SNR',
                        hue_order=[f'{s} dB' for s in snr_levels], order=phase_order,
                        estimator='mean', errorbar='sd', capsize=0.1, palette='Blues_d', ax=axs[row, 1])
            axs[row, 1].set_xticks(range(len(phase_order)))
            axs[row, 1].set_xticklabels([phase_labels[p] for p in phase_order], rotation=20, ha='right')
        axs[row, 1].set(ylabel='Mean RoA (%)', title=f'Phase RoA (triangular) -- {cfg}', ylim=(0, 105), xlabel='')

    return axs


def plot_optimisation_landscape(
    study: optuna.Study,
    title: str,
    loss_cols: Tuple[str, str, str] = ('user_attrs_wh_loss', 'user_attrs_sv_loss', 'user_attrs_total_loss'),
    roa_col: str = 'user_attrs_roa_mean_pooled',
    best_trial: Optional[optuna.trial.FrozenTrial] = None,
    showlegend: bool = True,
) -> List[List[go.Scatter]]:
    """Plotly optimisation-landscape traces: one column per loss, x=loss (log), y=RoA.

    Marks the max-RoA trial and best_trial distinctly (star vs diamond); they may coincide.
    Returns raw trace lists rather than a Figure so callers can arrange several of these
    into one composed figure via plotly.subplots.make_subplots.

    Args:
        study (optuna.Study): Completed study (single-objective or Pareto alike, compute_roa=True).
        title (str): Legend/hover label for this study's trials, e.g. the lr_mode token.
        loss_cols (Tuple[str, str, str], optional): trials_dataframe() columns for
            wh_loss/sv_loss/total_loss. Defaults to the standard user_attrs names.
        roa_col (str, optional): trials_dataframe() column for pooled mean RoA. Defaults to
            'user_attrs_roa_mean_pooled'.
        best_trial (Optional[optuna.trial.FrozenTrial], optional): The selected/winning trial
            to mark with a diamond. Defaults to None (no marker).
        showlegend (bool, optional): Whether the 'max RoA'/'selected' marker traces contribute
            a legend entry (set False for every call but the first when composing several of
            these into one figure, since those two markers mean the same thing in every call
            and would otherwise add one duplicate-looking legend entry per call). The main
            per-trial scatter trace always gets its own entry (named `title`), independent of
            this flag. Defaults to True.

    Returns:
        List[List[go.Scatter]]: One inner list of traces per available loss column. A cached
            study run before per-trial wh_loss/sv_loss/total_loss logging existed carries none
            of loss_cols -- such columns are silently skipped rather than raising, falling back
            to the single 'value' column (the study's own optimized surrogate loss) if none of
            loss_cols are present at all.
    """
    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df['state'] == 'COMPLETE']

    available_cols = [c for c in loss_cols if c in trials_df.columns]
    if not available_cols and 'value' in trials_df.columns:
        available_cols = ['value']

    has_roa = roa_col in trials_df.columns
    roa_series = trials_df[roa_col] if has_roa else pd.Series(np.nan, index=trials_df.index)
    max_roa_idx = roa_series.idxmax() if has_roa and roa_series.notna().any() else None
    columns: List[List[go.Scatter]] = []
    for loss_col in available_cols:
        traces = [go.Scatter(
            x=trials_df[loss_col], y=roa_series, mode='markers',
            marker=dict(size=8, color='#4C72B0'), text=trials_df['number'],
            name=title, legendgroup=title, showlegend=(loss_col == available_cols[0]),
        )]
        if max_roa_idx is not None:
            row = trials_df.loc[max_roa_idx]
            traces.append(go.Scatter(
                x=[row[loss_col]], y=[row[roa_col]], mode='markers',
                marker=dict(size=16, color='gold', symbol='star', line=dict(width=1, color='black')),
                name='max RoA', legendgroup='max_roa',
                showlegend=(showlegend and loss_col == available_cols[0]),
            ))
        if best_trial is not None and loss_col.replace('user_attrs_', '') in best_trial.user_attrs:
            key = loss_col.replace('user_attrs_', '')
            traces.append(go.Scatter(
                x=[best_trial.user_attrs[key]], y=[best_trial.user_attrs.get('roa_mean_pooled')],
                mode='markers', marker=dict(size=14, color='red', symbol='diamond',
                                             line=dict(width=1, color='black')),
                name='selected', legendgroup='selected',
                showlegend=(showlegend and loss_col == available_cols[0]),
            ))
        columns.append(traces)
    return columns


def plot_pareto_scatter(
    study: optuna.Study,
    pareto_front: List[optuna.trial.FrozenTrial],
    selected_trial: optuna.trial.FrozenTrial,
    title: str,
    objectives: Tuple[str, str] = ('wh_loss', 'sv_loss'),
    roa_col: str = 'user_attrs_roa_mean_pooled',
    showlegend: bool = True,
) -> List[go.Scatter]:
    """Plotly Pareto-front traces: wh_loss vs sv_loss, colour=RoA, front line + two markers.

    Marks selected_trial (the selection_rule's winner) and the max-RoA trial among ALL
    completed trials (not just the front) distinctly; they may coincide.

    Args:
        study (optuna.Study): Completed Pareto-front study (compute_roa=True).
        pareto_front (List[optuna.trial.FrozenTrial]): study.best_trials (the Pareto front).
        selected_trial (optuna.trial.FrozenTrial): The trial selection_rule chose.
        title (str): Legend/hover label, e.g. the lr_mode token.
        objectives (Tuple[str, str], optional): trials_dataframe() 'values_<x>'/'values_<y>'
            suffixes. Defaults to ('wh_loss', 'sv_loss').
        roa_col (str, optional): trials_dataframe() column for pooled mean RoA. Defaults to
            'user_attrs_roa_mean_pooled'.
        showlegend (bool, optional): Whether these traces contribute legend entries (set False
            for every panel but the first when composing several side by side). Defaults to True.

    Returns:
        List[go.Scatter]: The traces for one panel.
    """
    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df['state'] == 'COMPLETE']
    x_col, y_col = f'values_{objectives[0]}', f'values_{objectives[1]}'
    front_numbers = {t.number for t in pareto_front}

    traces = [go.Scatter(
        x=trials_df[x_col], y=trials_df[y_col], mode='markers',
        marker=dict(size=8, color=trials_df[roa_col], colorscale='Viridis',
                    showscale=showlegend, colorbar=dict(title='RoA') if showlegend else None),
        text=trials_df['number'], name='trials', showlegend=False,
    )]

    front_df = trials_df[trials_df['number'].isin(front_numbers)].sort_values(x_col)
    traces.append(go.Scatter(
        x=front_df[x_col], y=front_df[y_col], mode='lines+markers',
        line=dict(color='black', width=1, dash='dot'), marker=dict(size=4, color='black'),
        name='Pareto front', showlegend=showlegend,
    ))

    max_roa_row = trials_df.loc[trials_df[roa_col].idxmax()]
    traces.append(go.Scatter(
        x=[max_roa_row[x_col]], y=[max_roa_row[y_col]], mode='markers',
        marker=dict(size=16, color='gold', symbol='star', line=dict(width=1, color='black')),
        name='max RoA', showlegend=showlegend,
    ))

    sel_number = selected_trial.number
    sel_row = trials_df[trials_df['number'] == sel_number].iloc[0]
    traces.append(go.Scatter(
        x=[sel_row[x_col]], y=[sel_row[y_col]], mode='markers',
        marker=dict(size=14, color='red', symbol='diamond', line=dict(width=1, color='black')),
        name='selected', showlegend=showlegend,
    ))
    return traces


def plot_optimisation_landscape_grid(
    studies: Dict[str, optuna.Study],
    titles: Dict[str, str],
    best_trials: Dict[str, optuna.trial.FrozenTrial],
) -> go.Figure:
    """Compose plot_optimisation_landscape() panels for several studies into one grid.

    One row per study (e.g. per lr_mode), one column per available loss column
    (wh_loss/sv_loss/total_loss, or a fallback -- see plot_optimisation_landscape).

    Args:
        studies (Dict[str, optuna.Study]): One study per row key (e.g. lr_mode token).
        titles (Dict[str, str]): Row title per row key.
        best_trials (Dict[str, optuna.trial.FrozenTrial]): The selected/winning trial
            per row key (study.best_trial for a single-objective study, or the
            selection_rule's chosen front member for a Pareto study).

    Returns:
        go.Figure: The composed figure.
    """
    loss_cols = ('user_attrs_wh_loss', 'user_attrs_sv_loss', 'user_attrs_total_loss')
    row_keys = list(studies)

    # Not every cached study logged all three per-trial losses -- older studies (predating
    # unconditional wh_loss/sv_loss/total_loss user_attrs) may carry none of them, in which
    # case plot_optimisation_landscape falls back to the single 'value' column. Determine the
    # actual column count from the first study so the subplot grid matches what will render.
    first_df = studies[row_keys[0]].trials_dataframe()
    available = [c for c in loss_cols if c in first_df.columns]
    if not available:
        available = ['value'] if 'value' in first_df.columns else []
    col_titles = [c.replace('user_attrs_', '') for c in available] or ['(no logged loss columns)']
    n_cols = max(len(col_titles), 1)

    fig = make_subplots(rows=len(row_keys), cols=n_cols,
                         subplot_titles=[f'{titles[k]} -- {c}' for k in row_keys for c in col_titles])
    for row, key in enumerate(row_keys, start=1):
        panel = plot_optimisation_landscape(studies[key], titles[key], loss_cols=loss_cols,
                                             best_trial=best_trials[key], showlegend=(row == 1))
        for col, trace_group in enumerate(panel, start=1):
            for trace in trace_group:
                fig.add_trace(trace, row=row, col=col)
            fig.update_xaxes(title_text=col_titles[col - 1], type='log', row=row, col=col)
            fig.update_yaxes(title_text='pooled RoA (%)', row=row, col=col)
    fig.update_layout(height=450 * len(row_keys), width=1500,
                       title='Optimisation landscape -- RoA vs pooled loss')
    return fig


def plot_pareto_scatter_grid(
    studies: Dict[str, optuna.Study],
    pareto_fronts: Dict[str, List[optuna.trial.FrozenTrial]],
    selected_trials: Dict[str, optuna.trial.FrozenTrial],
) -> go.Figure:
    """Compose plot_pareto_scatter() panels for several studies side by side.

    One column per study (e.g. per lr_mode).

    Args:
        studies (Dict[str, optuna.Study]): One multi-objective study per column key.
        pareto_fronts (Dict[str, List[optuna.trial.FrozenTrial]]): study.best_trials
            per column key.
        selected_trials (Dict[str, optuna.trial.FrozenTrial]): The selection_rule's
            chosen front member per column key.

    Returns:
        go.Figure: The composed figure.
    """
    col_keys = list(studies)
    fig = make_subplots(rows=1, cols=len(col_keys), subplot_titles=col_keys)
    for col, key in enumerate(col_keys, start=1):
        traces = plot_pareto_scatter(studies[key], pareto_fronts[key],
                                      selected_trials[key], key, showlegend=(col == 1))
        for trace in traces:
            fig.add_trace(trace, row=1, col=col)
        fig.update_xaxes(title_text='wh_loss', type='log', row=1, col=col)
        fig.update_yaxes(title_text='sv_loss', type='log', row=1, col=col)
    fig.update_layout(height=550, width=650 * len(col_keys),
                       title='Real Pareto front -- wh_loss vs sv_loss, coloured by RoA')
    return fig