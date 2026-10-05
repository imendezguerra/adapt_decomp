"""Pareto fronts: dominance, the running front, and the rules that pick one member."""

from __future__ import annotations

from typing import Callable, Dict, List, Literal, Tuple, Union

import numpy as np
import optuna


FrontSelector = Callable[[List[optuna.trial.FrozenTrial]], optuna.trial.FrozenTrial]


SelectionName = Literal["min_sv_loss", "knee", "max_roa_mean"]


def _dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
    """True iff objective vector a Pareto-dominates b (both minimised, equal length).

    Args:
        a (Tuple[float, ...]): Candidate dominator's objective values.
        b (Tuple[float, ...]): Candidate dominated point's objective values.

    Returns:
        bool: True iff a is no worse than b in every dimension and strictly
        better in at least one -- matching Optuna's own study.best_trials
        convention, so a tie dominates neither point (both stay on the front).
    """
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def update_front(
    front: Dict[int, Tuple[float, ...]],
    trial_number: int,
    values: Tuple[float, ...],
    keep_ties: bool = True,
) -> Tuple[bool, List[int]]:
    """Join trial_number onto the resident front, evicting anything it now dominates.

    With one objective and keep_ties=False the front is the single best
    trial so far, replaced only on strict improvement (as study.best_trial).

    Args:
        front (Dict[int, Tuple[float, ...]]): Currently resident front
            members, trial_number -> objective values. Mutated in place.
        trial_number (int): This trial's Optuna trial.number.
        values (Tuple[float, ...]): This trial's objective values, ordered
            to match objectives.
        keep_ties (bool, optional): Whether a trial tying a resident member
            joins the front. Defaults to True (Pareto convention).

    Returns:
        Tuple[bool, List[int]]: joined, False if some resident member
        already dominates (or, without keep_ties, ties) this trial (front
        left unchanged), True otherwise, in which case trial_number is added
        to front; evicted, the trial numbers removed from front because this
        trial dominates them (always empty when joined is False).
    """
    if any(
        _dominates(existing, values) or (not keep_ties and existing == values)
        for existing in front.values()
    ):
        return False, []
    evicted = [n for n, existing in front.items() if _dominates(values, existing)]
    for n in evicted:
        del front[n]
    front[trial_number] = values
    return True, evicted


def front_mask(values: np.ndarray) -> np.ndarray:
    """Which rows of an objective table are on its Pareto front, every objective minimised.

    The table-based counterpart of update_front, for a finished search read back
    from its trials table (e.g. study.trials_dataframe()'s values_* columns).

    Args:
        values (np.ndarray): Objective values with shape (trials, objectives).

    Returns:
        np.ndarray: Boolean mask with shape (trials,): True for every row that no
        other row dominates (ties stay on the front, as in study.best_trials);
        rows with a non-finite value are never on it.
    """
    values = np.asarray(values, dtype=float)
    candidates = np.flatnonzero(np.all(np.isfinite(values), axis=1))
    mask = np.zeros(len(values), dtype=bool)
    for i in candidates:
        mask[i] = not any(
            _dominates(tuple(values[j]), tuple(values[i])) for j in candidates if j != i
        )
    return mask


def _select_min_sv_loss(pareto_front: List[optuna.trial.FrozenTrial]) -> optuna.trial.FrozenTrial:
    """Default Pareto-front selection: the front's own minimum pooled sv_loss member.

    Reads trial.user_attrs["sv_loss"] (always logged pooled, regardless of
    which dimensions objectives actually optimised) rather than
    trial.values, so this works even when "sv_loss" isn't itself one of
    objectives. Always Pareto-optimal by construction and needs no ground
    truth, though it sits at the front's sv_loss extreme.

    Args:
        pareto_front (List[optuna.trial.FrozenTrial]): study.best_trials
            from a completed Pareto search.

    Returns:
        optuna.trial.FrozenTrial: The front member with the lowest
        "sv_loss" user_attr.
    """
    return min(pareto_front, key=lambda t: t.user_attrs["sv_loss"])


def _select_max_roa_mean(pareto_front: List[optuna.trial.FrozenTrial]) -> optuna.trial.FrozenTrial:
    """Oracle Pareto-front selection: the front's own highest mean RoA member.

    Only meaningful when compute_roa was True (or "roa" was in objectives)
    for the search that produced pareto_front -- otherwise every member's
    "roa_mean_pooled" user_attr is absent and this falls back to picking
    arbitrarily among ties at float("-inf").

    Args:
        pareto_front (List[optuna.trial.FrozenTrial]): study.best_trials
            from a completed Pareto search.

    Returns:
        optuna.trial.FrozenTrial: The front member with the highest
        "roa_mean_pooled" user_attr.
    """
    return max(pareto_front, key=lambda t: t.user_attrs.get("roa_mean_pooled", float("-inf")))


def _select_knee(pareto_front: List[optuna.trial.FrozenTrial]) -> optuna.trial.FrozenTrial:
    """Knee-point Pareto-front selection for two objectives.

    Min-max normalises both objectives over the front and returns the member
    farthest from the line through its two extremes: the point where
    improving one objective starts costing the most of the other, rather
    than an extreme of the front. Needs no ground truth. Falls back to
    _select_min_sv_loss for fronts of fewer than three members (no interior
    point).

    Args:
        pareto_front (List[optuna.trial.FrozenTrial]): study.best_trials
            from a completed two-objective Pareto search.

    Raises:
        ValueError: If the front's trials don't have exactly two values.

    Returns:
        optuna.trial.FrozenTrial: The front member farthest from the chord.
    """
    values = np.array([t.values for t in pareto_front], dtype=float)
    if values.shape[1] != 2:
        raise ValueError(f"Knee selection needs exactly 2 objectives, got {values.shape[1]}.")
    if len(pareto_front) < 3:
        return _select_min_sv_loss(pareto_front)
    span = values.max(0) - values.min(0)
    norm = (values - values.min(0)) / np.where(span > 0, span, 1.0)
    start, end = norm[norm[:, 0].argmin()], norm[norm[:, 1].argmin()]
    chord = end - start
    if not np.linalg.norm(chord) > 0:
        return _select_min_sv_loss(pareto_front)
    offsets = norm - start
    distance = np.abs(chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0])
    return pareto_front[int(distance.argmax())]


SELECTION_RULES: Dict[str, FrontSelector] = {
    "min_sv_loss": _select_min_sv_loss,
    "knee": _select_knee,
    "max_roa_mean": _select_max_roa_mean,
}


def validate_selection(
    selection: Union[SelectionName, FrontSelector], objectives: Tuple[str, ...]
) -> None:
    """Check selection is a rule name in SELECTION_RULES or a callable, and fits objectives.

    Args:
        selection (Union[SelectionName, FrontSelector]): Rule name or callable.
        objectives (Tuple[str, ...]): The search's objective names.

    Raises:
        ValueError: If selection is an unknown name, or "knee" without exactly
            2 objectives.

    Returns:
        None
    """
    if not callable(selection) and selection not in SELECTION_RULES:
        raise ValueError(
            f"Unknown selection: {selection!r}; expected a callable or one of "
            f"{tuple(SELECTION_RULES)}"
        )
    if selection == "knee" and len(objectives) != 2:
        raise ValueError(f"selection='knee' needs exactly 2 objectives, got {objectives!r}")
