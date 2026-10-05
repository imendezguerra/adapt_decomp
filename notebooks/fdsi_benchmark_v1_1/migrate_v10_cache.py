"""One-off migration of the cached v1.0 adaptation results read by 05_comparison_v1_0_vs_v1_1.

The v1.0 pickles under data/fdsi_benchmark/outputs/adaptation/ were written by
earlier revisions of adapt_decomp.adaptation.data_structures.AdaptationResult.
pickle restores an instance __dict__ directly, never calling __init__ or
__post_init__, so fields added to the dataclass since then are simply absent on
the restored object and AdaptationResult.to_dict() raises AttributeError on the
first one it reads. That breaks every dict-style access (outputs[key]), not just
the field that happens to be missing.

Five classes of staleness are present (counts from the survey this was written
against):

    763 files  "ipts" absent                     -- deprecated alias for sources
    306 files  "preprocess_time_ms" absent       -- added in 1.0.0
    306 files  "wh_loss_total"/"sv_loss_total" absent
    206 of those carry "wh_loss_median"/"sv_loss_median", the pre-74d1ab7 names
    100 files  "total_loss" is a per-batch vector, not the guarded scalar
     63 files  "sil" absent                      -- optimisation-pool results
    763 files  "centroid_loss" present           -- field since removed

Each is repaired in place and atomically (temp file plus os.replace), leaving
every measured array untouched. The script is idempotent and dry-run by default.

Notes:
    The two destructive steps are both lossless. Renaming
    wh_loss_median/sv_loss_median is exactly the rename commit 74d1ab7 applied to
    the class. Replacing the per-batch total_loss vector with the scalar is safe
    because the vector equals wh_loss + nanmean(sv_loss, dim=1), which is
    recomputable from fields this script retains.

    The totals are rebuilt to match AdaptDecomp._compute_losses, except that its
    wh_trace/trace_cal divergence guard cannot be replicated: trace_cal lives on
    Decomposition and was never serialised. Only the NaN half of the guard is
    applied (it fires on none of the files surveyed). Where the legacy
    wh_loss_median/sv_loss_median fields exist their values are carried over
    verbatim rather than recomputed, so no convention is imposed on them.

    sv_loss_total is reduced with nansum for the files that need recomputation,
    matching the rest of the v1.0 cache. v1.1 writes nanmean
    (AdaptConfig.sv_loss_reduction defaults to "mean"), so stored sv_loss_total
    scalars are NOT comparable across the two versions -- a v1.0-vs-v1.1 loss
    comparison should reduce the per-batch sv_loss arrays itself under one
    convention. This script deliberately does not paper over that.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adapt_decomp import AdaptationResult  # noqa: E402

DEFAULT_ROOT = Path("../../data/fdsi_benchmark/outputs/adaptation")
LEGACY_RENAMES = {"wh_loss_median": "wh_loss_total", "sv_loss_median": "sv_loss_total"}
DROP_FIELDS = ("centroid_loss",)
DIVERGED = 1e10


def plan_patch(state: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Work out the field changes one restored instance dict needs.

    Args:
        state (Dict[str, Any]): The unpickled AdaptationResult's attribute dict.
            Not mutated.

    Returns:
        Tuple[Dict[str, Any], List[str]]: The patched state, and the change tags
        describing what was done (empty when the file is already current).
    """
    new = dict(state)
    changes: List[str] = []
    declared = {f.name for f in dataclasses.fields(AdaptationResult)}

    for legacy, current in LEGACY_RENAMES.items():
        if legacy in new:
            value = new.pop(legacy)
            if new.get(current) is None:
                new[current] = value
                changes.append(f"rename:{legacy}->{current}")
            else:
                changes.append(f"drop-legacy:{legacy}")

    for field in DROP_FIELDS:
        if field in new:
            value = new.pop(field)
            changes.append(f"drop:{field}" if value is None else f"drop:{field}(NON-NULL)")

    for field in sorted(declared - set(new)):
        if field == "preprocess_time_ms":
            new[field] = torch.zeros_like(new["total_time_ms"])
            changes.append("fill:preprocess_time_ms=zeros")
        elif field not in ("wh_loss_total", "sv_loss_total"):
            new[field] = None
            changes.append(f"fill:{field}=None")

    wh_loss, sv_loss = new.get("wh_loss"), new.get("sv_loss")
    needs_totals = new.get("wh_loss_total") is None or new.get("sv_loss_total") is None
    if needs_totals and torch.is_tensor(wh_loss) and torch.is_tensor(sv_loss):
        if torch.isnan(wh_loss).any():
            new["wh_loss_total"] = torch.tensor(DIVERGED)
            new["sv_loss_total"] = torch.tensor(DIVERGED)
            changes.append("compute:totals=1e10(nan-guard)")
        else:
            new["wh_loss_total"] = wh_loss.median()
            new["sv_loss_total"] = torch.nansum(sv_loss, dim=1).median()
            changes.append("compute:totals=median(wh),median(nansum(sv))")
    else:
        new.setdefault("wh_loss_total", None)
        new.setdefault("sv_loss_total", None)

    total = new.get("total_loss")
    if torch.is_tensor(total) and total.ndim > 0:
        wh_total, sv_total = new.get("wh_loss_total"), new.get("sv_loss_total")
        if torch.is_tensor(wh_total) and torch.is_tensor(sv_total):
            new["total_loss"] = wh_total + sv_total
            changes.append("scalarise:total_loss")

    return new, changes


def migrate_file(path: Path, apply: bool) -> Tuple[str, List[str]]:
    """Inspect, and optionally rewrite, one cached result.

    Args:
        path (Path): The .pkl to migrate.
        apply (bool): Write the patched object back when True; otherwise only
            report what would change.

    Returns:
        Tuple[str, List[str]]: An outcome tag ("current", "migrated",
        "would-migrate", "skipped:<type>", "failed:<reason>") and the change tags
        from plan_patch.
    """
    try:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
    except Exception as exc:  # noqa: BLE001 -- report and carry on over the tree
        return f"failed:unpickle:{type(exc).__name__}", []

    if not isinstance(obj, AdaptationResult):
        return f"skipped:{type(obj).__name__}", []

    patched_state, changes = plan_patch(vars(obj))
    if not changes:
        return "current", []

    patched = AdaptationResult.__new__(AdaptationResult)
    patched.__dict__.update(patched_state)
    patched.to_dict()  # the failure this migration exists to fix

    if not apply:
        return "would-migrate", changes

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as handle:
            pickle.dump(patched, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return f"failed:write:{type(exc).__name__}", changes

    with path.open("rb") as handle:
        pickle.load(handle).to_dict()
    return "migrated", changes


def main(argv: Optional[List[str]] = None) -> int:
    """Run the migration over a cache tree.

    Args:
        argv (Optional[List[str]]): Command-line arguments. Defaults to None,
            which reads sys.argv.

    Returns:
        int: 0 on success, 1 if any file failed.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Adaptation cache root to migrate (default: the v1.0 tree).")
    parser.add_argument("--apply", action="store_true",
                        help="Rewrite the files. Without this the run is a dry run.")
    parser.add_argument("--verbose", action="store_true", help="List every file.")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    paths = sorted(p for p in root.rglob("*.pkl") if p.name != "study.pkl")
    if not paths:
        raise ValueError(f"No .pkl files (other than study.pkl) under {root}")

    print(f"{'Applying' if args.apply else 'Dry run'}: {len(paths)} candidate files under {root}\n")

    outcomes: Counter = Counter()
    change_tags: Counter = Counter()
    failures: List[Tuple[Path, str]] = []

    for path in paths:
        outcome, changes = migrate_file(path, args.apply)
        outcomes[outcome] += 1
        change_tags.update(changes)
        if outcome.startswith("failed"):
            failures.append((path, outcome))
        if args.verbose:
            print(f"  {outcome:16} {path.relative_to(root)}  {','.join(changes)}")

    print("Outcomes")
    for outcome, count in outcomes.most_common():
        print(f"  {count:5d}  {outcome}")
    print("\nChanges")
    for tag, count in change_tags.most_common():
        print(f"  {count:5d}  {tag}")

    nonnull_dropped = sum(n for tag, n in change_tags.items() if tag.endswith("(NON-NULL)"))
    if nonnull_dropped:
        print(f"\nWARNING: {nonnull_dropped} files carry a non-None value in a dropped field. "
              f"Inspect before rerunning with --apply.")
    if failures:
        print(f"\n{len(failures)} failures:")
        for path, outcome in failures[:20]:
            print(f"  {outcome}  {path.relative_to(root)}")
        return 1
    if not args.apply:
        print("\nNothing written. Rerun with --apply to migrate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
