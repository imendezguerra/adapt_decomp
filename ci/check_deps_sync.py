"""Check that the three dependency specs agree with each other.

- pyproject.toml          -> lower bounds (what pip users get)
- environment.yaml        -> exact pins (the reproducible environment)
- ci/constraints-min.txt  -> the lower bounds, pinned (tested by the minimum-deps CI job)

Fails (exit 1) if a runtime dependency is missing from any of them, if an
environment.yaml pin is not exact or falls outside pyproject's range, or if a
constraints-min pin differs from pyproject's lower bound.
"""

import sys
from pathlib import Path

import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
# conda-forge name -> PyPI name, where they differ.
CONDA_TO_PYPI = {"pytorch": "torch"}
# environment.yaml entries that are not pyproject runtime dependencies.
NOT_RUNTIME = {"python", "pip", "pytest", "ruff", "pre-commit", "ipykernel"}


def lower_bound(req: Requirement) -> Version | None:
    """Return the version in a requirement's >= or == specifier, if any."""
    for spec in req.specifier:
        if spec.operator in (">=", "=="):
            return Version(spec.version)
    return None


def main() -> int:
    errors = []

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    runtime = {
        canonicalize_name(r.name): r for r in map(Requirement, pyproject["project"]["dependencies"])
    }

    env = yaml.safe_load((ROOT / "environment.yaml").read_text())
    pins = {}
    for dep in env["dependencies"]:
        if not isinstance(dep, str):  # the pip: sub-list
            continue
        name, _, version = dep.partition("==")
        name = canonicalize_name(name.split("=")[0].strip())
        if name in NOT_RUNTIME:
            continue
        if not version:
            errors.append(f"environment.yaml: {dep!r} is not pinned with ==")
            continue
        pins[CONDA_TO_PYPI.get(name, name)] = Version(version.strip())

    constraints = {}
    for line in (ROOT / "ci" / "constraints-min.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, _, version = line.partition("==")
            constraints[canonicalize_name(name)] = Version(version)

    for name, req in runtime.items():
        bound = lower_bound(req)
        if bound is None:
            errors.append(f"pyproject.toml: {req} has no lower bound (>=)")
        if name not in pins:
            errors.append(f"environment.yaml: missing {name} (required by pyproject.toml)")
        elif not req.specifier.contains(pins[name], prereleases=True):
            errors.append(f"environment.yaml: {name}=={pins[name]} does not satisfy {req}")
        if name not in constraints:
            errors.append(f"ci/constraints-min.txt: missing {name}")
        elif bound is not None and constraints[name] != bound:
            errors.append(
                f"ci/constraints-min.txt: {name}=={constraints[name]} != pyproject lower bound {bound}"
            )

    for name in sorted(pins.keys() - runtime.keys()):
        errors.append(f"environment.yaml: {name} is not a pyproject.toml dependency")
    for name in sorted(constraints.keys() - runtime.keys()):
        errors.append(f"ci/constraints-min.txt: {name} is not a pyproject.toml dependency")

    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
