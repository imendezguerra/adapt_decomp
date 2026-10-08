# Contributing

Thanks for helping improve adapt_decomp! This guide covers setting up a development
environment, the checks that run on your machine and on GitHub, and what to do when one fails.

- [Development setup](#development-setup)
- [Making a change](#making-a-change)
- [Running tests](#running-tests)
- [Documentation](#documentation)
- [Dependencies](#dependencies)
- [Reproducibility across operating systems](#reproducibility-across-operating-systems)
- [Automated checks](#automated-checks)
- [When a check fails](#when-a-check-fails)

Maintainers: the release process is in [`.github/RELEASING.md`](.github/RELEASING.md).

## Development setup

1. Fork the repository on GitHub, then clone your fork and move into it:
    ```sh
    git clone https://github.com/<your-username>/adapt_decomp.git
    cd adapt_decomp
    ```
2. Create and activate the conda environment. It pins every dependency to an exact version
   (see [Reproducibility](#reproducibility-across-operating-systems)) and installs the package
   in editable mode with its `dev` and `docs` extras:
    ```sh
    conda env create -f environment.yaml
    conda activate adapt_decomp
    ```
    Without conda, use a virtual environment with Python 3.10 to 3.12, install the CPU build of
    PyTorch first, then the package:
    ```sh
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -e ".[dev,docs]"
    ```
    Dependency versions are then not pinned.
3. Install the git hooks (see [pre-commit](#local-git-hooks-pre-commit)):
    ```sh
    pre-commit install
    ```
4. Download the data the reproducibility test, the paper example and the docs examples use (see
   [Running tests](#running-tests)):
    ```sh
    adapt-decomp data get fdsi_example-data   # 70 MB: the docs examples
    adapt-decomp data get neuromotion-data    # 1.6 GB: the reproducibility test and the paper example
    ```

## Making a change

1. Create a feature branch: `git checkout -b feature/newfeature`.
2. Make your change and add tests for it under `tests/`, in the folder matching the
   subpackage (`tests/cbss/`, `tests/adaptation/`, ...).
3. Make sure the tests pass (`make test`, and `make test-all` for changes to the pipelines).
4. Add an entry to the `[Unreleased]` section of `CHANGELOG.md`.
5. Commit. The pre-commit hooks check the staged files first.
6. Push the branch (the pre-push hook runs the fast tests) and open a pull request.
7. CI and the docs build run on the PR automatically. A PR is ready to merge when all checks
   are green.

Keep in mind:

- **Docstrings.** The API reference is generated from them (Google style), so update a
  function's docstring when you change its arguments. Config fields are documented in their
  class's `Attributes:` section.
- **Docs code.** The code on the quickstart and how-to pages comes from `docs/snippets/*.py`.
  Edit the snippet, not the page, and run it to check it still works.
- **The FDSI benchmark.** Its outputs are only recomputed when missing. When a change alters
  what a stage of `benchmarks/fdsi/` writes, run it under a new `version` in `config.yaml` (or
  delete the old outputs), so old and new results never mix.

## Running tests

```sh
make test       # fast tests: pytest -m "not slow and not repro"
make test-all   # every test, including the slow ones
make repro      # the reproducibility test: pytest tests/reproducibility -m repro
```

| Marker | Tests | Needs |
|---|---|---|
| (none) | Unit tests of every subpackage, and calibration, adaptation and a search end to end on a synthetic recording with ground truth (`tests/test_pipeline.py`) | Nothing |
| `slow` | Optuna searches across worker processes, multi-batch adaptation loops, the FDSI benchmark stages end to end on synthetic recordings | Nothing |
| `repro` | The paper example's adaptation against a stored reference | `neuromotion-data` (`make data`); skipped without it |

## Documentation

The site (https://imendezguerra.github.io/adapt_decomp/) is built with MkDocs Material:

- `docs/`: the pages. `getting-started/`, `guide/` (the API concepts), `how-to/` (the user
  guide), `benchmarks/` and `reference/` (the API reference, generated from the docstrings by mkdocstrings).
- `README.md`: the overview, installation and citation sections are included in the docs from
  between the `<!-- --8<-- [start:...] -->` markers. Keep its links absolute: it is also the
  PyPI page.
- Notebooks: the paper example (`notebooks/original_tutorial/`) and the benchmark's dataset and results notebooks are added from
  outside `docs/` by `docs/hooks.py`, and rendered from their stored outputs (they are not
  executed by the build). Re-run a notebook and commit its outputs to update its page.
- Figures: the plots on the *Plot results* page (`docs/assets/how-to/*.png`) are the ones
  `docs/snippets/workflow.py` saves to `data/fdsi_example/outputs/docs-example/`. After changing
  its plot sections, run it and copy them over.

To preview the site, run `make docs` (`mkdocs serve`) and open the address it prints. It
reloads when you edit `docs/`, `README.md`, `CHANGELOG.md` or the source. `make docs-build`
(`mkdocs build --strict`) builds it as CI does: any warning, such as a broken link or anchor,
fails it.

## Dependencies

Dependencies are declared in three places that must agree:

| File | Holds | Tested by |
|---|---|---|
| `pyproject.toml` | Lower bounds (what `pip install` resolves) | CI on Python 3.10 to 3.12, Linux, macOS and Windows; weekly against new releases |
| `environment.yaml` | Exact pins of every direct dependency (the reproducible environment) | CI on Linux, macOS and Windows |
| `ci/constraints-min.txt` | The lower bounds, pinned | CI `minimum-deps` job |

To add or change a dependency, update all three. `python ci/check_deps_sync.py` checks they
agree; it also runs in pre-commit and CI. `numpy<2` is pinned deliberately: don't relax it
without checking PyTorch and SciPy compatibility first.

## Reproducibility across operating systems

`environment.yaml` pins every direct dependency to an exact version, without build strings, so
the same file resolves on Linux, macOS and Windows. With it:

- **On one machine**, a CPU run reproduces bit for bit: calibration with a fixed
  `random_seed`, adaptation (which has no randomness), and searches with a fixed `random_seed`
  and `n_jobs`.
- **Across operating systems**, PyTorch uses different BLAS libraries (MKL, OpenBLAS,
  Accelerate), so results differ in the last digits. `tests/reproducibility/` checks that this
  stays negligible: it re-runs the paper example's adaptation and compares it with a stored
  reference (per-unit spike agreement of at least 0.99, rate of agreement with the ground truth
  within 0.5 percentage points, losses within a relative 1e-3). CI runs it on all three OSes.
- **Threads** change the order of floating-point sums. Pin one thread per run
  (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `torch.set_num_threads(1)`) when results must not depend on the machine's core count, as the
  FDSI benchmark does.
- **GPUs** are not bit-for-bit deterministic. Use the CPU for results that must reproduce.

Regenerate the reference (`make reference`) only when results are *meant* to change, and say
so in `CHANGELOG.md`.

## Automated checks

There are two kinds of automation, and they run in different places:

- **The git hooks (pre-commit)** run **on your machine** during `git commit` and `git push`.
- **GitHub Actions workflows** (`.github/workflows/*.yml`) run **on GitHub's servers** when
  you push or open a PR. Results appear as checks on the PR and in the repository's **Actions**
  tab.

| Event | pre-commit (local) | `ci.yml` | `docs.yml` |
|---|---|---|---|
| `git commit` on your machine | ✅ on staged files | | |
| `git push` from your machine | ✅ fast tests | | |
| Push to a PR / open a PR | | ✅ | build |
| Push / merge to `main` | | ✅ | build |
| Every Monday | | ✅ | |

Releases use two more steps, `publish.yml` and the docs deploy, described in
[`.github/RELEASING.md`](.github/RELEASING.md).

### Local git hooks: pre-commit

Configured in `.pre-commit-config.yaml`. `pre-commit install` installs both hooks:

- **On commit**, on the staged files only:
  - file hygiene: trailing whitespace, final newlines, valid YAML and TOML, no merge-conflict
    markers, no files over 1 MB (notebooks excepted), no leftover `breakpoint()`;
  - `ruff check --fix` and `ruff format`, with the rules in `pyproject.toml`;
  - the dependency sync check, when a dependency file changes.
- **On push**: the fast tests (`pytest -m "not slow and not repro"`).

If a hook fails or fixes something, **the commit is aborted**. Review the changes, `git add`
them, and commit again.

```sh
pre-commit run --all-files   # check the whole repository, not just staged files
git commit --no-verify       # skip the hooks once (CI will still run them)
```

### CI: `ci.yml`

**Triggers:** every PR, every push to `main`, a weekly run (catching breakage from new
upstream releases), manual runs, and `publish.yml` before a release. A newer push to the same
branch cancels the run still in progress.

| Job | What it does | Catches |
|---|---|---|
| **Lint** | Runs the pre-commit hooks on all files | Style and lint errors, commits made with `--no-verify` |
| **Test** (matrix) | `pip install -e ".[dev]"` with CPU PyTorch, then the fast tests, on Linux for Python 3.10, 3.11 and 3.12, plus macOS and Windows on 3.12; one Linux job also runs the slow tests | Bugs, Python-version and OS-specific breakage |
| **Minimum deps** | Python 3.10 with the oldest supported versions from `ci/constraints-min.txt`, on all three OSes | Code that silently needs a newer dependency than `pyproject.toml` claims |
| **Build** | Builds the sdist and wheel, runs `twine check --strict`, installs the wheel in a clean environment and loads every preset | Packaging mistakes: missing files, broken metadata, a README PyPI cannot render |
| **Pinned environment** | Creates `environment.yaml` with micromamba on all three OSes, runs the fast tests and the reproducibility test | Changes in results across OSes; a pin that no longer resolves |

Jobs run in parallel, and one failing job does not cancel the others.

### Docs build: `docs.yml`

On every PR and push to `main` it runs `mkdocs build --strict`. The site is only *deployed*
when a release is published, so it always describes the latest release.

## When a check fails

- **The commit is aborted by pre-commit:** the hook either fixed files itself (`git add` them
  and commit again) or printed the errors to fix.
- **Lint is red in CI:** run `pre-commit run --all-files` locally and commit the result.
- **A test fails on one Python version or OS only:** reproduce it in an environment with that
  version. With [uv](https://docs.astral.sh/uv/):
    ```sh
    uv venv -p 3.X .venv-3X && source .venv-3X/bin/activate
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install -e ".[dev]" && pytest -m "not slow and not repro"
    ```
- **Minimum deps fails:** the change uses a feature newer than the lowest version allowed in
  `pyproject.toml`. Either avoid it, or raise the lower bound in `pyproject.toml`, the pin in
  `ci/constraints-min.txt`, and check `environment.yaml` still satisfies it.
- **The reproducibility test fails:** the results changed. If that was intended, regenerate
  the reference (`make reference`) and record the change in `CHANGELOG.md`. If it fails on one
  OS only, CI uploads that platform's fingerprint as an artifact to compare.
- **Build fails:** run `python -m build` and `twine check --strict dist/*` locally (needs
  `pip install build twine`).
- **Docs build fails:** run `make docs-build` locally. The warning names the file and line,
  often a broken link or anchor, or a docstring argument that does not match the signature.
