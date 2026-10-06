# Releasing

Maintainer notes for publishing a new version: the package on [PyPI](https://pypi.org/project/adapt-decomp/),
its data on [Zenodo](https://zenodo.org), the benchmark results and the documentation.
Day-to-day development checks (pre-commit, `ci.yml`, the docs build) are described in
[`CONTRIBUTING.md`](../CONTRIBUTING.md#automated-checks).

- [Overview](#overview)
- [Versions](#versions)
- [Data archives](#data-archives)
- [Benchmark results](#benchmark-results)
- [Release checklist](#release-checklist)
- [One-off setup](#one-off-setup)
- [Maintenance](#maintenance)
- [When a release fails](#when-a-release-fails)

## Overview

A release ships code, data and a citation together. They reference each other, so they are
published in dependency order: the data first (its DOIs are baked into the code), then the
code, then the back-links from the data to the release.

Publishing a GitHub release, or clicking *Run workflow*, starts:

| Event | `ci.yml` | `docs.yml` | `publish.yml` |
|---|---|---|---|
| Publish a GitHub release | ✅ (called by publish) | build + **deploy** | upload to **PyPI** |
| *Run workflow* button | ✅ (called by publish) | build + **deploy** | upload to **TestPyPI** |

`publish.yml` runs the whole CI suite, builds the wheel and sdist once, checks that the tag
matches `__version__`, `CITATION.cff` and a `CHANGELOG.md` section, then uploads through PyPI
Trusted Publishing (no stored token) once you approve the `pypi` environment, and attaches the
files to the GitHub release. With Zenodo's GitHub integration on, the release is also archived
with a DOI.

> **Versions are permanent.** PyPI never accepts the same version twice, even after deleting
> it. A broken release can only be *yanked* and followed by a new version. Rehearse on TestPyPI
> first.

## Versions

The package follows [semantic versioning](https://semver.org): `MAJOR` for breaking changes
(removing or renaming a public function, changing results on purpose), `MINOR` for new
features, `PATCH` for bug fixes. The version lives in:

- `src/adapt_decomp/__init__.py` (`__version__`), which `pyproject.toml` reads;
- `CITATION.cff` (`version:` and `date-released:`);
- `CHANGELOG.md` (a `## [X.Y.Z] - YYYY-MM-DD` section, from `[Unreleased]`).

Each data archive has its own version, set on its Zenodo record and bumped only when that
archive's contents change, so a code release doesn't force a new upload of unchanged data.

## Data archives

`adapt-decomp-data` downloads the archives listed in `ARCHIVES`
(`src/adapt_decomp/utils/download.py`), each pinned to a Zenodo **version** DOI. An archive with
an empty DOI is packed but not published yet. Archives are named `adapt_decomp-<dataset>-<kind>.zip`
and every entry carries its full path from `data/`, so any subset unpacks with
`unzip '*.zip' -d data/`.

For each archive that is new or changed in this release (for 1.1.0: `fdsi_example-data`, new,
and `fdsi_benchmark-outputs`, with the v1.1 benchmark outputs):

1. **Pack** it into `dist/`, and check the entries (they must start with the dataset folder,
   never `data/` or an absolute path):
    ```sh
    python scripts/pack_data.py fdsi_example-data
    python -c "import zipfile; print(zipfile.ZipFile('dist/adapt_decomp-fdsi_example-data.zip').namelist()[:5])"
    ```
2. **Rehearse** on [sandbox.zenodo.org](https://sandbox.zenodo.org), a throwaway copy of
   Zenodo: create a record, upload, and download it with `adapt-decomp-data` pointed at it.
3. **Create the draft** on Zenodo (*New upload*, or *New version* of an existing record), fill
   in the metadata (version, description, licence, related identifiers) and **reserve its
   DOI**.
4. **Upload** the zip through the API rather than the browser, which fails quietly on large
   files. With a token scoped to `deposit:write` only, so it can upload but not publish:
    ```sh
    BUCKET=$(curl -s "https://zenodo.org/api/deposit/depositions/<ID>?access_token=$ZENODO_TOKEN" | python -c "import json,sys; print(json.load(sys.stdin)['links']['bucket'])")
    curl --upload-file dist/adapt_decomp-fdsi_example-data.zip "$BUCKET/adapt_decomp-fdsi_example-data.zip?access_token=$ZENODO_TOKEN"
    ```
    Check the MD5 Zenodo shows against `md5sum dist/adapt_decomp-fdsi_example-data.zip`.
5. **Bake the version DOI** into `ARCHIVES` in `src/adapt_decomp/utils/download.py` (and its
   size), into the data table of `docs/getting-started/installation.md`, and the concept DOI
   into the dataset's README if it cites itself.
6. **Publish** the draft, then check the download end to end:
    ```sh
    adapt-decomp-data get fdsi_example-data --dest /tmp/check
    ```

## Benchmark results

The [Results](https://imendezguerra.github.io/adapt_decomp/benchmarks/fdsi/report/) page is
rendered from the report notebook's stored outputs, which come from the full benchmark run:

1. Commit everything: every output records its commit and whether the tree was dirty.
2. On the cluster: `bash benchmarks/fdsi/pbs/submit.sh`. When it finishes, `verify` a few
   tasks of each stage, e.g. `python -m benchmarks.fdsi verify apply --tasks 0,250,599`.
3. Execute the report against the full tables and keep its outputs:
   `jupyter nbconvert --to notebook --execute --inplace benchmarks/fdsi/report.ipynb`.
   Remove its "Results pending" note, then commit it.
4. Pack and publish the outputs as a new version of `fdsi_benchmark-outputs`
   ([Data archives](#data-archives)).

## Release checklist

1. **Choose the version** ([Versions](#versions)).
2. **Prepare a PR to `main`** that:
    - publishes the new or changed data archives ([Data archives](#data-archives)) and bakes
      their DOIs into `ARCHIVES`;
    - updates the benchmark results if they changed ([Benchmark results](#benchmark-results));
    - bumps `__version__` in `src/adapt_decomp/__init__.py`;
    - bumps `version:` and `date-released:` in `CITATION.cff`;
    - renames `## [Unreleased]` in `CHANGELOG.md` to `## [X.Y.Z] - YYYY-MM-DD` and opens a new
      empty `[Unreleased]` section.
3. **Merge** once CI and the docs build are green.
4. **Rehearse on TestPyPI:** *Actions → publish → Run workflow* on `main`. Then, in a fresh
   environment:
    ```sh
    python -m venv /tmp/ad-test && source /tmp/ad-test/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ adapt-decomp
    python -c "import adapt_decomp; print(adapt_decomp.__version__)"
    adapt-decomp-data list
    ```
    TestPyPI also refuses a version it has already seen. To rehearse twice, use a pre-release
    version such as `1.1.0rc1`. The manual run also deploys the docs from `main`.
5. **Tag the release commit:**
    ```sh
    git checkout main && git pull                 # tag exactly what is on GitHub
    git tag -a vX.Y.Z -m "Release vX.Y.Z"
    git push origin vX.Y.Z
    ```
    Pushing a tag does **not** publish anything. Only publishing a GitHub release starts
    `publish.yml`.
6. **Release:** *Releases → Draft a new release*, choose the tag, paste the `CHANGELOG.md`
   section as the notes (flagging deprecations and the data DOIs), and click **Publish
   release**.
7. **Approve** the `pypi` deployment when the workflow pauses at *Upload to PyPI*.
8. **Verify:**
    - `pip install adapt-decomp` installs the new version;
    - the [PyPI page](https://pypi.org/project/adapt-decomp/) and the
      [docs](https://imendezguerra.github.io/adapt_decomp/) show it;
    - Zenodo has archived the release under the software's concept DOI.
9. **Back-link the data:** on each data record published for this release, add a *related
   identifier* ("is supplement to") pointing at the release's DOI. This edits metadata only,
   so no new data version is needed.

## One-off setup

1. **Accounts:** create accounts with two-factor authentication on
   [pypi.org](https://pypi.org) and [test.pypi.org](https://test.pypi.org). They are separate
   sites.
2. **Trusted publishers:** on each site, go to *Your projects → Publishing → Add a new pending
   publisher* (GitHub tab):

    | Field | Value |
    |---|---|
    | PyPI project name | `adapt-decomp` |
    | Owner | `imendezguerra` |
    | Repository name | `adapt_decomp` |
    | Workflow name | `publish.yml` |
    | Environment name | `pypi` (on PyPI) / `testpypi` (on TestPyPI) |

    A pending publisher does not reserve the name. The project is created by the first upload.

3. **GitHub environments:** in *Settings → Environments*, create `pypi` and `testpypi`. Add
   yourself as a **required reviewer** on `pypi`, so every upload waits for your approval,
   and leave *Prevent self-review* off. Under *Deployment branches and tags* on `pypi`, choose
   *Selected branches and tags* and add the **tag** rule `v*`: releases run on the tag, so a
   branch-only rule blocks the upload. `testpypi` can stay unrestricted.
4. **GitHub Pages:** in *Settings → Pages*, set *Source* to **GitHub Actions**. In *Settings →
   Environments → github-pages*, add the **tag** rule `v*` next to `main`, or the docs deploy on
   release fails.
5. **Zenodo:** sign in to [Zenodo](https://zenodo.org) with GitHub and switch on this
   repository. Every GitHub release is then archived with a DOI, using the metadata in
   `CITATION.cff`.

## Maintenance

Dependabot is off, so these updates are manual:

- **Action versions:** in all three workflows, check the actions' GitHub release pages a few
  times a year. Keep `upload-artifact` and `download-artifact` on majors released together.
- **Hook versions:** `pre-commit autoupdate`, then `pre-commit run --all-files`.
- **Dependency pins:** update `environment.yaml` when moving to new versions, check the
  reproducibility test, and regenerate its reference only if results are meant to change.
- **Python versions:** the test matrix stops at 3.12 because `numpy<2` has no wheels for 3.13.
  When the numpy pin is lifted, extend the matrix in `ci.yml`, the classifiers in
  `pyproject.toml` and the `minimum-deps` Python version.
- **MkDocs:** capped at `<2`, because MkDocs 2.0 drops the plugin system (mkdocstrings,
  mkdocs-jupyter). [Zensical](https://zensical.org) reads `mkdocs.yml` and is the likely
  migration path.
- **Deprecations:** `optimize_adapt_decomp_pooled_memory`, `optimize_adapt_decomp_pooled_disk`
  and their `_pareto` variants, and the `AdaptDecomp(emg=...).run()` path, warn with
  `FutureWarning`; remove them in the next major version.

## When a release fails

- **Tests fail in the `test` job:** nothing was uploaded. Fix the problem on `main` and publish
  a new release. Delete the failed release and its tag first to reuse the version number.
- **Version check fails:** the tag disagrees with `__version__`, `CITATION.cff` or
  `CHANGELOG.md`, and nothing was uploaded. Delete the release and its tag, fix the version on
  `main`, and release again:
    ```sh
    git push origin --delete vX.Y.Z
    git tag -d vX.Y.Z
    git checkout main && git pull
    git tag -a vX.Y.Z -m "Release vX.Y.Z"
    git push origin vX.Y.Z
    ```
    Only reuse a tag if nothing reached PyPI. Once a version is uploaded, release the next one.
- **Upload fails with an OIDC / "invalid publisher" error:** the trusted publisher on PyPI
  doesn't match exactly. Check the workflow file name, the environment name and the
  repository owner and name.
- **The run waits forever at *Upload to PyPI*:** it is waiting for approval. Open the run and
  click *Review deployments*.
- **Upload blocked by environment rules:** the `pypi` environment is missing the `v*` tag rule
  (one-off setup, step 3).
- **Docs deploy fails on release:** the `github-pages` environment is missing the `v*` tag
  rule (one-off setup, step 4).
- **A bad version reached PyPI:** on PyPI, *Manage → Releases → Yank* it, then fix the problem
  and release a new patch version.
