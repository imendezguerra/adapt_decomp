# Containers

Maintainer notes: the plan for publishing `adapt_decomp`'s CBSS and adaptation as a CPU-only
container image that runs on Linux, macOS and Windows, is tested in CI and released alongside
the PyPI package. Nothing here is implemented yet. The release process it extends is in
[`RELEASING.md`](RELEASING.md).

- [Background](#background)
- [Design decisions](#design-decisions)
- [Implementation steps](#implementation-steps)
- [Passing inputs: YAML and the CLI](#passing-inputs-yaml-and-the-cli)
- [CI/CD](#cicd)
- [User documentation](#user-documentation)
- [Kubernetes](#kubernetes)
- [Order of work](#order-of-work)

## Background

**The pieces:**
- **Dockerfile:** a recipe, one instruction per line. Each instruction makes a cached
  *layer*, and a rebuild reuses every layer above the first one that changed, so slow,
  stable steps go first.
- **Image:** the result of `docker build`, identified by a tag (`adapt-decomp:1.2.0`) and a
  digest (`sha256:…`). Only the digest is immutable.
- **Container:** a running instance of an image, thrown away after the run (`--rm`).
  Anything worth keeping is written to a mounted folder.
- **Registry:** stores images (GitHub Container Registry, GHCR). `docker pull` copies an
  image to the machine that runs it.
- **Apptainer:** runs the same images on HPC clusters, without root access.

**Where a container runs:** wherever it is started. It is an ordinary process on the host's
Linux kernel, isolated by namespaces and capped by cgroups. It is not a virtual machine.
On macOS and Windows, Docker Desktop runs a small Linux VM (on Windows through WSL2) and the
containers run inside it. On the HPC, the job scheduler allocates a node and Apptainer runs
the image inside that job.

**What the image fixes, and what the host decides:**

| Fixed by the image | Decided by the host at run time |
|---|---|
| OS libraries, Python, torch, numpy, the math library, `adapt_decomp`, their versions | CPU count, memory, GPU, CPU architecture, disk |

An image doesn't request resources. Mismatches are handled as follows:

| Mismatch | What happens | In this repo |
|---|---|---|
| CPU architecture | An amd64 image on arm64 runs under emulation (slow); without emulation it fails with `exec format error` | Publish a multi-architecture image: each host pulls its own variant |
| Memory | Not reserved, only capped; above the limit the kernel kills the process (exit 137) | The search reads the cgroup limit and fits to it; single `decompose` / `process_data` runs have no check |
| CPUs and threads | The container sees the cores its limits allow | `available_cores()` (`joblib.cpu_count`) respects container limits; a single run under `--cpus` alone may start too many torch threads (slower, not wrong) |
| GPU | Needs host support (`--gpus all`, `apptainer --nv`); MPS never reaches Linux containers | CPU-only image; `device: cuda` or `mps` in a YAML raises |
| Disk | Apptainer images are read-only; Docker discards writes outside mounts | Outputs must go to a mounted folder; `/dev/shm` is 64 MB by default (`--shm-size`) |

**Linux, macOS and Windows:** all three run the same Linux image; macOS and Windows run it in
Docker Desktop's VM. What differs:

| Topic | Linux | macOS | Windows |
|---|---|---|---|
| Image variant | amd64 or arm64 | arm64 (Apple Silicon) | amd64 (or arm64) |
| Resources | The machine, or `--cpus` / `--memory` | Docker Desktop settings | `%UserProfile%\.wslconfig` (WSL2 takes up to half the RAM by default) |
| Mount syntax | `-v "$PWD/data:/data"` | same | PowerShell `-v "${PWD}\data:/data"`, cmd `-v "%cd%\data:/data"` |
| Output file ownership | Needs `--user "$(id -u):$(id -g)"` | Handled by Docker Desktop | Handled by Docker Desktop |
| Mounted-file speed | Native | Good (VirtioFS) | Slow from `C:\`: keep large data in the WSL2 filesystem |
| GPU in containers | NVIDIA, with the container toolkit | Never | NVIDIA, through WSL2 |
| Apptainer | Yes (HPC) | No | No |

Docker Desktop is free for personal use, education, non-commercial open source and small
companies. Podman Desktop, Rancher Desktop and Colima (macOS) run the same images.

**Numerical reproducibility:** amd64 and arm64 use different math libraries, so their
results agree within tolerance (what the repro test checks), not bit for bit. The reference
for published numbers is **amd64**. Within amd64, `OPENBLAS_CORETYPE` (Step 7) makes results
identical across Intel and AMD CPUs.

## Design decisions

1. **CPU only, one Linux image for `linux/amd64` and `linux/arm64`.** The docs already say the
   GPU isn't much faster, and a CUDA image would be amd64-only and over 5 GB.
2. **The image contains the software only.** Data, configs and outputs stay with the user and
   are mounted at three fixed paths: `/data` (read-only), `/configs` (read-only) and
   `/outputs`.
3. **The image is built from the wheel, not from `src/` or from PyPI** (see below).
4. **Tests live in a separate `test` stage** of the Dockerfile, which CI builds and runs and
   which is never published.
5. **The `adapt-decomp` CLI is the image's interface.** It is the entry point, so a container
   takes the same arguments as the CLI.
6. **The hyperparameter search is in the image but not tested or documented there.** The CLI
   imports `optuna` and the `optimize` module at startup, so the package goes in whole.

### Build from the repository, from PyPI, or from the wheel?

| Source | Pros | Cons |
|---|---|---|
| `src/` in the repository (`pip install .`) | Works on any commit and PR | Builds the package a second time, separately from the PyPI wheel; needs `pyproject.toml`, `README.md`, `LICENSE` and a pinned setuptools in the image |
| PyPI (`pip install adapt-decomp==X.Y.Z`) | The exact file users install | Only exists after the PyPI upload, so PR images and pre-release tests are impossible; plain pip resolves pyproject's *ranges*, losing the conda pins and the math-library choice |
| **The wheel built once in CI (recommended)** | The **same file** goes to PyPI and into the image ("build once, publish everywhere"); works on PRs and pre-releases; the image needs no source, README or build tools | Building locally takes two commands (`python -m build --wheel`, then `docker build`) |

**How PyPI and the image relate:** both carry the same version and the same wheel, from the
same tagged commit.
- **PyPI:** the wheel with pyproject's lower bounds, for users who manage their own
  environment.
- **The image:** that wheel installed with `--no-deps` into the exact pins of
  `environment.runtime.yaml`, plus the OS libraries. It's the frozen, reproducible way to run
  that release.

Neither depends on the other: `publish.yml` builds the wheel once, and both publishing jobs
download that one artifact.

## Implementation steps

### Step 1. `environment.runtime.yaml`: the runtime pins

**Purpose:** `environment.yaml` is the development environment: it includes pytest, ruff,
pre-commit, mkdocs and wandb, and an editable install (`-e .`) that only works next to a
checkout. The image needs the runtime pins only.

```yaml
# Runtime pins for the container image: the runtime block of environment.yaml, unchanged
# (checked by ci/check_deps_sync.py), without the dev, docs and wandb tools.
name: adapt_decomp
channels: [conda-forge, nodefaults]
dependencies:
  - python=3.12.3
  - pip
  - libblas=*=*openblas    # one math library on both architectures, chosen explicitly
  - pytorch==2.11.0
  - numpy==1.26.4
  # ... every other runtime pin of environment.yaml, same versions
```

**Check that every pin exists on both Linux architectures** before writing the Dockerfile.
Linux on ARM is not covered by CI today:

```sh
micromamba create -n check --dry-run --platform linux-64      -f environment.runtime.yaml
micromamba create -n check --dry-run --platform linux-aarch64 -f environment.runtime.yaml
```

**Keep one source of truth:** extend `ci/check_deps_sync.py` so it fails if a pin in this
file differs from `environment.yaml`.

### Step 2. `.dockerignore`: what the build can see

**Purpose:** `docker build .` sends the folder (the *build context*) to Docker, including 28 GB
of `data/`. An allowlist keeps builds fast and the image minimal by construction.

```gitignore
# Exclude everything, then allow only what the image needs
*
!environment.runtime.yaml
!dist/*.whl
!tests/
**/__pycache__
```

`tests/` is only used by the `test` stage.

### Step 3. `Dockerfile`: the recipe

```dockerfile
# syntax=docker/dockerfile:1

# ---------- runtime: the published image ----------
# Pin by digest: docker buildx imagetools inspect mambaorg/micromamba:<tag>
FROM mambaorg/micromamba:2.3-debian12-slim@sha256:<digest> AS runtime

# 1. Environment: the slow step, re-run only when the file changes
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.runtime.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && micromamba clean --all --yes

# 2. Package: the wheel built by `python -m build --wheel` (one wheel in dist/)
ARG MAMBA_DOCKERFILE_ACTIVATE=1
COPY --chown=$MAMBA_USER:$MAMBA_USER dist/*.whl /tmp/dist/
RUN pip install --no-deps --no-cache-dir /tmp/dist/*.whl && rm -rf /tmp/dist

# 3. Runtime behaviour
ENV PATH=/opt/conda/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    ADAPT_DECOMP_DEVICE=cpu
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "adapt-decomp"]
CMD ["--help"]

# ---------- test: CI only, never published ----------
FROM runtime AS test
RUN micromamba install -y -n base -c conda-forge pytest=9.0.3 && micromamba clean --all --yes
COPY --chown=$MAMBA_USER:$MAMBA_USER tests/ /opt/adapt_decomp/tests/
WORKDIR /opt/adapt_decomp
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "-m", "pytest"]
CMD ["-m", "not slow and not repro", "--ignore=tests/benchmarks"]
```

What each choice does:
- **`@sha256:` on the base image:** tags can move; a digest can't.
- **`--no-deps`:** pip must not replace any conda-pinned package.
- **The wheel instead of `src/`:** the image holds the same file as PyPI, and no source,
  README or build tools.
- **`ENV PATH`:** `docker run --entrypoint` and `apptainer exec` skip `_entrypoint.sh`, which
  is what normally activates the environment.
- **`MPLCONFIGDIR`, `XDG_CACHE_HOME`:** writable for any user ID (`--user` on Linux, always
  under Apptainer).
- **`ADAPT_DECOMP_DEVICE`:** needs the small CLI change described in
  [Passing inputs](#passing-inputs-yaml-and-the-cli).
- **The `test` stage:** tests the installed wheel, like `ci.yml`'s wheel smoke test.
  `tests/benchmarks` imports `benchmarks/`, which isn't in the image. Check
  `tests/utils/test_download.py` in Step 5 too.

### Step 4. Build and run locally

```sh
rm -rf dist && python -m build --wheel                  # the wheel the image installs
docker build --target runtime -t adapt-decomp:dev .
docker run --rm adapt-decomp:dev                         # prints the CLI help
docker run --rm adapt-decomp:dev data list
```

Wrap the first two lines in a `make image` target.

With your own data, on Linux:
```sh
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/data:/data:ro" -v "$PWD/configs:/configs:ro" -v "$PWD/outputs:/outputs" \
  adapt-decomp:dev calibrate_and_process /data/rec_emg.npz --calib_stop 10240 \
  --calib_config /configs/cbss.yaml --preset muniverse --out_dir /outputs/rec
```
On macOS, the same command without `--user`. In Windows PowerShell:
```powershell
docker run --rm `
  -v "${PWD}\data:/data:ro" -v "${PWD}\configs:/configs:ro" -v "${PWD}\outputs:/outputs" `
  adapt-decomp:dev calibrate_and_process /data/rec_emg.npz --calib_stop 10240 `
  --calib_config /configs/cbss.yaml --preset muniverse --out_dir /outputs/rec
```

### Step 5. Test inside the image

```sh
docker build --target test -t adapt-decomp:test .
docker run --rm adapt-decomp:test                        # fast tests
docker run --rm -v "$PWD/data:/opt/adapt_decomp/data:ro" adapt-decomp:test \
  tests/reproducibility -m repro                         # NeuroMotion data mounted
```

`tests/reproducibility/fingerprint.py` looks for `<root>/data/neuromotion`, with the root
computed from its own location. That's why the data is mounted under `/opt/adapt_decomp`.

### Step 6. Both architectures

```sh
docker buildx create --use                                # one-off: a builder for several platforms
docker buildx build --platform linux/amd64,linux/arm64 --target runtime -t adapt-decomp:dev .
docker run --rm --platform linux/amd64 adapt-decomp:dev   # the x86 variant, emulated on a Mac
```

Run the quickstart in both variants and compare the numbers with the native run.

### Step 7 (optional). Identical numbers across x86 CPUs

`OPENBLAS_CORETYPE=Haswell` makes OpenBLAS use the same routines on every Intel and AMD CPU.
It is an x86 name, so it must only be set in the amd64 variant. Dockerfile instructions can't
be conditional, so a small entrypoint wrapper would set it based on the architecture.
Measure the speed cost first.

### Step 8. HPC (Apptainer)

```sh
apptainer pull adapt-decomp_1.2.0.sif docker://ghcr.io/imendezguerra/adapt-decomp:1.2.0
apptainer exec --bind data:/data:ro,outputs:/outputs adapt-decomp_1.2.0.sif \
  adapt-decomp calibrate_and_process /data/rec_emg.npz ...
```

Pull the `.sif` once to shared storage; every job reuses it. Jobs keep their PBS or SLURM
allocation, and the CLI respects it.

## Passing inputs: YAML and the CLI

**Keep YAML config files (mounted read-only), plus CLI arguments for data paths and per-run
options.** This is how the CLI already works, and it suits the repo:
- `CBSSConfig` and `AdaptConfig` serialise natively to YAML (`to_yaml` / `from_yaml`). The
  same files work natively and in the container.
- YAMLs can be partial (missing fields take their defaults), and `--preset` needs no file.
- For CBSS and the adaptation, configs contain no file paths, because data paths are CLI
  arguments. The container-path problem only affects the search's pool YAML, which is out of
  scope.

The alternatives are worse:
- **One flag per field:** over 50 fields, drifting from the dataclasses.
- **Environment variables:** flat strings, which can't express `ch_map` or `ch_mask`.
- **JSON:** a second format, no better than YAML.
- **Standard input:** no file to cite or re-run.
- **A single "run spec" file:** duplicates the CLI and brings container paths back.

**Gaps to fix before the first image release:**
1. **Device.** A YAML with `device: cuda` or `mps` fails in a CPU-only image. And
   `CBSSConfig` writes its resolved device into `calibration_config.yaml`: a calibration made
   natively on a Mac with `device: None` records `"mps"`, which then fails in the container
   (`calibrate_and_process` re-applies it through `CBSS.apply`). Fix: in `cli.py`'s
   `_calib_config` and `_adapt_config`, override `device` with `ADAPT_DECOMP_DEVICE` when it is
   set. The image sets it to `cpu`.
2. **Provenance.** `calibrate_and_process` and `process_data` save the CBSS config but not the
   AdaptConfig they used, so the defaults behind a partial YAML go unrecorded. Each command
   should write the resolved `adapt_config.yaml` and a `run.yaml` next to its outputs:
   package version and commit; image tag, and digest when passed as `ADAPT_DECOMP_IMAGE`; the
   command; SHA-256 checksums of the input files; and the environment (OS, architecture,
   torch, numpy, math library, threads).
3. **Pickled outputs.** They're tied to the package version, so users must re-load them with
   the same image tag. Unpickling someone else's file can also run code. Consider an `npz`
   export for sharing results.
4. **Input format.** Only `npz` (an array under the key `emg`) and the NeuroMotion HDF5 are
   read. Document `np.savez("rec_emg.npz", emg=array)`, or add `.mat` and `.csv` loaders.
5. **Seeds.** `random_seed` must stay fixed (1909 by default); `random_seed: null` gives a
   different decomposition on every run.

## CI/CD

The image follows the existing pattern: `ci.yml` tests every PR and push, and `publish.yml`,
on a GitHub release, re-runs CI, checks the version and waits for approval before
publishing.

### `container.yml`: build and test, never publish

- **Triggers:** `pull_request` and `push` to `main`, filtered to changes in `Dockerfile`,
  `.dockerignore`, `environment.runtime.yaml`, `src/**`, `tests/**` and `pyproject.toml`;
  plus `workflow_call` (for `publish.yml`) and `workflow_dispatch`.
- **Job `wheel`:** `python -m build --wheel`, uploaded as an artifact.
- **Job `image`:** a matrix of native runners, with no emulation:

  | Runner | Platform |
  |---|---|
  | `ubuntu-latest` | `linux/amd64` |
  | `ubuntu-24.04-arm` | `linux/arm64` |

  Each runner downloads the wheel and builds the `test` target, then runs it. It then builds
  `runtime` and smoke-tests it: `--help`, `data list`, and a `calibrate_and_process` on a
  synthetic recording, written to a mounted temporary folder with `--user`.
- **Cache:** `docker/build-push-action` with `cache-from` / `cache-to: type=gha,scope=<arch>`,
  so an unchanged environment layer rebuilds in seconds.
- **Job `repro`:** restore the `neuromotion-data-zenodo.22880910` cache, as `ci.yml`'s
  `pinned-env` job does, and run Step 5's repro command on both architectures.

Skeleton of the image job:
```yaml
  image:
    needs: wheel
    strategy:
      fail-fast: false
      matrix:
        include:
          - { runner: ubuntu-latest,    platform: linux/amd64, arch: amd64 }
          - { runner: ubuntu-24.04-arm, platform: linux/arm64, arch: arm64 }
    runs-on: ${{ matrix.runner }}
    steps:
      - uses: actions/checkout@v7
      - uses: actions/download-artifact@v8
        with: { name: wheel, path: dist/ }
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v6
        with:
          context: .
          target: test
          platforms: ${{ matrix.platform }}
          load: true
          tags: adapt-decomp:test
          cache-from: type=gha,scope=${{ matrix.arch }}
          cache-to: type=gha,mode=max,scope=${{ matrix.arch }}
      - run: docker run --rm adapt-decomp:test
```
Check the latest majors of the Docker actions when writing it.

### `publish.yml`: push only on a release

- **Gates:** call `container.yml` next to the existing `ci.yml` call. The image jobs depend on
  the existing `build` job, so they run only after the tag matches `__version__`,
  `CITATION.cff` and `CHANGELOG.md`, and they download its `dist` artifact: the same wheel
  PyPI receives.
- **Approval:** a new `ghcr` environment, with you as required reviewer and the `v*` tag rule,
  like `pypi`.
- **Publishing:**
  - each architecture builds `runtime` natively and pushes it **by digest only**
    (`outputs: type=image,push-by-digest=true,push=true`);
  - a final job joins the digests into one multi-architecture tag:
    `docker buildx imagetools create`;
  - `docker/metadata-action` derives the tags (`X.Y.Z`, `X.Y`, `latest`) and the OCI labels
    (source, version, revision, licences).
- **Permissions:** `packages: write` (push with `GITHUB_TOKEN`, no stored secret),
  `id-token: write` and `attestations: write`.
- **Supply chain:** `provenance: true` and `sbom: true` in `build-push-action`, plus
  `actions/attest-build-provenance`. Users can then verify the image was built by this
  workflow from this commit.
- **Citation:** attach `image-digest.txt` (the final digest) to the GitHub release, next to
  the wheel and the sdist.
- **Rehearsal:** `workflow_dispatch` pushes a pre-release tag (`X.Y.Zrc1`), the equivalent of
  the TestPyPI rehearsal.

### Policies (add to `RELEASING.md`)

- **Never re-push a released tag.** GHCR allows overwriting tags, unlike PyPI. A broken image
  gets a new patch version.
- **One-off setup:** the first push creates a *private* GHCR package. Make it public and link
  it to the repository. Create the `ghcr` environment.
- **Maintenance:** Dependabot is off, so add the base-image digest and the Docker actions to
  the manual-update list.
- **macOS and Windows:** hosted macOS runners can't run Docker, and Windows runners only run
  Windows containers. CI tests the Linux image, which is exactly what Docker Desktop runs.
  Release-checklist item: run the user docs' commands once on a Mac and once on Windows.

## User documentation

### New page: `docs/how-to/run-in-a-container.md`

Add it to `mkdocs.yml`'s *User guide*, after *Run on a cluster*. Outline:

1. **When to use it:** to reproduce results with a frozen software stack, or to run without
   installing Python. For speed and accurate timings, the native install is better.
2. **Install a container engine:** Docker Desktop (macOS, Windows), Docker Engine (Linux),
   Podman or Rancher Desktop as alternatives, Apptainer on clusters. Include the
   resource settings: Docker Desktop's *Resources*, and `.wslconfig` on Windows.
3. **Get the image:** `docker pull ghcr.io/imendezguerra/adapt-decomp:X.Y.Z`. Pin the version;
   don't use `latest` for work you will publish.
4. **Organise a study folder:** `data/`, `configs/`, `outputs/`, and the three mount points.
5. **Prepare the data:** the `npz` format, with a conversion snippet from MATLAB and CSV.
6. **Write the configs:** partial YAMLs, presets, `device` (always CPU in the image), and why
   `random_seed` must stay fixed.
7. **Run:** `decompose`, `process_data` and `calibrate_and_process`, with OS tabs (the theme
   already enables `pymdownx.tabbed`):

   ````markdown
   === "Linux"

       ```sh
       docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/data:/data:ro" ...
       ```

   === "macOS"

       ```sh
       docker run --rm -v "$PWD/data:/data:ro" ...
       ```

   === "Windows (PowerShell)"

       ```powershell
       docker run --rm -v "${PWD}\data:/data:ro" ...
       ```
   ````

8. **Outputs:** `calibration.pkl`, `calibration_config.yaml`, `adapted.pkl`, plus
   `adapt_config.yaml` and `run.yaml` once provenance lands. Pickles must be re-loaded with the
   same image tag.
9. **Use the Python API:** mount a script and run it with `--entrypoint python`.
10. **Reproduce and cite:** cite the image tag and digest
    (`docker inspect --format '{{index .RepoDigests 0}}' <image>`), keep `run.yaml`, and expect
    the same numbers on amd64, and agreement within tolerance on arm64.
11. **On a cluster:** `apptainer pull` / `apptainer exec`, with a PBS and a SLURM job example
    that links to *Run on a cluster*.
12. **Troubleshooting:**

    | Symptom | Cause | Fix |
    |---|---|---|
    | Exit code 137, no error | Out of memory in the container or VM | Raise Docker Desktop's or WSL2's memory, or `--memory` |
    | `exec format error` | Wrong architecture, no emulation | Pull without `--platform`; the image is multi-architecture |
    | `Permission denied` writing outputs (Linux) | Container runs as another user | Add `--user "$(id -u):$(id -g)"` |
    | Very slow on Windows | Data mounted from `C:\` | Keep data in the WSL2 filesystem |
    | Device error (`cuda`, `mps`) | GPU set in a YAML | The image is CPU-only: set `device: cpu`, or rely on `ADAPT_DECOMP_DEVICE` |
    | `No such file` for a path that exists | Host path used instead of a container path | Use `/data/...`, `/configs/...`, `/outputs/...` |

### Other pages to update

- **`README.md` install section** (shown in `installation.md`): a third route, "Run with a
  container", linking to the new page.
- **`docs/how-to/run-on-a-cluster.md`:** an Apptainer section next to the conda one.
- **`docs/how-to/use-the-command-line.md`:** `ADAPT_DECOMP_DEVICE`, and the files each command
  writes.
- **`docs/how-to/index.md`:** a row for the new page.
- **`CHANGELOG.md`:** the image, `environment.runtime.yaml`, `ADAPT_DECOMP_DEVICE` and the
  provenance files.
- **`RELEASING.md`:** the image steps in the overview table and the checklist, the one-off
  GHCR setup, and the policies above.
- **`CONTRIBUTING.md`:** building and testing the image locally (`make image`, Step 5).

## Kubernetes

Kubernetes orchestrates containers across a cluster of machines: it schedules them, restarts
failures and scales services. It runs the same images unchanged, so nothing in this plan
blocks it. This repo doesn't need it:
- users run batch jobs on laptops and HPC clusters, where PBS or SLURM plus Apptainer is the
  standard;
- the adaptation isn't deployed as a long-running service.

**Where it would become relevant:**
- a lab whose compute is a Kubernetes cluster: one *Job* per recording, or an *Indexed Job*
  for a batch;
- serving the online adaptation as a streaming service.

**How the image would behave there:** a Job's `resources.requests` and `resources.limits`
become cgroup limits, which the search's planner already reads. Data would come from a
*PersistentVolumeClaim* mounted at `/data`. A minimal Job sets the image by digest, the
`args` (the CLI command), CPU and memory limits, and volume mounts.

**Recommendation:** don't maintain Kubernetes manifests. If a user asks, add one example Job
to the container page as an *advanced* note.

## Order of work

1. Steps 1–5 locally: runtime environment, `.dockerignore`, Dockerfile, build, test.
2. Step 6: both architectures, and compare the quickstart numbers.
3. `container.yml`: build and test on every PR.
4. The CLI gaps: `ADAPT_DECOMP_DEVICE` and the provenance files.
5. The release jobs in `publish.yml`, with the `ghcr` environment and a pre-release rehearsal.
6. User documentation: the new page and the updates above.
7. Optional: Step 7 (`OPENBLAS_CORETYPE`).
