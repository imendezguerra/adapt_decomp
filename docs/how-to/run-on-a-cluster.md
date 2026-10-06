# Run on a cluster

## A search on a scheduled node

`optimize_adapt_decomp` uses every core it may (`n_cores=None`), honouring CPU affinity and
the scheduler's allocation: `SLURM_CPUS_PER_TASK` on SLURM, `NCPUS` on PBS Pro and
`PBS_NUM_PPN` on Torque. Before starting, it checks the predicted memory against the job's
limit (SLURM's, or the cgroup's on PBS). Request the cores and memory the search needs:

```sh
#PBS -l select=1:ncpus=12:ompthreads=12:mem=64gb   # PBS Pro (NCPUS comes from ompthreads)
#SBATCH --cpus-per-task=12 --mem=64G     # SLURM
```

`n_jobs` (trials suggested together) is part of the search's definition, while `n_cores` only
sets its speed. With `n_cores = n_jobs x pool size`, every run gets exactly one thread; extra
cores become torch threads per run (on FDSI, with identical spikes). See
[Speed and resources](../guide/optimisation.md#speed-and-resources).

## Many recordings: array jobs

The [FDSI benchmark](../benchmarks/fdsi.md) is a complete template:

- a spec file declares the experiment;
- `python -m benchmarks.fdsi <stage>` runs one task per array index (`--array-index`,
  defaulting to `$PBS_ARRAY_INDEX`);
- outputs are cached by content, and each one records its provenance;
- `benchmarks/fdsi/pbs/submit.sh` chains the stages on PBS Pro.

```sh
bash benchmarks/fdsi/pbs/submit.sh --dry-run   # print the qsub commands
bash benchmarks/fdsi/pbs/submit.sh
```

To benchmark your own data, copy `benchmarks/fdsi/`, replace `fdsi.py`'s paths and loaders
with your dataset's, and edit the spec.

## Reproducibility

Pin one thread per run (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `torch.set_num_threads(1)`) and run on the CPU from
`environment.yaml`: outputs then reproduce bit for bit. Across BLAS libraries (MKL vs
OpenBLAS) they differ in the last digits.
