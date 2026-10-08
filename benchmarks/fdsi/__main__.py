"""python -m benchmarks.fdsi STAGE: run one stage of the FDSI benchmark (see pipeline.py).

python -m benchmarks.fdsi calibrate --workers 8         # every recording, 8 at a time
python -m benchmarks.fdsi apply --index 3 --chunk 4     # one array job's tasks ($PBS_ARRAY_INDEX)
python -m benchmarks.fdsi apply --count                 # how many tasks the stage has
python -m benchmarks.fdsi search --quick                # the quick check (config.yaml's quick)
"""

import argparse
import os

# One BLAS/OpenMP thread per process, fixed before numpy and torch load, as in the published runs
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

from benchmarks.fdsi import pipeline  # noqa: E402

parser = argparse.ArgumentParser(
    prog="python -m benchmarks.fdsi", description=__doc__.split("\n")[0]
)
parser.add_argument("stage", choices=pipeline.STAGES)
parser.add_argument("--config", default=pipeline.DEFAULT_CONFIG, help="Benchmark config YAML")
parser.add_argument("--quick", action="store_true", help="Run the config's quick section")
parser.add_argument(
    "--index", type=int, default=None, help="Run one chunk of tasks (default: $PBS_ARRAY_INDEX)"
)
parser.add_argument("--chunk", type=int, default=1, help="Tasks per index")
parser.add_argument("--workers", type=int, default=1, help="Calibrate or apply tasks run at once")
parser.add_argument("--count", action="store_true", help="Print the number of tasks and exit")
args = parser.parse_args()

cfg = pipeline.load_config(args.config, quick=args.quick)
if args.count:
    print(len(pipeline.tasks(cfg, args.stage)))
else:
    index = args.index
    if index is None and os.environ.get("PBS_ARRAY_INDEX"):
        index = int(os.environ["PBS_ARRAY_INDEX"])
    pipeline.run(cfg, args.stage, index=index, chunk=args.chunk, n_workers=args.workers)
