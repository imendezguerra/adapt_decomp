#!/bin/bash
# Submit the FDSI benchmark to PBS: calibrate -> search -> apply -> collect, each stage an
# array job (one subjob per CHUNK tasks) starting once the previous one succeeded. Tasks whose
# outputs exist return at once, so resubmitting is safe.
#
# From the repository root, with the adapt_decomp environment active:
#   bash benchmarks/fdsi/pbs/submit.sh [--from STAGE] [--quick] [--dry-run]
set -euo pipefail

FROM=calibrate; QUICK=; DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --quick) QUICK=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown argument: $1 (expected --from, --quick or --dry-run)" >&2; exit 1 ;;
    esac
done

# Resources per stage (edit for your cluster): cores, memory, walltime, tasks per subjob. A search
# uses every core it gets (a 100-trial search takes ~3 h on 12); its random start-up trials run
# together, ~2 GB per pool recording each. An application's peak varies from 1.5 to 4.7 GB.
RESOURCES=(
    "calibrate 1 4gb 06:00:00 1"
    "search 12 32gb 08:00:00 1"
    "apply 1 8gb 02:00:00 4"
    "collect 1 8gb 02:00:00 1"
)
LOG_DIR="$PWD/.job_outputs"   # git-ignored: fdsi_<stage>_<submitted>[.<index>].log
SUBMITTED=$(date +%Y%m%d-%H%M%S)

started=0; previous=""
for line in "${RESOURCES[@]}"; do
    read -r stage ncpus mem walltime chunk <<< "$line"
    if [ "$stage" = "$FROM" ]; then started=1; fi
    if [ "$started" = 0 ]; then continue; fi
    n=$(python -m benchmarks.fdsi "$stage" --count ${QUICK:+--quick})
    n=$(( (n + chunk - 1) / chunk ))
    # ompthreads too: PBS Pro sets NCPUS from it, and a site may default it to 1
    args=(-N "fdsi_${stage}" -l "select=1:ncpus=${ncpus}:ompthreads=${ncpus}:mem=${mem}"
          -l "walltime=${walltime}" -v "STAGE=${stage},CHUNK=${chunk},QUICK=${QUICK}")
    log="${LOG_DIR}/fdsi_${stage}_${SUBMITTED}"
    if [ "$n" -gt 1 ]; then args+=(-J "0-$((n - 1))" -o "${log}.^array_index^.log"); else args+=(-o "${log}.log"); fi
    if [ -n "$previous" ]; then args+=(-W "depend=afterok:${previous}"); fi
    if [ "$DRY_RUN" = 1 ]; then
        echo "qsub ${args[*]} benchmarks/fdsi/pbs/stage.pbs"; previous="<${stage}_job_id>"
    else
        mkdir -p "$LOG_DIR"; previous=$(qsub "${args[@]}" benchmarks/fdsi/pbs/stage.pbs); echo "${stage}: ${previous}"
    fi
done
if [ "$started" = 0 ]; then echo "Unknown stage for --from: ${FROM}" >&2; exit 1; fi
