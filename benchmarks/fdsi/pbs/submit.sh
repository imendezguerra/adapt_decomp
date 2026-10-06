#!/bin/bash
# Submit the FDSI benchmark to PBS Pro: calibrate -> search -> apply -> collect. Each stage is an
# array job (one subjob per CHUNK tasks) that starts once the previous stage finished successfully.
# Tasks whose outputs are already current return immediately, so resubmitting is safe; outputs
# made from other settings (stale) are recomputed only with --force.
#
# Run from the repository root, with the adapt_decomp environment active (the array sizes come
# from the spec through the CLI):
#   bash benchmarks/fdsi/pbs/submit.sh [--spec PATH] [--from STAGE] [--force] [--dry-run]
set -euo pipefail

SPEC=benchmarks/fdsi/benchmark.yaml
FROM=calibrate
DRY_RUN=0
FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --spec) SPEC="$2"; shift 2 ;;
        --from) FROM="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown argument: $1 (expected --spec, --from, --force or --dry-run)" >&2; exit 1 ;;
    esac
done

# Resources per stage (edit for your cluster). A search's cores come from the spec
# (n_jobs x pool size x threads_per_run). Memory is sized from the
# FDSI recordings (100 channels x ext_fact 10): a search runs its random start-up trials
# threads_per_run at a time, 3 workers of ~2 GB each per trial (adapt_decomp predicts it and
# stops early if the request is too small), and a calibration is one such process. With
# n_jobs: 1 and 4 threads per run, a search of 100 trials takes ~2.5-3.5 h. An application
# adapts the whole recording, and heap fragmentation makes its peak vary from run to run
# (measured 1.5-4.7 GB), so it gets twice the calibration's. On Imperial's CX3 all of these
# route to small24.
CALIBRATE_MEM=4gb;  CALIBRATE_WALLTIME=06:00:00; CALIBRATE_CHUNK=1
SEARCH_MEM=32gb;    SEARCH_WALLTIME=08:00:00
APPLY_MEM=8gb;      APPLY_WALLTIME=02:00:00;     APPLY_CHUNK=4
COLLECT_MEM=8gb;    COLLECT_WALLTIME=02:00:00

TEMPLATE=benchmarks/fdsi/pbs/stage.pbs
# Each (sub)job's stdout and stderr, git-ignored: .job_outputs/fdsi_<stage>_<submitted>[.<index>].log
LOG_DIR="$PWD/.job_outputs"
SUBMITTED=$(date +%Y%m%d-%H%M%S)
STAGES=(calibrate search apply collect)

# Subjobs a stage needs: its task count divided by the chunk size, rounded up
n_subjobs() {
    local n
    n=$(python -m benchmarks.fdsi tasks "$1" --spec "$SPEC" --count)
    echo $(( (n + $2 - 1) / $2 ))
}

# Submit one stage after the job id in $6 (if any) and print its job id
submit() {
    local stage=$1 ncpus=$2 mem=$3 walltime=$4 chunk=$5 dependency=$6 n=$7
    # ompthreads too: PBS Pro sets NCPUS from it, not from ncpus, and a site may default it to 1
    local args=(-N "fdsi_${stage}" -l "select=1:ncpus=${ncpus}:ompthreads=${ncpus}:mem=${mem}" -l "walltime=${walltime}"
                -v "STAGE=${stage},SPEC=${SPEC},CHUNK=${chunk},FORCE=${FORCE}")
    local log="${LOG_DIR}/fdsi_${stage}_${SUBMITTED}"
    if [ "$n" -gt 1 ]; then
        args+=(-J "0-$((n - 1))" -o "${log}.^array_index^.log")  # PBS fills in each subjob's index
    else
        args+=(-o "${log}.log")
    fi
    if [ -n "$dependency" ]; then args+=(-W "depend=afterok:${dependency}"); fi
    if [ "$DRY_RUN" = 1 ]; then
        echo "qsub ${args[*]} ${TEMPLATE}" >&2
        echo "<${stage}_job_id>"
    else
        mkdir -p "$LOG_DIR"
        qsub "${args[@]}" "$TEMPLATE"
    fi
}

started=0
previous=""
for stage in "${STAGES[@]}"; do
    if [ "$stage" = "$FROM" ]; then started=1; fi
    if [ "$started" = 0 ]; then continue; fi
    case "$stage" in
        calibrate) previous=$(submit calibrate 1 "$CALIBRATE_MEM" "$CALIBRATE_WALLTIME" "$CALIBRATE_CHUNK" "$previous" "$(n_subjobs calibrate "$CALIBRATE_CHUNK")") ;;
        search)    previous=$(submit search "$(python -m benchmarks.fdsi tasks search --spec "$SPEC" --ncpus)" "$SEARCH_MEM" "$SEARCH_WALLTIME" 1 "$previous" "$(n_subjobs search 1)") ;;
        apply)     previous=$(submit apply 1 "$APPLY_MEM" "$APPLY_WALLTIME" "$APPLY_CHUNK" "$previous" "$(n_subjobs apply "$APPLY_CHUNK")") ;;
        collect)   previous=$(submit collect 1 "$COLLECT_MEM" "$COLLECT_WALLTIME" 1 "$previous" 1) ;;
    esac
    echo "${stage}: ${previous}"
done
if [ "$started" = 0 ]; then
    echo "Unknown stage for --from: ${FROM} (expected one of ${STAGES[*]})" >&2
    exit 1
fi
