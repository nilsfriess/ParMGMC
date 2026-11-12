#!/bin/bash
# Run several single-core wrench_refinement.py runs side by side on one exclusive Helix node.
#
# Each run gets its own 8-core block (one memory domain of the 64-core node) and runs
# single-threaded in it, so up to 8 runs execute concurrently; further runs wait for a
# free block.  Runs files are processed one after the other (all runs of one file must
# finish before the next file starts), which allows e.g. a run on an otherwise idle node
# followed by a batch of concurrent runs.
#
# Runs file format: one run per line, "<name> <wrench_refinement.py options...>";
# empty lines and lines starting with # are ignored.  For every run <name> the results
# directory gets <name>.out (stdout/stderr), <name>.json, <name>.npy (QOI chain) and
# <name>.log_view.txt.
#
# Usage (from anywhere):
#   sbatch --time=02:00:00 --output=<results>/slurm_%j.out job_packed.sh <results> <runs-file>...
#   sbatch --partition=devel --time=00:30:00 ...    # smoke tests
# Environment: ROOT (default $HOME/algebraic_mgmc_paper), RUN_MEM (memory per run,
#   default 28G)
#
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=230G
#SBATCH --export=NONE

set -uo pipefail

[ $# -ge 2 ] || { echo "usage: $0 <results-dir> <runs-file>..." >&2; exit 1; }
results=$1
shift

ROOT=${ROOT:-$HOME/algebraic_mgmc_paper}
DRIVER=$ROOT/ParMGMC/examples/paper/wrench/wrench_refinement.py
RUN_MEM=${RUN_MEM:-28G}

module load compiler/gnu/11.3 devel/python/3.13.1 mpi/openmpi/4.1.8 numlib/mkl/2022.2
source "$ROOT/venv-firedrake/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=$ROOT/ParMGMC/build/python

MKLLIB=/opt/bwhpc/common/compiler/intel/2022.2/mkl/2022.1.0/lib/intel64
OMPILIB=$(dirname "$(which mpicc)")/../lib64
PRELOAD=/opt/bwhpc/common/compiler/gnu/11.3.0/lib64/libgomp.so.1:$OMPILIB/libmpi.so.40:$MKLLIB/libmkl_core.so.2:$MKLLIB/libmkl_gnu_thread.so.2:$MKLLIB/libmkl_intel_lp64.so.2:$MKLLIB/libmkl_blacs_openmpi_lp64.so.2:$MKLLIB/libmkl_rt.so.2

export SLURM_EXPORT_ENV=ALL
export PMIX_MCA_psec=native # authentication without munge
export PMIX_MCA_gds=hash    # data store without shmem/shmem2

mkdir -p "$results"
{
  echo "job $SLURM_JOB_ID on $(hostname), $(date -Iseconds)"
  echo "driver $DRIVER (md5 $(md5sum "$DRIVER" | cut -d' ' -f1))"
  echo
  lscpu
  echo
  numactl -H 2>/dev/null || echo "numactl not available"
} > "$results/node_$SLURM_JOB_ID.txt" 2>&1

for runs_file in "$@"; do
  cp "$runs_file" "$results/$(basename "$runs_file").$SLURM_JOB_ID"
  echo "$(date -Iseconds) starting runs from $runs_file"
  # The runs file is read through fd 3 and srun's stdin is /dev/null: srun forwards
  # its stdin to the task and would otherwise swallow the rest of the runs file.
  while read -r name args <&3; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    # shellcheck disable=SC2086 # $args holds the driver options, split on purpose
    srun --mpi=pmix --exact --ntasks=1 --cpus-per-task=8 --mem="$RUN_MEM" \
      --cpu-bind=verbose,cores --job-name="$name" --output="$results/$name.out" \
      env LD_PRELOAD="$PRELOAD" python "$DRIVER" $args \
      -wr_out "$results/$name.json" -wr_chain "$results/$name.npy" \
      -log_view ":$results/$name.log_view.txt" -options_left < /dev/null &
    sleep 1 # keep the step launch order (and hence the core blocks) deterministic
  done 3< "$runs_file"
  wait
  echo "$(date -Iseconds) finished runs from $runs_file"
done

sacct -j "$SLURM_JOB_ID" --format=JobID,JobName%40,Elapsed,MaxRSS,State,ExitCode \
  > "$results/sacct_$SLURM_JOB_ID.txt"
