#!/bin/bash
# One parallel run of toroid_weak_scaling.py on Helix (submitted by submit_weak_scaling.sh,
# which sets the number of nodes and tasks).
#
# Usage: sbatch [resources] job_scaling.sh <results-dir> <run-name> <driver options...>
# Writes <run-name>.out, .json, .npy, .log_view.txt and .node.txt to the results directory.
#
#SBATCH --exclusive
#SBATCH --export=NONE

set -uo pipefail

[ $# -ge 2 ] || { echo "usage: $0 <results-dir> <run-name> [driver options...]" >&2; exit 1; }
results=$1
name=$2
shift 2

ROOT=$HOME/algebraic_mgmc_paper
DRIVER=$ROOT/ParMGMC/examples/paper/toroid/toroid_weak_scaling.py

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
  echo "job $SLURM_JOB_ID, $(date -Iseconds)"
  echo "nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
  echo "tasks $SLURM_NTASKS, tasks per node ${SLURM_NTASKS_PER_NODE:-?}, cpus per task ${SLURM_CPUS_PER_TASK:-1}"
  echo "driver $DRIVER (md5 $(md5sum "$DRIVER" | cut -d' ' -f1))"
  echo "options: $*"
} > "$results/$name.node.txt"

# srun does not inherit --cpus-per-task from sbatch; block:cyclic alternates the tasks of a
# node between the two sockets, and each task is bound to its block of cores.
srun --mpi=pmix --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" --cpu-bind=verbose,cores \
  --distribution=block:cyclic \
  env LD_PRELOAD="$PRELOAD" python "$DRIVER" "$@" \
  -tt_out "$results/$name.json" -tt_chain "$results/$name.npy" \
  -log_view ":$results/$name.log_view.txt" -options_left \
  > "$results/$name.out" 2>&1 < /dev/null
status=$?

sacct -j "$SLURM_JOB_ID" --format=JobID,JobName%30,NNodes,NTasks,Elapsed,MaxRSS,State,ExitCode \
  >> "$results/$name.node.txt"
exit $status
