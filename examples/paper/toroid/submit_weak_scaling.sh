#!/bin/bash
# Submit the weak-scaling runs of toroid_weak_scaling.py on Helix: one job per (ranks, smoother),
# 16 ranks per node (each bound to 4 cores, i.e. two ranks per memory domain), exclusive nodes.
# Each step multiplies the number of ranks and (by one more uniform refinement) the number of
# DOFs by about 8, so the DOFs per rank stay at about 170-190k.
#
# Usage: submit_weak_scaling.sh <results-dir> [extra driver options...]
# Environment: RANKS (default "1 8 64 512"), SMOOTHERS (default "gibbs hogwild"),
#   PARTITION (overrides the partition, e.g. devel), TIME (overrides the walltime),
#   DRY_RUN=1 (only print the sbatch commands)
set -euo pipefail

[ $# -ge 1 ] || { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
results=$(realpath -m "$1")
shift

here=$(dirname "$(realpath "$0")")
ranks_per_node=16
cpus_per_task=$((64 / ranks_per_node))

# ranks -> "refine prerefine walltime"
declare -A setup=(
  [1]="2 0 00:30:00"
  [8]="3 1 00:30:00"
  [64]="4 2 01:00:00"
  [512]="5 2 01:00:00"
)

mkdir -p "$results"
for ranks in ${RANKS:-1 8 64 512}; do
  [ -n "${setup[$ranks]:-}" ] || { echo "no setup for $ranks ranks" >&2; exit 1; }
  read -r refine prerefine walltime <<< "${setup[$ranks]}"
  nodes=$(( (ranks + ranks_per_node - 1) / ranks_per_node ))
  per_node=$(( ranks < ranks_per_node ? ranks : ranks_per_node ))
  partition=${PARTITION:-$([ "$nodes" -eq 1 ] && echo cpu-single || echo cpu-multi)}
  for smoother in ${SMOOTHERS:-gibbs hogwild}; do
    name=$(printf "np%04d_%s" "$ranks" "$smoother")
    cmd=(sbatch --partition="$partition" --nodes="$nodes" --ntasks="$ranks" --ntasks-per-node="$per_node"
         --cpus-per-task="$cpus_per_task" --time="${TIME:-$walltime}" --job-name="tt_$name"
         --output="$results/slurm_${name}_%j.out"
         "$here/job_scaling.sh" "$results" "$name"
         -tt_refine "$refine" -tt_prerefine "$prerefine" -tt_smoother "$smoother" "$@")
    echo "${cmd[*]}"
    if [ "${DRY_RUN:-0}" != 1 ]; then "${cmd[@]}"; fi
  done
done
