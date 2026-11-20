#!/bin/bash
# Exact parallel Gibbs (PCPARSOR) variants for the toroid, all in ONE allocation (same nodes), each as one
# job_scaling.sh run with the weak3 settings and per-level PARSOR statistics: processor colouring
# (Jones-Plassmann / greedy / smallest-last) and MID message handling (MPI_Waitany / MPI_Waitsome).
# The mesh refinement follows the weak-scaling series (1/8/64/512 ranks -> refine 2/3/4/5).
#
# Usage: sbatch --partition=cpu-multi --nodes=32 --ntasks=512 --ntasks-per-node=16 --cpus-per-task=4 \
#          --time=01:00:00 sweep_parsor.sh <results-dir>      (or e.g. --nodes=4 --ntasks=64 for 64 ranks)
#SBATCH --exclusive
#SBATCH --export=NONE
set -uo pipefail
R=$1
J=$HOME/algebraic_mgmc_paper/ParMGMC/examples/paper/toroid/job_scaling.sh
declare -A REF=([1]="2 0" [8]="3 1" [64]="4 2" [512]="5 2")
read -r refine prerefine <<< "${REF[$SLURM_NTASKS]}"
tag=$(printf "np%04d" "$SLURM_NTASKS")
P=-gamgmc_mg_levels_sorgibbs_pc_parsor_
BASE="-tt_refine $refine -tt_prerefine $prerefine -tt_smoother gibbs -tt_ntheta 33 -tt_nsamples 2000 -tt_nburnin 100 -tt_verbose -tt_view_end -no_signal_handler ${P}stats"
configs=(
  "jp_waitany|${P}proc_coloring jp ${P}waitany"
  "jp|${P}proc_coloring jp"
  "greedy|${P}proc_coloring greedy"
  "sl|${P}proc_coloring sl"
  "sl_sweeps2|${P}proc_coloring sl -gamgmc_mg_levels_ksp_max_it 2"
  "jp_waitany_end|${P}proc_coloring jp ${P}waitany"
)
for c in "${configs[@]}"; do
  name=${c%%|*}
  opts=${c#*|}
  echo "$(date -Iseconds) $name: $opts"
  # shellcheck disable=SC2086
  bash "$J" "$R" "${tag}_$name" $BASE $opts
done
