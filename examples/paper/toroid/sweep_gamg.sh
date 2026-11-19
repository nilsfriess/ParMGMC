#!/bin/bash
# GAMG parameter sweep for the toroid: all configurations run one after another in ONE allocation
# (same nodes), each as one job_scaling.sh run (weak3 settings, exact Gibbs). The mesh refinement
# follows the weak-scaling series (1/8/64/512 ranks -> refine 2/3/4/5, prerefine 0/1/2/2).
#
# Usage: sbatch --partition=cpu-multi --nodes=32 --ntasks=512 --ntasks-per-node=16 --cpus-per-task=4 \
#          --time=01:30:00 sweep_gamg.sh <results-dir>      (or e.g. --nodes=4 --ntasks=64 for 64 ranks)
#SBATCH --exclusive
#SBATCH --export=NONE
set -uo pipefail
R=$1
J=$HOME/algebraic_mgmc_paper/ParMGMC/examples/paper/toroid/job_scaling.sh
declare -A REF=([1]="2 0" [8]="3 1" [64]="4 2" [512]="5 2")
read -r refine prerefine <<< "${REF[$SLURM_NTASKS]}"
tag=$(printf "np%04d" "$SLURM_NTASKS")
BASE="-tt_refine $refine -tt_prerefine $prerefine -tt_smoother gibbs -tt_ntheta 33 -tt_nsamples 2000 -tt_nburnin 100 -tt_verbose -no_signal_handler"
P=-gamgmc_pc_gamg_
configs=(
  "default|"
  "nsmooths0|${P}agg_nsmooths 0"
  "nsmooths2|${P}agg_nsmooths 2"
  "threshold0.02|${P}threshold 0.02"
  "threshold0.05|${P}threshold 0.05"
  "aggressive0|${P}aggressive_coarsening 0"
  "aggressive2|${P}aggressive_coarsening 2"
  "nsmooths2_threshold0.05|${P}agg_nsmooths 2 ${P}threshold 0.05"
  "sweeps2|-gamgmc_mg_levels_ksp_max_it 2"
  "eqlimit1000|${P}process_eq_limit 1000"
  "eqlimit5000|${P}process_eq_limit 5000"
  "default_end|"
)
for c in "${configs[@]}"; do
  name=${c%%|*}
  opts=${c#*|}
  echo "$(date -Iseconds) $name: $opts"
  # shellcheck disable=SC2086
  bash "$J" "$R" "${tag}_$name" $BASE $opts
done
