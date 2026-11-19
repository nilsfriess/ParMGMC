#!/bin/bash
# GAMG parameter sweep for the toroid at 512 ranks: all configurations run one after another in
# ONE allocation (same nodes), each as one job_scaling.sh run (weak3 settings, exact Gibbs).
#
# Usage: sbatch --partition=cpu-multi --nodes=32 --ntasks=512 --ntasks-per-node=16 --cpus-per-task=4 \
#          --time=01:30:00 sweep_gamg.sh <results-dir>
#SBATCH --exclusive
#SBATCH --export=NONE
set -uo pipefail
R=$1
J=$HOME/algebraic_mgmc_paper/ParMGMC/examples/paper/toroid/job_scaling.sh
BASE="-tt_refine 5 -tt_prerefine 2 -tt_smoother gibbs -tt_ntheta 33 -tt_nsamples 2000 -tt_nburnin 100 -tt_verbose -no_signal_handler"
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
  "default_end|"
)
for c in "${configs[@]}"; do
  name=${c%%|*}
  opts=${c#*|}
  echo "$(date -Iseconds) $name: $opts"
  # shellcheck disable=SC2086
  bash "$J" "$R" "np0512_$name" $BASE $opts
done
