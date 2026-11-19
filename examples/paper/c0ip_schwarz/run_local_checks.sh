#!/bin/bash
# Local checks for the alpha = 2 example (small problems, laptop): validation of the sampler
# configuration used on Helix, one-level vs two-level convergence and h-robustness.
# Usage: run_local_checks.sh [launcher]   e.g. run_local_checks.sh "memcap -m 14"
# Output: one RESULT block per run on stdout.
set -u
L=${1:-}
D=$(dirname "$0")/c0ip_mgmc.py
run() { np=$1; shift; echo "### np=$np $*"; $L mpiexec --oversubscribe -n "$np" python "$D" "$@" 2>&1 | grep -A6 "^RESULT\|rror"; }

# 1. Validation: QoI and pointwise variances against the exact covariance, 20k samples
S="-mode sample -nsamples 20000 -nburnin 100"
for np in 1 4; do
  run $np -N 32 -M 4 $S
  run $np -N 32 -M 8 $S
  run $np -dim 3 -N 8 -M 2 $S
  run $np -dim 3 -N 16 -M 4 $S
  run $np -mode spde -N 64 -nsamples 20000 -nburnin 0
  run $np -mode spde -dim 3 -N 16 -nsamples 20000 -nburnin 0
done

# 2. One-level (Schwarz sweep alone) vs two-level, growing number of subdomains of fixed size
for lev in "" "-one_level"; do
  for nm in "64 8" "128 16" "256 32"; do set -- $nm; run 8 -N $1 -M $2 -mode rho $lev; done
  for nm in "16 4" "32 8"; do set -- $nm; run 8 -dim 3 -N $1 -M $2 -mode rho -rho_its 80 $lev; done
done

# 3. h-robustness: fixed number of subdomains, growing H/h
for n in 32 64 128 256; do run 8 -N $n -M 4 -mode rho; done
for n in 8 16 24; do run 8 -dim 3 -N $n -M 2 -mode rho -rho_its 80; done
