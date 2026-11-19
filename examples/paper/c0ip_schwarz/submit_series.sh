#!/bin/bash
# Submit the alpha = 2 series on Helix: growing number of box subdomains of fixed size (H/h = 8 in 2D,
# 4 in 3D), 64 (2D) / 32 (3D) subdomains per rank. Per point: MGMC timing and IACT (-nsamples, V(1,0)),
# convergence factor (rho), and the SPDE sampler on the same mesh (N) and with the same number of
# unknowns (2N), on the same number of ranks. Exclusive nodes, 16 ranks per node.
#
# Usage: submit_series.sh <results-dir>     Environment: DRY_RUN=1 (print only), NSAMPLES (10000)
set -euo pipefail
[ $# -eq 1 ] || { sed -n '2,8p' "$0"; exit 1; }
R=$(realpath -m "$1")
J=$(dirname "$(realpath "$0")")/job_c0ip.sh
mkdir -p "$R"

sub() { # name ranks walltime driver-options...
  local name=$1 np=$2 t=$3; shift 3
  local nn=$(( (np + 15) / 16 )) npn=$(( np < 16 ? np : 16 ))
  local part=cpu-multi; [ $nn -eq 1 ] && part=cpu-single
  local cmd=(sbatch --parsable --partition=$part --nodes=$nn --ntasks=$np --ntasks-per-node=$npn --cpus-per-task=4
             --time=$t --job-name=a2_$name --output="$R/slurm_${name}_%j.out" "$J" "$R" "$name" "$@")
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "${cmd[*]}"; else echo "$("${cmd[@]}") $name"; fi
}

MG="-mode sample -nsamples ${NSAMPLES:-10000} -nburnin 100 -timing_only 1 -gamgmc_pc_mg_log"
SP="-mode spde -nsamples 1000 -nburnin 10 -timing_only 1 -spde_pc_mg_log"
# dim N M ranks
for p in "2 64 8 1" "2 128 16 4" "2 256 32 16" "2 512 64 64" "2 1024 128 256" "3 16 4 2" "3 32 8 16" "3 64 16 128"; do
  read -r d n m np <<< "$p"
  tag=$(printf "d%d_N%d_M%d_np%04d" "$d" "$n" "$m" "$np")
  its=$([ "$d" = 3 ] && echo 80 || echo 50)
  sub "${tag}_mgmc" "$np" 01:00:00 -dim "$d" -N "$n" -M "$m" $MG
  sub "${tag}_rho" "$np" 00:45:00 -dim "$d" -N "$n" -M "$m" -mode rho -rho_its "$its"
  sub "$(printf "d%d_N%d_np%04d_spde" "$d" "$n" "$np")" "$np" 00:30:00 -dim "$d" -N "$n" $SP
  sub "$(printf "d%d_N%d_np%04d_spde" "$d" $((2 * n)) "$np")" "$np" 00:30:00 -dim "$d" -N $((2 * n)) $SP
done
# cube rho only (no rank count gives 32 subdomains per rank)
sub d3_N24_M6_np0008_rho 8 00:45:00 -dim 3 -N 24 -M 6 -mode rho -rho_its 80
sub d3_N48_M12_np0032_rho 32 00:45:00 -dim 3 -N 48 -M 12 -mode rho -rho_its 80
