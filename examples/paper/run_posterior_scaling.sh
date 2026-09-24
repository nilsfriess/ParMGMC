#!/usr/bin/env bash
# Strong scaling of posterior sampling on the vessel graph: algebraic MGMC vs Cholesky (Woodbury), 1-32 ranks.
#
# One log and one CSV per (sampler, ranks) in $OUT; combine the CSVs with pandas, e.g.
#   pd.concat(pd.read_csv(f) for f in glob.glob("results/posterior/*.csv"))
#
# Everything can be overridden from the environment, e.g. on a cluster:
#   MPIRUN=srun RANKS="1 2 4 8 16 32 64" ./run_posterior_scaling.sh
# or for a quick test:
#   GRAPH=vessel_1-64.dat RANKS="1 2" NSAMPLES=20 KAPPAS=0.01 ./run_posterior_scaling.sh

MPIRUN=${MPIRUN:-mpirun}
GRAPH=${GRAPH:-vessel.dat} # written by: python preprocess_graph.py
RANKS=${RANKS:-"1 2 4 8 16 32"}
KAPPAS=${KAPPAS:-0.01,0.001}
NSAMPLES=${NSAMPLES:-1000}
SIGMA=${SIGMA:-0.01}     # observation noise std; fixed, so every run solves the same problem
OBS_SCALE=${OBS_SCALE:-0.05} # observed values: OBS_SCALE * N(0,1) with a fixed seed (do not affect the cost)
OUT=${OUT:-results/posterior}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Samplers, run in this order. The Cholesky/Woodbury sampler draws exact independent samples, hence no burn-in.
SAMPLER_NAMES=(mgmc mgmc_default_gamg cholesky)
declare -A SAMPLERS=(
  [mgmc]="-pc_type gamgmc -gamgmc_pc_gamg_aggressive_coarsening 2 -nburnin 100"
  [mgmc_default_gamg]="-pc_type gamgmc -nburnin 100"
  [cholesky]="-pc_type woodbury -pc_woodbury_sampler cholsampler -nburnin 0"
  # Plain Gibbs mixes too slowly at small kappa for a meaningful IACT with NSAMPLES samples; add it by hand if needed:
  # [gibbs]="-pc_type sorgibbs -nburnin 1000"
)
# Common options: the exact posterior mean/variance of the QoI (for checking) does not need 1e-10.
COMMON="-graph $GRAPH -kappas $KAPPAS -nsamples $NSAMPLES -sigma $SIGMA -obs_scale $OBS_SCALE -exact_ksp_rtol 1e-8"

if [ ! -f "$GRAPH" ]; then
  echo "$GRAPH not found; create it with: python preprocess_graph.py" >&2
  exit 1
fi
mkdir -p "$OUT"

for np in $RANKS; do
  for name in "${SAMPLER_NAMES[@]}"; do
    tag="${name}_np${np}"
    echo "=== $tag ($(date +%T))"
    # shellcheck disable=SC2086
    $MPIRUN -n "$np" python graph_posterior.py $COMMON ${SAMPLERS[$name]} 2>&1 \
      | tee "$OUT/$tag.log" | sed -n '/^# CSV/,$p' | tail -n +2 > "$OUT/$tag.csv"
    if [ ! -s "$OUT/$tag.csv" ]; then
      echo "!!! $tag produced no CSV, see $OUT/$tag.log" >&2
      rm -f "$OUT/$tag.csv"
    fi
  done
done
