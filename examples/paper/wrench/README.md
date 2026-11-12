# Wrench examples (paper Sections 7.1 and 7.2)

- `wrench_refinement.py`: one sampling run (one sampler, one mesh level, prior or posterior), writes a JSON
  record and the QOI chain. See its docstring for the options and the samplers.
- `job_packed.sh`: Slurm script for Helix, runs up to 8 single-core runs side by side on one exclusive node
  (one per memory domain).
- `runs/*.txt`: the production runs (one run per line).
- `collect.py`: collects the JSON records into a CSV/Markdown table.

## Production runs on Helix

```bash
J=$HOME/algebraic_mgmc_paper/ParMGMC/examples/paper/wrench
R=$HOME/algebraic_mgmc_paper/results/wrench/production
mkdir -p $R && cd $R
# 7.1: alpha = 1
sbatch --time=04:00:00 --job-name=wr_mgmc_k0.1  --output=$R/slurm_%j.out $J/job_packed.sh $R $J/runs/runs_mgmc_k0.1.txt
sbatch --time=04:00:00 --job-name=wr_mgmc_k0.01 --output=$R/slurm_%j.out $J/job_packed.sh $R $J/runs/runs_mgmc_k0.01.txt
sbatch --time=03:00:00 --job-name=wr_gibbs_chol --output=$R/slurm_%j.out $J/job_packed.sh $R $J/runs/runs_gibbs_chol.txt
# 7.2: alpha = 2
sbatch --time=03:00:00 --job-name=wr_a2         --output=$R/slurm_%j.out $J/job_packed.sh $R $J/runs/runs_a2.txt
# both: runs that need more memory than one memory domain has (two at a time)
sbatch --time=08:00:00 --job-name=wr_highmem --output=$R/slurm_%j.out \
  $J/job_packed.sh --run-mem 110G $R $J/runs/runs_highmem.txt
```

Afterwards: `python $J/collect.py $R --csv wrench.csv --md wrench.md`.
