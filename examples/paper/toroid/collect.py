"""Collect the JSON records of toroid_weak_scaling.py into a CSV and a Markdown table.

Usage: python collect.py <results-dir>... [--csv out.csv] [--md out.md]

The weak-scaling efficiency is t_iter(1 rank) / t_iter(P ranks) for the same smoother (it
is only meaningful if the DOFs per rank are about the same, see the dofs/rank column).
Flags: iact (fewer than 50 IACTs in the chain), var / mean (QOI variance / mean more than
3 standard errors from the exact value).
"""

import argparse
import csv
import glob
import json
import os

COLUMNS = ["smoother", "gamg", "ranks", "nodes", "refine", "ndofs", "dofs_per_rank", "imbalance",
           "mg_levels", "t_mesh", "t_setup", "t_iter", "efficiency", "iact", "t_indep",
           "var_ratio", "var_ratio_se", "mean_z", "flags", "job", "file"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--csv")
    parser.add_argument("--md")
    args = parser.parse_args()

    rows = []
    for d in args.dirs:
        for path in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            with open(path) as fh:
                rec = json.load(fh)
            if rec.get("driver") != "toroid_weak_scaling.py":
                continue
            p, s, par, t, c = rec["problem"], rec["sampler"], rec["parallel"], rec["timings"], rec["chain"]
            ref = rec.get("reference") or {}
            flags = []
            if not c["reliable_iact"]:
                flags.append("iact")
            if ref and abs(ref["var_ratio"] - 1) > 3 * ref["var_ratio_se"]:
                flags.append("var")
            if ref and abs(ref["mean_z"]) > 3:
                flags.append("mean")
            rows.append({
                "smoother": s["smoother"], "gamg": s["gamg"], "ranks": par["nranks"],
                "nodes": par["nnodes"], "refine": p["refine"], "ndofs": p["ndofs"],
                "dofs_per_rank": par["dofs_per_rank_mean"],
                "imbalance": par["dofs_per_rank_max"] / par["dofs_per_rank_mean"],
                "mg_levels": s["mg_levels"], "t_mesh": t["mesh"],
                "t_setup": t["setup"] + t.get("first_iteration", 0.0),
                "t_iter": t["per_iteration"], "efficiency": None, "iact": c["iact"],
                "t_indep": t["per_indep_sample"], "var_ratio": ref.get("var_ratio"),
                "var_ratio_se": ref.get("var_ratio_se"), "mean_z": ref.get("mean_z"),
                "flags": ",".join(flags), "job": rec["provenance"].get("slurm_job_id"),
                "file": os.path.relpath(path),
            })

    rows.sort(key=lambda r: (r["gamg"], r["smoother"], r["ranks"]))
    base = {(r["gamg"], r["smoother"]): r["t_iter"] for r in rows if r["ranks"] == 1}
    for r in rows:
        b = base.get((r["gamg"], r["smoother"]))
        r["efficiency"] = b / r["t_iter"] if b else None

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    head = ("| smoother | GAMG | ranks | nodes | refine | DOFs | DOFs/rank | imbal. | levels | mesh [s] "
            "| setup [s] | t/iter [s] | eff. | IACT | var ratio | mean z | flags |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        eff = f"{r['efficiency']:.2f}" if r["efficiency"] is not None else "-"
        ratio = f"{r['var_ratio']:.3f} ± {r['var_ratio_se']:.3f}" if r["var_ratio"] is not None else "-"
        z = f"{r['mean_z']:+.2f}" if r["mean_z"] is not None else "-"
        lines.append(f"| {r['smoother']} | {r['gamg']} | {r['ranks']} | {r['nodes']} | {r['refine']} "
                     f"| {r['ndofs']} | {r['dofs_per_rank']:.0f} | {r['imbalance']:.2f} | {r['mg_levels']} "
                     f"| {r['t_mesh']:.1f} | {r['t_setup']:.2f} | {r['t_iter']:.3e} | {eff} "
                     f"| {r['iact']:.2f} | {ratio} | {z} | {r['flags']} |")
    table = "\n".join(lines)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
