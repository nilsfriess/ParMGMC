"""Collect the JSON records written by wrench_refinement.py into a CSV and a Markdown table.

Usage: python collect.py <results-dir>... [--csv out.csv] [--md out.md]

Flags (column "flags"):
    iact      fewer than 50 IACTs in the chain, the IACT estimate is unreliable
    early     the run hit -wr_max_time and has fewer samples than requested
    var       sample variance of the QOI more than 3 standard errors from the exact one
    mean      sample mean of the QOI more than 3 standard errors from the exact one
    dirty     the ParMGMC checkout had uncommitted changes to tracked files
"""

import argparse
import csv
import glob
import json
import os
import sys

COLUMNS = [
    "alpha", "kappa", "mode", "sampler", "refine", "ndofs", "mg_levels",
    "t_setup", "t_iter", "iact", "t_indep", "nsamples", "n_over_iact",
    "var_ratio", "var_ratio_se", "mean_z", "flags", "job", "file",
]

SAMPLER_ORDER = {"gibbs": 0, "amgmc": 1, "amgmc_tuned": 2, "gmgmc": 3, "cholesky": 4}


def row_from_record(rec, path):
    p, s, r, t, c = (rec["problem"], rec["sampler"], rec["run"], rec["timings"],
                     rec["chain"])
    ref = rec.get("reference") or {}
    flags = []
    if not c["reliable_iact"]:
        flags.append("iact")
    if r["stopped_early"]:
        flags.append("early")
    if ref:
        if abs(ref["var_ratio"] - 1.0) > 3 * ref["var_ratio_se"]:
            flags.append("var")
        if abs(ref["mean_z"]) > 3:
            flags.append("mean")
    if (rec["provenance"].get("git") or {}).get("dirty"):
        flags.append("dirty")
    return {
        "alpha": p["alpha"], "kappa": p["kappa"], "mode": p["mode"],
        "sampler": s.get("label", s["name"]), "refine": p["refine"], "ndofs": p["ndofs"],
        "mg_levels": s["mg_levels"],
        "t_setup": t["setup"], "t_iter": t["per_iteration"], "iact": c["iact"],
        "t_indep": t["per_indep_sample"], "nsamples": r["nsamples"],
        "n_over_iact": c["n_over_iact"],
        "var_ratio": ref.get("var_ratio"), "var_ratio_se": ref.get("var_ratio_se"),
        "mean_z": ref.get("mean_z"),
        "flags": ",".join(flags),
        "job": rec["provenance"].get("slurm_job_id"),
        "file": os.path.relpath(path),
    }


def fmt(val, spec):
    if val is None or val == "":
        return "-"
    return format(val, spec)


def markdown(rows):
    head = ("| alpha | kappa | mode | sampler | refine | DOFs | levels | setup [s] "
            "| t/iter [s] | IACT | t/indep [s] | N | var ratio | mean z | flags |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        ratio = (f"{r['var_ratio']:.3f} ± {r['var_ratio_se']:.3f}"
                 if r["var_ratio"] is not None else "-")
        lines.append(
            f"| {r['alpha']} | {r['kappa']:g} | {r['mode']} | {r['sampler']} "
            f"| {r['refine']} | {r['ndofs']} | {fmt(r['mg_levels'], 'd')} "
            f"| {r['t_setup']:.3g} | {r['t_iter']:.3e} | {r['iact']:.2f} "
            f"| {r['t_indep']:.3e} | {r['nsamples']} | {ratio} "
            f"| {fmt(r['mean_z'], '+.2f')} | {r['flags']} |")
    return "\n".join(lines)


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
            if rec.get("driver") != "wrench_refinement.py":
                continue
            rows.append(row_from_record(rec, path))

    rows.sort(key=lambda r: (r["alpha"], -r["kappa"], r["mode"] != "prior",
                             SAMPLER_ORDER.get(r["sampler"], 99), r["sampler"], r["refine"]))

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    table = markdown(rows)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(table + "\n")
    print(table)
    nflagged = sum(1 for r in rows if r["flags"] and r["flags"] != "dirty")
    print(f"\n{len(rows)} runs, {nflagged} flagged (ignoring 'dirty')", file=sys.stderr)


if __name__ == "__main__":
    main()
