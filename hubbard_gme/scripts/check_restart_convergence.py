#!/usr/bin/env python3
"""
check_restart_convergence.py
----------------------------
Does increasing the restart budget still lower C_GME?

Runs the GCG solver on every cached rotation of a given cluster size at
several restart budgets, and reports:

  * the value each rotation reaches at each budget
  * the gain from one budget to the next
  * the spread across rotations (they must be EQUAL by translation
    symmetry, so the spread is pure optimiser error)

This answers the question "is n settled at N restarts, or does more help?".

Usage
-----
    # the supervisor's question: does 80 still help at n=5?
    python scripts/check_restart_convergence.py --n 5 --restarts 40 80

    # both body orders, three budgets, write a CSV
    python scripts/check_restart_convergence.py --n 4 5 --restarts 20 40 80 \
        --csv restart_convergence.csv

    # match the production solver settings
    python scripts/check_restart_convergence.py --n 5 --restarts 40 80 \
        --kmax 6 --max-iter 300

Runtime scales with the SUM of the restart budgets, times the number of
rotations, times the number of body orders. `--kmax` is the other big lever:
cost grows with k^2, so kmax=6 is several times kmax=2.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
CACHE = os.path.join(REPO, "results", "rdm_pkl_results", "rdm_cache")

from hubbard_gme.convex_roof import convex_roof_gcg      # noqa: E402


def rotations(n: int, L: int, Ne: int, U, t) -> list:
    """(sites_label, rho) for every cached rotation of an n-site cluster."""
    pat = os.path.join(
        CACHE, f"rdm_v1_L{L}_Ne{Ne}_U{U}_t{t}_mode-uniform_state-gs_n{n}_*.pkl")
    out = []
    for path in sorted(glob.glob(pat)):
        base = os.path.basename(path)
        if "sites-" not in base:
            continue
        sites = base.split("sites-")[-1].split("_")[0]
        rho = pickle.load(open(path, "rb"))["rho"]
        tr = np.trace(rho).real
        if abs(tr) > 1e-14:
            rho = rho / tr
        out.append((sites, rho))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", nargs="+", type=int, default=[5],
                    help="cluster sizes to test (default: 5)")
    ap.add_argument("--restarts", nargs="+", type=int, default=[40, 80],
                    help="restart budgets, ascending (default: 40 80)")
    ap.add_argument("--kmax", type=int, default=2,
                    help="k_max_extra (default 2; production uses 5-6, and "
                         "cost grows with k^2)")
    ap.add_argument("--max-iter", type=int, default=300, dest="max_iter")
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--Ne", type=int, default=6)
    ap.add_argument("--U", default="4")
    ap.add_argument("--t", default="1")
    ap.add_argument("--csv", default=None, help="write every row to CSV")
    a = ap.parse_args(argv)

    budgets = sorted(a.restarts)
    rows = []
    t_start = time.time()

    print(f"L={a.L} Ne={a.Ne} U={a.U} t={a.t}  "
          f"kmax={a.kmax}  max_iter={a.max_iter}")
    print(f"restart budgets: {budgets}\n", flush=True)

    for n in a.n:
        rots = rotations(n, a.L, a.Ne, a.U, a.t)
        if not rots:
            print(f"n={n}: no cached RDMs found, skipping\n", flush=True)
            continue

        print(f"===== n={n}  ({len(rots)} rotations) " + "=" * 34, flush=True)
        hdr = f"{'rotation':>16}" + "".join(f"{f'r={r}':>15}" for r in budgets)
        if len(budgets) > 1:
            hdr += f"{'last gain':>12}"
        print(hdr, flush=True)
        print("-" * len(hdr), flush=True)

        per_budget = {r: [] for r in budgets}
        for sites, rho in rots:
            vals = []
            for r in budgets:
                t0 = time.time()
                res = convex_roof_gcg(rho, n=n, n_restarts=r,
                                      max_iter=a.max_iter,
                                      k_max_extra=a.kmax, verbose=False)
                v = float(res["C_gcg"])
                vals.append(v)
                per_budget[r].append(v)
                rows.append(dict(n=n, rotation=sites, restarts=r, value=v,
                                 seconds=round(time.time() - t0, 1),
                                 kmax=a.kmax, max_iter=a.max_iter))
            line = f"{sites:>16}" + "".join(f"{v:15.9f}" for v in vals)
            if len(vals) > 1:
                line += f"{vals[-2] - vals[-1]:12.2e}"
            print(line, flush=True)

        print("-" * len(hdr), flush=True)
        for r in budgets:
            v = np.array(per_budget[r])
            spread = v.max() - v.min()
            print(f"  r={r:<4} lowest {v.min():.9f}   spread {spread:.2e}"
                  f"   ({100 * spread / v.mean():.4f}% of value)", flush=True)
        if len(budgets) > 1:
            lo_prev = np.array(per_budget[budgets[-2]]).min()
            lo_last = np.array(per_budget[budgets[-1]]).min()
            verdict = ("STILL IMPROVING" if lo_prev - lo_last > 1e-9
                       else "settled")
            print(f"  best {budgets[-2]} -> {budgets[-1]}: "
                  f"{lo_prev - lo_last:.2e}   -> {verdict}", flush=True)
        print(flush=True)

    print(f"total wall time: {(time.time() - t_start) / 60:.1f} min")

    if a.csv and rows:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"CSV written: {a.csv}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
