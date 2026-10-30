"""
Collect the tail-rerun results (thermal_tail_hpc.py) and build the new Fig. 3(b) values.

For every (T, n): new C_GME = min(old value, best fresh start, best warm start).
Every value is the cost of an explicit decomposition, i.e. an upper bound on C_GME, so the
lowest one is kept. Points that were not rerun keep the old value.

Writes, next to the production PKL (which is not changed):
  <production name>_tailrerun.pkl   same structure as the production PKL, C_GME replaced, plus
                                    result["tail_rerun"] with the per-point details
  tail_rerun_L8/report.txt          old vs new values, per point
Run from the repo:  python hpc_tail/merge_thermal_tail.py
"""
import argparse, glob, os, pickle, sys
from collections import defaultdict
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thermal_tail_hpc import PKL, RUNS

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=RUNS)
ap.add_argument("--pkl", default=PKL)
ap.add_argument("--out", default=os.path.join(RUNS, "tail_rerun_L8"))
A = ap.parse_args()

d = pickle.load(open(os.path.join(A.runs, A.pkl), "rb"))
r = d["result"]
T = np.asarray(r["T_values"])
res = defaultdict(lambda: dict(fresh=[], warm=[]))
for f in glob.glob(os.path.join(A.out, "n*", "*.pkl")):
    t = pickle.load(open(f, "rb"))
    res[(t["T_index"], t["n"])][t["kind"]].append(t)

lines = ["Fig. 3(b) tail rerun (L = 8): old value vs best fresh / warm start; new = lowest of the three",
         f"production PKL: {A.pkl}", ""]
detail = {}
for n in sorted({k[1] for k in res}):
    lines.append(f"n = {n}")
    lines.append(f"  {'k_BT/t':>8} {'old':>10} {'fresh':>10} {'(tasks, q)':>12} {'warm':>10} {'(tasks)':>8} {'new':>10}  iter-limit hits")
    for ti in sorted(k[0] for k in res if k[1] == n):
        R = res[(ti, n)]
        old = float(r["C_GME"][n][ti])
        bf = min(R["fresh"], key=lambda x: x["h_best"]) if R["fresh"] else None
        bw = min(R["warm"], key=lambda x: x["h_best"]) if R["warm"] else None
        new = min([old] + [x["h_best"] for x in (bf, bw) if x is not None])
        hits = sum(x["termination"] == "max_iter" for x in R["fresh"] + R["warm"])
        detail[(ti, n)] = dict(old=old, new=new, fresh_best=bf and bf["h_best"], fresh_q=bf and bf["k"],
                               fresh_seed=bf and bf.get("seed"), warm_best=bw and bw["h_best"],
                               warm_from_q=bw and bw.get("from_q"), n_fresh=len(R["fresh"]), n_warm=len(R["warm"]),
                               max_iter_hits=hits)
        if new < old:
            r["C_GME"][n][ti] = new
        fs = f"{bf['h_best']:10.6f} {len(R['fresh']):>5}, q={bf['k']:<4}" if bf else f"{'-':>10} {'':>12}"
        ws = f"{bw['h_best']:10.6f} {len(R['warm']):>8}" if bw else f"{'-':>10} {'':>8}"
        lines.append(f"  {T[ti]:8.4f} {old:10.6f} {fs} {ws} {new:10.6f}  {hits}/{len(R['fresh']) + len(R['warm'])}")
    lines.append("")

r["tail_rerun"] = dict(source_pkl=A.pkl, points=detail,
                       note="C_GME at these (T index, n) = min(old, fresh, warm); see hpc_tail/merge_thermal_tail.py")
new_pkl = os.path.join(A.runs, A.pkl.replace(".pkl", "_tailrerun.pkl"))
pickle.dump(d, open(new_pkl, "wb"))
lines.append(f"written {new_pkl}")
open(os.path.join(A.out, "report.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
