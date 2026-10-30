"""
k-convergence scan of every figure PKL.

For every point (x, n) solved with GCG, the stored k_records give the best C_GME at each number of
pure states k = r, r+1, ...  A point is "converged" when the k loop stopped because two consecutive
k agreed within tol_k (1e-6); otherwise it stopped at the limit k = r + k_max_extra.
For each point we record: rank r, last k, whether the limit was hit, and the change of C_GME
in the last k step (C(k_last-1) - C(k_last)), i.e. how much the value was still moving.

Run from the hubbard_gme repo:
    PYTHONPATH=src python /mnt/d/Dropbox/PhDWork/hubbard_gme/audit_reports/table_tests/kconv_scan.py
Writes kconv_report.txt and kconv_points.pkl next to this file.
"""
import os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tables_lib import SPECS, RUNS

THERMAL = [
    ("Fig. 3 (a) L=6", "thermal_v1_L6_Ne6_U10_t1_mode-uniform_tpall_t2all_tperpall_nmin2_solver-gcg_full_restarts10_iter300_kmax5_tol1em09_cut10_adaptive1_step5_clusteravg1_T0to1_N20-148a4f8ecb18_knone_diag0.pkl"),
    ("Fig. 3 (b) L=8", "thermal_v1_L8_Ne8_U10_t1_mode-uniform_tpall_t2all_tperpall_nmin2_solver-gcg_full_restarts8_iter200_kmax4_tol1em09_cut10_adaptive1_step5_clusteravg1_T0to0.5_N20-3c87501b406b_knone_diag0.pkl"),
]
TOL_K = 1e-6
OUT = os.path.join(HERE, "kconv_report.txt")
fh = open(OUT, "w")


def out(s=""):
    print(s, flush=True); fh.write(s + "\n")


def records(conv):
    if not isinstance(conv, dict):
        return None
    return conv.get("k_records")


runs = [(s["fig"] + " " + s["panel"], s["pkl"]) for s in SPECS] + THERMAL
allpts = {}
out("k-convergence of the GCG results stored in the figure PKLs")
out(f"converged = two consecutive k agree within {TOL_K:g}; otherwise the loop stopped at k = r + k_max_extra")
out("last step = C(k_last - 1) - C(k_last): how much C_GME still dropped at the last k\n")
for label, f in runs:
    d = pickle.load(open(RUNS + f, "rb"))
    m, r = d["meta"], d["result"]
    xs = None
    for key in ("U_over_t", "V_over_t", "N_values", "T_over_t", "T_values"):
        if key in r:
            xs = np.asarray(r[key], float); break
    conv = r.get("convergence", {})
    if not conv and "gme_job_paths" in r:                       # thermal runs: one job file per (T index, n)
        JC = RUNS + "job_cache/"
        conv = {}
        for (ti, n), p in r["gme_job_paths"].items():
            loc = JC + os.path.basename(p)
            if not os.path.exists(loc):
                continue
            jd = pickle.load(open(loc, "rb"))
            rf = jd.get("result_full", {})
            recs = (jd.get("decompositions") or {}).get("gcg")
            conv.setdefault(n, {})[ti] = dict(k_records=recs, rank=rf.get("rank"), k_max_extra=jd["meta"].get("k_max_extra"),
                                              tol_k=rf.get("tol_k", TOL_K), C_gcg=rf.get("C_gcg"))
        conv = {n: [v.get(i) for i in range(len(xs))] for n, v in conv.items()}
    if not conv:
        out("   no per-point k records stored in this PKL (older run): convergence in k cannot be checked")
    out("=" * 100)
    out(f"{label}:  restarts={m.get('n_restarts')}  iterations={m.get('max_iter')}  k_max_extra={m.get('k_max_extra')}")
    out(f"   {f[:95]}")
    out(f"   {'n':>3} {'points':>7} {'GCG':>5} {'conv':>5} {'at limit':>9} {'max last step':>14} {'median last':>12}  worst point (x, r, k_last, C)")
    pts = []
    for n in sorted(conv.keys()) if isinstance(conv, dict) else []:
        lst = conv[n]
        nconv = nlim = ngcg = 0
        steps, worst = [], None
        for i, c in enumerate(lst):
            recs = records(c)
            if not recs:
                continue
            ngcg += 1
            Cs = [rec["C_GME"] for rec in recs]
            ks = [rec["k"] for rec in recs]
            r0 = c.get("rank", ks[0])
            limit = r0 + int(c.get("k_max_extra", m.get("k_max_extra", 0)))
            converged = len(Cs) > 1 and abs(Cs[-1] - Cs[-2]) < c.get("tol_k", TOL_K)
            at_limit = (ks[-1] >= limit) and not converged
            step = (Cs[-2] - Cs[-1]) if len(Cs) > 1 else 0.0
            nconv += converged; nlim += at_limit
            if at_limit:
                steps.append(step)
                if worst is None or step > worst[-1]:
                    worst = (xs[i] if xs is not None else i, r0, ks[-1], float(c.get("C_gcg", min(Cs))), step)
            pts.append(dict(n=n, i=i, x=(xs[i] if xs is not None else i), rank=r0, k_last=ks[-1], limit=limit,
                            converged=converged, at_limit=at_limit, last_step=step, C=float(c.get("C_gcg", min(Cs)))))
        if ngcg == 0:
            out(f"   {n:>3} {len(lst):>7} {0:>5}   (pure state, closed formula: no k loop)")
            continue
        ms = f"{max(steps):.1e}" if steps else "-"
        md = f"{np.median(steps):.1e}" if steps else "-"
        w = f"x={worst[0]:g} r={worst[1]} k={worst[2]} C={worst[3]:.4f}" if worst else ""
        out(f"   {n:>3} {len(lst):>7} {ngcg:>5} {nconv:>5} {nlim:>9} {ms:>14} {md:>12}  {w}")
    allpts[label] = dict(pkl=f, restarts=m.get("n_restarts"), max_iter=m.get("max_iter"), k_max_extra=m.get("k_max_extra"), points=pts)
    out("")
pickle.dump(allpts, open(os.path.join(HERE, "kconv_points.pkl"), "wb"))
out("written " + OUT)
