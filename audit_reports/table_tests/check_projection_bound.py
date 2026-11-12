"""
Check a projected T = 0 scan against the matching unprojected scan.

Removing the doublons on the n detection sites splits the nBRDM as Gamma = Gamma_P + (pieces that are
the identity on the doublon sites). The pieces are biseparable and the convex roof is convex, so

    C_old <= (omega_P / omega) * C_P

for every point. Both C values are GCG upper bounds for n >= 3, so a small violation means an unconverged point.
The script also recomputes omega_P from the state and compares it with the value stored in the projected PKL.

Usage:  python audit_reports/table_tests/check_projection_bound.py --old OLD.pkl --new NEW_proj1.pkl [--tol 2e-3]
Exit code 1 if a stored omega_P is wrong or the bound is violated by more than tol. T = 0 scans only (uscan, vscan, nscan).
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
from tables_lib import point, build_states, omega  # noqa: E402

KIND = {"uscan": ("U", "U_over_t"), "vscan": ("V", "V_over_t"), "nscan": ("K", "N_values")}


def check(old_path, new_path, tol=2e-3, omega_tol=1e-9):
    """Return (rows, n_bad). One row per (x, n): x, n, omega, omega_P, C_old, C_P, bound, status."""
    old, new = pickle.load(open(old_path, "rb")), pickle.load(open(new_path, "rb"))
    m_old, r_old, r_new = old["meta"], old["result"], new["result"]
    name = os.path.basename(old_path)
    if not r_new.get("project_doublons") or r_old.get("project_doublons"):
        raise ValueError("--old must be an unprojected run and --new a projected one")
    prefix = name.split("_")[0]
    if prefix not in KIND:
        raise ValueError(f"not a T = 0 scan: {name}")
    kind, xkey = KIND[prefix]
    spec = dict(kind=kind, state="es" if "_es_" in name else "gs")
    sp = "_sz0_" in name
    xs_old, xs_new = np.asarray(r_old[xkey], float), np.asarray(r_new[xkey], float)
    if xs_old.shape != xs_new.shape or not np.allclose(xs_old, xs_new):
        raise ValueError("the two runs have different grids")
    rows, n_bad = [], 0
    for i, x in enumerate(xs_new):
        L, K, t, U, V = point(spec, m_old, r_old, sp, float(x))
        psis = build_states(spec, L, K, t, U, V, sp)[0]
        for n in r_new["n_values"]:
            if n > min(L, K):
                continue
            w, wP = omega(psis, L, K, n, sp), omega(psis, L, K, n, sp, project_doublons=True)
            stored, C_old, C_P = float(r_new["omega"][n][i]), float(r_old["C_GME"][n][i]), float(r_new["C_GME"][n][i])
            bound = wP / w * C_P if w > 1e-14 else float("nan")
            status = "ok"
            if abs(stored - wP) > omega_tol:
                status = "OMEGA MISMATCH"
            elif C_old > bound + tol:
                status = "BOUND VIOLATED"
            n_bad += status != "ok"
            rows.append((float(x), n, w, wP, C_old, C_P, bound, status))
    return rows, n_bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--tol", type=float, default=2e-3)
    A = ap.parse_args()
    rows, n_bad = check(A.old, A.new, A.tol)
    print(f"{'x':>9} {'n':>2} {'omega':>8} {'omega_P':>8} {'C_old':>8} {'C_P':>8} {'bound':>8}  status")
    for x, n, w, wP, C_old, C_P, bound, status in rows:
        print(f"{x:9.4g} {n:2d} {w:8.4f} {wP:8.4f} {C_old:8.4f} {C_P:8.4f} {bound:8.4f}  {status}")
    print(f"\n{len(rows)} points, {n_bad} failed")
    sys.exit(1 if n_bad else 0)
