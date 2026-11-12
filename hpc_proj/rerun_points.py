"""
Strong GCG rerun of single (U/t, n) points of projected T = 0 U-scans.

For each point the projected rho_s is rebuilt exactly as run_U_dependence_gme builds it (same eigenstate or
degenerate average, same mask); the run stops if its purity or trace differs from the stored values. The lower of
the stored and the rerun C_GME is kept (both are upper bounds) and written to <name>_rerun.pkl next to the
original, which is not changed.

Run from the repo:  python hpc_proj/rerun_points.py [--workers 4] [--dry-run]
"""
import argparse, os, sys, pickle, copy, contextlib, io, numpy as np, multiprocessing as mp
from datetime import datetime
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import hubbard_gme.thermal_pipeline as tp
from hubbard_gme.hamiltonian import compute_all_eigenstates
R = os.path.join(REPO, "results", "rdm_pkl_results", "thermal_runs") + os.sep
POINTS = [  # file, U/t, n
    ("uscan_v1_L6_Ne6_t1_mode-uniform_nmin2_solver-gcg_sz0_proj1_restarts10_iter200_kmax4_tol1em09_U1to20_N20-d03a74da789e.pkl", 2.0, 4),
    ("uscan_v1_L6_Ne6_t0p1_mode-uniform_nmin2_solver-gcg_full_es_proj1_restarts10_iter300_kmax5_tol1em09_U0.1to5_N21-8081b758e7cc.pkl", 3.45, 4),
    ("uscan_v1_L8_Ne6_t1_mode-uniform_nmin2_solver-gcg_sz0_proj1_restarts10_iter300_kmax4_tol1em09_U1to20_N20-d03a74da789e.pkl", 6.0, 4),
    ("uscan_v1_L8_Ne6_t1_mode-uniform_nmin2_solver-gcg_sz0_proj1_restarts10_iter300_kmax4_tol1em09_U1to20_N20-d03a74da789e.pkl", 9.0, 4),
]
SETTINGS = STRONG = dict(solver="gcg", n_restarts=30, max_iter=1500, tol=1e-9, k_max_extra=6)


def rho_as_in_production(f, i, n):
    """Same steps as run_U_dependence_gme for grid point i."""
    r = pickle.load(open(R + f, "rb"))["result"]; si = r["system_info"]
    L, t, V, Ne = si["L"], si["t"], si.get("V", 0.0), si["Ne"]; U = float(r["U_values"][i])
    sp = "_sz0_" in f; es = bool(r["excited_singlet"])
    with contextlib.redirect_stdout(io.StringIO()):
        energies, eigvecs, n_so = compute_all_eigenstates(L, t, U, "uniform", Ne, spin_preserving=sp, V=V, classify_spin=False)
        if es:
            info = tp.find_first_excited_singlet(L=L, t=t, U=U, V=V, mode="uniform", n_electrons=Ne, k=r["singlet_k"], spin_preserving=sp)
            E0, psi0 = float(info["energy"]), np.asarray(info["psi"])
        else:
            E0, psi0 = energies[0], eigvecs[:, 0]
    degen = np.where(np.abs(energies - E0) < 1e-10)[0]
    ops = tp._precompute_ops(tuple(range(n)), n_so, Ne, spin_preserving=sp, project_doublons=True)
    raw = (tp._rdm_from_ops(ops, psi0, n) if len(degen) <= 1
           else sum(tp._rdm_from_ops(ops, eigvecs[:, d], n) for d in degen) / len(degen))
    tr = np.real(np.trace(raw))
    return raw / tr, tr, r


def job(pt):
    f, x, n = pt
    r0 = pickle.load(open(R + f, "rb"))["result"]
    i = int(np.argmin(np.abs(np.asarray(r0["U_over_t"]) - x)))
    rho, tr, r = rho_as_in_production(f, i, n)
    same = abs(np.trace(rho @ rho).real - r["purity"][n][i]) < 1e-10 and abs(tr - r["omega"][n][i]) < 1e-10
    c, pur, det = tp.compute_gme(rho, n, **SETTINGS)
    return f, i, n, float(r["C_GME"][n][i]), float(c), same, [kr["k"] for kr in det["k_records"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--dry-run", action="store_true", help="light settings, nothing written (checks the rebuilt rho)")
    A = ap.parse_args()
    SETTINGS = dict(STRONG, n_restarts=2, max_iter=30, k_max_extra=1) if A.dry_run else STRONG
    with mp.Pool(min(A.workers, len(POINTS))) as pool:
        res = pool.map(job, POINTS)
    by_file = {}
    for f, i, n, old, new, same, qs in res:
        print(f"{f[:44]}  U/t={pickle.load(open(R+f,'rb'))['result']['U_over_t'][i]:.4g} n={n}: same rho {same}  stored {old:.4f}  strong {new:.4f}  (q tried {qs})")
        assert same, "rebuilt rho differs from the production one"
        by_file.setdefault(f, []).append((i, n, old, new))
    if A.dry_run:
        sys.exit(0)
    for f, pts in by_file.items():
        d = pickle.load(open(R + f, "rb")); r = d["result"]
        log = []
        for i, n, old, new in pts:
            keep = min(old, new)
            r["C_GME"][n][i] = keep
            log.append(dict(index=i, U_over_t=float(r["U_over_t"][i]), n=n, old=old, strong=new, kept=keep, settings=SETTINGS))
        r["point_rerun"] = dict(points=log, note="strong GCG rerun of single points; C_GME = lower of production and rerun (both are upper bounds)",
                                date=datetime.now().isoformat(timespec="seconds"))
        out = R + f.replace(".pkl", "_rerun.pkl")
        pickle.dump(d, open(out, "wb"))
        print("written", out.split("/")[-1][:90])
