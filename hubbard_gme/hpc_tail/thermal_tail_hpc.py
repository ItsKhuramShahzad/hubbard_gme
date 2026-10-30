"""
Rerun of the high-temperature tail of Fig. 3(b) (L = 8, U/t = 10, thermal n-spin RDMs).

The production run (8 restarts, 200 iterations, q = r..r+4) stopped every point at the iteration
limit, so its C_GME values at high T are not converged. This driver improves them, one n per call,
splitting the work into many independent GCG tasks that run in parallel on one node.

  fresh : new random starts, q = r .. r+k_extra, `restarts` per q, `max_iter` iterations.
          Seeds are the production ones, 31337 + 17 (q - r) + restart.
  warm  : start from the decompositions stored in the production run (one start per stored q),
          enlarged to q = r + k_extra, and run `warm_iter` more iterations.

rho_s is rebuilt from the decomposition stored in each job record (rho = sum_i p_i |psi_i><psi_i|),
so the 18 GB RDM bank is not needed. Every finished task is saved on its own; a killed job resumes
where it stopped when started again with the same arguments.

Example (one node, n = 7, both modes):
    OMP_NUM_THREADS=1 python hpc_tail/thermal_tail_hpc.py --n 7 --mode both --workers 52
"""
import argparse, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from hubbard_gme.convex_roof import _h_full_unitary, _gcg_gradient, _gcg_inner, _gcg_optimise_single

PKL = ("thermal_v1_L8_Ne8_U10_t1_mode-uniform_tpall_t2all_tperpall_nmin2_solver-gcg_full_restarts8_iter200_kmax4_"
       "tol1em09_cut10_adaptive1_step5_clusteravg1_T0to0.5_N20-3c87501b406b_knone_diag0.pkl")
RUNS = os.path.join(REPO, "results", "rdm_pkl_results", "thermal_runs")


# ── GCG loop: convex_roof._gcg_optimise_single line for line, but from a given start U ─────────────
def gcg_from(U, V, s, n, max_iter=1000, tol=1e-9, measure="cgme"):
    G = _gcg_gradient(U, V, s, n, measure)
    X = -G
    h_curr = _h_full_unitary(U, V, s, n, measure=measure)
    history = [h_curr]
    converged = False
    termination = "max_iter"
    last_t = 1.0
    for it in range(max_iter):
        def line_h(t):
            return _h_full_unitary(U @ expm(t * X), V, s, n, measure)
        t_bracket = max(0.1, min(2.0, 4.0 * abs(last_t)))
        try:
            res = minimize_scalar(line_h, bracket=(0.0, t_bracket), method="brent",
                                  options={"xtol": 1e-8, "maxiter": 50})
            t_new, h_new = float(res.x), float(res.fun)
        except Exception:
            t_new = 0.01
            h_new = line_h(t_new)
        if h_new >= h_curr - 1e-15 or t_new <= 0:
            X = -G
            try:
                res = minimize_scalar(line_h, bracket=(0.0, 1.0), method="brent",
                                      options={"xtol": 1e-8, "maxiter": 50})
                t_new, h_new = float(res.x), float(res.fun)
            except Exception:
                termination = "line_search_exception"
                break
            if h_new >= h_curr - 1e-15 or t_new <= 0:
                termination = "line_search_stalled"
                break
        U_new = U @ expm(t_new * X)
        G_new = _gcg_gradient(U_new, V, s, n, measure)
        half_step = expm(0.5 * t_new * X)
        T_transp = half_step @ G @ half_step.conj().T
        denom = _gcg_inner(G, G)
        gamma = 0.0 if abs(denom) < 1e-20 else _gcg_inner(G_new - T_transp, G_new) / denom
        gamma = max(0.0, gamma)
        X_new = -G_new + gamma * X
        if _gcg_inner(G_new, X_new) >= 0.0:
            X_new = -G_new
        grad_norm = np.sqrt(_gcg_inner(G_new, G_new))
        improvement = h_curr - h_new
        if grad_norm < tol and improvement < tol:
            U, G, X = U_new, G_new, X_new
            h_curr = h_new
            history.append(h_curr)
            converged = True
            termination = "gradient_and_step_tol"
            break
        U, G, X = U_new, G_new, X_new
        h_curr = h_new
        last_t = t_new
        history.append(h_curr)
    return dict(h_best=max(0.0, h_curr), U_best=U, n_iters=len(history) - 1, history=history,
                converged=converged, termination=termination)


def random_start(k, seed):
    """Same random start as _gcg_optimise_single(seed=seed)."""
    rng = np.random.default_rng(seed)
    H0 = rng.standard_normal((k, k)) + 1j * rng.standard_normal((k, k))
    return expm(1j * 0.5 * (H0 + H0.conj().T))


def warm_start(rec, V, s, k_new, eps, seed):
    """Unitary (k_new x k_new) in our eigenbasis reproducing a stored decomposition, enlarged to k_new states.
    The added states start with zero weight; a small random rotation (eps) gives them a nonzero gradient."""
    k, r = rec["k"], len(s)
    W = np.zeros((V.shape[0], k), complex)
    for e in rec["decomp"]:
        W[:, e["component_index"]] = np.sqrt(e["p"]) * e["psi"]
    A = ((V.conj().T @ W) / s[:, None]).T             # U[i, j] = <v_j|w_i> / sqrt(lambda_j)
    A = np.vstack([A, np.zeros((k_new - k, r))])       # added states: zero weight
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(np.hstack([A, rng.standard_normal((k_new, k_new - r))]))
    U = np.hstack([A, Q[:, r:]])
    if eps > 0:
        H = rng.standard_normal((k_new, k_new)) + 1j * rng.standard_normal((k_new, k_new))
        U = U @ expm(1j * eps * 0.5 * (H + H.conj().T) / np.sqrt(k_new))
    return U


# ── one (T, n) point of the production run ───────────────────────────────────────────────────────
def load_point(runs, pkl, ti, n):
    r = pickle.load(open(os.path.join(runs, pkl), "rb"))["result"]
    jd = pickle.load(open(os.path.join(runs, "job_cache", os.path.basename(r["gme_job_paths"][(ti, n)])), "rb"))
    recs = jd["decompositions"]["gcg"]
    rho = sum(e["p"] * np.outer(e["psi"], e["psi"].conj()) for e in recs[-1]["decomp"])
    rho = 0.5 * (rho + rho.conj().T)
    lam, V = np.linalg.eigh(rho)
    m = lam > 1e-10                                     # same rank cut as convex_roof_gcg
    return dict(T=float(r["T_values"][ti]), V=V[:, m], s=np.sqrt(lam[m]), r=int(m.sum()), recs=recs,
                C_old=float(r["C_GME"][n][ti]), purity=float(np.real(np.sum(np.abs(rho) ** 2))))


_cache = {}


def point(ti):
    key = (A.runs, A.pkl, ti, A.n)
    if key not in _cache:
        _cache[key] = load_point(*key)
    return _cache[key]


def run_task(task):
    name, kind, ti, k, idx = task
    path = os.path.join(OUT, name + ".pkl")
    if os.path.exists(path):
        return name, "done before"
    P = point(ti)
    t0 = time.time()
    if kind == "fresh":
        seed = 31337 + 17 * (k - P["r"]) + idx
        out = gcg_from(random_start(k, seed), P["V"], P["s"], A.n, max_iter=A.max_iter)
        out.pop("U_best")                               # deterministic: rebuilt from the seed if needed
        out["seed"] = seed
    else:
        rec = P["recs"][idx]
        U0 = warm_start(rec, P["V"], P["s"], k, A.eps, seed=1000 + idx)
        out = gcg_from(U0, P["V"], P["s"], A.n, max_iter=A.warm_iter)
        out.update(from_q=rec["k"], C_stored=rec["C_GME"])
    out["history"] = np.asarray(out["history"], np.float32)
    out.update(kind=kind, T=P["T"], T_index=ti, n=A.n, k=k, r=P["r"], idx=idx, C_old=P["C_old"],
               purity=P["purity"], elapsed=time.time() - t0,
               settings=dict(max_iter=A.max_iter, warm_iter=A.warm_iter, restarts=A.restarts, k_extra=A.k_extra, eps=A.eps))
    pickle.dump(out, open(path + ".tmp", "wb"))
    os.replace(path + ".tmp", path)
    return name, f"T={P['T']:.3f} q={k}: h={out['h_best']:.6f} (old {P['C_old']:.6f}) it={out['n_iters']} {out['termination']} {out['elapsed']:.0f}s"


def tasks(T_idx):
    out = []
    for ti in T_idx:
        P = point(ti)
        r, kmax = P["r"], P["r"] + A.k_extra
        if A.mode in ("fresh", "both") and (A.fresh_T is None or ti in A.fresh_T):
            qs = range(r, kmax + 1) if A.fresh_q is None else [r + e for e in A.fresh_q]
            for k in qs:
                for rs in range(A.restarts):
                    out.append((k ** 2 * A.max_iter, (f"fresh_n{A.n}_T{ti}_q{k}_r{rs}", "fresh", ti, k, rs)))
        if A.mode in ("warm", "both"):
            recs = P["recs"]
            use = range(len(recs)) if A.warm_from == "all" else [int(np.argmin([x["C_GME"] for x in recs]))]
            for j in use:
                out.append((kmax ** 2 * A.warm_iter, (f"warm_n{A.n}_T{ti}_from{recs[j]['k']}", "warm", ti, kmax, j)))
    out.sort(key=lambda x: -x[0])                       # longest first
    return [t for _, t in out]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--mode", choices=["fresh", "warm", "both"], default="both")
    ap.add_argument("--T-min", type=float, default=0.15, help="rerun only temperatures k_BT/t >= this")
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--max-iter", type=int, default=1000, help="iterations of each fresh start")
    ap.add_argument("--warm-iter", type=int, default=2000, help="iterations of each warm start")
    ap.add_argument("--k-extra", type=int, default=10, help="q goes up to r + k_extra")
    ap.add_argument("--fresh-T", type=int, nargs="+", default=None, help="T indices for fresh starts (default: all tail T)")
    ap.add_argument("--fresh-q", type=int, nargs="+", default=None, help="q - r values for fresh starts (default 0..k_extra)")
    ap.add_argument("--warm-from", choices=["all", "best"], default="all", help="warm-start from every stored q or only the best")
    ap.add_argument("--eps", type=float, default=0.02, help="size of the random rotation added to a warm start")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--runs", default=RUNS, help="thermal_runs folder (with job_cache/)")
    ap.add_argument("--pkl", default=PKL)
    ap.add_argument("--out", default=os.path.join(RUNS, "tail_rerun_L8"))
    ap.add_argument("--list", action="store_true", help="only print the task list")
    A = ap.parse_args()
    OUT = os.path.join(A.out, f"n{A.n}")
    os.makedirs(OUT, exist_ok=True)

    T_all = np.asarray(pickle.load(open(os.path.join(A.runs, A.pkl), "rb"))["result"]["T_values"])
    T_idx = [i for i, T in enumerate(T_all) if T >= A.T_min - 1e-12]
    for ti in T_idx:
        P = point(ti)
        best = min(P["recs"], key=lambda x: x["C_GME"])
        h0 = _h_full_unitary(warm_start(best, P["V"], P["s"], best["k"], 0.0, 0), P["V"], P["s"], A.n)
        print(f"n={A.n} T={P['T']:.4f} (index {ti}): r={P['r']} purity={P['purity']:.4f} old C={P['C_old']:.6f}"
              f"  stored decomposition gives {h0:.6f}", flush=True)
    todo = tasks(T_idx)
    print(f"{len(todo)} tasks, {A.workers} workers, results in {OUT}", flush=True)
    if A.list:
        for t in todo:
            print("  ", t[0])
        sys.exit()
    t0 = time.time()
    with Pool(A.workers) as pool:
        for i, (name, msg) in enumerate(pool.imap_unordered(run_task, todo), 1):
            print(f"[{i}/{len(todo)} {time.time() - t0:8.0f}s] {name}: {msg}", flush=True)
    print("all tasks finished", flush=True)
