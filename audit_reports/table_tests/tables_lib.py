"""
Shared definitions for make_tables.py and validate_tables.py.

Every table of the paper is described by one spec: which PKL it comes from, which state
(ground state or first excited singlet), the sector, and which rows are printed.
The state is always rebuilt exactly as the production scan built it:
  * ground state : compute_k_eigenstates, all states within 1e-10 of E0 averaged with equal weights
  * excited      : find_first_excited_singlet (full Hamiltonian), one state
"""
import contextlib, io, os, pickle
import numpy as np
from openfermion import FermionOperator, get_number_preserving_sparse_operator
from hubbard_gme.hamiltonian import compute_k_eigenstates
from hubbard_gme.thermal_pipeline import find_first_excited_singlet

RUNS = "/mnt/d/Dropbox/PhDWork/hubbard_gme/results/rdm_pkl_results/thermal_runs/"
FIGDATA = "/mnt/d/Dropbox/PhDWork/hubbard_gme/latex_figs/pdf/FIGURES_DATA.txt"
DEG_TOL = 1e-10          # same degeneracy tolerance as the production scans
ZERO_TRACE = 1e-14       # production threshold (rho = Gamma/tr only if |tr| > 1e-14); below it the trace is zero to round-off

SPECS = [
    # key, kind, PKL, x name, rows, figure, panel
    dict(key="uscan_gs_L6", kind="U", state="gs", fig="Fig. 1", panel="(a) L=6", rows=[1, 2, 4, 6, 10, 15, 20],
         pkl="uscan_v1_L6_Ne6_t1_mode-uniform_nmin2_solver-gcg_sz0_restarts10_iter200_kmax4_tol1em09_U1to20_N20-d03a74da789e.pkl"),
    dict(key="uscan_gs_L8", kind="U", state="gs", fig="Fig. 1", panel="(b) L=8", rows=[1, 2, 4, 6, 10, 15, 20],
         pkl="uscan_v1_L8_Ne8_t1_mode-uniform_nmin2_solver-gcg_full_restarts8_iter200_kmax4_tol1em09_U1to20_N20-d03a74da789e.pkl"),
    dict(key="uscan_es_L6", kind="U", state="es", fig="Fig. 2", panel="(a) L=6", rows=[1, 5.9, 10.8, 20.6, 30.4, 40.2, 50],
         pkl="uscan_v1_L6_Ne6_t0p1_mode-uniform_nmin2_solver-gcg_full_es_restarts10_iter300_kmax5_tol1em09_U0.1to5_N21-8081b758e7cc.pkl"),
    dict(key="uscan_es_L8", kind="U", state="es", fig="Fig. 2", panel="(b) L=8", rows=[1, 5.9, 10.8, 20.6, 30.4, 40.2, 50],
         pkl="uscan_v1_L8_Ne8_t0p1_mode-uniform_nmin2_solver-gcg_full_es_restarts10_iter300_kmax5_tol1em09_U0.1to5_N21-8081b758e7cc.pkl"),
    dict(key="vscan_L6", kind="V", state="gs", fig="Fig. 4", panel="(a) L=6", rows=[0, 2, 4, 6, 8, 10, 12],
         pkl="vscan_v1_L6_Ne6_U10_t1_mode-uniform_nmin2_solver-gcg_full_restarts10_iter300_kmax5_tol1em09_V0to12_N25-ebabade1a2a1.pkl"),
    dict(key="vscan_L8", kind="V", state="gs", fig="Fig. 4", panel="(b) L=8", rows=[0, 2, 4, 6, 8, 10, 12],
         pkl="vscan_v1_L8_Ne8_U10_t1_mode-uniform_nmin2_solver-gcg_full_restarts10_iter300_kmax5_tol1em09_V0to12_N25-ebabade1a2a1.pkl"),
    dict(key="doped_K", kind="K", state="gs", fig="Fig. 6", panel="single panel", rows=[3, 4, 5, 6, 7, 8],
         pkl="nscan_v1_L8_U10_t1_mode-uniform_nmin2_solver-gcg_full_restarts10_iter300_kmax2_tol1em09_N3to8_n6-d2c9afae4595.pkl"),
    dict(key="doped_U", kind="U", state="gs", fig="Fig. 5", panel="(a) L=8, K=6", rows=[1, 2, 4, 6, 10, 15, 20],
         pkl="uscan_v1_L8_Ne6_t1_mode-uniform_nmin2_solver-gcg_sz0_restarts10_iter300_kmax4_tol1em09_U1to20_N20-d03a74da789e.pkl"),
]


def load(spec):
    d = pickle.load(open(RUNS + spec["pkl"], "rb"))
    m, r = d["meta"], d["result"]
    sp = "_sz0_" in spec["pkl"]                      # sector from the run name (meta lacks it in older runs)
    xs = np.asarray(r["U_over_t"] if spec["kind"] == "U" else r["V_over_t"] if spec["kind"] == "V" else r["N_values"], float)
    return m, r, sp, xs


def point(spec, m, r, sp, x):
    """Parameters (L, K, t, U, V) of one row."""
    L, t = m["L"], m["t"]
    if spec["kind"] == "U":
        return L, m["Ne"], t, x * t, float(m.get("V") or 0.0)
    if spec["kind"] == "V":
        return L, m["Ne"], t, float(m["U"]), x * t
    return L, int(x), t, float(m["U"]), float(m.get("V") or 0.0)


def build_states(spec, L, K, t, U, V, sp, singlet_k=16):
    """Return (list of state vectors to average, energies, info dict)."""
    with contextlib.redirect_stdout(io.StringIO()):
        if spec["state"] == "gs":
            E, vecs, _ = compute_k_eigenstates(L, t, U, "uniform", K, k=12, spin_preserving=sp, V=V)
            deg = np.where(np.abs(E - E[0]) < DEG_TOL)[0]
            return [vecs[:, j] for j in deg], E, dict(g=len(deg), gap=E[len(deg)] - E[0])
        info = find_first_excited_singlet(L, t, U, V, n_electrons=K, k=singlet_k, spin_preserving=False)
        E = info["energies"]
        g_level = int(np.sum(np.abs(E - info["energy"]) < DEG_TOL))     # all states at the singlet's energy
        return [info["psi"]], E, dict(g=g_level, gap=info["energy"] - E[0], energy=info["energy"], idx=info["idx"])


def spin_info(psis, L, K, sp):
    """Total spin S and the S_z values present in the span of psis (S^2 and S_z diagonalised in that span)."""
    from openfermion import s_squared_operator, sz_operator
    P = np.column_stack(psis)
    S2 = get_number_preserving_sparse_operator(s_squared_operator(L), 2 * L, K, spin_preserving=sp)
    Sz = get_number_preserving_sparse_operator(sz_operator(L), 2 * L, K, spin_preserving=sp)
    s2 = np.linalg.eigvalsh(P.conj().T @ (S2 @ P))
    sz = np.linalg.eigvalsh(P.conj().T @ (Sz @ P))
    S = sorted({round(0.5 * (-1 + np.sqrt(1 + 4 * max(v, 0))) * 2) / 2 for v in s2})
    Szs = sorted({round(2 * v) / 2 for v in sz})
    return S, Szs


def occ_op(sites):
    op = FermionOperator("")
    for s in sites:
        op = op * (FermionOperator(((2 * s, 1), (2 * s, 0))) + FermionOperator(((2 * s + 1, 1), (2 * s + 1, 0))))
    return op


def single_occ_op(sites, L, K, sp):
    """prod_s (n_up + n_dn - 2 n_up n_dn): 1 if each site holds exactly one electron, else 0."""
    op = FermionOperator("")
    for s in sites:
        up, dn = FermionOperator(((2 * s, 1), (2 * s, 0))), FermionOperator(((2 * s + 1, 1), (2 * s + 1, 0)))
        op = op * (up + dn - 2 * up * dn)
    return get_number_preserving_sparse_operator(op, 2 * L, K, spin_preserving=sp)


def omega(psis, L, K, n, sp, project_doublons=False):
    """omega_n = <prod_{s<n} n_s>, averaged over the degenerate states (= Tr of the averaged Gamma).
    project_doublons=True: probability of exactly one electron on each of the n sites."""
    if project_doublons:
        return float(np.mean([np.real(np.vdot(p, single_occ_op(range(n), L, K, sp) @ p)) for p in psis]))
    M = get_number_preserving_sparse_operator(occ_op(range(n)), 2 * L, K, spin_preserving=sp)
    return float(np.mean([np.real(np.vdot(p, M @ p)) for p in psis]))


def rho_s(psis, L, K, n, sp, project_doublons=False):
    """rho_s on sites 0..n-1, built as run_U_dependence_gme does (degenerate states averaged, divided by the trace)."""
    from hubbard_gme.thermal_pipeline import _precompute_ops, _rdm_from_ops
    ops = _precompute_ops(tuple(range(n)), 2 * L, K, spin_preserving=sp, project_doublons=project_doublons)
    raw = sum(_rdm_from_ops(ops, p, n) for p in psis) / len(psis)
    tr = np.real(np.trace(raw))
    return raw / tr if abs(tr) > ZERO_TRACE else raw
