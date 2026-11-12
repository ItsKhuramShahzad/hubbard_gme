"""
Tables of C_GME and detection probability without doublons (amplitudes with a doublon on any of the n detection sites set to 0 in the state, not renormalised) next to the values of the full state (with doublons).

One table per figure, same rows as the paper's tables (SPECS in tables_lib). Columns:
  x, n, omega_n (paper), omega_n^P, gamma_n^P, C_GME (paper), C_GME^P
omega_n and omega_n^P are recomputed from the states with the package ED (tables_lib.build_states, as make_tables.py
does); the run stops if omega_n^P differs from the value stored in the projected PKL. C_GME and the purities are read
from the PKLs; for the projected runs the *_rerun.pkl is used when it exists.

Run from the hubbard_gme repo:
    PYTHONPATH=src python audit_reports/table_tests/make_projected_tables.py [--spec KEY]
Without --spec all tables are written to projected_tables.tex next to this file; with --spec only that table, to
projected_table_<KEY>.tex.
"""
import argparse, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tables_lib import SPECS, ZERO_TRACE, load, point, build_states, omega

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ_RUNS = os.path.join(HERE, "..", "..", "results", "rdm_pkl_results", "thermal_runs")
XLABEL = {"U": "$U/t$", "V": "$V/t$", "K": "$K$"}


def projected_result(spec):
    base = spec["pkl"].replace("_restarts", "_proj1_restarts", 1) if "_es_" not in spec["pkl"] else \
        spec["pkl"].replace("_es_", "_es_proj1_", 1)
    if spec["kind"] == "V" or spec["kind"] == "K":
        base = spec["pkl"].replace("_full_", "_full_proj1_", 1)
    for name in (base.replace(".pkl", "_rerun.pkl"), base):
        path = os.path.join(PROJ_RUNS, name)
        if os.path.exists(path):
            return pickle.load(open(path, "rb"))["result"], name
    raise FileNotFoundError(base)


def fmt(v, digits=3, sci=True):
    """sci=False for C_GME and purity: values below 1e-3 are optimiser noise around 0, printed as 0.000 like the paper."""
    if v is None or not np.isfinite(v):
        return "--"
    if sci and 0 < abs(v) < 10 ** -digits:
        m, e = f"{v:.0e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    return f"{v:.{digits}f}"


def table(spec):
    m, r, sp, xs = load(spec)
    rp, pname = projected_result(spec)
    xp = np.asarray(rp["U_over_t"] if spec["kind"] == "U" else rp["V_over_t"] if spec["kind"] == "V" else rp["N_values"], float)
    lines = []
    for x in spec["rows"]:
        i, j = int(np.argmin(np.abs(xs - x))), int(np.argmin(np.abs(xp - x)))
        assert abs(xs[i] - x) < 1e-6 and abs(xp[j] - x) < 1e-6, (spec["key"], x)
        L, K, t, U, V = point(spec, m, r, sp, float(x))
        psis = build_states(spec, L, K, t, U, V, sp)[0]
        first = True
        for n in r["n_values"]:
            if n > min(L, K):
                continue
            w, wP = omega(psis, L, K, n, sp), omega(psis, L, K, n, sp, project_doublons=True)
            stored = float(rp["omega"][n][j])
            if abs(stored - wP) > 1e-9:
                raise RuntimeError(f"{spec['key']} x={x} n={n}: omega_P {wP} differs from the stored {stored}")
            defined = wP > ZERO_TRACE
            CP = float(rp["C_GME"][n][j]) if defined else None
            lines.append(" & ".join([fmt(x, 3 if spec["kind"] == "U" and x % 1 else 0) if first else "", str(n), fmt(w),
                                     fmt(wP), fmt(float(rp["purity"][n][j]) if defined else None, sci=False),
                                     fmt(float(r["C_GME"][n][i]) if w > ZERO_TRACE else None, sci=False),
                                     fmt(CP, sci=False) if CP is not None else "--"]) + r" \\")
            first = False
        lines.append(r"\hline")
    lines[-1] = lines[-1] if lines[-1] != r"\hline" else ""
    state = "first excited singlet" if spec["state"] == "es" else "ground state"
    cap = (rf"{spec['fig']} {spec['panel']}, {state}: $\mathcal{{C}}_{{\mathrm{{GME}}}}$ and detection probability without "
           r"doublons, compared with the full state (with doublons). "
           r"$\omega_n$ and "
           r"$\mathcal{C}_{\mathrm{GME}}$: full state, with doublons; $\omega_n^{P}$: trace of the $n$BRDM without doublons, i.e. the probability "
           r"of exactly one electron on each detection site; $\gamma_n^{P}$ and $\mathcal{C}_{\mathrm{GME}}^{P}$: purity and "
           r"GME concurrence of the normalized spin state. Entries -- : $\omega_n^{P}=0$, the state is not defined.")
    head = (rf"{XLABEL[spec['kind']]} & $n$ & $\omega_n$ & $\omega_n^{{P}}$ & $\gamma_n^{{P}}$ & "
            r"$\mathcal{C}_{\mathrm{GME}}$ & $\mathcal{C}_{\mathrm{GME}}^{P}$ \\")
    body = "\n".join(l for l in lines if l)
    return (f"% {spec['key']}: paper PKL {spec['pkl']}\n% projected PKL {pname}\n"
            "\\begin{table*}\n\\caption{" + cap + "}\n\\label{tab:proj_" + spec["key"] + "}\n"
            "\\begin{ruledtabular}\n\\begin{tabular}{ccccccc}\n" + head + "\n\\hline\n" + body +
            "\n\\end{tabular}\n\\end{ruledtabular}\n\\end{table*}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=None, help="one key of tables_lib.SPECS; default: all")
    A = ap.parse_args()
    specs = [s for s in SPECS if A.spec in (None, s["key"])]
    if not specs:
        sys.exit(f"unknown --spec {A.spec}; choose from {[s['key'] for s in SPECS]}")
    tex = "\n".join(table(s) for s in specs)
    out = os.path.join(HERE, "projected_tables.tex" if A.spec is None else f"projected_table_{A.spec}.tex")
    open(out, "w").write(tex)
    print("written", out)
