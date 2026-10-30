"""
Deep validation of every number in all_tables.tex (U/t gs and es, V/t, doped K, doped U/t).

Per row:
  state   : rebuilt as in production (tables_lib.build_states); ground states averaged over the
            degenerate manifold (|E-E0| < 1e-10); cross-checks: dense eigensolver (L=6), the other
            sector (Sz=0 runs vs full Hamiltonian), excited singlet energy vs PKL, <S^2>, degeneracy.
  omega_n : (a) table method <prod n_s> (FermionOperator), (b) production Tr Gamma(n)
            (_precompute_ops + _rdm_from_ops, sites 0..n-1, degenerate average),
            (c) direct count over the occupation basis, (d) direct full-ring Gamma for n = L = 8.
  purity  : Tr rho_s^2 from (b)/(d) vs the purity stored in the PKL; zero-trace rows: the PKL must
            hold the zero-matrix result (C = 0) and the table must show "--".
  C_GME   : table file = PKL = FIGURES_DATA.txt; closed-form pure-state formula when purity = 1;
            a GCG re-run from scratch for one (x, n) per table.
  doped   : S from <S^2>, g, and g = (2S+1) x (number of momenta) for K = 3..8.
  file    : every number printed in all_tables.tex is compared with the computed value.

Run from the hubbard_gme repo:
    PYTHONPATH=src python /mnt/d/Dropbox/PhDWork/hubbard_gme/audit_reports/table_tests/validate_tables.py
Writes validate_tables_report.txt next to this file.
"""
import contextlib, io, os, re, sys, time
import numpy as np
from openfermion import get_number_preserving_sparse_operator, s_squared_operator
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "trace_check_L8_U10"))
from tables_lib import SPECS, ZERO_TRACE, FIGDATA, load, point, build_states, omega
import rdm_full_ring_direct as direct
from hubbard_gme.hamiltonian import compute_k_eigenstates, build_hubbard_hamiltonian
from hubbard_gme.thermal_pipeline import _precompute_ops, _rdm_from_ops
from hubbard_gme.gme import compute_C_GME_pure
from hubbard_gme.convex_roof import convex_roof_gcg

OUT = os.path.join(HERE, "validate_tables_report.txt")
GCG_SPOT = {"uscan_gs_L6": (10, 4), "uscan_gs_L8": (10, 3), "uscan_es_L6": (10.8, 4), "uscan_es_L8": (10.8, 3),
            "vscan_L6": (4, 4), "vscan_L8": (4, 3), "doped_K": (6, 4), "doped_U": (10, 4)}
DOPED_EXPECT = {3: (1.5, 4), 4: (1.0, 3), 5: (0.5, 4), 6: (0.0, 1), 7: (0.5, 4), 8: (0.0, 1)}   # (S, g)
log = open(OUT, "w")
FAILS, NCHECK = [], [0]


def out(s=""):
    print(s, flush=True); log.write(s + "\n"); log.flush()


def check(cond, msg):
    NCHECK[0] += 1
    if not cond:
        FAILS.append(msg); out("   FAIL: " + msg)


def count_omega(psis, occ, n):
    site = occ[:, 0::2] + occ[:, 1::2]
    f = np.prod(site[:, :n], axis=1)
    return float(np.mean([np.sum(np.abs(p) ** 2 * f) for p in psis]))


def figdata(fig, panel):
    txt = open(FIGDATA).read()
    blk = txt[txt.index(fig + "  "):]
    blk = blk[blk.index("--- " + panel):]
    blk = blk[blk.index("C_GME (value per x and n):"):]
    lines = blk.split("\n")[1:]
    ns = [int(x) for x in re.findall(r"n=(\d+)", lines[0])]
    res = {}
    for ln in lines[1:]:
        p = ln.split()
        if not p or not re.match(r"^[0-9.]+$", p[0]):
            break
        res[float(p[0])] = {n: (float(v) if v != "-" else np.nan) for n, v in zip(ns, p[1:])}
    return res


def num(c):
    c = c.strip().rstrip("\\").strip()
    if c in ("--", ""):
        return np.nan
    m = re.match(r"\$?([0-9.]+)\\times10\^\{(-?\d+)\}\$?", c)
    return float(m.group(1)) * 10 ** int(m.group(2)) if m else float(c)


def table_file():
    txt = open(os.path.join(HERE, "all_tables.tex")).read()
    tabs = {}
    for lab in re.findall(r"\\label\{tab:(\w+)\}", txt):
        body = txt[txt.index(r"\label{tab:" + lab):]
        body = body[body.index(r"\hline") + 6: body.index(r"\end{tabular}")]
        tabs[lab] = [[c.strip() for c in ln.split("&")] for ln in body.strip().split("\n") if ln.strip() != r"\hline"]
    return tabs


out(f"Deep validation of all_tables.tex   ({time.strftime('%Y-%m-%d %H:%M')})")
FILE = table_file()
T0 = time.time()
for spec in SPECS:
    m, r, sp, xs = load(spec)
    fd = figdata(spec["fig"], spec["panel"])
    out("\n" + "=" * 112)
    out(f"{spec['key']}: {spec['fig']} {spec['panel']}, sector {'Sz=0' if sp else 'full'}, PKL {spec['pkl'][:60]}...")
    out("=" * 112)
    ops_cache, frows = {}, FILE[spec["key"]]
    fi = 0
    for x in spec["rows"]:
        i = int(np.argmin(np.abs(xs - x)))
        L, K, t, U, V = point(spec, m, r, sp, x)
        psis, E, info = build_states(spec, L, K, t, U, V, sp, int(r.get("singlet_k", 16) or 16))
        out(f"\n x={x:g}  (L={L} K={K} t={t:g} U={U:g} V={V:g})  g={info['g']}  gap={info['gap']:.6g}")
        # ---- state checks
        if spec["state"] == "gs":
            if L == 6:
                H = get_number_preserving_sparse_operator(build_hubbard_hamiltonian(L, t, U, V=V), 2 * L, K, spin_preserving=sp).toarray()
                Ed = np.linalg.eigvalsh(H)
                check(abs(Ed[0] - E[0]) < 1e-9 and abs((Ed[info['g']] - Ed[0]) / t - info["gap"]) < 1e-7,
                      f"{spec['key']} x={x}: dense spectrum disagrees")
            if sp:
                with contextlib.redirect_stdout(io.StringIO()):
                    E2, _, _ = compute_k_eigenstates(L, t, U, "uniform", K, k=6, spin_preserving=False, V=V)
                check(abs(E2[0] - E[0]) < 1e-8, f"{spec['key']} x={x}: E0(Sz=0) != E0(full)")
                check(info["g"] == 1, f"{spec['key']} x={x}: Sz=0 run but ground state degenerate")
        else:
            check(abs(info["energy"] - r["singlet_energy"][i]) < 1e-8, f"{spec['key']} x={x}: singlet energy vs PKL")
            if int(info["idx"]) != int(r["singlet_idx"][i]):
                out(f"   note: eigenstate index {info['idx']} vs PKL {r['singlet_idx'][i]} (same energy; ordering only)")
        S2 = get_number_preserving_sparse_operator(s_squared_operator(L), 2 * L, K, spin_preserving=sp)
        s2 = [float(np.real(np.vdot(p, S2 @ p))) for p in psis]
        S = 0.5 * (-1 + np.sqrt(1 + 4 * max(s2[0], 0)))
        out(f"   <S^2> = {', '.join(f'{v:.6f}' for v in s2)}  ->  S = {S:.3f}")
        if spec["kind"] == "K":
            Sx, gx = DOPED_EXPECT[K]
            check(abs(S - Sx) < 1e-6 and info["g"] == gx, f"doped K={K}: S={S}, g={info['g']} (expected {Sx}, {gx})")
        elif spec["state"] == "es" or not spec["kind"] == "K":
            check(all(abs(v) < 1e-6 for v in s2), f"{spec['key']} x={x}: state is not a singlet")
        # ---- per n
        ns = [n for n in m["n_values"] if (n >= 3 or spec["kind"] == "K") and n <= (K if spec["kind"] == "K" else L)]
        occ, index = direct.basis(L, K, sp)
        for n in ns:
            wa = omega(psis, L, K, n, sp)
            wc = count_omega(psis, occ, n)
            Cp, Pp = float(r["C_GME"][n][i]), float(r["purity"][n][i])
            msg = f"   n={n}: omega a={wa:.10e} c={wc:.10e}"
            check(abs(wa - wc) <= 1e-12 + 1e-9 * wc, f"{spec['key']} x={x} n={n}: omega (a) vs (c)")
            G = None
            if L == 6 or n <= 5:
                key = (K, n)
                if key not in ops_cache:
                    ops_cache[key] = _precompute_ops(tuple(range(n)), 2 * L, K, spin_preserving=sp)
                G = sum(_rdm_from_ops(ops_cache[key], p, n) for p in psis) / len(psis)
            elif n == L:
                direct.prepare(L, occ)
                G = sum(direct.gamma_full_ring(p, L, occ, index) for p in psis) / len(psis)
            if G is not None:
                wb = float(np.real(np.trace(G)))
                msg += f" b={wb:.10e}"
                check(abs(wb - wc) <= 1e-12 + 1e-9 * abs(wc), f"{spec['key']} x={x} n={n}: production trace vs count")
                if abs(wb) > ZERO_TRACE:
                    rho = G / wb
                    pur = float(np.real(np.trace(rho @ rho)))
                    msg += f" | purity {pur:.8f} (PKL {Pp:.8f})"
                    check(abs(pur - Pp) < 1e-7, f"{spec['key']} x={x} n={n}: purity {pur} vs PKL {Pp}")
                    if abs(pur - 1) < 1e-6:
                        Cf = compute_C_GME_pure(rho, n)
                        msg += f" | closed-form C {Cf:.6f}"
                        check(abs(Cf - Cp) < 1e-6, f"{spec['key']} x={x} n={n}: closed-form C {Cf} vs PKL {Cp}")
            if wc < ZERO_TRACE:
                msg += "  [zero trace]"
                check(Cp < 1e-6, f"{spec['key']} x={x} n={n}: zero trace but PKL C = {Cp}")
            # FIGURES_DATA
            xk = min(fd, key=lambda z: abs(z - x))
            fdc = fd[xk].get(n, np.nan) if abs(xk - x) < 1e-3 else np.nan
            msg += f" | C PKL {Cp:.6f} FIG {fdc:.6f}"
            check(np.isnan(fdc) or abs(fdc - Cp) < 2e-6, f"{spec['key']} x={x} n={n}: FIGURES_DATA {fdc} vs PKL {Cp}")
            # table file
            if spec["kind"] == "K":
                row = frows[fi]; fi += 1
                tw, tP, tC = num(row[4]), num(row[5]), num(row[6])
                check(int(row[3]) == n, f"doped table row order K={K} n={n}")
                check(abs(tP - round(Pp, 3)) < 1.5e-3, f"doped K={K} n={n}: table purity {tP} vs {Pp}")
            else:
                if n == ns[0]:
                    row = frows[fi]; fi += 1
                    check(abs(num(row[0]) - x) < 1e-9, f"{spec['key']}: row order at x={x}")
                    gt = num(row[1])
                    check(abs(gt - info["gap"]) <= max(1.5e-3, 0.006 * info["gap"]), f"{spec['key']} x={x}: table gap {gt} vs {info['gap']}")
                j = ns.index(n)
                tw, tC = num(row[2 + 2 * j]), num(row[3 + 2 * j])
            if wc < ZERO_TRACE:
                check(np.isnan(tw) and np.isnan(tC), f"{spec['key']} x={x} n={n}: zero trace not marked -- in table")
            else:
                check(abs(tw - wc) <= max(1.5e-3 if wc >= 0.0995 else 0.0, 0.006 * wc), f"{spec['key']} x={x} n={n}: table omega {tw} vs {wc}")
                check(abs(tC - round(Cp, 3)) < 1.5e-3, f"{spec['key']} x={x} n={n}: table C {tC} vs PKL {Cp}")
            out(msg)
    # ---- GCG spot check
    x, n = GCG_SPOT[spec["key"]]
    i = int(np.argmin(np.abs(xs - x)))
    L, K, t, U, V = point(spec, m, r, sp, x)
    psis, E, info = build_states(spec, L, K, t, U, V, sp, int(r.get("singlet_k", 16) or 16))
    ops = _precompute_ops(tuple(range(n)), 2 * L, K, spin_preserving=sp)
    G = sum(_rdm_from_ops(ops, p, n) for p in psis) / len(psis)
    rho = G / np.real(np.trace(G))
    t1 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        res = convex_roof_gcg(rho, n, n_restarts=m["n_restarts"], max_iter=m["max_iter"], k_max_extra=m["k_max_extra"], verbose=False)
    Cp = float(r["C_GME"][n][i])
    out(f"\n GCG re-run x={x:g} n={n}: now {res['C_gcg']:.6f}  PKL {Cp:.6f}  diff {res['C_gcg'] - Cp:+.1e}  ({time.time() - t1:.0f} s)")
    check(abs(res["C_gcg"] - Cp) < 2e-3, f"{spec['key']}: GCG re-run {res['C_gcg']} vs PKL {Cp}")
    out(f" ({time.time() - T0:.0f} s so far)")

out("\n" + "=" * 112)
out(f"RESULT: {NCHECK[0]} checks, {len(FAILS)} failed" + ("" if not FAILS else ":\n  " + "\n  ".join(FAILS)))
out(f"total time {time.time() - T0:.0f} s")
