"""Write latex_figs/pdf/FIGURES_DATA.txt: per figure/panel, the source PKL and
every number that was plotted (x values, C_GME, purity), read from the PKLs."""
import datetime
import hashlib
import os
import pickle
import re

import numpy as np

ROOT = "/mnt/d/Dropbox/PhDWork/hubbard_gme"
PKL_DIR = f"{ROOT}/results/rdm_pkl_results/thermal_runs"
TEX_DIR = f"{ROOT}/latex_figs"
OUT = f"{TEX_DIR}/pdf/FIGURES_DATA.txt"

# (figure, pdf, panel letter, tex stem, legend shown?) -- tex header names the PKL
F1A = "uscan_v1_L6_Ne6_t1_mode-uniform_nmin2_solver-gcg_sz0_restarts10_iter200_kmax4_tol1em09_U1to20_N20-d03a74da789e"
F1B = "uscan_v1_L8_Ne8_t1_mode-uniform_nmin2_solver-gcg_full_restarts8_iter200_kmax4_tol1em09_U1to20_N20-d03a74da789e"
F1A0 = "uscan_v1_L6_Ne6_t0p1_mode-uniform_nmin2_solver-gcg_full_restarts8_iter300_kmax2_tol1em09_U0to15_N16-2f7c06936fa2"
F2A = "uscan_v1_L6_Ne6_t0p1_mode-uniform_nmin2_solver-gcg_full_es_restarts10_iter300_kmax5_tol1em09_U0.1to5_N21-8081b758e7cc"
F2B = "uscan_v1_L8_Ne8_t0p1_mode-uniform_nmin2_solver-gcg_full_es_restarts10_iter300_kmax5_tol1em09_U0.1to5_N21-8081b758e7cc"
F3A = "thermal_v1_L6_Ne6_U10_t1_mode-uniform_nmin2_solver-gcg_full_restarts10_iter300_kmax5_tol1em09_step5_clusteravg1_T0to1_N20-148a4f8ecb18_knone_diag0"
F3B = "thermal_v1_L8_Ne8_U10_t1_mode-uniform_nmin2_solver-gcg_full_restarts8_iter200_kmax4_tol1em09_clusteravg1_T0to0.5_N20-3c87501b406b_knone_diag0"
F3A_OLD = "thermal_v1_L6_Ne6_U10_t1_mode-uniform_nmin2_solver-gcg_restarts2_iter200_kmax4_tol1em09_step5_clusteravg1_T0to1_N40-98c4f0fd07f2_knone_diag0"
F4A = "vscan_v1_L6_Ne6_U10_t1_mode-uniform_nmin2_solver-gcg_full_restarts10_iter300_kmax5_tol1em09_V0to12_N25-ebabade1a2a1"
F4B = "vscan_v1_L8_Ne8_U10_t1_mode-uniform_nmin2_solver-gcg_full_restarts10_iter300_kmax5_tol1em09_V0to12_N25-ebabade1a2a1"
F4A15 = "vscan_v1_L6_Ne6_U10_t1_mode-uniform_nmin2_solver-gcg_full_restarts8_iter300_kmax4_tol1em09_V0to15_N16-2f7c06936fa2"
F4B15 = "vscan_v1_L8_Ne8_U10_t1_mode-uniform_nmin2_solver-gcg_sz0_restarts8_iter200_kmax4_tol1em09_V0to15_N16-2f7c06936fa2"
F5 = "uscan_v1_L8_Ne6_t1_mode-uniform_nmin2_solver-gcg_sz0_restarts10_iter300_kmax4_tol1em09_U1to20_N20-d03a74da789e"
F6 = "nscan_v1_L8_U10_t1_mode-uniform_nmin2_solver-gcg_full_restarts10_iter300_kmax2_tol1em09_N3to8_n6-d2c9afae4595"

FIGURES = [   # (title, whole-figure PDF, [(panel label, tex stem of the C_GME panel)])
    ("Fig. 1  ground state vs U/t, K = L, V = 0", "fig1.pdf", [("(a) L=6", F1A), ("(b) L=8", F1B)]),
    ("Fig. 2  first excited singlet vs U/t, K = L, V = 0", "fig2.pdf", [("(a) L=6", F2A), ("(b) L=8", F2B)]),
    ("Fig. 3  thermal state vs k_B T / t, U = 10, t = 1, K = L, V = 0 (plotted up to 0.5)", "fig3.pdf",
     [("(a) L=6", F3A), ("(b) L=8", F3B)]),
    ("Fig. 4  ground state vs V/t, U = 10, t = 1, K = L", "fig4.pdf", [("(a) L=6", F4A), ("(b) L=8", F4B)]),
    ("Fig. 5  ground state vs U/t, L = 8, K = 6, V = 0", "fig5.pdf", [("(a) L=8, K=6", F5)]),
    ("Fig. 6  ground state vs electron number K, L = 8, U/t = 10, V = 0", "fig6.pdf", [("single panel", F6)]),
]
COMPANIONS = [("purity Tr(rho_s^2)", "_purity"), ("linear entropy 1 - Tr(rho_s^2)", "_linear_entropy"),
              ("von Neumann entropy S(rho_s) = -Tr(rho_s ln rho_s)", "_vN_entropy")]


def tex_header(stem):
    txt = open(f"{TEX_DIR}/{stem}.tex").read()
    get = lambda k: (re.search(rf"% {k}\s*: (.+)", txt) or [None, None])[1]
    return dict(pkl=get("source PKL").strip(), kind=get("scan type").strip(),
                state=(get("state") or "").strip(), quantity=(get("quantity") or "cgme").strip(),
                cap=(get("x-max cap") or "").strip(),
                letter=bool(re.search(r"after end axis.*\{\((a|b)\)\}", txt)),
                legend="\\addlegendentry" in txt, ncurves=txt.count("\\addplot["))


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def table(x, data, ns, xname="x", fmt="{:9.6f}"):
    head = f"  {xname:>10s} " + "".join(f"   n={int(n):<6d}" for n in ns)
    rows = [head]
    for i, xv in enumerate(x):
        rows.append(f"  {xv:10.5g} " + "".join(
            "   " + (fmt.format(np.ravel(data[n])[i]) if np.isfinite(np.ravel(data[n])[i]) else "      -  ")
            for n in ns))
    return "\n".join(rows)


def panel_block(label, stem, whole):
    h = tex_header(stem)
    path = f"{PKL_DIR}/{h['pkl']}"
    d = pickle.load(open(path, "rb"))
    r, meta = d["result"], d.get("meta", {}) or {}
    si = r.get("system_info", {}) or {}
    t = float(si.get("t", re.search(r"_t(\d+(?:p\d+)?)_", h["pkl"]).group(1).replace("p", ".")
                     if re.search(r"_t(\d+(?:p\d+)?)_", h["pkl"]) else 1.0))
    kind = h["kind"]
    if kind == "uscan":
        x = np.asarray(r["U_over_t"], float) if "U_over_t" in r else np.asarray(r["U_values"], float) / t
        xname, raw = "U/t", f"U = {np.asarray(r['U_values']).min():g} ... {np.asarray(r['U_values']).max():g} at t = {t:g}"
    elif kind == "vscan":
        x = np.asarray(r["V_over_t"], float) if "V_over_t" in r else np.asarray(r["V_values"], float) / t
        xname, raw = "V/t", f"V = {np.asarray(r['V_values']).min():g} ... {np.asarray(r['V_values']).max():g} at t = {t:g}"
    elif kind == "Tscan":
        x = np.asarray(r["T_values"], float) / t
        xname, raw = "k_B T / t", f"T = {np.asarray(r['T_values']).min():g} ... {np.asarray(r['T_values']).max():g} (k_B = 1) at t = {t:g}"
    elif kind == "nscan":
        x = np.asarray(r["N_values"], float)
        xname, raw = "K (electrons)", "electron numbers swept"
    else:
        raise SystemExit(f"unhandled kind {kind}")
    keep = np.ones(x.shape, bool)
    if h["cap"]:
        keep = ~(x > float(h["cap"]) + 1e-12)
    xs = x[keep]
    C = {n: np.ravel(np.asarray(v, float))[keep] for n, v in r["C_GME"].items()}
    P = {n: np.ravel(np.asarray(v, float))[keep] for n, v in r["purity"].items()}
    ent_src, Sv = "stored in the source PKL", r.get("entropy")
    derived = f"{TEX_DIR}/pdf/derived_pkl/{os.path.splitext(h['pkl'])[0]}_vN_from_rdm_bank.pkl"
    if Sv is None and os.path.exists(derived):
        Sv = pickle.load(open(derived, "rb"))["result"]["entropy"]
        ent_src = ("from the ground-state RDMs in rdm_bank/ (same rho_s: purity matches to 1e-15); "
                   f"- = no bank entry; file: latex_figs/pdf/derived_pkl/{os.path.basename(derived)}")
    S = ({n: np.ravel(np.asarray(v, float))[keep] for n, v in Sv.items()} if Sv is not None else None)
    # points whose spin RDM was a zero matrix (purity < 1/2^n) are not plotted: show them as "-"
    nzero = 0
    for n in list(C):
        z = np.ravel(np.asarray(r["purity"][n], float))[keep] < 1.0 / 2 ** int(n) - 1e-12
        if z.any():
            nzero += int(z.sum()) if 3 <= int(n) else 0
            C[n] = np.where(z, np.nan, C[n]); P[n] = np.where(z, np.nan, P[n])
            if S is not None and n in S:
                S[n] = np.where(z, np.nan, S[n])
    L = int(si.get("L", re.search(r"_L(\d+)_", h["pkl"]).group(1)))
    Ne = si.get("Ne")
    if Ne is None:
        m = re.search(r"_Ne(\d+)_", h["pkl"])
        Ne = int(m.group(1)) if m else None
    nmax = L if kind == "nscan" else (int(Ne) if Ne else L)
    ns = [n for n in sorted(C) if 3 <= int(n) <= nmax]          # n=2 is not plotted
    mU = re.search(r"_U([\d.]+)_", h["pkl"])
    Ustr = si.get("U", mU.group(1) if mU else "swept (see x axis)")
    sector = "all Sz (full Hamiltonian)" if "_full" in h["pkl"] else "Sz = 0 sector"
    rs = re.search(r"restarts(\d+)", h["pkl"]).group(1)
    it = re.search(r"iter(\d+)", h["pkl"]).group(1)
    km = re.search(r"kmax(\d+)", h["pkl"]).group(1)
    cmd = (meta.get("environment", {}) or {}).get("command") or "(not stored in this PKL)"
    conv = r.get("convergence") or {}
    tot = cap = 0
    for n in ns:
        for j, c in enumerate(conv.get(n, []) or []):
            if keep[j] and isinstance(c, dict) and "k_records" in c:
                tot += 1
                cap += int(not c.get("converged", True))
    quantity = {"cgme": "C_GME (GME concurrence)", "linear-entropy": "linear entropy 1 - Tr(rho_s^2)",
                "purity": "purity Tr(rho_s^2)", "vn-entropy": "von Neumann entropy"}[h["quantity"]]
    lines = [
        f"  --- {label}",
        f"  PDF file        : latex_figs/pdf/{whole}, panel {label.split()[0]}",
        f"  LaTeX file      : latex_figs/{stem}.tex",
        f"  source PKL      : {path}",
        f"  PKL date / md5  : {datetime.datetime.fromtimestamp(os.path.getmtime(path)):%Y-%m-%d %H:%M}   {md5(path)}",
        f"  system          : L = {L}, K = {Ne if Ne else 'swept'}, U = {Ustr}, "
        f"t = {t:g}, V = {si.get('V', 'swept' if kind == 'vscan' else 0)}, mode = {si.get('mode', 'uniform')}",
        f"  state / sector  : {'first excited singlet' if (r.get('excited_singlet') or '_es_' in h['pkl']) else ('thermal (Gibbs) state' if kind == 'Tscan' else 'ground state')} / {sector}",
        f"  solver          : GCG convex roof, {rs} restarts, {it} iterations, k_max_extra = {km}"
        + (f";  {cap} of {tot} (point, n) values stopped at the q limit" if tot else ""),
        f"  run command     : {cmd}",
        f"  x axis          : {xname}, {len(xs)} points from {xs.min():g} to {xs.max():g}   ({raw})"
        + (f"; points above {h['cap']} not plotted" if h["cap"] else ""),
        f"  y axis          : {quantity}",
        *([f"  left out        : {nzero} point(s) where the spin RDM is a zero matrix (trace ~1e-20; shown as - below)"] if nzero else []),
        f"  curves          : n = {', '.join(str(int(n)) for n in ns)}   (n = 2 not plotted)"
        + ("" if h["legend"] else ";  no legend in this panel (same colours as the other panel)"),
        "",
        "  C_GME (value per x and n):",
        table(xs, C, ns, xname),
        "",
        "  purity Tr(rho_s^2) (linear entropy = 1 - purity):",
        table(xs, P, ns, xname),
        "",
    ]
    if S is not None:
        lines += [f"  von Neumann entropy S(rho_s) = -Tr(rho_s ln rho_s), natural log ({ent_src}):",
                  table(xs, S, ns, xname), ""]
    else:
        lines += ["  von Neumann entropy: not stored for this run.", ""]
    return "\n".join(lines)


out = [
    "FIGURE DATA RECORD - which PKL each figure was made from, and every plotted number",
    f"written {datetime.datetime.now():%Y-%m-%d %H:%M} from the PKLs themselves (latex_figs/pdf/make_FIGURES_DATA.py)",
    "",
    "PDFs: latex_figs/pdf/figN.pdf = C_GME figure (all panels in one file); figN_purity / figN_linear_entropy /",
    "      figN_vN_entropy.pdf = the same figure with the purity or an entropy instead of C_GME.",
    "LaTeX: latex_figs/<source-PKL name>.tex, generated by scripts/pkl_to_latex.py with",
    "       --bare --panel-letter (a)|(b) --legend none (L=6 panels) | auto (L=8 panels)",
    "       [--x-max 0.5 for Fig. 3]  [--quantity purity|linear-entropy|vn-entropy for the companion PDFs].",
    "Each table: first column = the x-axis value of the plotted point (U/t, V/t, k_B T/t or K),",
    "            one column per cluster size n = the plotted value at that point.",
    f"PKL directory: {PKL_DIR}",
    "Sector tag in the PKL name: _full_ = all Sz (full Hamiltonian), _sz0_ or no tag = Sz = 0 only.",
    "At T = 0 the two sectors give identical states and RDMs (checked 2026-10-23 for L = 6 and 8);",
    "for thermal states they do not, which is why Fig. 3 uses _full_ runs only.",
    "",
]
def summary(stems, u_values=(10, 20)):
    rows = ["=" * 100, "SUMMARY - Fig. 1 cases (ground state, K = L, V = 0) at fixed U/t: C_GME vs mixedness of rho_s",
            "=" * 100,
            "  1 - Tr(rho_s^2) = linear entropy, S_vN = von Neumann entropy (nats). Odd n: purity <= 1/2 by spin symmetry.", ""]
    for u in u_values:
        rows.append(f"  U/t = {u}")
        for lab, stem in stems:
            h = tex_header(stem); r = pickle.load(open(f"{PKL_DIR}/{h['pkl']}", "rb"))["result"]
            der = f"{TEX_DIR}/pdf/derived_pkl/{os.path.splitext(h['pkl'])[0]}_vN_from_rdm_bank.pkl"
            Sv = r.get("entropy") or (pickle.load(open(der, "rb"))["result"]["entropy"] if os.path.exists(der) else None)
            x = np.asarray(r["U_over_t"], float); i = int(np.argmin(abs(x - u)))
            ns = [n for n in sorted(r["C_GME"]) if int(n) >= 3]
            g = lambda d, n: np.ravel(np.asarray(d[n], float))[i]
            rows.append(f"    {lab:5s} n          " + "".join(f"{int(n):>9d}" for n in ns))
            rows.append(f"          C_GME      " + "".join(f"{g(r['C_GME'], n):9.3f}" for n in ns))
            rows.append(f"          1-Tr rho^2 " + "".join(f"{1 - g(r['purity'], n):9.3f}" for n in ns))
            if Sv is not None:
                rows.append(f"          S_vN       " + "".join(f"{g(Sv, n):9.3f}" for n in ns))
        rows.append("")
    return rows


out += summary([("L=6", F1A), ("L=8", F1B)])
for title, whole, panels in FIGURES:
    out += ["=" * 100, title, "=" * 100, ""]
    comp = [(q, whole[:-4] + sfx + ".pdf") for q, sfx in COMPANIONS
            if os.path.exists(f"{TEX_DIR}/pdf/{whole[:-4]}{sfx}.pdf")]
    missing = [q for q, sfx in COMPANIONS if (q, whole[:-4] + sfx + ".pdf") not in comp]
    out.append(f"  PDFs            : latex_figs/pdf/{whole}  (C_GME)")
    for q, f in comp:
        out.append(f"                    latex_figs/pdf/{f}  ({q}; same runs, same layout)")
    if missing:
        out.append(f"  not made        : {', '.join(missing)} - not stored in every PKL of this figure")
    if whole == "fig1.pdf":
        out.append("  NOTE fig1_vN_entropy.pdf, panel (a): the L=6 run did not store the entropy. It was computed from the")
        out.append("       ground-state RDMs stored in results/rdm_pkl_results/rdm_bank/ (rdmbank_v1_L6_Ne6_U*_t*_..._sz0_clusteravg1.pkl;")
        out.append("       U/t = U/t of the bank file, exact at T = 0). Same rho_s as the figure: purity matches the stored purity to")
        out.append("       1e-15 at every point. U/t = 11, 13, 17, 19 are not in the bank and are left out. Derived PKL with the")
        out.append("       source list: latex_figs/pdf/derived_pkl/..._vN_from_rdm_bank.pkl")
    out.append("")
    for label, stem in panels:
        out.append(panel_block(label, stem, whole))
open(OUT, "w").write("\n".join(out) + "\n")
print("wrote", OUT, os.path.getsize(OUT), "bytes")
