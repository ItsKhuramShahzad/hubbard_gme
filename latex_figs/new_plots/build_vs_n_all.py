"""C_GME against the number of spins n for every paper figure, one line per value of the figure's scan parameter.
Values are read directly from the paper PKLs (read in place in Dropbox), the same PKLs as Figs. 1-6; each md5 is
checked against latex_figs/pdf/FIGURES_DATA.txt. Fig. 3 uses the merged tail-rerun PKLs, cut at k_B T/t <= 0.3.
Colour: one hue per parameter (U/t blue, k_B T/t red, V/t purple, K green), light = small value, dark = large value.
Output: figN_vs_n.pdf and figN_vs_n_data.txt in this folder.
Run: ~/miniconda3/envs/qchem/bin/python build_vs_n_all.py"""
import os, hashlib, pickle, subprocess, numpy as np, fitz
HERE = os.path.dirname(os.path.abspath(__file__))
R = "/mnt/d/Dropbox/PhDWork/hubbard_gme/results/rdm_pkl_results/thermal_runs/"
TEX = os.path.expanduser("~/miniconda3/envs/texbuild/bin/tectonic")
RAMP = {"U": [(107, 174, 214), (33, 113, 181), (8, 48, 107)],
        "T": [(252, 146, 114), (222, 45, 38), (103, 0, 13)],
        "V": [(158, 154, 200), (106, 81, 163), (63, 0, 125)],
        "K": [(116, 196, 118), (35, 139, 69), (0, 68, 27)]}
XKEY = {"U": "U_over_t", "T": "T_values", "V": "V_over_t", "K": "N_values"}
LABEL = {"U": r"$U/t$", "T": r"$k_BT/t$", "V": r"$V/t$", "K": r"$K$"}
U1 = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
# round colour-bar ticks per parameter range
TICKS = {"fig1": "1,5,10,15,20", "fig2": "1,10,20,30,40,50", "fig3": "0,0.1,0.2", "fig4": "0,3,6,9,12",
         "fig5": "1,5,10,15,20", "fig6": "3,4,5,6,7,8"}
FIGS = {
 "fig1": ("U", U1, [(6, "uscan_v1_L6_Ne6_t1_mode-uniform_nmin2_solver-gcg_sz0_restarts10_iter200_kmax4_tol1em09_U1to20_N20-d03a74da789e.pkl", "0f958a55ccbda2c48771b9915fe9ee04", "(a)"),
                  (8, "uscan_v1_L8_Ne8_t1_mode-uniform_nmin2_solver-gcg_full_restarts8_iter200_kmax4_tol1em09_U1to20_N20-d03a74da789e.pkl", "0fad681095a933ec6c9ed44efe70415b", "(b)")]),
 "fig2": ("U", [1, 3.45, 5.9, 8.35, 10.8, 15.7, 20.6, 30.4, 40.2, 50],
          [(6, "uscan_v1_L6_Ne6_t0p1_mode-uniform_nmin2_solver-gcg_full_es_restarts10_iter300_kmax5_tol1em09_U0.1to5_N21-8081b758e7cc.pkl", "3a196c3228020ba0d6ee0f9cfd23b9ed", "(a)"),
           (8, "uscan_v1_L8_Ne8_t0p1_mode-uniform_nmin2_solver-gcg_full_es_restarts10_iter300_kmax5_tol1em09_U0.1to5_N21-8081b758e7cc.pkl", "4a9ce86dd28452a13e101feded4cc911", "(b)")]),
 "fig3": ("T", [0, 0.0263158, 0.0526316, 0.0789474, 0.105263, 0.131579, 0.157895, 0.210526, 0.289474],
          [(6, "thermal_v1_L6_Ne6_U10_t1_mode-uniform_tpall_t2all_tperpall_nmin2_solver-gcg_full_restarts10_iter300_kmax5_tol1em09_cut10_adaptive1_step5_clusteravg1_T0to0.289474_N12-f3dd7936eb54_knone_diag0_tailrerun.pkl", "c461b23073a9856bb4b47d90b859b9c5", "(a)"),
           (8, "thermal_v1_L8_Ne8_U10_t1_mode-uniform_tpall_t2all_tperpall_nmin2_solver-gcg_full_restarts8_iter200_kmax4_tol1em09_cut10_adaptive1_step5_clusteravg1_T0to0.5_N20-3c87501b406b_knone_diag0_tailrerun.pkl", "3667b16a32b01ca4b795718f35e4195b", "(b)")]),
 "fig4": ("V", [0, 2, 4, 4.5, 5, 5.5, 6, 8, 12],
          [(6, "vscan_v1_L6_Ne6_U10_t1_mode-uniform_nmin2_solver-gcg_full_restarts10_iter300_kmax5_tol1em09_V0to12_N25-ebabade1a2a1.pkl", "94efa59fe8bb2207c0902b6f488de0ae", "(a)"),
           (8, "vscan_v1_L8_Ne8_U10_t1_mode-uniform_nmin2_solver-gcg_full_restarts10_iter300_kmax5_tol1em09_V0to12_N25-ebabade1a2a1.pkl", "1a26ac6c36a7dab7b76ca2d0a98b5fb7", "(b)")]),
 "fig5": ("U", U1, [(6, "uscan_v1_L8_Ne6_t1_mode-uniform_nmin2_solver-gcg_sz0_restarts10_iter300_kmax4_tol1em09_U1to20_N20-d03a74da789e.pkl", "3df20c2a3732f5c262454f8aeaf34706", "(a)")]),
 "fig6": ("K", [3, 4, 5, 6, 7, 8], [(8, "nscan_v1_L8_U10_t1_mode-uniform_nmin2_solver-gcg_full_restarts10_iter300_kmax2_tol1em09_N3to8_n6-d2c9afae4595.pkl", "9bd0e3b5ed61246521fc8cd3066b1ec2", None)]),
}

def load(name, md5):
    raw = open(R + name, "rb").read()
    assert hashlib.md5(raw).hexdigest() == md5, f"{name}: md5 differs from FIGURES_DATA"
    return pickle.loads(raw)["result"]

def colour(ramp, f):
    x = f * (len(ramp) - 1); i = min(int(x), len(ramp) - 2); w = x - i
    return tuple(int(round((1 - w) * a + w * b)) for a, b in zip(ramp[i], ramp[i + 1]))

def panel(fig, kind, xs, nmax, name, md5, letter, defs, log):
    r = load(name, md5); X = np.asarray(r[XKEY[kind]], float)
    lo, hi = min(xs), max(xs); ns = [n for n in sorted(r["C_GME"]) if 3 <= int(n) <= nmax]
    log.append(f"{letter or ''} PKL {name}\n  n = {ns}")
    plots = []
    for x in xs:
        j = int(np.argmin(np.abs(X - x))); assert abs(X[j] - x) < 1e-5, (fig, x, X[j])
        pts = [(n, float(r["C_GME"][n][j])) for n in ns]
        pts = [(n, v) for n, v in pts if np.isfinite(v) and not (kind == "K" and n > X[j])]
        rgb = colour(RAMP[kind], (X[j] - lo) / (hi - lo)); cname = f"c{fig}{len(defs)}"
        defs.append(rf"\definecolor{{{cname}}}{{RGB}}{{{rgb[0]},{rgb[1]},{rgb[2]}}}")
        log.append(f"  {LABEL[kind].strip('$')}={X[j]:g}: " + "  ".join(f"n={n}: {v:.6f}" for n, v in pts))
        plots.append(rf"    \addplot[color={cname}, mark=*, mark size=1.1pt, line width=1.2pt] coordinates {{"
                     + " ".join(f"({n},{v:g})" for n, v in pts) + "};")
    cmap = " ".join(f"rgb255=({a},{b},{c})" for a, b, c in RAMP[kind])
    ticks = TICKS[fig]
    letter_code = (r"    after end axis/.append code={\node[anchor=east, font=\normalsize\rmfamily] at (axis description cs:-0.13,0.954) {"
                   + letter + "};},\n") if letter else ""
    return (r"""\begin{tikzpicture}
  \begin{axis}[
    width=\columnwidth, height=0.78\columnwidth,
    xlabel={number of spins $n$}, ylabel={$C_{\mathrm{GME}}$},
    xmin=2.7, xmax=""" + f"{nmax + 0.3}" + r""", ymin=-0.03, ymax=1.05, xtick={""" + ",".join(map(str, ns)) + r"""},
    grid=both, grid style={line width=.1pt, draw=gray!20},
    tick label style={font=\footnotesize\rmfamily, /pgf/number format/fixed},
    label style={font=\normalsize\rmfamily},
    colormap={ramp}{""" + cmap + r"""},
    colorbar, point meta min=""" + f"{lo:g}" + ", point meta max=" + f"{hi:g}" + r""",
    colorbar style={width=2.5mm, ytick={""" + ticks + r"""}, ylabel={""" + LABEL[kind] + r"""},
                    yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2},
                    tick label style={font=\footnotesize\rmfamily}, ylabel style={font=\normalsize\rmfamily}},
""" + letter_code + r"""  ]
""" + "\n".join(plots) + r"""
  \end{axis}
\end{tikzpicture}""")

for fig, (kind, xs, panels) in FIGS.items():
    defs, log = [], [f"{fig}: C_GME against n, read directly from the PKLs; one line per {LABEL[kind].strip('$')}"]
    body = (r"\\[2pt]" + "\n").join(panel(fig, kind, xs, L, name, md5, letter, defs, log) for L, name, md5, letter in panels)
    tex = (r"\documentclass[border=4pt]{standalone}" "\n" r"\usepackage{pgfplots}\pgfplotsset{compat=1.18}" "\n"
           + "\n".join(defs) + "\n" r"\begin{document}" "\n" r"\setlength{\columnwidth}{8.6cm}%" "\n"
           r"\begin{tabular}{@{}c@{}}" "\n" + body + "\n" r"\end{tabular}" "\n" r"\end{document}" "\n")
    open(os.path.join(HERE, f"{fig}_vs_n.tex"), "w").write(tex)
    open(os.path.join(HERE, f"{fig}_vs_n_data.txt"), "w").write("\n".join(log) + "\n")
    res = subprocess.run([TEX, f"{fig}_vs_n.tex"], cwd=HERE, capture_output=True, text=True)
    if res.returncode: raise SystemExit(fig + "\n" + res.stderr[-2500:])
    src = fitz.open(os.path.join(HERE, f"{fig}_vs_n.pdf")); p = src[0]; bb = fitz.Rect()
    for d in p.get_drawings(): bb |= d["rect"]
    for b in p.get_text("dict")["blocks"]: bb |= fitz.Rect(b["bbox"])
    bb = (bb + (-2, -2, 2, 2)) & p.rect
    o = fitz.open(); q = o.new_page(width=bb.width, height=bb.height); q.show_pdf_page(q.rect, src, 0, clip=bb)
    src.close(); tmp = os.path.join(HERE, "_c.pdf"); o.save(tmp); os.replace(tmp, os.path.join(HERE, f"{fig}_vs_n.pdf"))
    print("built", fig)
