"""Rebuild the paper figures latex_figs/pdf/figN*.pdf from the Dropbox panel .tex files.
mode=orig: unchanged panels (to check the rebuild matches the current PDFs); mode=10pt: label/letter font \small -> \normalsize, data line width 0.7pt -> 1.2pt."""
import os, sys, shutil, subprocess, fitz
D = "/mnt/d/Dropbox/PhDWork/hubbard_gme/latex_figs"
mode, out = sys.argv[1], sys.argv[2]
work = os.path.join(out, "build"); os.makedirs(work, exist_ok=True)
P = {1: ["uscan_v1_L6_Ne6_t1_mode-uniform_nmin2_solver-gcg_sz0_restarts10_iter200_kmax4_tol1em09_U1to20_N20-d03a74da789e",
         "uscan_v1_L8_Ne8_t1_mode-uniform_nmin2_solver-gcg_full_restarts8_iter200_kmax4_tol1em09_U1to20_N20-d03a74da789e"],
     2: ["uscan_v1_L6_Ne6_t0p1_mode-uniform_nmin2_solver-gcg_full_es_restarts10_iter300_kmax5_tol1em09_U0.1to5_N21-8081b758e7cc",
         "uscan_v1_L8_Ne8_t0p1_mode-uniform_nmin2_solver-gcg_full_es_restarts10_iter300_kmax5_tol1em09_U0.1to5_N21-8081b758e7cc"],
     4: ["vscan_v1_L6_Ne6_U10_t1_mode-uniform_nmin2_solver-gcg_full_restarts10_iter300_kmax5_tol1em09_V0to12_N25-ebabade1a2a1",
         "vscan_v1_L8_Ne8_U10_t1_mode-uniform_nmin2_solver-gcg_full_restarts10_iter300_kmax5_tol1em09_V0to12_N25-ebabade1a2a1"],
     5: ["uscan_v1_L8_Ne6_t1_mode-uniform_nmin2_solver-gcg_sz0_restarts10_iter300_kmax4_tol1em09_U1to20_N20-d03a74da789e"],
     6: ["nscan_v1_L8_U10_t1_mode-uniform_nmin2_solver-gcg_full_restarts10_iter300_kmax2_tol1em09_N3to8_n6-d2c9afae4595"]}
Q = {"": "", "_purity": "_purity", "_linear_entropy": "_SL", "_vN_entropy": "_SvN"}
jobs = []   # (pdf name, [panel files with dir])
for n, bases in P.items():
    for q, suf in Q.items():
        files = []
        for i, b in enumerate(bases):
            s = "_vN_from_rdm_bank_SvN" if (n == 1 and i == 0 and q == "_vN_entropy") else suf
            files.append(os.path.join(D, b + s + ".tex"))
        jobs.append((f"fig{n}{q}", files))
F3 = os.path.join(D, "fig3b_tail_rerun")
for q, (a, b) in {"": ("fig3a_cut", "fig3b_tailrerun"), "_purity": ("fig3a_purity", "fig3b_purity"),
                  "_linear_entropy": ("fig3a_SL", "fig3b_SL")}.items():
    jobs.append((f"fig3{q}", [os.path.join(F3, a + ".tex"), os.path.join(F3, b + ".tex")]))
TEX = "~/miniconda3/envs/texbuild/bin/tectonic"
for name, files in sorted(jobs):
    names = []
    for k, f in enumerate(files):
        s = open(f).read()
        if mode == "10pt":
            c = s.count(r"font=\small\rmfamily")
            assert c >= 1, f
            s = s.replace(r"font=\small\rmfamily", r"font=\normalsize\rmfamily")
            assert "line width=0.7pt" in s, f
            s = s.replace("line width=0.7pt", "line width=1.2pt")
        local = f"{name}_p{k}.tex"; open(os.path.join(work, local), "w").write(s); names.append(local)
    body = "\\\\[2pt]\n".join(f"\\input{{{x}}}" for x in names)
    wrap = ("\\documentclass[border=4pt]{standalone}\n\\usepackage{pgfplots}\\pgfplotsset{compat=1.18}\n\\begin{document}\n"
            "\\setlength{\\columnwidth}{8.6cm}%\n\\begin{tabular}{@{}c@{}}\n" + body + "\n\\end{tabular}\n\\end{document}\n")
    open(os.path.join(work, name + ".tex"), "w").write(wrap)
    r = subprocess.run(f"cd {work} && {TEX} {name}.tex", shell=True, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-2000:]); sys.exit(f"compile failed {name}")
    src = fitz.open(os.path.join(work, name + ".pdf")); pg = src[0]
    bb = None
    for b in pg.get_bboxlog():
        bb = fitz.Rect(b[1]) if bb is None else bb | fitz.Rect(b[1])
    bb = (bb + (-2, -2, 2, 2)) & pg.rect
    o = fitz.open(); p = o.new_page(width=bb.width, height=bb.height); p.show_pdf_page(p.rect, src, 0, clip=bb)
    o.save(os.path.join(out, name + ".pdf")); print(name, round(bb.width), round(bb.height))
