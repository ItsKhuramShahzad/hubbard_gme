"""Copies of all figures (paper 10pt, projected, merged Fig. 1+2) with a soft drop shadow behind each data line.
Sources: ../pdf_10pt/build, ../projected, ../fig12_combined (left unchanged). Run from this folder:
    ~/miniconda3/envs/qchem/bin/python build_shadow.py"""
import os, re, glob, shutil, subprocess, sys, fitz
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(HERE)
sys.path.insert(0, UP)
import figtools
TEX = os.path.expanduser("~/miniconda3/envs/texbuild/bin/tectonic")
SHADOW = ("line width=1.2pt, preaction={draw=black, draw opacity=0.22, line width=2.0pt, "
          "transform canvas={xshift=0.6pt, yshift=-0.6pt}}]")

def add_shadow(path):
    s = open(path).read(); k = s.count("line width=1.2pt]")
    assert k >= 1, path
    name = os.path.basename(path)
    s = figtools.fig2_range_to_40(s, name)            # fig2: U/t axis ends at 40
    s = figtools.split_legend(s, name)                # fig1: colour code split 3+3 over the panels
    if name in figtools.STACKED:
        s = figtools.stacked_legend(s, name)          # figs 1, 2, 6: labels on top, line samples below
    elif name in figtools.COMPACT:
        s = figtools.compact_legend(s, name)
    else:
        s = figtools.legend_inside(s, name)           # 10 pt legend inside the axes, clear of the curves
    s = figtools.add_L_label(s, name)                 # figs 1, 2, 3, 5: "(a) L=6" / "(b) L=8"
    s = s.replace(r"tick label style={font=\footnotesize\rmfamily", r"tick label style={font=\normalsize\rmfamily")   # ticks 10 pt
    s = s.replace(r"legend style={font=\scriptsize\rmfamily", r"legend style={font=\normalsize\rmfamily")         # legend 10 pt
    open(path, "w").write(s.replace("line width=1.2pt]", SHADOW))

def crop(src_pdf, dst_pdf):
    src = fitz.open(src_pdf); p = src[0]; bb = fitz.Rect()
    for d in p.get_drawings(): bb |= d["rect"]
    for b in p.get_text("dict")["blocks"]: bb |= fitz.Rect(b["bbox"])
    bb = (bb + (-2, -2, 2, 2)) & p.rect
    o = fitz.open(); q = o.new_page(width=bb.width, height=bb.height); q.show_pdf_page(q.rect, src, 0, clip=bb)
    o.save(dst_pdf, garbage=4, deflate=True)

def compile_all(work, wrappers, outdir):
    def one(w):
        r = subprocess.run([TEX, w], cwd=work, capture_output=True, text=True)
        if r.returncode: raise RuntimeError(w + "\n" + r.stderr[-1500:])
        crop(os.path.join(work, w[:-4] + ".pdf"), os.path.join(outdir, w[:-4] + ".pdf"))
    with ThreadPoolExecutor(6) as ex: list(ex.map(one, wrappers))
    print(outdir, len(wrappers))

# 1. paper figures (Figs. 1-6, 10 pt)
out = os.path.join(HERE, "paper"); work = os.path.join(out, "build"); os.makedirs(work, exist_ok=True)
for f in glob.glob(os.path.join(UP, "pdf_10pt", "build", "*.tex")): shutil.copy(f, work)
for f in glob.glob(os.path.join(work, "*_p[0-9].tex")): add_shadow(f)
compile_all(work, sorted(os.path.basename(f) for f in glob.glob(os.path.join(work, "fig*.tex")) if "_p" not in os.path.basename(f)[-7:]), out)

# 2. projected figures
out = os.path.join(HERE, "projected"); work = os.path.join(out, "build"); os.makedirs(work, exist_ok=True)
for f in glob.glob(os.path.join(UP, "projected", "fig*_proj*.tex")): shutil.copy(f, work)
for f in glob.glob(os.path.join(work, "*_panel.tex")): add_shadow(f)
compile_all(work, sorted(os.path.basename(f) for f in glob.glob(os.path.join(work, "fig*.tex")) if not f.endswith("_panel.tex")), out)

# 3. merged Fig. 1 + Fig. 2
out = os.path.join(HERE, "fig12_combined"); work = os.path.join(out, "build"); os.makedirs(work, exist_ok=True)
for f in glob.glob(os.path.join(UP, "fig12_combined", "*.tex")): shutil.copy(f, work)
for f in glob.glob(os.path.join(work, "panel_*.tex")): add_shadow(f)
compile_all(work, ["fig12_combined.tex"], out)

# 4. overviews of the projected figures
os.chdir(os.path.join(HERE, "projected"))
def overview(outname, title, suffix, notes):
    pages = [[("fig1a_proj", "Fig. 1(a)  ground state, L=6"), ("fig1b_proj", "Fig. 1(b)  ground state, L=8"),
              ("fig2a_proj", "Fig. 2(a)  excited singlet, L=6"), ("fig2b_proj", "Fig. 2(b)  excited singlet, L=8"),
              ("fig3a_proj", "Fig. 3(a)  thermal, L=6"), (None, "Fig. 3(b)  thermal, L=8")],
             [("fig4a_proj", "Fig. 4(a)  V/t scan, L=6"), ("fig4b_proj", "Fig. 4(b)  V/t scan, L=8"),
              ("fig5_proj", "Fig. 5  L=8, K=6"), ("fig6_proj", "Fig. 6  K scan, L=8")]]
    o = fitz.open(); W, H = 595, 842; cw = 240; pitch = 195
    for pi, items in enumerate(pages):
        pg = o.new_page(width=W, height=H); pg.insert_text((45, 35), f"{title}   (page {pi+1}/2)", fontsize=9.5)
        for k, (n, cap) in enumerate(items):
            x, y = 45 + (k % 2) * (cw + 30), 55 + (k // 2) * pitch; note = notes.get(n, "")
            pg.insert_text((x, y + 8), cap + (f"   ({note})" if note else ""), fontsize=9 if not note else 8.2)
            if n is None: pg.insert_text((x, y + 60), "Still running on HPC (job 150447_2).", fontsize=9); continue
            src = fitz.open(n + suffix + ".pdf"); r = src[0].rect; s = cw / r.width
            pg.show_pdf_page(fitz.Rect(x, y + 12, x + r.width * s, y + 12 + r.height * s), src, 0)
    o.save(outname, garbage=4, deflate=True)
overview("projected_cgme_overview_updated.pdf", "Projected nBRDM (doublons on the n detection sites removed): C_GME, updated 11 Nov 2026", "",
         {"fig1a_proj": "n=4 at U/t=2 rerun", "fig2b_proj": "new", "fig3a_proj": "tail above 0.2 not yet rerun",
          "fig4b_proj": "new; zero-RDM points not shown", "fig5_proj": "n=4 at U/t=6, 9 rerun"})
overview("projected_omega_overview.pdf", "Projected nBRDM: detection probability omega_n = Tr of the projected nBRDM, 11 Nov 2026", "_omega", {})
print("overviews done")
