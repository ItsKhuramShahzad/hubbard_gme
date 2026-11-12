"""Combine fig1_vs_n.pdf ... fig6_vs_n.pdf into one review PDF: A4 landscape, two figures side by side per page,
every figure placed at 100 % size so the labels keep their 10 pt.
Run after build_vs_n_all.py: ~/miniconda3/envs/qchem/bin/python combine_vs_n.py"""
import os, fitz
HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS = [("fig1_vs_n", "Fig. 1  ground state, K = L, V = 0", "lines: U/t = 1 ... 20"),
         ("fig2_vs_n", "Fig. 2  first excited singlet, K = L, V = 0", "lines: U/t = 1 ... 50"),
         ("fig3_vs_n", "Fig. 3  thermal state, U/t = 10", "lines: k_B T/t = 0 ... 0.29"),
         ("fig4_vs_n", "Fig. 4  ground state, U/t = 10, V/t scan", "lines: V/t = 0 ... 12"),
         ("fig5_vs_n", "Fig. 5  ground state, L = 8, K = 6 (doped)", "lines: U/t = 1 ... 20"),
         ("fig6_vs_n", "Fig. 6  ground state, L = 8, U/t = 10, K scan", "lines: K = 3 ... 8")]
W, H, TOP, HEAD = 842, 595, 34, 26
out = fitz.open(); npages = (len(ITEMS) + 1) // 2
for k in range(npages):
    pg = out.new_page(width=W, height=H)
    pg.insert_text((40, 22), f"C_GME against the number of spins n   (page {k + 1}/{npages})", fontsize=9.5)
    pair = ITEMS[2 * k:2 * k + 2]
    rects = [fitz.open(os.path.join(HERE, n + ".pdf"))[0].rect for n, _, _ in pair]
    gap = (W - sum(r.width for r in rects)) / (len(rects) + 1)
    x = gap
    for (name, title, sub), r in zip(pair, rects):
        assert TOP + HEAD + r.height <= H, name
        src = fitz.open(os.path.join(HERE, name + ".pdf"))
        pg.insert_text((x, TOP + 10), title, fontsize=9); pg.insert_text((x, TOP + 21), sub, fontsize=8, color=(0.35, 0.35, 0.35))
        pg.show_pdf_page(fitz.Rect(x, TOP + HEAD, x + r.width, TOP + HEAD + r.height), src, 0)   # scale 1:1
        x += r.width + gap
out.save(os.path.join(HERE, "vs_n_all_figures.pdf"), garbage=4, deflate=True)
print("written vs_n_all_figures.pdf, pages:", len(out))
