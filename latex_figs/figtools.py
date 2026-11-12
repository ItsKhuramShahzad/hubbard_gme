"""Shared helpers for the shadow figure builds (build_shadow.py, build_shadow_cosmetic.py):

* legend_inside: place the 10 pt legend INSIDE the axes, automatically, where it does not touch any curve
  (scored over a grid of positions and 1-3 columns; fill=white). A position can be pinned by hand in PINNED
  after visual review.
* add_L_label: panel labels "(a) L = 6" / "(b) L = 8" for the paper figures 1, 2, 3 and "(a) L = 8" for
  figure 5 (supervisor request, 12 Nov 2026). Applied to the C_GME figure and its purity/entropy variants.
"""
import re
from math import ceil

import numpy as np

AXW, AXH = 194.7, 134.6          # axis box in pt, measured on the compiled 10 pt panels
ENTRY_W, COL_SEP, PAD_W = 52.0, 6.0, 8.0     # one legend entry (marker + "n = 3") at 10 pt
ROW_H, PAD_H = 13.5, 7.0

# name -> (x_rel, y_rel of the box's top-left corner, columns); overrides the automatic spot
PINNED = {}


def _axis_limits(s):
    out = []
    for k in ("xmin", "xmax", "ymin", "ymax"):
        m = re.search(rf"\b{k}=([-0-9.]+)", s)
        out.append(float(m.group(1)) if m else None)
    return out


def _dense_points(s):
    """Every curve of the panel sampled about once per pt, in axis-box pt coordinates."""
    xmin, xmax, ymin, ymax = _axis_limits(s)
    pts = []
    for block in re.findall(r"coordinates \{(.*?)\};", s, re.S):
        q = np.array([((float(a) - xmin) / (xmax - xmin) * AXW,
                       (float(b) - ymin) / (ymax - ymin) * AXH)
                      for a, b in re.findall(r"\(([^,]+),([^)]+)\)", block)])
        if len(q) == 0:
            continue
        if len(q) == 1:
            pts.append(q)
            continue
        for a, b in zip(q[:-1], q[1:]):
            pts.append(np.linspace(a, b, max(2, int(np.hypot(*(b - a))))))
    return np.vstack(pts) if pts else np.zeros((0, 2))


def _clearance(box, P):
    if len(P) == 0:
        return 99.0
    x0, y0, x1, y1 = box
    dx = np.maximum(np.maximum(x0 - P[:, 0], 0), P[:, 0] - x1)
    dy = np.maximum(np.maximum(y0 - P[:, 1], 0), P[:, 1] - y1)
    return float(np.hypot(dx, dy).min())


def legend_inside(s, name, verbose=True):
    """Give the panel an inside legend position (10 pt, fill=white) clear of the curves."""
    n = s.count(r"\addlegendentry")
    if n == 0:
        return s
    # drop the author-tuned inside overrides: they were set for the 7 pt legend
    s = re.sub(r"\n\s*legend style=\{at=\{\([0-9.]+,[0-9.]+\)\}, anchor=[a-z ]+(?:, fill=white)?\},", "", s)
    if name in PINNED:
        xr, yr, cols = PINNED[name]
        return _append_axis_options(s, xr, yr, cols)
    P = _dense_points(s)
    best = None
    for cols in (3, 2, 1):
        rows = ceil(n / cols)
        w = cols * ENTRY_W + (cols - 1) * COL_SEP + PAD_W
        h = rows * ROW_H + PAD_H
        if w > AXW - 8 or h > AXH - 8:
            continue
        for tlx in np.arange(4.0, AXW - 4.0 - w + 1e-9, 4.0):
            for tly in np.arange(AXH - 4.0, h + 4.0 - 1e-9, -4.0):
                box = (tlx, tly - h, tlx + w, tly)
                score = (min(_clearance(box, P), 18.0)
                         + 1.2 * tly / AXH + 0.8 * (tlx + w) / AXW - 0.02 * rows)
                if best is None or score > best[0]:
                    best = (score, box, cols)
    assert best is not None, name
    _, box, cols = best
    clear = _clearance(box, P)
    if clear < 1.0:
        # no spot clears the curves: give the panel headroom above the data and put the legend there
        if verbose:
            print(f"    [legend] {name}: no clear spot (best {clear:.1f} pt) - headroom added above the data")
        return _with_headroom(s, n)
    if verbose and clear < 3.0:
        print(f"    [legend] {name}: inside spot with clearance {clear:.1f} pt - review")
    return _append_axis_options(s, round(box[0] / AXW, 3), round(box[3] / AXH, 3), cols)


def _with_headroom(s, n):
    """Extend ymax so an empty band as tall as the legend sits above the highest curve, keep the
    ticks on the data range, and centre the legend in the band. The axis box itself keeps its size."""
    from matplotlib.ticker import MaxNLocator
    xmin, xmax, ymin, ymax = _axis_limits(s)
    ytop = max(float(b) for _, b in re.findall(r"\(([^,]+),([^)]+)\)",
               " ".join(re.findall(r"coordinates \{(.*?)\};", s, re.S))))
    cols = 3 if 3 * ENTRY_W + 2 * COL_SEP + PAD_W <= AXW - 8 else 2
    rows = ceil(s.count(r"\addlegendentry") / cols)
    w = cols * ENTRY_W + (cols - 1) * COL_SEP + PAD_W
    h = rows * ROW_H + PAD_H
    new_ymax = ymin + (ytop - ymin) / ((AXH - h - 12.0) / AXH)
    s = re.sub(r"\bymax=[-0-9.]+", f"ymax={new_ymax:.3f}", s, count=1)
    if "ytick" not in s:
        ticks = [t for t in MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(max(0.0, ymin), ytop)
                 if ymin <= t <= ymax + 1e-9]
        tick_str = ",".join(f"{t:g}" for t in ticks)
        s = re.sub(r"(\n(\s*)ymax=[-0-9.]+,)", rf"\1\n\2ytick={{{tick_str}}},", s, count=1)
    return _append_axis_options(s, round((AXW - w) / 2 / AXW, 3), round((AXH - 4) / AXH, 3), cols)


def _append_axis_options(s, xr, yr, cols):
    extra = (f"    legend style={{at={{({xr},{yr})}}, anchor=north west, fill=white}},\n"
             f"    legend columns={cols},\n")
    i = s.index("\n  ]\n")                            # end of the axis options
    return s[:i + 1] + extra + s[i + 1:]


# panels whose six-entry legend fits inside only as one compact row (Fig. 6; Figs. 1 and 2: the whole legend in
# panel (b), the L = 8 panel that shows all six curves)
COMPACT = set()                 # (replaced by STACKED for Figs. 1, 2 and 6)
COMPACT_W, COMPACT_H = 150.0, 17.0          # measured on the compiled legend (147.8 x 15.9 pt) plus a margin


def compact_legend(s, name):
    """One row inside the top band: "n: 3 4 5 6 7 8" with short line samples (Fig. 6, where no other
    inside spot exists and the y axis must not be extended). The k-th entry stays with the k-th plot."""
    if name not in COMPACT:
        return s
    s = re.sub(r"\\addlegendentry\{\$n=(\d+)\$\}", r"\\addlegendentry{$\1$}", s)
    s = re.sub(r"\n\s*legend style=\{at=\{\([0-9.]+,[0-9.]+\)\}, anchor=[a-z ]+(?:, fill=white)?\},", "", s)
    s = re.sub(r"\n\s*legend columns=\d+,", "", s)
    first = s.index("\\addplot[")
    s = s[:first] + "\\addlegendimage{empty legend}\\addlegendentry{$n$:}\n    " + s[first:]
    # automatic spot: the free place for the compact box with the largest clearance from every curve
    P = _dense_points(s)
    best = None
    for tlx in np.arange(4.0, AXW - 4.0 - COMPACT_W + 1e-9, 2.0):
        for tly in np.arange(AXH - 4.0, COMPACT_H + 4.0 - 1e-9, -2.0):
            box = (tlx, tly - COMPACT_H, tlx + COMPACT_W, tly)
            score = min(_clearance(box, P), 12.0) + 0.5 * tly / AXH
            if best is None or score > best[0]:
                best = (score, box)
    box = best[1]
    if _clearance(box, P) < 1.0:
        print(f"    [legend] {name}: compact legend has clearance {_clearance(box, P):.1f} pt - review")
    xr, yr = round(box[0] / AXW, 3), round(box[3] / AXH, 3)
    extra = (f"    legend style={{at={{({xr},{yr})}}, anchor=north west, fill=white,\n"
             "                  /tikz/every even column/.append style={column sep=2pt}},\n"
             "    legend columns=-1,\n"
             "    legend image post style={xscale=0.55},\n")
    i = s.index("\n  ]\n")
    return s[:i + 1] + extra + s[i + 1:]


SPLIT = ()                      # figures whose legend is split 3 + 3 over the two panels (none now)


STACKED = {"fig1_p1.tex", "fig2_p1.tex", "fig6_p0.tex"}   # legend as a table: labels on top, line samples below
STACKED_W, STACKED_H = 145.5, 20.3                        # measured on the compiled node (144.1 x 19.3 pt) plus a margin


def stacked_legend(s, name):
    """Legend drawn as a small table: one column per curve, "n=3" on top and the curve's line sample with its
    marker underneath (pgfplots legends can only put the label beside the sample). Colours and markers are
    read from each \\addplot, so they always match the curves. Placed automatically in a free spot."""
    if name not in STACKED:
        return s
    entries = []
    for m in re.finditer(r"\\addplot\[(.*?)\] coordinates \{.*?\};\s*\\addlegendentry\{\$n=(\d+)\$\}", s, re.S):
        opts = m.group(1)
        col = re.search(r"color=(\w+)", opts).group(1)
        mark = re.search(r"mark=([^,\]]+)", opts).group(1)
        mopt = re.search(r"mark options=\{([^}]*)\}", opts)
        msize = re.search(r"mark size=([0-9.]+pt)", opts).group(1)
        entries.append((m.group(2), col, mark, mopt.group(1) if mopt else "", msize))
    assert len(entries) >= 2, name
    s = re.sub(r"[ \t]*\\addlegendentry\{[^}]*\}\n", "", s)
    s = re.sub(r"\n\s*legend style=\{at=\{\([0-9.]+,[0-9.]+\)\}, anchor=[a-z ]+(?:, fill=white)?\},", "", s)
    s = re.sub(r"\n\s*legend columns=\d+,", "", s)
    labels = " & ".join(f"$n{{=}}{n}$" for n, *_ in entries)
    bars = " & ".join(
        rf"\tikz[baseline=-0.5ex, x=1pt, y=1pt]{{\draw[color={c}, line width=1.2pt, line cap=butt] (0,0) -- (14,0); "
        rf"\path[color={c}] plot[mark={mk}, mark size={ms}, mark options={{{mo}}}] coordinates {{(7,0)}};}}"
        for _, c, mk, mo, ms in entries)
    # automatic spot for the box
    P = _dense_points(s)
    best = None
    for tlx in np.arange(2.0, AXW - 2.0 - STACKED_W + 1e-9, 1.0):
        for tly in np.arange(AXH - 2.0, STACKED_H + 2.0 - 1e-9, -1.0):
            box = (tlx, tly - STACKED_H, tlx + STACKED_W, tly)
            score = min(_clearance(box, P), 12.0) + 0.5 * tly / AXH
            if best is None or score > best[0]:
                best = (score, box)
    box = best[1]
    if _clearance(box, P) < 1.0:
        print(f"    [legend] {name}: stacked legend has clearance {_clearance(box, P):.1f} pt - review")
    xr, yr = round(box[0] / AXW, 3), round(box[3] / AXH, 3)
    node = (rf"    after end axis/.append code={{\node[anchor=north west, fill=white, draw=gray!40, inner sep=1pt, "
            rf"font=\normalsize\rmfamily] at (rel axis cs:{xr},{yr}) "
            rf"{{\setlength{{\tabcolsep}}{{2.5pt}}\renewcommand{{\arraystretch}}{{0.7}}"
            rf"\begin{{tabular}}{{{'c' * len(entries)}}}{labels}\\{bars}\end{{tabular}}}};}},") + "\n"
    i = s.index("\n  ]\n")
    return s[:i + 1] + node + s[i + 1:]


def split_legend(s, name):
    """Figs. 1 and 2 C_GME: the six-entry legend does not fit inside either panel at 10 pt without extending the
    y axis, but split 3 + 3 it does: n = 3, 4, 5 go to panel (a) and n = 6, 7, 8 stay in panel (b), each a
    single row in the free band under the flat top curve. legend_inside then places each half as usual."""
    m = re.match(r"(fig\d)_p([01])\.tex$", name)
    if not m or m.group(1) not in SPLIT:
        return s
    entries = re.findall(r"[ \t]*\\addlegendentry\{[^}]*\}\n", s)
    if m.group(2) == "1":
        for e in entries[:3]:                        # n = 3, 4, 5 move to panel (a)
            s = s.replace(e, "", 1)
        # pgfplots gives the k-th legend entry to the k-th plot, so the first three curves (n = 3, 4, 5)
        # must be left out of this panel's legend, or n = 6, 7, 8 would get their markers
        for _ in range(3):
            s = re.sub(r"\\addplot\[(?!forget plot)", r"\\addplot[forget plot, ", s, count=1)
        return s
    # panel (a): attach entries to its first three plots (n = 3, 4, 5 in file order)
    assert not entries, name
    plots = [mm.end() for mm in re.finditer(r"\\addplot\[.*?\] coordinates \{.*?\};\n", s, flags=re.S)]
    assert len(plots) >= 3, name
    for k, n_lab in zip((2, 1, 0), (5, 4, 3)):       # insert back to front so offsets stay valid
        s = s[:plots[k]] + f"    \\addlegendentry{{$n={n_lab}$}}\n".replace("\\\\", "\\") + s[plots[k]:]
    return s


def fig2_range_to_40(s, name):
    """Fig. 2 (and its purity/entropy versions): U/t axis ends at 40. Same 2 % margin on the left as
    before; the data beyond 40 (the grid point 40.2 and up) is cut off by the axis frame."""
    if not re.match(r"fig2(_\w+)?_p\d\.tex$", name):
        return s
    s, a = re.subn(r"\bxmin=[-0-9.]+,", "xmin=0.22,", s, count=1)
    s, b = re.subn(r"\bxmax=[-0-9.]+,", "xmax=40,", s, count=1)
    assert a == b == 1, name
    return s


def add_L_label(s, name):
    """Figs 1, 2, 3, 4: L = 6 in the top panel and L = 8 in the bottom one; Fig. 5: L = 8 (its one panel)."""
    m = re.match(r"fig(\d)\w*_p(\d)\.tex$", name)
    if not m or m.group(1) not in "12345":
        return s
    lab = "L=8" if m.group(1) == "5" else ("L=6" if m.group(2) == "0" else "L=8")
    out, k = re.subn(r"\{\((\w)\)\};\},", r"{(\1)\\hspace{0.45em}$" + lab + r"$};},", s)
    assert k == 1, (name, k)
    return out
