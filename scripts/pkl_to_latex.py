#!/usr/bin/env python3
"""
pkl_to_latex.py
---------------
Turn a cached scan PKL into a standalone LaTeX figure whose DATA LIVES IN THE
.tex FILE.

The numbers are written as PGFPlots ``\\addplot coordinates {...}`` lists, so the
plot is drawn by LaTeX at compile time and every value is visible and editable
in the source.  Nothing is re-solved: this reads the PKLs already produced by
thermal_pipeline.py.  The matplotlib PDFs are untouched.

Supported scans (auto-detected from the result keys, same tests the pipeline
uses in plot_scan_finalized)
---------------------------------------------------------------------------
    V_values     -> V-scan            x = V/t
    U_values     -> U-scan            x = U/t
    t_values     -> t-scan            x = t/U
    N_values     -> K-scan            x = K (electron number)
    C_surface    -> tT surface        x = t/U, y = T          (surf + heatmap)
    C_GME_pure   -> excited singlet   bar chart over n, + C(T) if present

NOTATION.  K is the electron number (the K-fermion system) and n is the
cluster / RDM size that labels the curves; the code's internal ``Ne`` is K.

Curve selection and the exact-line purity gate reproduce
``thermal_pipeline.plot_scan_finalized`` exactly:
  * only n <= Ne is drawn (an n > Ne cluster RDM has zero trace)
  * the exact n=L reference is dropped wherever the largest-cluster RDM is
    mixed (stored purity < 1 - 1e-6), because the pure formula does not apply

Usage
-----
    # one file
    python scripts/pkl_to_latex.py path/to/vscan_....pkl -o latex_figs/

    # every scan in the cache directory
    python scripts/pkl_to_latex.py results/rdm_pkl_results/thermal_runs/ -o latex_figs/

    # restrict to some scan types, and emit a master file that inputs them all
    python scripts/pkl_to_latex.py <dir> -o latex_figs/ --kind vscan uscan --master

Each PKL yields ``<name>.tex`` (a figure environment ready to \\input) and,
with --standalone, a compilable document you can build directly:

    lualatex -output-directory=latex_figs latex_figs/<name>_standalone.tex

Requires in your preamble (the generated snippets assume these):

    \\usepackage{pgfplots}
    \\pgfplotsset{compat=1.18}
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import re
import os
import pickle
import sys
from typing import Any, Optional

import numpy as np

# ── PRB / revtex4-2 [aps,prb,reprint] geometry ───────────────────────────────
# Single column is 246pt = 3.404in; both columns 510pt = 7.057in.  Figures are
# declared in \columnwidth units so LaTeX never rescales them.
COL_W = r"\columnwidth"
FULL_W = r"\textwidth"

# EXACT mirror of thermal_pipeline._COLORS / _MARKERS (the matplotlib "tab10"
# cycle), so curve k in the LaTeX figure is the same colour and marker as curve
# k in the matplotlib PDF of the same scan.  Named xcolor colours are NOT used:
# "red!75!black" is not #ff7f0e, and an approximate palette silently reorders
# the cycle (tab10 has orange at index 1 and red at index 3, which the obvious
# name mapping swaps).  The hex values are emitted as HTML colours instead.
_TEX_COLORS = [
    "1f77b4", "ff7f0e", "2ca02c", "d62728",
    "9467bd", "8c564b", "e377c2", "17becf",
]
# matplotlib marker -> closest PGFPlots mark.  Filled variants are used to match
# matplotlib's default filled markers.
#   o -> *        s -> square*    ^ -> triangle*   D -> diamond*
#   v -> triangle* (rotated 180)  P -> pentagon*   X -> x        * -> asterisk
_TEX_MARKS = [
    "*", "square*", "triangle*", "diamond*",
    "triangle*", "pentagon*", "x", "asterisk",
]
# extra per-mark options (rotation for the downward triangle)
_TEX_MARK_OPTS = [
    "", "", "", "",
    "rotate=180", "", "", "",
]

class NothingToPlot(ValueError):
    """Every curve in this PKL was filtered out (e.g. an n=2-only scan while
    n=2 is hidden).  Expected, not an error."""


# y-axis symbol for every figure: the convex roof evaluated on the spin-reduced
# density matrix.  Defined once so all six builders stay in step.
# \mathrm, not \mathcal: the subscript must be the same upright font as
# "GME" in the text and captions (\mathcal drew it in script letters).
YLABEL = r"$C_{\mathrm{GME}}$"
# temperature scans use the same symbol
YLABEL_T = YLABEL
# T axis: the dimensionless ratio k_B T / t, like U/t and V/t in the other
# figures. The stored T (k_B = 1) is in the units of t, so the plotted value
# is T / t: unchanged for t = 1, multiplied by 10 for t = 0.1.
XLABEL_T = r"$k_B T/t$"

# purity above this counts as a pure RDM (matches plot_scan_finalized)
_PURE_TOL = 1e-6

# Cluster sizes never drawn.  n=2 is excluded by default: for two qubits the
# convex roof IS the Wootters concurrence, so the n=2 curve carries no genuine
# MULTIPARTITE information and only crowds the figure.  Dropping it also drops
# the separate Wootters reference line, which plots the same quantity.
# Override on the command line with --include-n 2 (or --drop-n to hide more).
DROP_N = {2}

# When True the generated file holds ONLY the tikzpicture (no figure float, no
# caption, no label).  Required when the snippet is \input inside a subfigure:
# a figure float nested in another float is illegal in LaTeX and is silently
# dropped, which shows up as an empty panel.
BARE = False

# Legend placement: "auto" puts it INSIDE the axis, in the largest empty
# region (falls back to above the axis if no region is big enough); "top"
# always puts it above the axis; "none" drops it (the colour code is the same
# in every panel, so one legend per figure is enough).
LEGEND_MODE = "auto"

# Panel letter, e.g. "(a)", drawn by the figure itself so it sits next to the
# panel rather than under it.  Default position: left of the "1" tick label.
PANEL_LETTER: Optional[str] = None
PANEL_LETTER_AT: Optional[tuple] = None   # (x, y) in axis description cs

# Points with x above this are dropped and the axis ends there (e.g. 0.5 for
# a T-scan where nothing happens beyond it).
X_CAP: Optional[float] = None

# What the line scans plot. "cgme" is the GME concurrence (default); the others
# read the purity / von Neumann entropy stored for the same spin states, so a
# figure can be redrawn with an identical layout, range and colour code.
#   key : (y-axis label, filename suffix, PKL key, transform)
QUANTITIES = {
    "cgme":           (None,                              "",     "C_GME",   None),
    "purity":         (r"$\mathrm{Tr}\,\rho_s^2$",         "_purity", "purity",  None),
    "linear-entropy": (r"$1-\mathrm{Tr}\,\rho_s^2$",       "_SL",     "purity",  lambda p: 1.0 - p),
    "vn-entropy":     (r"$S(\rho_s)$",                     "_SvN",    "entropy", None),
}
QUANTITY = "cgme"


def _zero_rdm_mask(result: dict, n) -> np.ndarray:
    """True where the n-site spin RDM was a zero matrix (trace ~ 0).

    The PKLs do not store the trace, but a zero matrix leaves its mark in the
    stored purity: a normalised n-spin state has purity >= 1/2^n, while a zero
    matrix (trace ~1e-20, e.g. the charge-density-wave phase of the extended
    model) gives ~1e-40. The spin state does not exist there, so the point is
    left out of the plot instead of being drawn as C_GME = 0.
    """
    P = (result.get("purity") or {}).get(n)
    if P is None:
        return np.zeros(0, bool)
    return np.asarray(P, float) < 1.0 / 2 ** int(n) - 1e-12


def _quantity_data(result: dict) -> dict:
    """{n: array} of the selected QUANTITY, from the PKL's stored arrays."""
    _, _, key, fn = QUANTITIES[QUANTITY]
    if key not in result:
        raise NothingToPlot(f"this PKL stores no '{key}' (needed for --quantity {QUANTITY})")
    return {n: (fn(np.asarray(v, float)) if fn else np.asarray(v, float))
            for n, v in result[key].items()}


def color_defs() -> str:
    """\definecolor lines for the tab10 cycle used by both back ends.

    Emitted into every generated file so a snippet is self-contained: \input
    twice is harmless, \definecolor simply redefines the same value.
    """
    return "".join(
        f"\\definecolor{{gmeC{i}}}{{HTML}}{{{c.upper()}}}\n"
        for i, c in enumerate(_TEX_COLORS))


# ── small helpers ────────────────────────────────────────────────────────────

def _tex_escape(s: str) -> str:
    """Escape the characters that would break a LaTeX caption."""
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def _fmt(v: float, nd: int = 6) -> str:
    """Fixed-point number for a coordinate list; NaN/inf are dropped upstream."""
    return f"{float(v):.{nd}f}".rstrip("0").rstrip(".") or "0"


def _coords(x: np.ndarray, y: np.ndarray, nd: int = 6) -> str:
    """``(x,y) (x,y) ...`` skipping non-finite points, wrapped for readability."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    m = min(x.size, y.size)
    out, line = [], []
    for i in range(m):
        if not (np.isfinite(x[i]) and np.isfinite(y[i])):
            continue                      # a gap in the curve, not a zero
        line.append(f"({_fmt(x[i], nd)},{_fmt(y[i], nd)})")
        if len(line) == 4:                # 4 pairs per source line
            out.append(" ".join(line)); line = []
    if line:
        out.append(" ".join(line))
    return "\n        ".join(out) if out else ""


def state_tag(result: dict, kind: str) -> str:
    """'gs' or 'ex_singlet' for this pkl.

    The eigenstate a figure describes is not visible anywhere in the pkl file
    NAME, so a ground-state and an excited-singlet run of the same system would
    otherwise produce two .tex files that differ only by a hash.  This tag goes
    into the output filename, the \\label and the header comment.
    """
    return ("ex_singlet" if (result.get("excited_singlet")
                             or kind == "excited_singlet") else "gs")


# Overleaf rejects a filename longer than this.
MAX_NAME = 150

# Dropped in this order when a name is too long.  Every one of these is either
# constant across all runs or recorded inside the .tex header, so removing them
# loses nothing that distinguishes two figures.
_DROPPABLE = [
    "_tpall_t2all_tperpall", "_cut10", "_adaptive1", "_adaptive0", "_step5",
    "_clusteravg1", "_knone", "_diag0", "_mode-uniform", "_tol1em09",
    "_iter200", "_iter300", "_iter60", "_iter50", "_iter40",
]


def _fit_name(stem: str, limit: int = MAX_NAME) -> str:
    """Shorten a pkl stem to `limit` chars (minus the .tex suffix).

    Redundant segments go first; only if the name is still too long is it
    truncated, and then a hash is appended so it stays unique.
    """
    budget = limit - len(".tex")
    if len(stem) <= budget:
        return stem
    out = stem
    for seg in _DROPPABLE:
        if len(out) <= budget:
            break
        out = out.replace(seg, "")
    if len(out) > budget:                     # still too long: truncate + hash
        tag = hashlib.sha1(stem.encode()).hexdigest()[:6]
        keep = budget - len(tag) - 1
        out = f"{out[:keep]}_{tag}"
    return out


def _stem_tag(stem: str) -> str:
    """Short stable tag for a PKL stem; keeps \\label unique between runs that
    differ only deep inside a long filename (nmin2 vs nmin3, ...)."""
    return hashlib.sha1(stem.encode()).hexdigest()[:6]


def _slug(*parts: Any) -> str:
    """Deterministic \\label key: fig:vscan-L6-Ne6-U10-a1b2c3."""
    s = "-".join(str(p) for p in parts if p not in (None, ""))
    keep = [c if (c.isalnum() or c == "-") else "-" for c in s]
    out = "".join(keep)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-").lower()


def _g(d: dict, *names, default=None):
    """First present key among *names*."""
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


def _sysinfo(result: dict, meta: dict) -> dict:
    si = result.get("system_info") or {}
    if not isinstance(si, dict):
        si = {}
    merged = dict(meta)
    merged.update({k: v for k, v in si.items() if v is not None})
    return merged


# ── figure assembly ──────────────────────────────────────────────────────────

def _axis_block(body: str, *, xlabel: str, ylabel: str, title: str = "",
                width: str = COL_W, height: str = "0.78",
                ymin: Optional[float] = -0.03, ymax: Optional[float] = 1.05,
                xmin: Optional[float] = None, xmax: Optional[float] = None,
                legend_cols: int = 1, extra: str = "") -> str:
    opts = [
        f"width={width}",
        f"height={height}{width}",
        f"xlabel={{{xlabel}}}",
        f"ylabel={{{ylabel}}}",
        "grid=both",
        "grid style={line width=.1pt, draw=gray!20}",
        # \footnotesize etc. set the SIZE only.  Without an explicit \rmfamily
        # the tick labels render in the sans-serif default and the figure reads
        # as a pasted-in image rather than part of the document; the axis text
        # must be the same serif face as the body.
        "tick label style={font=\\footnotesize\\rmfamily, "
        "/pgf/number format/fixed}",
        "label style={font=\\small\\rmfamily}",
        "scaled ticks=false",
        # legend sits ABOVE the axis: placed inside, it covers the curves that
        # approach C_GME = 1 (the whole-ring and exact references)
        "legend style={font=\\scriptsize\\rmfamily, draw=gray!40, "
        "at={(0.5,1.03)}, anchor=south, /tik"
        "z/every even column/.append style={column sep=6pt}}",
        "legend cell align={left}",
        f"legend columns={legend_cols}",
    ]
    if ymin is not None:
        opts.append(f"ymin={ymin}")
    if ymax is not None:
        opts.append(f"ymax={ymax}")
    if xmin is not None:
        opts.append(f"xmin={xmin:g}")
    if xmax is not None:
        opts.append(f"xmax={xmax:g}")
    if title:
        opts.append(f"title={{{title}}}")
        opts.append("title style={font=\\scriptsize, align=center}")
    if extra:
        opts.append(extra)
    return ("\\begin{tikzpicture}\n"
            "  \\begin{axis}[\n    " + ",\n    ".join(opts) + ",\n  ]\n"
            + body +
            "  \\end{axis}\n"
            "\\end{tikzpicture}")


def _x_range(x: np.ndarray, pad_frac: float = 0.02):
    """xmin/xmax that hug the data, with just enough padding (d = pad_frac of
    the data span) that a marker sitting exactly on the first/last point is
    not clipped by the axis border.  Without this, PGFPlots' own auto-range
    leaves a wide, empty margin on both sides.
    """
    xf = x[np.isfinite(x)]
    if xf.size == 0:
        return None, None
    lo, hi = float(np.min(xf)), float(np.max(xf))
    d = (hi - lo) * pad_frac if hi > lo else max(abs(lo), 1.0) * pad_frac
    return lo - d, hi + d


def _cap_mask(x: np.ndarray) -> np.ndarray:
    """Points kept under X_CAP (all of them when no cap is set)."""
    if X_CAP is None:
        return np.ones(x.shape, bool)
    return ~(x > X_CAP + 1e-12)


# Size of the plot box of a \columnwidth axis and of one legend entry at
# \scriptsize, in pt.  Approximate on purpose: they only need to be good
# enough to tell an empty region from a crowded one.
_AX_W_PT, _AX_H_PT = 200.0, 150.0
_LEG_ROW_PT, _LEG_ENTRY_PT, _LEG_PAD_PT = 9.5, 46.0, 4.0
_LEG_CLEAR_PT = 3.0          # minimum gap between legend box and any curve


def _auto_legend(curves, xmin, xmax, ymin, ymax, n_entries):
    """Centre (axis description cs) and column count for a legend placed in
    the emptiest part of the axis, or None if no position clears the curves.

    Every curve is sampled along its segments (not only at the markers, so a
    line crossing the box counts), then a grid of box positions is scored by
    the distance from the box to the nearest sample; the best one wins.
    """
    W, H = _AX_W_PT, _AX_H_PT
    pts = []
    for x, y in curves:
        m = np.isfinite(x) & np.isfinite(y)
        u = (x[m] - xmin) / (xmax - xmin) * W
        v = (y[m] - ymin) / (ymax - ymin) * H
        if u.size == 1:
            pts.append(np.c_[u, v])
        s = np.linspace(0.0, 1.0, 12)[:, None]
        for k in range(u.size - 1):
            pts.append(np.c_[u[k] + s * (u[k + 1] - u[k]),
                             v[k] + s * (v[k + 1] - v[k])])
    if not pts or n_entries == 0:
        return None
    P = np.vstack(pts)
    best = None
    for cols in sorted({min(c, n_entries) for c in (3, 2, 1)}, reverse=True):
        rows = -(-n_entries // cols)
        bw = cols * _LEG_ENTRY_PT + 2 * _LEG_PAD_PT
        bh = rows * _LEG_ROW_PT + 2 * _LEG_PAD_PT
        if bw > W - 6 or bh > H - 6:
            continue
        for cx in np.linspace(bw / 2 + 3, W - bw / 2 - 3, 40):
            dx = np.maximum(np.abs(P[:, 0] - cx) - bw / 2, 0.0)
            for cy in np.linspace(bh / 2 + 3, H - bh / 2 - 3, 40):
                dy = np.maximum(np.abs(P[:, 1] - cy) - bh / 2, 0.0)
                gap = float(np.min(np.hypot(dx, dy)))
                if best is None or gap > best[0] + 1e-9:
                    best = (gap, cx / W, cy / H, cols)
    if best is None or best[0] < _LEG_CLEAR_PT:
        return None
    return (best[1], best[2]), best[3]


def _finish_axis(body: str, curves, *, xlabel: str, ylabel: str,
                 n_entries: int, ymin: float = -0.03,
                 ymax: float = 1.05) -> str:
    """_axis_block plus the per-figure layout: x range hugging the data,
    legend inside the axis (or dropped), panel letter."""
    xs = (np.concatenate([np.asarray(c[0], float) for c in curves])
          if curves else np.array([]))
    xmin, xmax = _x_range(xs)
    extra, cols = [], 2
    if LEGEND_MODE == "none":
        body = re.sub(r"^[ \t]*\\addlegendentry\{.*\}[ \t]*\n", "", body,
                      flags=re.M)
    elif LEGEND_MODE == "auto" and xmin is not None:
        fit = _auto_legend(curves, xmin, xmax, ymin, ymax, n_entries)
        if fit is None:
            print("  legend: no empty region inside the axis, kept above it",
                  file=sys.stderr)
        else:
            (cx, cy), cols = fit
            extra.append(f"legend style={{at={{({cx:.3f},{cy:.3f})}}, "
                         f"anchor=center, fill=white}}")
    if PANEL_LETTER:
        if PANEL_LETTER_AT:
            lx, ly = PANEL_LETTER_AT
        else:
            # left of the top tick label "1", level with it, with a gap; on an
            # axis that runs well past 1 (von Neumann entropy) "1" sits halfway
            # up, next to the y label, so use the top of the axis instead
            lx = -0.13
            ly = (1.0 - ymin) / (ymax - ymin) if ymax <= 1.2 else 0.97
        # after end axis: drawn outside the clip region, so the letter is
        # not cut off when it sits left of the axis box
        extra.append("after end axis/.append code={\\node[anchor=east, "
                     "font=\\small\\rmfamily] at (axis description cs:"
                     f"{lx:g},{ly:.3f}) {{{PANEL_LETTER}}};}}")
    return _axis_block(body, xlabel=xlabel, ylabel=ylabel, xmin=xmin,
                       xmax=xmax, ymin=ymin, ymax=ymax, legend_cols=cols,
                       extra=",\n    ".join(extra))


def _figure_env(inner: str, caption: str, label: str, *,
                star: bool = False) -> str:
    if BARE:
        # caption/label belong to the surrounding subfigure in this mode
        return inner + "\n"
    env = "figure*" if star else "figure"
    return (f"\\begin{{{env}}}[htbp]\n"
            "  \\centering\n"
            + "\n".join("  " + ln if ln.strip() else ln
                        for ln in inner.splitlines()) + "\n"
            f"  \\caption{{{caption}}}\n"
            f"  \\label{{fig:{label}}}\n"
            f"\\end{{{env}}}\n")


def _plot_line(x, y, idx: int, legend: str, *, style: str = "") -> str:
    i = idx % len(_TEX_COLORS)
    c = _TEX_COLORS[i]
    m = _TEX_MARKS[i]
    mo = _TEX_MARK_OPTS[i]
    co = _coords(x, y)
    if not co:
        return ""
    mark_opts = f", mark options={{{mo}}}" if mo else ""
    # Sizes are chosen for a \columnwidth (3.4in) figure, NOT copied from the
    # matplotlib defaults: those apply to a 9x6in canvas, and reusing them here
    # gives markers that swamp the data.  mark size is a RADIUS in PGFPlots.
    sty = style or (f"color=gmeC{i}, mark={m}, mark size=1.1pt{mark_opts}, "
                    f"line width=0.7pt")
    # A curve reduced to ONE surviving point (the purity-gated exact line on an
    # Ne-scan is the usual case) draws nothing at all when the style says
    # mark=none: there is no segment to stroke. Give it a visible marker.
    if co.count("(") == 1 and "mark=none" in sty:
        sty = sty.replace("mark=none", "mark=*, mark size=1.4pt, only marks")
    # NOTE: \addplot+ (with the plus) takes the next style from PGFPlots'
    # default cycle list and THEN applies these options. That cycle carries
    # dash patterns at later positions, so a curve could come out dashed even
    # though only colour/mark/width are set here (the supervisor spotted this
    # on n=8 of the T-scan). Plain \addplot ignores the cycle entirely.
    return (f"    \\addplot[{sty}] coordinates {{\n        {co}\n    }};\n"
            f"    \\addlegendentry{{{legend}}}\n")


# ── scan-type detection ──────────────────────────────────────────────────────

def detect_kind(result: dict) -> str:
    if "C_surface" in result:
        return "tTsurface"
    if "C_GME_pure" in result:
        return "excited_singlet"
    if "N_values" in result:
        return "nscan"
    if "V_values" in result:
        return "vscan"
    if "t_values" in result and "C_GME" in result:
        return "tscan"
    if "U_values" in result:
        return "uscan"
    # temperature scan: C_GME(T) at fixed U, t (the thermal_v1_* caches)
    if "T_values" in result and "C_GME" in result:
        return "Tscan"
    return "unknown"


# ── the four 1-D scans (V, U, t, Ne) ─────────────────────────────────────────

def build_linescan(result: dict, meta: dict, kind: str, stem: str) -> str:
    si = _sysinfo(result, meta)
    L = int(_g(si, "L", default=0) or 0)
    Ne = int(_g(si, "Ne", "n_electrons", default=L) or L)
    mode = str(_g(si, "mode", default="uniform"))
    V = float(_g(si, "V", default=0.0) or 0.0)
    spin_pres = bool(result.get("spin_preserving", False))

    C = _quantity_data(result)          # C_GME unless --quantity says otherwise
    if kind == "nscan":
        # Ne is the SWEPT axis here, not a fixed number, so a single n <= Ne cut
        # is meaningless: a curve is valid wherever n <= N at that x point. The
        # per-point cut is already encoded as NaN in the stored C_GME (a cluster
        # larger than the electron count has a zero-trace RDM), and _coords()
        # drops non-finite points, so each curve simply starts at its own
        # smallest feasible N. Cap only by the ring size.
        ns = [n for n in sorted(C) if int(n) <= L]
    else:
        ns = [n for n in sorted(C) if int(n) <= Ne]  # n <= Ne cap
    ns = [n for n in ns if int(n) not in DROP_N]     # hidden cluster sizes
    if not ns:
        lim = L if kind == "nscan" else Ne
        raise NothingToPlot(f"only n in {sorted(DROP_N)} available, all hidden")

    # x axis, matching plot_scan_finalized
    if kind == "nscan":
        t = float(_g(si, "t", default=1.0)); U = float(_g(si, "U", default=0.0))
        x = np.asarray(result["N_values"], float)
        # K is the electron number (the K-fermion system), matching the paper's
        # notation; n stays the cluster/RDM size that labels the curves.
        xlabel, fixed, xkind = r"$K$", f"$U={U:g}$, $t={t:g}$", "K"
    elif kind == "vscan":
        t = float(_g(si, "t", default=1.0)); U = float(_g(si, "U", default=0.0))
        x = (np.asarray(result["V_over_t"], float) if "V_over_t" in result
             else np.asarray(result["V_values"], float) / t)
        xlabel, fixed, xkind = r"$V/t$", f"$U={U:g}$, $t={t:g}$", "V"
    elif kind == "tscan":
        U = float(_g(si, "U", default=1.0))
        x = (np.asarray(result["t_over_U"], float) if "t_over_U" in result
             else np.asarray(result["t_values"], float) / U)
        xlabel, fixed, xkind = r"$t/U$", f"$U={U:g}$", "t"
    else:                                            # uscan
        t = float(_g(si, "t", default=1.0))
        x = (np.asarray(result["U_over_t"], float) if "U_over_t" in result
             else np.asarray(result["U_values"], float) / t)
        xlabel, fixed, xkind = r"$U/t$", f"$t={t:g}$", "U"

    # Colour index is taken from the position in the FULL n list, not the
    # filtered one, so hiding a curve does not recolour the ones that remain:
    # n=3 keeps its colour whether or not n=2 is drawn.
    all_ns = [n for n in sorted(C) if int(n) <= (L if kind == "nscan" else Ne)]
    keep = _cap_mask(x)
    x = x[keep]
    body, curves = "", []
    for n in ns:
        k = all_ns.index(n)
        # legend carries the bare symbol: annotations such as "(whole ring,
        # pure)" belong in the caption, not on every entry
        lab = f"$n={n}$"
        y = np.asarray(C[n], float)[keep]
        zm = _zero_rdm_mask(result, n)
        if zm.size == keep.size and zm[keep].any():
            y = np.where(zm[keep], np.nan, y)
            print(f"  n={n}: {int(zm[keep].sum())} point(s) with a zero spin RDM left out",
                  file=sys.stderr)
        body += _plot_line(x, y, k, lab)
        curves.append((x, y))

    # The exact n=L reference line is deliberately NOT drawn: the n=L curve
    # already carries that value, so the extra line duplicates information.

    # n=2 Wootters concurrence (exact convex roof for two qubits).  Skipped when
    # n=2 is hidden: it is the same quantity as the n=2 curve, so drawing it
    # would put n=2 data back on a figure that is meant to exclude it.
    if 2 in C and 2 not in DROP_N and QUANTITY == "cgme":
        w = result.get("C_wootters_n2")
        w = (np.asarray(w, float) if w is not None
             and np.isfinite(np.asarray(w, float)).any()
             else np.asarray(C[2], float))[keep]
        body += _plot_line(
            x, w, 0, r"$n=2$ Wootters $C$",
            style="color=black, dotted, mark=none, line width=1.0pt")
        curves.append((x, w))

    # Every point can be dropped as non-finite - e.g. a U-scan stored with
    # t=0, where U/t is infinite everywhere. _coords() correctly discards those,
    # but the figure would then be written empty, with no curves and an empty
    # legend. Skip it instead.
    if not body.strip():
        raise NothingToPlot(
            f"no finite points to plot (check for t=0 or U=0 in the stored grid)")

    state = ("1st excited singlet" if result.get("excited_singlet")
             else "ground state")
    sector = "all $S_z$" if not spin_pres else "$S_z=0$"
    v_tag = "" if kind == "vscan" else (f", $V={V:g}$" if abs(V) > 1e-12 else "")
    ne_tag = "" if kind == "nscan" else f"$K={Ne}$, "

    ylab = QUANTITIES[QUANTITY][0] or YLABEL
    # the von Neumann entropy can exceed 1 (up to n ln 2): scale the y range to
    # the data, with the same 3% / 5% margins as the [0, 1] quantities
    ylim = {}
    if QUANTITY == "vn-entropy":
        top = max(float(np.nanmax(c[1])) for c in curves if np.isfinite(c[1]).any())
        ylim = dict(ymin=round(-0.03 * top, 3), ymax=round(1.05 * top, 3))
    axis = _finish_axis(body, curves, xlabel=xlabel, ylabel=ylab,
                        n_entries=body.count("\\addlegendentry"), **ylim)

    caption = (
        rf"Genuine multipartite entanglement $C_{{\mathrm{{GME}}}}$ of the "
        rf"$n$-site reduced density matrix versus {xlabel} for the "
        rf"$L={L}$ Hubbard ring ({ne_tag}{fixed}{v_tag}, {mode} mode, "
        rf"{state}, {sector}). "
        rf"Curves are shown for $n=\{{{', '.join(str(int(n)) for n in ns)}\}}$; "
        rf"{len(x)} points."
    )
    label = _slug(kind, state_tag(result, kind), f"L{L}", f"K{Ne}",
                  _stem_tag(stem))
    caption = resolve_caption(kind, result, si, caption)
    return _figure_env(axis, caption, label)


# ── temperature scan: C_GME(T) at fixed U and t ──────────────────────────────

def build_Tscan(result: dict, meta: dict, stem: str) -> str:
    """C_GME versus temperature (the thermal_v1_* caches).

    Mirrors thermal_pipeline._make_figure_T: the exact ground-state reference is
    drawn only when the T -> 0 state really is pure, which is read from the
    stored purity of the largest cluster at the lowest temperature.
    """
    si = _sysinfo(result, meta)
    L = int(_g(si, "L", default=0) or 0)
    Ne = int(_g(si, "Ne", "n_electrons", default=L) or L)
    U = float(_g(si, "U", default=0.0) or 0.0)
    t = float(_g(si, "t", default=1.0) or 1.0)
    mode = str(_g(si, "mode", default="uniform"))
    V = float(_g(si, "V", default=0.0) or 0.0)

    # k_B = 1 in the solver (rdm.thermal_probabilities uses beta = 1/T with T
    # "in the same units as energies"), so the stored T is k_B T in the units
    # of the hopping. Plot the dimensionless k_B T / t.
    T = np.asarray(result["T_values"], float) / (t if t else 1.0)
    keep = _cap_mask(T)
    T = T[keep]
    C = _quantity_data(result)          # C_GME unless --quantity says otherwise
    all_ns = [n for n in sorted(C) if int(n) <= Ne]
    ns = [n for n in all_ns if int(n) not in DROP_N]
    if not ns:
        raise NothingToPlot(f"only n in {sorted(DROP_N)} available, all hidden")

    body, curves = "", []
    for n in ns:
        k = all_ns.index(n)          # stable colour when a curve is hidden
        lab = f"$n={n}$"
        y = np.asarray(C[n], float)[keep]
        body += _plot_line(T, y, k, lab)
        curves.append((T, y))

    # The exact T=0 reference line is deliberately NOT drawn: the n=L curve
    # already starts at that value, so it duplicates information.
    ylab = QUANTITIES[QUANTITY][0] or YLABEL_T
    ylim = {}
    if QUANTITY == "vn-entropy":          # can exceed 1: scale to the data
        top = max(float(np.nanmax(c[1])) for c in curves if np.isfinite(c[1]).any())
        ylim = dict(ymin=round(-0.03 * top, 3), ymax=round(1.05 * top, 3))
    axis = _finish_axis(body, curves, xlabel=XLABEL_T, ylabel=ylab,
                        n_entries=len(ns), **ylim)
    v_tag = f", $V={V:g}$" if abs(V) > 1e-12 else ""
    caption = (
        rf"Temperature dependence of the genuine multipartite entanglement "
        rf"$C_{{\mathrm{{GME}}}}$ of the $n$-site reduced density matrix for "
        rf"the $L={L}$, $K={Ne}$ Hubbard ring at "
        rf"$U={U:g}$, $t={t:g}${v_tag} ({mode} mode). "
        rf"Curves are shown for $n=\{{{', '.join(str(int(n)) for n in ns)}\}}$; "
        rf"{T.size} temperature points."
    )
    caption = resolve_caption("Tscan", result, si, caption)
    return _figure_env(axis, caption,
                       _slug("Tscan", state_tag(result, "Tscan"), f"L{L}",
                             f"K{Ne}", f"U{U:g}", _stem_tag(stem)))


# ── tT surface: heatmap + 3D surface ─────────────────────────────────────────

def build_tTsurface(result: dict, meta: dict, stem: str) -> str:
    si = _sysinfo(result, meta)
    L = int(_g(si, "L", default=0) or 0)
    Ne = int(_g(si, "Ne", default=L) or L)
    U = float(_g(si, "U", default=0.0) or 0.0)

    t = np.asarray(_g(result, "t_over_U", "t_values"), float)
    T = np.asarray(result["T_values"], float)
    surf = result["C_surface"]

    blocks = []
    for n in sorted(surf):
        if int(n) > Ne or int(n) in DROP_N:
            continue
        Z = np.asarray(surf[n], float)          # shape (len(t), len(T))
        if Z.ndim != 2:
            continue
        rows = []
        for i in range(min(Z.shape[0], t.size)):
            for j in range(min(Z.shape[1], T.size)):
                if np.isfinite(Z[i, j]):
                    rows.append(f"        {_fmt(t[i])} {_fmt(T[j])} "
                                f"{_fmt(Z[i, j])}")
            rows.append("")                      # blank line = scanline break
        table = "\n".join(rows)

        inner = (
            "\\begin{tikzpicture}\n"
            "  \\begin{axis}[\n"
            f"    width={COL_W},\n"
            f"    height=0.85{COL_W},\n"
            "    view={0}{90},\n"                # top-down = heatmap
            "    xlabel={$t/U$},\n"
            "    ylabel={$T$},\n"
            "    colorbar,\n"
            "    colorbar style={font=\\scriptsize\\rmfamily, "
            "ylabel={" + YLABEL + "}},\n"
            "    tick label style={font=\\footnotesize\\rmfamily},\n"
            "    label style={font=\\small\\rmfamily},\n"
            "    point meta min=0, point meta max=1,\n"
            "  ]\n"
            "    \\addplot3[surf, shader=interp, mesh/ordering=x varies,\n"
            f"              mesh/num points={{{min(Z.shape[1], T.size)}}}]\n"
            "    table {\n"
            f"{table}\n"
            "    };\n"
            "  \\end{axis}\n"
            "\\end{tikzpicture}"
        )
        caption = (
            rf"$C_{{\mathrm{{GME}}}}$ of the $n={int(n)}$ cluster as a function "
            rf"of hopping $t/U$ and temperature $T$ for the $L={L}$, "
            rf"$K={Ne}$ Hubbard ring at $U={U:g}$. "
            rf"Grid: ${t.size}\times{T.size}$ points."
        )
        caption = resolve_caption("tTsurface", result, si, caption)
        blocks.append(_figure_env(inner, caption,
                                  _slug("tTsurf", state_tag(result,
                                                            "tTsurface"),
                                        f"L{L}", f"n{int(n)}", f"U{U:g}",
                                        _stem_tag(stem))))
    if not blocks:
        raise NothingToPlot(f"no surface left after hiding n {sorted(DROP_N)}")
    return "\n".join(blocks)


# ── excited singlet: bar chart over n, plus C(T) if the PKL has it ───────────

def build_excited_singlet(result: dict, meta: dict, stem: str) -> str:
    si = _sysinfo(result, meta)
    L = int(_g(si, "L", default=0) or 0)
    Ne = int(_g(si, "Ne", default=L) or L)
    U = float(_g(si, "U", default=0.0) or 0.0)
    t = float(_g(si, "t", default=1.0) or 1.0)

    Cp = result["C_GME_pure"]
    ns = [n for n in sorted(Cp) if int(n) <= Ne and int(n) not in DROP_N]
    idx = result.get("singlet_idx")
    en = result.get("singlet_energy")

    bars = " ".join(f"({int(n)},{_fmt(float(np.ravel(Cp[n])[0]))})" for n in ns
                    if np.isfinite(float(np.ravel(Cp[n])[0])))
    body = (f"    \\addplot[ybar, fill=blue!55!black, draw=black, "
            f"bar width=10pt] coordinates {{{bars}}};\n"
            f"    \\addlegendentry{{1st excited singlet}}\n")

    axis = _axis_block(
        body, xlabel=r"cluster size $n$",
        ylabel=YLABEL,
        extra=(f"xtick={{{','.join(str(int(n)) for n in ns)}}}, "
               "ybar, enlarge x limits=0.15"),
    )
    e_tag = (rf", $E={float(en):.4f}$" if en is not None
             and np.isfinite(float(en)) else "")
    caption = (
        rf"$C_{{\mathrm{{GME}}}}$ of the first excited singlet "
        rf"(eigenstate index {idx}{e_tag}) as a function of cluster size $n$ "
        rf"for the $L={L}$, $K={Ne}$ Hubbard ring at "
        rf"$U={U:g}$, $t={t:g}$."
    )
    caption = resolve_caption("excited_singlet", result, si, caption)
    out = [_figure_env(axis, caption,
                       _slug("singlet", f"L{L}", f"Ne{Ne}", f"U{U:g}",
                             _stem_tag(stem)))]

    # optional C(T) panel stored under T_results
    Tr = result.get("T_results")
    if isinstance(Tr, dict) and "T_values" in Tr and "C_GME" in Tr:
        Tv = np.asarray(Tr["T_values"], float) / t     # k_B T / t, as in build_Tscan
        CT = Tr["C_GME"]
        b2 = ""
        ct_all = [x for x in sorted(CT) if int(x) <= Ne]
        for n in [x for x in ct_all if int(x) not in DROP_N]:
            b2 += _plot_line(Tv, np.asarray(CT[n], float),
                             ct_all.index(n), f"$n={int(n)}$")
        if b2:
            ax2 = _axis_block(b2, xlabel=XLABEL_T, ylabel=YLABEL_T,
                              legend_cols=2)
            cap2 = (rf"Temperature dependence of $C_{{\mathrm{{GME}}}}$ for the "
                    rf"$L={L}$, $K={Ne}$ Hubbard ring at "
                    rf"$U={U:g}$, $t={t:g}$ (thermal state built on the "
                    rf"excited-singlet run).")
            out.append(_figure_env(ax2, cap2,
                                   _slug("singletT", f"L{L}", f"Ne{Ne}",
                                         _stem_tag(stem))))
    return "\n".join(out)


# ── spin table -> booktabs (excited-singlet PKLs carry one) ──────────────────

def build_spin_table(result: dict, meta: dict, stem: str) -> Optional[str]:
    rows = result.get("spin_table")
    if rows is None or len(np.ravel(rows)) == 0:
        return None
    si = _sysinfo(result, meta)
    L = int(_g(si, "L", default=0) or 0)
    Ne = int(_g(si, "Ne", default=L) or L)

    lines = []
    for r in np.ravel(rows):
        if not isinstance(r, dict):
            continue
        lines.append(
            f"    {int(r.get('idx', -1))} & {float(r.get('energy', np.nan)):.6f} & "
            f"{float(r.get('S2', np.nan)):.4f} & {float(r.get('S', np.nan)):.2f} & "
            f"{float(r.get('multiplicity', np.nan)):.0f} & "
            f"{_tex_escape(str(r.get('spin_label', '')))} \\\\")
    if not lines:
        return None
    return (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        rf"  \caption{{Spin classification of the lowest eigenstates of the "
        rf"$L={L}$, $K={Ne}$ Hubbard ring.}}" "\n"
        f"  \\label{{tab:{_slug('spin', f'L{L}', f'Ne{Ne}', _stem_tag(stem))}}}\n"
        "  \\begin{tabular}{rrrrrl}\n"
        "    \\hline\\hline\n"
        "    Index & Energy & $\\langle S^2\\rangle$ & $S$ & $2S+1$ & Label \\\\\n"
        "    \\hline\n"
        + "\n".join(lines) + "\n"
        "    \\hline\\hline\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )


# ── standalone wrapper ───────────────────────────────────────────────────────

_PREAMBLE = r"""\documentclass[aps,prb,reprint]{revtex4-2}
\usepackage{amsmath,amssymb}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
% Make every plot inherit the document's serif face.  Without this the axis
% text falls back to the sans-serif default and the figure looks pasted in
% rather than typeset with the paper.
\pgfplotsset{
  every axis/.append style={font=\rmfamily},
  tick label style={font=\footnotesize\rmfamily},
  label style={font=\small\rmfamily},
  legend style={font=\scriptsize\rmfamily},
}
"""


def wrap_standalone(body: str) -> str:
    """A compilable document around one figure.

    The colour definitions must be emitted here too: the body references
    gmeC0..gmeC7, and without \definecolor in the preamble LaTeX stops with
    "Undefined color". They are idempotent, so the copy already inside the
    body is harmless.
    """
    return (_PREAMBLE + color_defs() + "\\begin{document}\n"
            + body + "\n\\end{document}\n")


# ── user-supplied captions ───────────────────────────────────────────────────

CAPTIONS: dict = {}


def caption_key(kind: str, state: str, L, Ne, U, t, V) -> str:
    """Identity a caption is looked up by.

    Deliberately built from PHYSICS only (scan type, state, L, Ne, U, t, V) and
    never from the file name, so a re-run of the same system keeps its caption.
    """
    def _n(v):
        if v is None:
            return "x"
        f = float(v)
        return str(int(f)) if f == int(f) else f"{f:g}"
    return f"{kind}|{state}|L{_n(L)}|Ne{_n(Ne)}|U{_n(U)}|t{_n(t)}|V{_n(V)}"


def load_captions(path: str) -> dict:
    """Read a caption file.

    Format is one entry per figure; blank lines and #-comments ignored:

        [tscan|gs|L6|Ne6|U10|tx|V0]
        Spin entanglement in the ground state ... $K=L$ and $V=0$.

    The body may span several lines and may contain any LaTeX.  A key may also
    be a PREFIX (e.g. "[tscan|gs]") which then applies to every figure whose
    key starts with it, so one entry can cover a whole family.
    """
    caps, key, buf = {}, None, []
    if not path:
        return caps
    with open(path) as fh:
        for raw in fh:
            ln = raw.rstrip("\n")
            if ln.strip().startswith("#"):
                continue
            m = re.match(r"^\s*\[(.+?)\]\s*$", ln)
            if m:
                if key and buf:
                    if key in caps:
                        # a repeated key silently discards the earlier text, so
                        # say so: it is nearly always a copy-paste slip
                        print(f"  WARNING duplicate caption key [{key}] "
                              f"- the later one wins", file=sys.stderr)
                    caps[key] = " ".join(b.strip() for b in buf).strip()
                key, buf = m.group(1).strip(), []
            elif key is not None:
                buf.append(ln)
    if key and buf:
        caps[key] = " ".join(b.strip() for b in buf).strip()
    return caps


# axis a caption of this scan type should be talking about
_KIND_AXIS = {
    "vscan": "$V/t$", "uscan": "$U/t$", "tscan": "$t/U$",
    "Tscan": "$T$", "nscan": "$K$",
}


def _check_caption_axis(kind: str, text: str, key: str) -> None:
    """Warn when a caption names a different axis than the figure plots.

    A caption copied between scan types is easy to miss by eye and impossible
    to see in the compiled PDF without reading it, so flag it at build time.
    """
    want = _KIND_AXIS.get(kind)
    if not want or want in text:
        return
    others = [a for k, a in _KIND_AXIS.items() if k != kind and a in text]
    if others:
        print(f"  WARNING caption [{key}] mentions {others[0]} but this is a "
              f"{kind} ({want} axis)", file=sys.stderr)


def resolve_caption(kind, result, si, auto, *, state=None):
    """User caption if one is supplied for this figure, else the auto text."""
    st = state or ("excited" if (result.get("excited_singlet")
                                 or kind == "excited_singlet") else "gs")
    key = caption_key(kind, st,
                      _g(si, "L"), _g(si, "Ne", "n_electrons"),
                      _g(si, "U"), _g(si, "t"), _g(si, "V", default=0) or 0)
    user = lookup_caption(key)
    if user:
        _check_caption_axis(kind, user, key)
        return user
    return auto


def lookup_caption(key: str):
    """Exact match first, then the longest matching prefix."""
    if key in CAPTIONS:
        return CAPTIONS[key]
    best = None
    for k, v in CAPTIONS.items():
        if key.startswith(k) and (best is None or len(k) > len(best[0])):
            best = (k, v)
    return best[1] if best else None


# ── latest-run selection ─────────────────────────────────────────────────────

# Knobs that record HOW hard the solver worked, not WHAT was computed.  Two
# files differing only in these describe the same physical scan, so only the
# newest is kept under --latest.
_EFFORT_PATTERNS = [
    r"_restarts\d+", r"_iter\d+", r"_kmax\d+", r"_tol\w+?(?=_)",
    r"_solver-[\w-]+", r"_cut\d+", r"_adaptive\d+", r"_step\d+",
    r"_clusteravg\d+", r"_k(?:\d+|none)(?=_|$)", r"_diag\d+", r"_sz\d+",
    r"_full(?=_|$)", r"_es(?=_|$)", r"_\d{8}_\d{6}",
]


def config_key(basename: str, kind: str) -> str:
    """Identity of the PHYSICAL scan: type + system + swept grid.

    Everything describing solver effort is stripped, so re-runs of one scan at
    different restart budgets collapse to a single key.  The trailing content
    digest is KEPT: it encodes the swept grid, and two different grids are two
    different figures.
    """
    s = basename[:-4] if basename.endswith(".pkl") else basename
    for pat in _EFFORT_PATTERNS:
        s = re.sub(pat, "", s)
    return f"{kind}|{s}"


def select_latest(files: list) -> list:
    """Keep only the newest PKL per physical config.

    NOTE ON "LATEST".  C_GME is a convex roof, so every solve returns an UPPER
    bound and a lower value is a better one.  Newest is therefore not always
    best: a recent low-restart run can sit above an older high-restart one.
    This picks by modification time because that is what "latest" means and it
    is reproducible; use --best to pick the tightest bound instead.
    """
    groups = {}
    for f in files:
        try:
            with open(f, "rb") as fh:
                r = pickle.load(fh).get("result", {})
        except Exception:
            continue
        k = detect_kind(r)
        if k == "unknown":
            continue
        groups.setdefault(config_key(os.path.basename(f), k), []).append(f)
    return [max(v, key=os.path.getmtime) for v in groups.values()]


def select_best(files: list) -> list:
    """Keep, per physical config, the run whose C_GME curves are lowest.

    The roof is a minimum, so among runs of the same scan the one with the
    smaller mean C_GME has found the better optimum.  Ties fall back to mtime.
    """
    groups = {}
    for f in files:
        try:
            with open(f, "rb") as fh:
                r = pickle.load(fh).get("result", {})
        except Exception:
            continue
        k = detect_kind(r)
        if k == "unknown":
            continue
        C = r.get("C_GME") or r.get("C_GME_pure") or r.get("C_surface") or {}
        vals = []
        if isinstance(C, dict):
            for arr in C.values():
                a = np.asarray(arr, float).ravel()
                if a.size and np.isfinite(a).any():
                    vals.append(np.nanmean(a))
        score = float(np.mean(vals)) if vals else np.inf
        groups.setdefault(config_key(os.path.basename(f), k),
                          []).append((score, os.path.getmtime(f), f))
    out = []
    for v in groups.values():
        best = min(v, key=lambda z: (z[0], -z[1]))   # lowest C, newest on tie
        out.append(best[2])
    return out


# ── driver ───────────────────────────────────────────────────────────────────

def convert(path: str, outdir: str, *, standalone: bool = False,
            verbose: bool = True) -> Optional[str]:
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    if not isinstance(d, dict) or "result" not in d:
        if verbose:
            print(f"  SKIP (not a scan pkl): {os.path.basename(path)}")
        return None

    result = d["result"]
    meta = d.get("meta", {}) or {}
    kind = detect_kind(result)
    stem = os.path.splitext(os.path.basename(path))[0]

    if kind == "unknown":
        if verbose:
            print(f"  SKIP (unrecognised): {os.path.basename(path)[:60]}")
        return None

    if kind in ("vscan", "uscan", "tscan", "nscan"):
        body = build_linescan(result, meta, kind, stem)
    elif kind == "Tscan":
        body = build_Tscan(result, meta, stem)
    elif kind == "tTsurface":
        body = build_tTsurface(result, meta, stem)
    else:
        body = build_excited_singlet(result, meta, stem)
        tab = build_spin_table(result, meta, stem)
        if tab:
            body += "\n" + tab

    os.makedirs(outdir, exist_ok=True)
    # The .tex keeps the pkl's name, so a figure can be traced back to the run
    # that produced it by reading the filename.  Overleaf refuses names longer
    # than 150 characters, so when the pkl name exceeds that, segments that are
    # constant across every run (and therefore identify nothing) are dropped in
    # a fixed order until it fits.  L, K, U, t, restarts, the swept range and
    # the digest are never touched.
    # a purity / entropy figure gets a suffix so it never overwrites the
    # C_GME figure made from the same PKL
    sfx = QUANTITIES[QUANTITY][1]
    out = os.path.join(outdir, _fit_name(stem, MAX_NAME - len(sfx)) + sfx + ".tex")
    header = (f"% generated by scripts/pkl_to_latex.py\n"
              f"% source PKL : {os.path.basename(path)}\n"
              f"% scan type  : {kind}\n"
              f"% state      : {state_tag(result, kind)}\n"
              f"% quantity   : {QUANTITY}\n"
              + (f"% x-max cap  : {X_CAP:g}\n" if X_CAP is not None else "")
              +
              f"% requires   : \\usepackage{{pgfplots}} "
              f"\\pgfplotsset{{compat=1.18}}\n"
              f"% colours match the matplotlib PDF (tab10 cycle)\n"
              + color_defs() + "\n")
    with open(out, "w") as fh:
        fh.write(header + body)

    if standalone:
        sa = os.path.join(outdir, _fit_name(stem, MAX_NAME - 11)
                          + "_standalone.tex")
        with open(sa, "w") as fh:
            fh.write(wrap_standalone(body))

    if verbose:
        print(f"  {kind:16s} -> {os.path.basename(out)}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert scan PKLs into LaTeX (PGFPlots) figures whose "
                    "data lives in the .tex source.")
    ap.add_argument("input", help="a .pkl file, or a directory of them")
    ap.add_argument("-o", "--outdir", default="latex_figs",
                    help="output directory (default: latex_figs)")
    ap.add_argument("--kind", nargs="+", default=None,
                    choices=["vscan", "uscan", "tscan", "Tscan", "nscan",
                             "tTsurface", "excited_singlet"],
                    help="only convert these scan types")
    ap.add_argument("--standalone", action="store_true",
                    help="also write a compilable <name>_standalone.tex")
    ap.add_argument("--master", action="store_true",
                    help="write all_figures.tex that \\inputs every snippet")
    ap.add_argument("--bare", action="store_true",
                    help="emit only the tikzpicture, with no figure float, "
                         "caption or label - use when the snippet is \\input "
                         "inside a subfigure (a nested float is dropped by "
                         "LaTeX and the panel comes out empty)")
    ap.add_argument("--legend", choices=["auto", "top", "none"],
                    default="auto",
                    help="auto: inside the axis, in the largest empty region "
                         "(default); top: above the axis; none: no legend "
                         "(use on all panels but one - the colours are the "
                         "same in every panel)")
    ap.add_argument("--panel-letter", metavar="TEXT",
                    help='letter drawn next to the panel, e.g. "(a)"; only '
                         "with a single input .pkl")
    ap.add_argument("--panel-letter-at", metavar="X,Y",
                    help="position of the letter in axis description "
                         "coordinates (default: left of the '1' tick label)")
    ap.add_argument("--x-max", type=float, metavar="X",
                    help="drop points with x > X and end the axis there")
    ap.add_argument("--quantity", choices=sorted(QUANTITIES), default="cgme",
                    help="what the line scans plot: cgme (default), purity, "
                         "linear-entropy (1 - purity) or vn-entropy")
    ap.add_argument("--captions", metavar="FILE",
                    help="file of user-written captions; see --dump-caption-keys")
    ap.add_argument("--dump-caption-keys", action="store_true",
                    help="print a caption-file skeleton for the selected pkls "
                         "and exit, so you can fill in the text")
    ap.add_argument("--drop-n", nargs="*", type=int, default=None,
                    metavar="N",
                    help="cluster sizes to hide (default: 2, since the n=2 "
                         "roof is just the Wootters concurrence). Pass with no "
                         "values to hide nothing.")
    ap.add_argument("--include-n", nargs="+", type=int, default=None,
                    metavar="N", help="un-hide these n (e.g. --include-n 2)")
    ap.add_argument("--latest", action="store_true",
                    help="keep only the NEWEST pkl per physical scan config "
                         "(collapses re-runs that differ only in solver effort)")
    ap.add_argument("--best", action="store_true",
                    help="keep the run with the LOWEST C_GME per config; the "
                         "roof is an upper bound, so lower is tighter")
    ap.add_argument("--list-only", action="store_true",
                    help="print the selected pkl paths and exit, write nothing")
    ap.add_argument("--limit", type=int, default=0,
                    help="convert at most N files (0 = no limit)")
    args = ap.parse_args(argv)

    global BARE, LEGEND_MODE, PANEL_LETTER, PANEL_LETTER_AT, X_CAP, QUANTITY
    BARE = bool(args.bare)
    QUANTITY = args.quantity
    LEGEND_MODE = args.legend
    X_CAP = args.x_max
    if args.panel_letter:
        if os.path.isdir(args.input):
            print("--panel-letter needs a single .pkl, not a directory",
                  file=sys.stderr)
            return 1
        PANEL_LETTER = args.panel_letter
    if args.panel_letter_at:
        PANEL_LETTER_AT = tuple(float(v) for v in
                                args.panel_letter_at.split(","))

    # resolve which cluster sizes are hidden, before anything is built
    global DROP_N
    if args.drop_n is not None:
        DROP_N = set(args.drop_n)
    if args.include_n:
        DROP_N = DROP_N - set(args.include_n)
    if DROP_N:
        print(f"hiding n = {sorted(DROP_N)}", file=sys.stderr)

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.pkl")),
                       key=os.path.getmtime, reverse=True)
    else:
        files = [args.input]
    if not files:
        print(f"no .pkl found in {args.input}", file=sys.stderr)
        return 1

    # --kind must be applied BEFORE --latest/--best, otherwise selection runs
    # over every scan type and the filter never reaches --list-only at all.
    if args.kind:
        kept = []
        for f in files:
            try:
                with open(f, "rb") as fh:
                    r = pickle.load(fh).get("result", {})
            except Exception:
                continue
            if detect_kind(r) in args.kind:
                kept.append(f)
        files = kept
        if not files:
            print(f"no pkl of kind {args.kind} in {args.input}", file=sys.stderr)
            return 1

    if args.latest or args.best:
        n_before = len(files)
        files = (select_best(files) if args.best else select_latest(files))
        files = sorted(files, key=os.path.getmtime, reverse=True)
        how = "lowest C_GME" if args.best else "newest mtime"
        print(f"{n_before} pkl -> {len(files)} distinct configs ({how})",
              file=sys.stderr)          # stderr: keeps --list-only pipeable

    if args.captions:
        global CAPTIONS
        CAPTIONS = load_captions(args.captions)
        print(f"loaded {len(CAPTIONS)} caption(s) from {args.captions}",
              file=sys.stderr)

    if args.dump_caption_keys:
        print("# caption file for pkl_to_latex.py --captions")
        print("# one [key] line, then the caption text (may span lines).")
        print("# A shorter key acts as a prefix default, e.g. [tscan|gs]")
        print("# tx / Nex / Ux mean that value is absent from the pkl.\n")
        seen = set()
        for f in files:
            try:
                d = pickle.load(open(f, "rb"))
            except Exception:
                continue
            r = d.get("result", {}); m = d.get("meta", {})
            k = detect_kind(r)
            if k == "unknown":
                continue
            si = _sysinfo(r, m)
            st = ("excited" if (r.get("excited_singlet")
                                or k == "excited_singlet") else "gs")
            key = caption_key(k, st, _g(si, "L"),
                              _g(si, "Ne", "n_electrons"), _g(si, "U"),
                              _g(si, "t"), _g(si, "V", default=0) or 0)
            if key in seen:
                continue
            seen.add(key)
            print(f"[{key}]")
            print("TODO caption text\n")
        return 0

    if args.list_only:
        for f in files:
            print(f)
        return 0

    print(f"scanning {len(files)} PKL file(s) -> {args.outdir}/")
    written, skipped, failed = [], 0, 0
    for f in files:
        if args.limit and len(written) >= args.limit:
            break
        try:
            # --kind was already applied above, before latest/best selection
            out = convert(f, args.outdir, standalone=args.standalone)
            if out:
                written.append(out)
            else:
                skipped += 1
        except NothingToPlot as e:      # every curve filtered out: not an error
            skipped += 1
            print(f"  SKIP {os.path.basename(f)[:46]}: {e}")
        except Exception as e:                      # keep going on a bad file
            failed += 1
            print(f"  FAIL {os.path.basename(f)[:50]}: "
                  f"{type(e).__name__}: {e}")

    if args.master and written:
        mp = os.path.join(args.outdir, "all_figures.tex")
        with open(mp, "w") as fh:
            fh.write("% generated by scripts/pkl_to_latex.py\n"
                     "% \\input this from your document, or copy the lines you want.\n"
                     "% needs \\usepackage{pgfplots} \\pgfplotsset{compat=1.18}\n\n"
                     + color_defs() + "\n")
            for w in written:
                fh.write(f"\\input{{{os.path.splitext(os.path.basename(w))[0]}}}\n")
        print(f"master file: {mp}")

    print(f"\nwritten {len(written)}   skipped {skipped}   failed {failed}")
    return 0 if written or not files else 1


if __name__ == "__main__":
    raise SystemExit(main())
