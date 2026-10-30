#!/usr/bin/env python3
"""
verify_latex_vs_pkl.py
----------------------
Cross-check: parse the numbers back OUT of every generated .tex and compare
them against the source PKL.  Catches a wrong array, a shifted x axis, a
mis-applied filter, or a formatting bug that silently corrupts a value.

The .tex header records which PKL produced it, so the pairing is exact.

    python scripts/verify_latex_vs_pkl.py latex_figs/ --pkl-dir <thermal_runs>

Exit status is 1 if any figure disagrees with its PKL.
"""
from __future__ import annotations
import argparse, glob, os, pickle, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "p2l", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pkl_to_latex.py"))
p2l = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(p2l)

# the .tex writes 6 decimals then strips trailing zeros
TOL = 5e-7


def parse_tex(path):
    """-> (source_pkl, kind, [(legend, np.array([[x,y],...])), ...])"""
    txt = open(path).read()
    src = re.search(r"% source PKL : (.+)", txt)
    kind = re.search(r"% scan type  : (\w+)", txt)
    cap = re.search(r"% x-max cap  : (\S+)", txt)
    parse_tex.cap = float(cap.group(1)) if cap else None
    q = re.search(r"% quantity   : (\S+)", txt)
    parse_tex.quantity = q.group(1) if q else "cgme"
    curves = []
    # each addplot ... coordinates { ... };  followed by its legend entry
    for m in re.finditer(
            r"\\addplot\+?\[[^\]]*\]\s*coordinates\s*\{(.*?)\};"
            r"(?:\s*\\addlegendentry\{(.*?)\})?", txt, re.S):
        pts = re.findall(r"\(([-\d.eE+]+),([-\d.eE+]+)\)", m.group(1))
        if pts:
            curves.append(((m.group(2) or "").strip(),
                           np.array([[float(a), float(b)] for a, b in pts])))
    # surfaces are emitted as "\\addplot3[...] table { x y z\\n ... };"
    for m in re.finditer(r"\\addplot3\[[^\]]*\]\s*table\s*\{(.*?)\};", txt, re.S):
        rows = []
        for ln in m.group(1).splitlines():
            f3 = ln.split()
            if len(f3) == 3:
                try:
                    rows.append([float(v) for v in f3])
                except ValueError:
                    pass
        if rows:
            curves.append(("SURFACE", np.array(rows)))
    return (src.group(1).strip() if src else None,
            kind.group(1) if kind else None, curves)


def expected_curves(result, meta, kind):
    """Rebuild what SHOULD be plotted, straight from the PKL."""
    si = p2l._sysinfo(result, meta)
    L = int(p2l._g(si, "L", default=0) or 0)
    Ne = int(p2l._g(si, "Ne", "n_electrons", default=L) or L)
    out = {}

    if kind in ("vscan", "uscan", "tscan", "nscan"):
        # same source array the exporter used for this --quantity
        _, _, qkey, qfn = p2l.QUANTITIES[getattr(parse_tex, "quantity", "cgme")]
        C = {n: (qfn(np.asarray(v, float)) if qfn else v) for n, v in result[qkey].items()}
        if kind == "nscan":
            x = np.asarray(result["N_values"], float)
            ns = [n for n in sorted(C) if int(n) <= L]
        else:
            if kind == "vscan":
                t = float(p2l._g(si, "t", default=1.0))
                x = (np.asarray(result["V_over_t"], float) if "V_over_t" in result
                     else np.asarray(result["V_values"], float) / t)
            elif kind == "tscan":
                U = float(p2l._g(si, "U", default=1.0))
                x = (np.asarray(result["t_over_U"], float) if "t_over_U" in result
                     else np.asarray(result["t_values"], float) / U)
            else:
                t = float(p2l._g(si, "t", default=1.0))
                x = (np.asarray(result["U_over_t"], float) if "U_over_t" in result
                     else np.asarray(result["U_values"], float) / t)
            ns = [n for n in sorted(C) if int(n) <= Ne]
        ns = [n for n in ns if int(n) not in p2l.DROP_N]
        for n in ns:
            y = np.asarray(C[n], float)
            zm = p2l._zero_rdm_mask(result, n)          # points the exporter leaves out
            if zm.size == y.size:
                y = np.where(zm, np.nan, y)
            out[f"$n={n}$"] = (x, y)
        ex = result.get("C_exact_nL")
        if ex is not None and ns:
            ex = np.asarray(ex, float).copy()
            pur = (result.get("purity", {}) or {}).get(max(ns))
            if pur is not None:
                ex = np.where(np.asarray(pur, float) > 1 - p2l._PURE_TOL, ex, np.nan)
            if np.isfinite(ex).any():
                out["EXACT"] = (x, ex)

    elif kind == "Tscan":
        # the exporter plots the dimensionless k_B T / t, so rescale here too
        _, _, qkey, qfn = p2l.QUANTITIES[getattr(parse_tex, "quantity", "cgme")]
        C = {n: (qfn(np.asarray(x, float)) if qfn else x) for n, x in result[qkey].items()}
        _t = float(p2l._g(si, "t", default=1.0) or 1.0)
        T = np.asarray(result["T_values"], float) / _t
        ns = [n for n in sorted(C) if int(n) <= Ne and int(n) not in p2l.DROP_N]
        for n in ns:
            out[f"$n={n}$"] = (T, np.asarray(C[n], float))

    elif kind == "tTsurface":
        t = np.asarray(p2l._g(result, "t_over_U", "t_values"), float)
        T = np.asarray(result["T_values"], float)
        surf = result["C_surface"]
        for n in sorted(surf):
            if int(n) > Ne or int(n) in p2l.DROP_N:
                continue
            Z = np.asarray(surf[n], float)
            if Z.ndim != 2:
                continue
            trip = []
            for i in range(min(Z.shape[0], t.size)):
                for j in range(min(Z.shape[1], T.size)):
                    if np.isfinite(Z[i, j]):
                        trip.append([t[i], T[j], Z[i, j]])
            if trip:
                out[f"SURF{int(n)}"] = np.array(trip)

    elif kind == "excited_singlet":
        Cp = result["C_GME_pure"]
        ns = [n for n in sorted(Cp) if int(n) <= Ne and int(n) not in p2l.DROP_N]
        xs = np.array([float(n) for n in ns])
        ys = np.array([float(np.ravel(Cp[n])[0]) for n in ns])
        out["BARS"] = (xs, ys)
        # the same file also carries a C(T) panel, whose curves are labelled
        # "$n=..$" exactly like a line scan; register them so they are checked
        # against T_results rather than being mis-matched onto the bar data
        Tr = result.get("T_results")
        if isinstance(Tr, dict) and "T_values" in Tr and "C_GME" in Tr:
            _t = float(p2l._g(si, "t", default=1.0) or 1.0)
            Tv = np.asarray(Tr["T_values"], float) / _t
            CT = Tr["C_GME"]
            for n in sorted(CT):
                if int(n) <= Ne and int(n) not in p2l.DROP_N:
                    out[f"$n={int(n)}$"] = (Tv, np.asarray(CT[n], float))
    return out


def cmp_curve(got, exp_x, exp_y, cap=None):
    """Compare one parsed curve against expected, ignoring dropped NaNs and
    points beyond an --x-max cap."""
    m = np.isfinite(exp_x) & np.isfinite(exp_y)
    if cap is not None:
        m &= ~(exp_x > cap + 1e-12)
    ex, ey = exp_x[m], exp_y[m]
    if got.shape[0] != ex.size:
        return False, f"point count {got.shape[0]} != {ex.size}"
    dx = np.max(np.abs(got[:, 0] - ex)) if ex.size else 0.0
    dy = np.max(np.abs(got[:, 1] - ey)) if ey.size else 0.0
    if dx > TOL or dy > TOL:
        return False, f"max|dx|={dx:.2e} max|dy|={dy:.2e}"
    return True, f"{got.shape[0]} pts  max|dx|={dx:.1e} max|dy|={dy:.1e}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("texdir")
    ap.add_argument("--pkl-dir", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    texs = sorted(f for f in glob.glob(os.path.join(a.texdir, "*.tex"))
                  if not f.endswith("all_figures.tex")
                  and not f.endswith("_standalone.tex"))
    print(f"checking {len(texs)} .tex against PKLs in {a.pkl_dir}\n")

    n_ok = n_bad = n_skip = 0
    bad_files = []
    for tf in texs:
        src, kind, curves = parse_tex(tf)
        if not src:
            n_skip += 1; continue
        pkl = os.path.join(a.pkl_dir, src)
        if not os.path.exists(pkl):
            print(f"  MISSING PKL for {os.path.basename(tf)[:44]}"); n_skip += 1; continue
        try:
            d = pickle.load(open(pkl, "rb"))
        except Exception as e:
            print(f"  UNREADABLE {src[:40]}: {e}"); n_skip += 1; continue
        exp = expected_curves(d.get("result", {}), d.get("meta", {}), kind)
        if not exp:
            n_skip += 1; continue

        # constant referenced by flat 2-point reference lines
        _res = d.get("result", {})
        result_exact_const = None
        for k in ("C_exact_singlet", "C_exact_gs"):
            v = _res.get(k)
            if v is not None:
                try:
                    result_exact_const = float(v)
                except (TypeError, ValueError):
                    pass
                break

        # match parsed curves to expected by legend; unlabeled -> EXACT/BARS
        problems = []
        checked = 0
        # a panel written with --legend none has no labels: its curves are
        # in n order, so pair them with the expected n keys in that order
        n_keys = iter(sorted((k for k in exp if k.startswith("$n=")),
                             key=lambda k: int(re.sub(r"\D", "", k))))
        for legend, pts in curves:
            key = None
            if legend == "" and pts.shape[0] > 2:
                legend = next(n_keys, "")
            if legend in exp:
                key = legend
            elif "Exact" in legend and "EXACT" in exp:
                key = "EXACT"
            elif legend == "1st excited singlet" and "BARS" in exp:
                key = "BARS"
            # Reference lines drawn as a flat 2-point segment (the singlet's
            # exact value, the T-scan's T=0 line) carry no per-point PKL array
            # to compare against; check the CONSTANT instead.
            elif "exact" in legend.lower() and pts.shape[0] == 2:
                c = result_exact_const
                if c is not None and np.isfinite(c):
                    ok = np.max(np.abs(pts[:, 1] - c)) <= TOL
                    checked += 1
                    if not ok:
                        problems.append(
                            f"exact const: tex={pts[0,1]:.9g} pkl={c:.9g}")
                    elif a.verbose:
                        print(f"    {os.path.basename(tf)[:34]:36s} "
                              f"{'EXACTCONST':12s} value={c:.9g} OK")
                continue
            if legend == "SURFACE":
                surf_keys = [k for k in exp if k.startswith("SURF")]
                if not surf_keys:
                    continue
                # match by row count; a file may hold several n
                cand = [k for k in surf_keys
                        if exp[k].shape[0] == pts.shape[0]]
                k2 = cand[0] if cand else surf_keys[0]
                e = exp[k2]
                checked += 1
                if e.shape != pts.shape:
                    problems.append(
                        f"{k2}: shape {pts.shape} != {e.shape}")
                else:
                    dd = np.max(np.abs(pts - e))
                    if dd > TOL:
                        problems.append(f"{k2}: max|d|={dd:.2e}")
                    elif a.verbose:
                        print(f"    {os.path.basename(tf)[:34]:36s} "
                              f"{k2:12s} {pts.shape[0]} pts max|d|={dd:.1e}")
                    exp.pop(k2, None)     # consume, so 2 surfaces both check
                continue
            if key is None:
                continue
            ok, msg = cmp_curve(pts, *exp[key], cap=parse_tex.cap)
            checked += 1
            if not ok:
                problems.append(f"{key}: {msg}")
            elif a.verbose:
                print(f"    {os.path.basename(tf)[:34]:36s} {key:12s} {msg}")
        if problems:
            n_bad += 1; bad_files.append(os.path.basename(tf))
            print(f"  MISMATCH {os.path.basename(tf)[:46]}")
            for p in problems: print(f"      {p}")
        elif checked:
            n_ok += 1
        else:
            n_skip += 1

    print(f"\n{'='*60}")
    print(f"  figures verified against PKL : {n_ok}")
    print(f"  mismatches                   : {n_bad}")
    print(f"  skipped (no comparable data) : {n_skip}")
    if bad_files:
        print("\n  files with mismatches:")
        for b in bad_files: print(f"    {b}")
    print(f"{'='*60}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
