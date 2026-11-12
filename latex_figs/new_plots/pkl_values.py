"""C_GME of the Fig. 1 runs read directly from the PKLs (read in place in Dropbox, not copied).
The md5 of each PKL is checked against the value recorded in latex_figs/pdf/FIGURES_DATA.txt."""
import hashlib, pickle, numpy as np
R = "/mnt/d/Dropbox/PhDWork/hubbard_gme/results/rdm_pkl_results/thermal_runs/"
PKL = {6: ("uscan_v1_L6_Ne6_t1_mode-uniform_nmin2_solver-gcg_sz0_restarts10_iter200_kmax4_tol1em09_U1to20_N20-d03a74da789e.pkl",
           "0f958a55ccbda2c48771b9915fe9ee04"),
       8: ("uscan_v1_L8_Ne8_t1_mode-uniform_nmin2_solver-gcg_full_restarts8_iter200_kmax4_tol1em09_U1to20_N20-d03a74da789e.pkl",
           "0fad681095a933ec6c9ed44efe70415b")}

def cgme(L, n_min=3):
    """{n: {U/t: C_GME}} for n = n_min..L, straight from result['C_GME'] of the Fig. 1 PKL of this L."""
    name, md5 = PKL[L]
    raw = open(R + name, "rb").read()
    assert hashlib.md5(raw).hexdigest() == md5, f"{name}: md5 differs from FIGURES_DATA"
    r = pickle.loads(raw)["result"]
    U = np.asarray(r["U_over_t"], float)
    return name, {int(n): {float(u): float(c) for u, c in zip(U, r["C_GME"][n])}
                  for n in sorted(r["C_GME"]) if n_min <= int(n) <= L}
