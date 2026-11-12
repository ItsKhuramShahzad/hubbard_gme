# Progress.md — Hubbard GME Project Progress Log

Generated/updated: 2026-06-29
Project: `hubbard_gme`  (package version 0.3.0)

---

## 1. Overall Status

The project has grown from a single-point ground-state GME pipeline into a multi-mode
toolkit. On top of the auditable, restartable convex-roof workflow (per-method solver
cache, effective settings, full audit metadata), the recent work adds a
**temperature- and parameter-dependent layer** in `thermal_pipeline.py`:

```text
Task 1  Thermal GME            C_GME(n, T) vs T          (--thermal)
Task 2  Excited-singlet GME    C_GME(n) bar + T-sweep    (--excited-singlet [--T-sweep])
Task 3  t-scan                 C_GME(n) vs t at T=0       (--t-scan)
Task 4  U-scan                 C_GME(n) vs U at T=0       (--u-scan)
```

These are orchestrated by `run_all_tasks.sh` for L=6, Ne=6, t=1, U=10, gcg solver.

Cache convention is **v1** (default; cache filenames are tagged `..._v1_...`).

---

## 2. Current Status Summary

| Area | Status | Notes |
|---|---:|---|
| Project file inventory | Done | Structure documented in README / Plan |
| Plan.md / Progress.md / Verify.md / README.md | Updated | Refreshed 2026-06-29 to match code |
| Topologies (uniform/open/nnn/ssh/ladder) | Done | `VALID_MODES` in `hamiltonian.py` |
| Single-point CLI | Done | `main.py` ground-state + n=N verification |
| System-state PKL cache | Done | `state_cache/` reload works |
| RDM PKL cache | Done | `rdm_cache/` load-before-recompute; 48 PKLs present |
| Per-method GME cache | Done | stiefel/gcg/stiefel_gcg independent reload/recompute |
| GME audit metadata | Done | integrity, runtime, success, decomposition vectors |
| PDF solver-source visualization | Done | PDF shows PKL / NEW / MISS->NEW per solver |
| Parallel convex roof | Done | per-cluster via `--workers` |
| n=N pure-state logic | Done | pure formula for n=N, convex roof for mixed |
| Thermal GME (Task 1) | Done | PDFs generated 2026-06-29 (T0→5, N40) |
| Excited-singlet GME (Task 2) | Done | bar + T-sweep PDFs generated 2026-06-29 |
| t-scan (Task 3) | Done | PDF generated 2026-06-29 (t 0.05→2, N20) |
| U-scan (Task 4) | Done | `run_U_dependence_gme` verified 2026-06-29: `L6_uscan_t1_task4_uscan_GME_*_U1to20_N20_*.pdf` + PKL produced (n=6 flat ≈0.958, n<6 grow/saturate with U) |
| Molecular extension | Done | `main_mol.py` + `molecular_*` modules |
| Fast unit tests | Pass | 77 tests pass (metrics/hamiltonian/rdm/gme/meyerwallach/contract) |
| Convex-roof integration tests | Slow | `test_canonical_parameters`, `test_convex_roof`, `test_mixed_states` run real optimisation (minutes); not part of the quick pass |
| Scientific numerical validation | Ongoing | Ground-state E0 confirmed; thermal/scan values to be cross-checked |
| Cross-check vs `gcg_reference` | Done | 9-step suite, `tests/compare_with_reference_gcg.py`; agree to 8e-13 on analytics, 1.2e-15 on pure states, 0 bracket violations |
| Restart-budget convergence | **Action needed** | 8-10 restarts insufficient for n>=4; 40 required (see Plan.md 8b) |
| `h(U)` value-only fix | Done 2026-08-22 | `_h_full_unitary` no longer computes discarded gradients: 1.24x, identical results |
| HPC job restructuring | Planned | `Plan_HPC_Scaling.md`: split by (point, n), 1 CPU/task, `merge_scan_pkls` |
| Production figure regeneration | **Pending** | all n>=4 figures must be re-run at 40 restarts |

---

## 3. Done

### 3.1 Thermal / excited / scan layer (`thermal_pipeline.py`)

Three public entry points wired into `main.py`:

```text
run_thermal_gme_analysis(L, t, U, T_values, ...)   -> --thermal
run_excited_singlet_gme(L, t, U, ...)              -> --excited-singlet [--T-sweep]
run_t_dependence_gme(L, t_values, U, V, ...)       -> --t-scan
run_U_dependence_gme(L, U_values, t, V, ...)       -> --u-scan
run_V_dependence_gme(L, V_values, t, U, ...)       -> --v-scan   (extended Hubbard)
find_first_excited_singlet(...)                    -> spin classification helper
```

Thermal mixed states use the GCG convex-roof solver (`--thermal-solver gcg`); a fast
analytical lower bound is available with `--thermal-solver supervisor`. The thermal
caches live under `results/rdm_pkl_results/thermal_runs/` (result PKLs, per-(T,n)
`job_cache/`, and per-T `rdm_bank_cache/`).

Outputs produced on 2026-06-29 (in `outputs/`):

```text
Task 1  L6_*_task1_thermal_GME_L6_t1.0_U10.0_T0to5_N40_*.pdf
Task 2  L6_singlet_*_task2_excited_singlet_GME_*_T0_*.pdf  (+ ..._thermal_*_T0.01to5_N30_*.pdf)
Task 3  L6_tscan_*_task3_tscan_GME_L6_U10.0_t0.05to2_N20_*.pdf
```

### 3.1b Extended Hubbard V (nearest-neighbour interaction)  (DONE 2026-07-29)

Added `V Σ_j n_j n_{j+1}` (n_j = n_{j↑}+n_{j↓}, PBC ring) to
`build_hubbard_hamiltonian`, threaded `V` (keyword, default 0.0) through
`compute_all_eigenstates` / `compute_k_eigenstates` /
`compute_ground_and_excited_states`, both scans, and `main.py --V`. `V=0`
reproduces the plain Hubbard model exactly (regression checked). Validated:
E0 shifts with V; a U-scan at V=5 changes C_GME on every cluster.

New V-scan (fix t and U, sweep V) — shows the Mott→CDW transition at V ~ U/2:

```text
run_V_dependence_gme(L, V_values, t, U, ...)  -> --v-scan  (+ --v-scan-min/max/steps)
plot_scan_finalized                            -> V/t x-axis branch; fixed V shown in
                                                  t-/U-scan titles when V != 0
```

Saved everywhere: V is in each scan PKL (`system_info` + the cache filename, so
V/t = 5, 10, 20 runs never collide) and in the auto-produced PDF title. Test
(L=6, U=10, V 0→12): C_GME(n<6) collapses between V=4 and V=8, i.e. across
V ≈ U/2 = 5 — the CDW transition, captured as a single curve.

### 3.2 Per-method GME solver control

Independent cache decisions for `stiefel`, `gcg`, `stiefel_gcg`:

```bash
--gme-cache-default {reload,recompute}     # default: reload
--stiefel-cache / --gcg-cache / --stiefel-gcg-cache {default,reload,recompute}
--force-gme-recompute                      # hard global recompute
```

Solver-source labels in PDFs: `S:PKL` (reused), `G:NEW` (recomputed),
`SG:MISS->NEW` (reload requested but missing, then recomputed).

### 3.3 RDM cache loading

Pipeline loads each RDM PKL before recomputing; recomputes only on
`--force-rdm-recompute`. Second-run message:

```text
[cache] RDM cache summary: GS loaded=24, GS computed=0
```

### 3.4 Effective settings saved

No ambiguous `iterauto`/`tolauto`. Saved defaults:

```text
n_restarts = 10, k_max_extra = 2, tol = 1e-9, tol_k = 1e-6
gcg max_iter = 300, stiefel_gcg max_iter = 300, stiefel max_iter = adaptive by k/n_params
```

### 3.5 GME audit metadata

Each GME cluster PKL includes RDM integrity (`rho_hash_sha256`, `trace`, `purity`,
`rank`, `eigvals`, `lambda_min/max`, `hermitian_max_error`, `trace_error`), solver
audit (`solver_runtime`, `solver_success`, `elapsed_by_stage`, `solver_cache_status`),
and the pure-state decomposition vectors with `psi_hash_sha256` per component.

### 3.6 n=N rule

```text
n=N pure  -> pure-state formula + n=N verification
n=N mixed -> convex roof
n<N mixed -> convex roof
```

For `--L 6 --U 4.0 --min-n 3 --max-n 6`: 18 mixed clusters via convex roof (not 24);
the 6 `n=6` clusters are pure and handled by the pure formula.

---

## 4. In Progress / To Verify

### 4.1 Task 4 — U-scan  (DONE 2026-06-29)

`run_U_dependence_gme` (`--u-scan`) verified with a full production run:

```bash
python main.py --u-scan --L 6 --t 1.0 --u-scan-min 1.0 --u-scan-max 20.0 \
  --u-scan-steps 20 --min-n 2 --thermal-solver gcg --n-restarts 10 \
  --max-iter 200 --k-max-extra 4 --workers 6 --use-pkl-cache \
  --output-prefix L6_uscan_t1
```

Produced `outputs/L6_uscan_t1_task4_uscan_GME_L6_t1.0_U1to20_N20_20260629_130036.pdf`
and `thermal_runs/uscan_v1_..._U1to20_N20-*.pkl`. Physics OK: n=6 C_GME flat ≈ 0.958
(no U-dependence for n=L), n<6 grow with U and saturate at large U.

### 4.2 verify_project.py cache-version default  (FIXED)

`scripts/verify_project.py` `--cache-version` default was changed `v2` → `v1` to match
the on-disk cache and `main.py`.

`scripts/verify_project.py` still defaults `--cache-version` to `v2`, but the on-disk
cache and `main.py` use `v1`. Either pass `--cache-version v1` to the verify script, or
change its default to `v1`.

### 4.3 Convex-roof integration tests are slow

`test_canonical_parameters.py` and the other real-optimisation tests time out of a
quick (sub-minute) pass. They should be run with a generous timeout in the `qchem`
environment as a separate, scheduled check.

---

## 5. Pending

### 5.1 Scientific validation

```text
Ground-state energy E0 = -3.6687061789 for L=6, U=4, t=1, uniform   (confirm)
RDM trace ≈ 1, Hermiticity error small, eigenvalues ≥ -tol
Pure n=N via pure formula; mixed n<N via convex roof
C_GME(n=L, T→0) -> exact ground-state value (thermal sanity check)
Thermal C_GME(n,T) monotonic / physically sensible
Excited-singlet state correctly identified (S=0 above triplet manifold)
```

### 5.2 Documentation hygiene  (FIXED)

`__init__.py` now imports and exports `run_U_dependence_gme` (added to the
`thermal_pipeline` import and to `__all__`). `run_all_tasks.sh` header comment corrected
from "three tasks" to "four tasks".

---

## 6. Recommended Next Actions  (updated 2026-08-22)

Ordered by what blocks the paper.

1. **Stage 0 of `Plan_HPC_Scaling.md`** — on the cluster, measure 1 vs 8 threads
   at L=8 and record `MaxRSS`. ~1 h, decides `--cpus-per-task` and `--mem`.
2. **Write `merge_scan_pkls`** in `thermal_pipeline.py` (template:
   `merge_tT_surface_pkls`, line 2877) and wire `--merge-scan` in `main.py`
   (template: `--merge-surface`, lines 329 / 734).
3. **Dry run** the split array with `--array=0-6` (one grid point, all n) and
   test the merge on those 7 parts.
4. **Launch** the full `--array=0-146` for the U-scan; ~3 days at 8 slots.
5. **Merge -> replot** and check the figure against the old one.
6. Repeat for the t-scan and V-scan, L=6 and L=8, gs and excited singlet.
7. Only then finalise `N_r` in the paper's numerical-approach section. Until the
   figures are regenerated the text must say `N_r = 10`, which is what the
   current data used.

Deferred (not blocking):

* align `verify_project.py` cache-version default with `v1`
* run the slow convex-roof integration tests with a long timeout

---

## 7. Lab Log Template

Append after every important run.

```text
Date:
Machine/environment:
Code version (package __version__ / file mtimes):
Cache version:
Task / mode:
Command:
Runtime:
Workers:
Output directory:
RDM cache summary:
GME solver policy:
Clusters / (T,n) jobs completed:
Important numerical values:
    E0 / E1 / gap:
    C_GME(n=3..L):
    C_GME(n, T) notes:
Errors/warnings:
Files produced (PDFs / PKLs):
Conclusion:
Next action:
```

---

## 8. Current Conclusion  (updated 2026-08-22)

The implementation is verified correct: it reproduces the analytic roofs (GHZ,
W, M-families) to machine precision and agrees with an independently written
implementation of the same algorithm to 8e-13 on analytic cases and 1.2e-15 on
pure states, with zero bracket violations.

The open issue is **not correctness but convergence**. The production runs used
8-10 restarts, and at n>=4 that is demonstrably too few: at n=5 a 20-restart run
sits in a spurious local minimum 3.6e-03 above the value 40 restarts reaches.
Every figure involving n>=4 (45 of the 49 cached scans) must be regenerated at
`--n-restarts 40`.

That regeneration is infeasible with the current job structure (~23 days for one
serial scan, over the 14-day limit). `Plan_HPC_Scaling.md` sets out the fix:
split the scan array by (grid point, n), drop to 1 CPU per task because BLAS
threading slows the solver, and add `merge_scan_pkls` to reassemble. Projected
~3 days per scan.
