"""
Tests for --sym-rdm: the nBRDM built inside the pipeline from its independent elements only (S_z blocks,
hermiticity, spin flip), with the symmetries checked exactly on the set of vectors used (one vector, a degenerate
level, one level of a thermal sum) and the elements computed instead whenever a check fails.

Strictness
  * References never come from the code under test: RDMs from rdm.compute_general_nRDM, occupations and S_z from
    number operators, the spin flip rebuilt from OpenFermion operator algebra (F O F^dag = O with up <-> down).
  * Element-by-element comparisons, absolute tolerance 1e-11 on O(1) values (the level sums of eigh vectors agree to
    ~1e-12; single vectors to ~1e-15).
  * Negative controls: forcing a symmetry where it does not hold must change the RDM, so the checks are needed.
  * The work is counted: the operators built must equal the closed-form count, and no operator is built twice.
  * Every production route and main.py run with the flag on and off on a small ring; results must agree, and the
    flag must be recorded. Result folders are redirected to a temporary directory.
"""
import importlib.util
import itertools
import pickle
import sys
from math import comb
from pathlib import Path

import numpy as np
import pytest
from openfermion import FermionOperator, sz_operator
from openfermion.linalg import get_number_preserving_sparse_operator as number_preserving

import hubbard_gme.rdm_sym as rs
import hubbard_gme.thermal_pipeline as tp
from hubbard_gme.hamiltonian import build_hubbard_hamiltonian, compute_all_eigenstates
from hubbard_gme.rdm import compute_general_nRDM, generate_site_clusters

REPO = Path(__file__).resolve().parents[1]
TOL = 1e-11
GCG = dict(solver="gcg", n_restarts=2, max_iter=30, tol=1e-9, k_max_extra=0, verbose=False)


@pytest.fixture(autouse=True)
def fresh_basis_cache():
    """The basis data are cached per Hilbert space; every test starts without them."""
    tp._SYM_BASIS_CACHE.clear()
    yield
    tp._SYM_BASIS_CACHE.clear()


# ═══════════════════════════════ independent references ═══════════════════════════════

def occupations(L, Ne, spin_preserving=True):
    return np.column_stack([
        number_preserving(FermionOperator(((k, 1), (k, 0))), 2 * L, Ne,
                          spin_preserving=spin_preserving).diagonal().real.round().astype(int)
        for k in range(2 * L)])


def mask_of(sites, L, Ne, spin_preserving=True):
    occ = occupations(L, Ne, spin_preserving)
    return np.all([occ[:, 2 * s] + occ[:, 2 * s + 1] < 2 for s in sites], axis=0).astype(float)


def reference_sum(vectors, sites, L, Ne, spin_preserving=True, project=False):
    """sum_k Gamma[v_k] with rdm.compute_general_nRDM (the code the pipeline replaced)."""
    m = mask_of(sites, L, Ne, spin_preserving) if project else 1.0
    return sum(compute_general_nRDM(list(sites), v * m, 2 * L, Ne, spin_preserving=spin_preserving)
               for v in np.asarray(vectors).T)


def hubbard(L, U, Ne, spin_preserving=True, V=0.0, zeeman=0.0):
    H = build_hubbard_hamiltonian(L, 1.0, U, "uniform", V=V)
    if zeeman:
        H += zeeman * sz_operator(L)
    M = number_preserving(H, 2 * L, Ne, spin_preserving=spin_preserving).toarray()
    E, W = np.linalg.eigh(M)
    return E, W


def random_hamiltonian(L, seed, zeeman=0.0, flux=0.0):
    """Spin-independent random hoppings, on-site energies and U_i; optional Zeeman field and flux."""
    rng = np.random.default_rng(seed)
    H = FermionOperator()
    for i, j in itertools.combinations(range(L), 2):
        t = rng.uniform(0.3, 1.2) * np.exp(1j * flux * rng.uniform(-np.pi, np.pi))
        for s in (0, 1):
            H += FermionOperator(((2 * i + s, 1), (2 * j + s, 0)), -t)
            H += FermionOperator(((2 * j + s, 1), (2 * i + s, 0)), -np.conj(t))
    for i in range(L):
        eps, Ui, h = rng.uniform(-1, 1), rng.uniform(2, 6), zeeman * rng.uniform(0.5, 1.5)
        H += FermionOperator(((2 * i, 1), (2 * i, 0)), eps + h)
        H += FermionOperator(((2 * i + 1, 1), (2 * i + 1, 0)), eps - h)
        H += FermionOperator(((2 * i, 1), (2 * i, 0), (2 * i + 1, 1), (2 * i + 1, 0)), Ui)
    return H


def levels(E, tol=1e-10):
    """Degenerate levels by the T = 0 rule, written independently of tp._degenerate_levels."""
    out, used = [], np.zeros(len(E), bool)
    for i in np.argsort(E, kind="stable"):
        if not used[i]:
            idx = np.flatnonzero((~used) & (E - E[i] < tol) & (E >= E[i]))
            used[idx] = True
            out.append(idx)
    return out


def flip_operator(L, Ne, spin_preserving):
    """F as a dense matrix from operator algebra: F c^dag_{j,s} F^dag = c^dag_{j,-s}, F|vac> = |vac>.
    Column c of F is the state prod_k c^dag_{flip(k)} |vac> for the occupied orbitals k of configuration c, read
    off by applying the corresponding normal-ordered creation string through OpenFermion."""
    occ = occupations(L, Ne, spin_preserving)
    code = {tuple(r): i for i, r in enumerate(occ)}
    F = np.zeros((len(occ), len(occ)))
    for c, row in enumerate(occ):
        orbitals = [k for k in range(2 * L) if row[k]]
        flipped = [k ^ 1 for k in orbitals]                          # 2j <-> 2j + 1
        target = np.zeros(2 * L, int)
        target[flipped] = 1
        if tuple(target) not in code:
            return None
        # sign of reordering the creation string c^dag_{f1} c^dag_{f2} ... into ascending order
        perm = np.argsort(flipped, kind="stable")
        inversions = sum(1 for a, b in itertools.combinations(range(len(perm)), 2) if perm[a] > perm[b])
        F[code[tuple(target)], c] = (-1) ** inversions
    return F


def assert_same(A, B, tol=TOL):
    err = float(np.max(np.abs(np.asarray(A) - np.asarray(B))))
    assert err < tol, f"max difference {err:.2e}"


# ═══════════════════════════════ A. basis data and the spin flip ═══════════════════════════════

@pytest.mark.parametrize("L,Ne,flag", [(4, 4, True), (4, 4, False), (4, 3, False), (4, 2, True), (6, 6, True), (6, 5, False)])
def test_A1_sz_of_every_configuration(L, Ne, flag):
    sz2, _, _ = tp._sym_basis(2 * L, Ne, flag)
    occ = occupations(L, Ne, flag)
    assert np.array_equal(sz2, occ[:, 0::2].sum(1) - occ[:, 1::2].sum(1))


@pytest.mark.parametrize("L,Ne,flag", [(4, 4, True), (4, 4, False), (4, 3, False), (4, 2, False), (6, 6, True)])
def test_A2_flip_equals_the_operator_algebra(L, Ne, flag):
    _, perm, sign = tp._sym_basis(2 * L, Ne, flag)
    F = flip_operator(L, Ne, flag)
    assert perm is not None and F is not None
    mine = np.zeros_like(F)
    mine[perm, np.arange(len(perm))] = sign
    assert np.array_equal(mine, F)
    assert np.array_equal(F @ F, np.eye(len(F)))                       # F^2 = 1


@pytest.mark.parametrize("L,Ne,flag", [(4, 4, True), (4, 3, False)])
def test_A3_flip_commutes_with_H_and_flips_every_cluster_operator(L, Ne, flag):
    F = flip_operator(L, Ne, flag)
    H = number_preserving(build_hubbard_hamiltonian(L, 1.0, 3.0, "uniform", V=0.7), 2 * L, Ne,
                          spin_preserving=flag).toarray()
    assert np.max(np.abs(F @ H @ F.T - H)) < 1e-12
    sites, n = (0, 2, 3), 3
    for a, b in [(0, 0), (1, 2), (3, 5), (6, 6), (1, 4), (2, 7)]:
        O = tp._cluster_operator(sites, a, b, 2 * L, Ne, flag).toarray()
        Of = tp._cluster_operator(sites, rs.flip(a, n), rs.flip(b, n), 2 * L, Ne, flag).toarray()
        assert np.max(np.abs(F @ O @ F.T - Of)) < 1e-14


def test_A4_flip_is_unavailable_outside_Sz_zero():
    assert tp._sym_basis(8, 3, True)[1] is None                        # S_z = +1/2 sector
    assert tp._sym_basis(8, 3, False)[1] is not None                   # full space is closed under the flip
    assert flip_operator(4, 3, True) is None


def test_A5_a_wrong_basis_order_is_refused(monkeypatch):
    real = tp._basis_occupations
    monkeypatch.setattr(tp, "_basis_occupations", lambda *a, **k: real(*a, **k)[::-1])
    with pytest.raises(RuntimeError, match="Basis order mismatch"):
        tp._sym_basis(8, 4, True)


def test_A6_cluster_operator_is_the_old_operator():
    L, Ne = 4, 4
    old = tp._precompute_ops((1, 3, 0), 2 * L, Ne, spin_preserving=False)
    for (a, b), op in old.items():
        assert (op != tp._cluster_operator((1, 3, 0), a, b, 2 * L, Ne, False)).nnz == 0


# ═══════════════════════════════ B. the checks ═══════════════════════════════

def info_for(L, Ne, flag, n=2):
    return tp._SymInfo(tuple(range(n)), 2 * L, Ne, flag)


@pytest.mark.parametrize("flag", [True, False])
def test_B1_non_degenerate_ground_state_passes_both(flag):
    E, W = hubbard(4, 4.0, 4, flag)
    assert E[1] - E[0] > 1e-3
    assert tp._sym_checks(info_for(4, 4, flag), W[:, :1]) == (True, True)


def test_B2_odd_electron_number():
    E, W = hubbard(4, 4.0, 3, True)                                     # S_z = +1/2 sector: no flip at all
    assert tp._sym_checks(info_for(4, 3, True), W[:, :1]) == (True, False)
    E, W = hubbard(4, 4.0, 3, False)                                    # full space: the level is a doublet (or more)
    level = levels(E)[0]
    assert len(level) >= 2
    assert tp._sym_checks(info_for(4, 3, False), W[:, level]) == (True, True)


def test_B3_single_vectors_of_a_degenerate_level_fail_but_the_level_passes():
    E, W = hubbard(4, 4.0, 4, False)
    level = next(l for l in levels(E) if len(l) == 3)                  # a triplet S_z = -1, 0, +1
    rng = np.random.default_rng(1)
    mixed = W[:, level] @ np.linalg.qr(rng.normal(size=(3, 3)))[0]     # an arbitrary orthonormal basis of the level
    info = info_for(4, 4, False)
    assert tp._sym_checks(info, mixed) == (True, True)
    assert tp._sym_checks(info, mixed[:, :1]) == (False, False)        # mixes S_z sectors
    assert tp._sym_checks(info, mixed[:, :2]) == (False, False)        # incomplete level


def zeeman_states(flag, seed=2, zeeman=0.4):
    """Eigenstates with a site-dependent Zeeman field (a uniform field is a constant inside S_z = 0)."""
    return np.linalg.eigh(number_preserving(random_hamiltonian(4, seed, zeeman=zeeman), 8, 4, spin_preserving=flag).toarray())


def test_B4_zeeman_field_breaks_only_the_flip():
    E, W = zeeman_states(True)
    assert tp._sym_checks(info_for(4, 4, True), W[:, :1]) == (True, False)
    E, W = zeeman_states(False)
    assert E[1] - E[0] > 1e-6
    assert tp._sym_checks(info_for(4, 4, False), W[:, :1]) == (True, False)


def test_B4b_uniform_field_keeps_the_flip_in_the_Sz_zero_sector():
    E0, W0 = hubbard(4, 4.0, 4, True)
    E1, W1 = hubbard(4, 4.0, 4, True, zeeman=0.3)
    assert np.allclose(E0, E1)                                             # h S_z = 0 in this sector
    assert tp._sym_checks(info_for(4, 4, True), W1[:, :1]) == (True, True)


def test_B5_random_vector_fails_both():
    v = np.random.default_rng(3).normal(size=70)
    assert tp._sym_checks(info_for(4, 4, False), v[:, None]) == (False, False)


# ═══════════════════════════════ C. the RDM against the full construction ═══════════════════════════════

CASES = [  # L, Ne, U, V, spin_preserving, sites
    (4, 4, 4.0, 0.0, True, (0, 1, 2)), (4, 4, 4.0, 0.0, False, (0, 1, 2)), (4, 4, 1.0, 0.0, False, (2, 0, 3)),
    (4, 3, 4.0, 0.0, False, (0, 1, 2)), (4, 3, 4.0, 0.0, True, (1, 2)), (4, 2, 4.0, 0.0, False, (0, 2)),
    (4, 4, 4.0, 1.5, False, (0, 1, 2, 3)), (6, 6, 10.0, 0.0, True, (0, 1, 2, 3)), (6, 4, 4.0, 0.0, False, (0, 1, 2)),
    (6, 5, 4.0, 0.0, False, (3, 4, 5)),
]


@pytest.mark.parametrize("project", [False, True])
@pytest.mark.parametrize("L,Ne,U,V,flag,sites", CASES)
def test_C1_levels_and_single_vectors(L, Ne, U, V, flag, sites, project):
    E, W = hubbard(L, U, Ne, flag, V=V)
    n = len(sites)
    ops = tp._precompute_ops(sites, 2 * L, Ne, spin_preserving=flag, project_doublons=project, sym_rdm=True)
    for level in levels(E)[:6]:
        assert_same(tp._sym_rdm_sum(ops, W[:, level], n), reference_sum(W[:, level], sites, L, Ne, flag, project))
        assert_same(tp._level_rdm_average(ops, W[:, level], n),
                    reference_sum(W[:, level], sites, L, Ne, flag, project) / len(level))
        for k in level:
            assert_same(tp._rdm_from_ops(ops, W[:, k], n), reference_sum(W[:, [k]], sites, L, Ne, flag, project))


@pytest.mark.parametrize("seed,zeeman,flux", [(0, 0.0, 0.0), (1, 0.0, 0.8), (2, 0.4, 0.0), (3, 0.4, 0.8)])
def test_C2_random_hamiltonians_zeeman_and_flux(seed, zeeman, flux):
    L, Ne, sites = 4, 4, (0, 2, 3)
    for flag in (True, False):
        M = number_preserving(random_hamiltonian(L, seed, zeeman, flux), 2 * L, Ne, spin_preserving=flag).toarray()
        E, W = np.linalg.eigh(M)
        ops = tp._precompute_ops(sites, 2 * L, Ne, spin_preserving=flag, sym_rdm=True)
        for level in levels(E)[:5]:
            assert_same(tp._sym_rdm_sum(ops, W[:, level], 3), reference_sum(W[:, level], sites, L, Ne, flag))
        if flux:
            assert np.abs(reference_sum(W[:, :1], sites, L, Ne, flag).imag).max() > 1e-6     # really complex


def test_C3_random_vectors_and_unequal_mixtures_are_exact():
    L, Ne, sites = 4, 4, (0, 1, 3)
    rng = np.random.default_rng(7)
    ops = tp._precompute_ops(sites, 2 * L, Ne, spin_preserving=False, sym_rdm=True)
    V = rng.normal(size=(70, 3)) + 1j * rng.normal(size=(70, 3))
    assert_same(tp._sym_rdm_sum(ops, V, 3), reference_sum(V, sites, L, Ne, False))


def test_C4_negative_control_forcing_the_flip_is_wrong(monkeypatch):
    """Zeeman ground state: the flip does not hold, and assuming it gives a different RDM."""
    L, Ne, sites = 4, 4, (0, 1, 2)
    E, W = zeeman_states(True)
    ref = reference_sum(W[:, :1], sites, L, Ne)
    ops = tp._precompute_ops(sites, 2 * L, Ne, sym_rdm=True)
    assert_same(tp._sym_rdm_sum(ops, W[:, :1], 3), ref)
    monkeypatch.setattr(tp, "_sym_checks", lambda info, V: (True, True))
    assert np.max(np.abs(tp._sym_rdm_sum(ops, W[:, :1], 3) - ref)) > 1e-3


def test_C5_negative_control_forcing_the_blocks_is_wrong(monkeypatch):
    L, Ne, sites = 4, 4, (0, 1)
    v = np.random.default_rng(5).normal(size=70)
    ref = reference_sum(v[:, None], sites, L, Ne, False)
    ops = tp._precompute_ops(sites, 2 * L, Ne, spin_preserving=False, sym_rdm=True)
    monkeypatch.setattr(tp, "_sym_checks", lambda info, V: (True, False))
    assert np.max(np.abs(tp._sym_rdm_sum(ops, v, 2) - ref)) > 1e-3


def test_C6_negative_control_single_vector_of_a_level_with_the_flip_assumed(monkeypatch):
    E, W = hubbard(4, 4.0, 4, False)
    level = next(l for l in levels(E) if len(l) == 3)
    v = W[:, level] @ np.array([0.6, 0.0, 0.8])
    ref = reference_sum(v[:, None], (0, 1, 2), 4, 4, False)
    ops = tp._precompute_ops((0, 1, 2), 8, 4, spin_preserving=False, sym_rdm=True)
    assert_same(tp._sym_rdm_sum(ops, v, 3), ref)
    monkeypatch.setattr(tp, "_sym_checks", lambda info, V: (True, True))
    assert np.max(np.abs(tp._sym_rdm_sum(ops, v, 3) - ref)) > 1e-3


def test_C7_structure_hermitian_trace_and_zeros():
    L, Ne, sites = 6, 6, (0, 1, 2, 3)
    E, W = hubbard(L, 10.0, Ne, True)
    G = tp._sym_rdm_sum(tp._precompute_ops(sites, 2 * L, Ne, sym_rdm=True), W[:, 0], 4)
    assert np.array_equal(G, G.conj().T)
    d = np.array([rs.n_down(a) for a in range(16)])
    assert np.all(G[d[:, None] != d[None, :]] == 0)
    assert abs(np.trace(G).real - reference_sum(W[:, :1], sites, L, Ne).trace().real) < 1e-13


# ═══════════════════════════════ D. the work done ═══════════════════════════════

def expected_count(n):
    return (comb(2 * n, n) + 2 ** n + (comb(n, n // 2) if n % 2 == 0 else 0)) // 4


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_D1_table_holds_exactly_the_independent_elements(n):
    ops = tp._precompute_ops(tuple(range(n)), 12, 6, sym_rdm=True)
    assert len(ops) == ops.sym.n_built == expected_count(n) == len(rs.independent_pairs(n, True, True))


def test_D2_no_operator_is_built_while_evaluating_a_symmetric_state(monkeypatch):
    E, W = hubbard(6, 10.0, 6, True)
    ops = tp._precompute_ops((0, 1, 2, 3), 12, 6, sym_rdm=True)
    calls = []
    real = tp.get_number_preserving_sparse_operator
    monkeypatch.setattr(tp, "get_number_preserving_sparse_operator", lambda *a, **k: calls.append(1) or real(*a, **k))
    tp._rdm_from_ops(ops, W[:, 0], 4)
    assert calls == [] and ops.sym.n_built == expected_count(4)
    assert ops.sym.calls[-1] == (1, True, True)


def test_D3_a_failed_check_builds_each_missing_operator_once():
    v = np.random.default_rng(2).normal(size=70)
    ops = tp._precompute_ops((0, 1, 2), 8, 4, spin_preserving=False, sym_rdm=True)
    tp._sym_rdm_sum(ops, v, 3)
    assert ops.sym.n_built == len(ops) == 8 * 9 // 2                       # hermiticity only: all a <= b
    tp._sym_rdm_sum(ops, v, 3)
    assert ops.sym.n_built == 8 * 9 // 2                                   # nothing built twice


def test_D4_flag_off_table_is_unchanged():
    ops = tp._precompute_ops((0, 1, 2), 8, 4)
    assert ops.sym is None and len(ops) == 64


def test_D5_degenerate_level_rule():
    # -1 + 1.4e-10 is within 1e-10 of its neighbour (-1 + 5e-11) but not of the lowest state of the level (-1):
    # anchored at the lowest energy (the T = 0 rule) it is a level of its own; a chained rule would merge it.
    E = np.array([0.3, -1.0, -1.0 + 5e-11, 0.3 + 2e-11, 2.0, -1.0 + 1.4e-10])
    got = [list(l) for l in tp._degenerate_levels(E)]
    assert got == [[1, 2], [5], [0, 3], [4]]
    assert sorted(i for l in got for i in l) == list(range(len(E)))


def test_D6_bank_without_energies_is_refused():
    with pytest.raises(ValueError, match="energies"):
        tp._compute_cluster_rdm_bank(2, (0, 1), np.eye(36)[:, :3], 8, 4, sym_rdm=True)


# ═══════════════════════════════ E. the routes, flag on against flag off ═══════════════════════════════

@pytest.fixture
def folders(tmp_path, monkeypatch):
    for name, sub in (("_thermal_runs_dir", "runs"), ("_rdm_bank_dir", "bank"), ("_thermal_job_cache_dir", "jobs")):
        d = tmp_path / sub
        d.mkdir()
        monkeypatch.setattr(tp, name, lambda d=d: d)
    monkeypatch.setenv("HUBBARD_GME_NO_LATEX", "1")
    return tmp_path


def same_results(off, on, keys=("omega", "purity", "entropy", "rank")):
    """Everything computed directly from rho must agree. C_GME is not compared: it comes from the convex-roof
    optimiser, which is unchanged code but not continuous in rho when it has not converged (the tests use 2
    restarts and 30 iterations), so the exact solver inputs are compared instead (solver_inputs)."""
    for key in keys:
        if key not in off:
            continue
        for n in off[key]:
            a, b = np.asarray(off[key][n], float), np.asarray(on[key][n], float)
            assert np.array_equal(np.isnan(a), np.isnan(b)), (key, n)
            assert np.nanmax(np.abs(a - b), initial=0.0) < 1e-10, (key, n, np.nanmax(np.abs(a - b), initial=0.0))
    for key in ("C_exact_nL", "C_wootters_n2", "C_exact_gs"):
        if key in off:
            a, b = np.atleast_1d(np.asarray(off[key], float)), np.atleast_1d(np.asarray(on[key], float))
            for x, y in zip(a, b):
                # The exact pure-state values are sqrt(2(1 - max purity)) (gme.compute_C_GME_pure): the square root
                # turns a 1e-16 rounding difference of the purity into 1e-8 near C = 0, and gives NaN when the purity
                # rounds to 1 + 1e-15 (gme.py is deliberately left unchanged). So they are compared through
                # C^2 / 2 = 1 - purity, with NaN accepted only for C = 0 on the other side.
                if np.isnan(x) != np.isnan(y):
                    assert abs(np.nan_to_num(x) + np.nan_to_num(y)) < 1e-7, (key, x, y)
                elif not np.isnan(x):
                    tol = 1e-10 if key == "C_wootters_n2" else 1e-13
                    assert abs((x * x - y * y) / 2 if key != "C_wootters_n2" else x - y) < tol, (key, x, y)
    assert set(off.get("C_GME", {})) == set(on.get("C_GME", {}))


@pytest.fixture
def solver_inputs(monkeypatch):
    """Records every matrix handed to the convex roof: rho in compute_gme (scans, in-process) and the normalised
    thermal nBRDMs (normalize_density_matrix, always in the parent process)."""
    record = {"current": None, "sym": 0}
    real_gme, real_norm, real_sym = tp.compute_gme, tp.normalize_density_matrix, tp._sym_rdm_sum

    def sym(ops, V, n):
        record["sym"] += 1
        return real_sym(ops, V, n)
    monkeypatch.setattr(tp, "_sym_rdm_sum", sym)

    def gme(rho, n, *a, **k):
        if record["current"] is not None:
            record["current"].append(("gme", int(n), np.array(rho, copy=True)))
        return real_gme(rho, n, *a, **k)

    def norm(G):
        out = real_norm(G)
        if record["current"] is not None:
            record["current"].append(("norm", int(np.log2(len(out))), np.array(out, copy=True)))
        return out
    monkeypatch.setattr(tp, "compute_gme", gme)
    monkeypatch.setattr(tp, "normalize_density_matrix", norm)

    def run(fn):
        record["current"], record["sym"] = [], 0
        out = fn()
        got, record["current"] = record["current"], None
        run.sym_calls = record["sym"]
        return out, got
    return run


def run_both(solver_inputs, fn, in_process=True):
    """Runs fn(False) and fn(True); the flag-off run must never touch the symmetric code and, when the RDMs are built
    in this process, the flag-on run must use it (so a route that silently drops the flag is caught)."""
    off = solver_inputs(lambda: fn(False))
    assert solver_inputs.sym_calls == 0
    on = solver_inputs(lambda: fn(True))
    if in_process:
        assert solver_inputs.sym_calls > 0
    return off, on


def same_inputs(a, b, tol=TOL):
    assert len(a) == len(b) > 0
    for (ka, na, A), (kb, nb, B) in zip(a, b):
        assert (ka, na) == (kb, nb)
        assert_same(A, B, tol)


def spy_levels(monkeypatch):
    sizes = []
    real = tp._sym_rdm_sum
    monkeypatch.setattr(tp, "_sym_rdm_sum", lambda ops, V, n: sizes.append(np.atleast_2d(np.asarray(V).T).shape[0]) or real(ops, V, n))
    return sizes


@pytest.mark.parametrize("project", [False, True])
@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("es", [False, True])
def test_E1_U_scan(folders, solver_inputs, project, flag, es):
    (off, a), (on, b) = run_both(solver_inputs, lambda sym: tp.run_U_dependence_gme(
        4, np.array([1.0, 4.0]), 1.0, n_electrons=4, output_dir=str(folders), save_pkl=False, spin_preserving=flag,
        excited_singlet=es, project_doublons=project, sym_rdm=sym, **GCG))
    assert off["sym_rdm"] is False and on["sym_rdm"] is True
    same_results(off, on)
    same_inputs(a, b)


def test_E2_V_scan(folders, solver_inputs):
    (off, a), (on, b) = run_both(solver_inputs, lambda sym: tp.run_V_dependence_gme(
        4, np.array([0.5, 3.0]), 1.0, 4.0, n_electrons=4, output_dir=str(folders), save_pkl=False,
        spin_preserving=False, sym_rdm=sym, **GCG))
    assert on["sym_rdm"] is True
    same_results(off, on)
    same_inputs(a, b)


@pytest.mark.parametrize("flag", [True, False])
def test_E3_N_scan_with_degenerate_levels_uses_the_whole_level(folders, monkeypatch, solver_inputs, flag):
    sizes = spy_levels(monkeypatch)
    (off, a), (on, b) = run_both(solver_inputs, lambda sym: tp.run_Ne_dependence_gme(
        4, [3, 4], 1.0, 4.0, output_dir=str(folders), save_pkl=False, spin_preserving=flag, sym_rdm=sym, **GCG))
    same_results(off, on)
    same_inputs(a, b)
    assert max(sizes) >= 2                                                 # Ne = 3 ground level evaluated as a whole


def test_E4_t_scan(folders, solver_inputs):
    (off, a), (on, b) = run_both(solver_inputs, lambda sym: tp.run_t_dependence_gme(
        4, np.array([0.5, 1.0]), 4.0, n_electrons=4, output_dir=str(folders), save_pkl=False,
        spin_preserving=False, sym_rdm=sym, **GCG))
    assert on["sym_rdm"] is True
    same_results(off, on)
    same_inputs(a, b)


@pytest.mark.parametrize("project", [False, True])
@pytest.mark.parametrize("flag,Ne", [(True, 4), (False, 4), (False, 3)])
@pytest.mark.parametrize("workers", [1, 2])
def test_E5_thermal(folders, solver_inputs, project, flag, Ne, workers):
    T = np.array([0.0, 0.3, 1.0])
    (off, a), (on, b) = run_both(solver_inputs, lambda sym: tp.run_thermal_gme_analysis(
        4, 1.0, 4.0, T, n_electrons=Ne, output_dir=str(folders), save_pkl=False, thermal_energy_cutoff=None,
        n_workers=workers, spin_preserving=flag, project_doublons=project, sym_rdm=sym, **GCG), in_process=(workers == 1))
    assert ("_sym1" in str(on["rdm_bank_cache_path"])) and ("_sym1" not in str(off["rdm_bank_cache_path"]))
    assert on["sym_rdm"] is True and off["sym_rdm"] is False
    same_results(off, on)
    norm_a, norm_b = [x for x in a if x[0] == "norm"], [x for x in b if x[0] == "norm"]
    assert len(norm_a) == len(T) * len(off["n_values"])
    same_inputs(norm_a, norm_b)


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("flag,Ne", [(False, 4), (False, 3), (True, 4)])
def test_E6_bank_stores_the_level_average(flag, Ne, workers):
    """With 2 workers the bank is built in child processes: the stored level averages show that the flag and the
    energies reached them (the thermal sums alone would not)."""
    E, W, nso = compute_all_eigenstates(4, 1.0, 4.0, "uniform", Ne, spin_preserving=flag, V=0.0, classify_spin=False)
    off = tp._build_rdm_bank([2, 3], W, nso, Ne, L=4, verbose=False, spin_preserving=flag, n_workers=workers)
    on = tp._build_rdm_bank([2, 3], W, nso, Ne, L=4, verbose=False, spin_preserving=flag, sym_rdm=True, energies=E,
                            n_workers=workers)
    if not flag:                                                           # full space: single vectors are not level averages
        assert any(np.max(np.abs(off[3][l] - off[3][l].mean(axis=0))) > 1e-6 for l in levels(E) if len(l) > 1)
    assert any(len(l) > 1 for l in levels(E))
    for n in (2, 3):
        for level in levels(E):
            assert_same(on[n][level], np.broadcast_to(off[n][level].mean(axis=0), on[n][level].shape))
        p = np.exp(-(E - E[0]) / 0.4)
        p /= p.sum()
        assert_same(np.einsum("m,mij->ij", p, on[n]), np.einsum("m,mij->ij", p, off[n]))


def test_E7_bank_cache_is_tagged_and_kept_apart(folders):
    kw = dict(L=4, t=1.0, U=4.0, n_electrons=4, spin_preserving=False, use_pkl_cache=True, verbose=False)
    off = tp.build_or_load_rdm_bank(**kw)
    on = tp.build_or_load_rdm_bank(sym_rdm=True, **kw)
    assert not on["loaded_from_cache"]
    assert "_sym1" in on["cache_path"].name and "_sym1" not in off["cache_path"].name
    assert pickle.load(open(on["cache_path"], "rb"))["sym_rdm"] is True
    assert tp.build_or_load_rdm_bank(sym_rdm=True, **kw)["loaded_from_cache"]
    payload = pickle.load(open(on["cache_path"], "rb"))
    payload["sym_rdm"] = False
    pickle.dump(payload, open(on["cache_path"], "wb"))
    assert not tp.build_or_load_rdm_bank(sym_rdm=True, **kw)["loaded_from_cache"]


def test_E8_only_the_bank_name_changes():
    base = dict(L=6, t=1.0, U=10.0, t_prime=None, t2=None, t_perp=None, mode="uniform", n_electrons=6, n_min=2,
                average_site_clusters=True, cache_version="v1", spin_preserving=False)
    assert tp._rdm_bank_cache_path(**base).name == "rdmbank_v1_L6_Ne6_U10_t1_mode-uniform_tpall_t2all_tperpall_nmin2_full_clusteravg1.pkl"
    assert tp._rdm_bank_cache_path(**base, sym_rdm=True).name.endswith("_clusteravg1_sym1.pkl")
    assert tp._rdm_bank_cache_path(**base, project_doublons=True, sym_rdm=True).name.endswith("_clusteravg1_proj1_sym1.pkl")


def test_E9_excited_singlet_route_with_thermal_part(folders, solver_inputs, monkeypatch):
    # The route takes its k = 12 states from a sparse eigensolver that, run to run, can miss one state of a degenerate
    # level (existing behaviour, flag on or off), so its T > 0 sums are not reproducible. Exact dense k lowest states
    # here make both runs see the same states and isolate the flag.
    def dense_k(L, t, U, mode, n_electrons, k=4, spin_preserving=True, V=0.0, **kw):
        H = number_preserving(build_hubbard_hamiltonian(L, t, U, mode, V=V), 2 * L, n_electrons,
                              spin_preserving=spin_preserving).toarray()
        E, W = np.linalg.eigh(H)
        return E[:k], W[:, :k], 2 * L
    monkeypatch.setattr(tp, "compute_k_eigenstates", dense_k)
    run = lambda sym: tp.run_excited_singlet_gme(
        4, 1.0, 4.0, n_electrons=4, output_dir=str(folders), T_values=np.array([0.0, 0.5]), thermal_energy_cutoff=None,
        save_pkl=False, spin_preserving=True, sym_rdm=sym, **GCG)
    # (S_z sector only: run_excited_singlet_gme does not pass spin_preserving to find_first_excited_singlet, so the
    #  standalone excited-singlet route fails in the full space with or without --sym-rdm; existing, not changed here)
    calls = spy_levels(monkeypatch)
    (off, a), (on, b) = run_both(solver_inputs, run)
    assert any(k > 1 for k in calls)                                       # thermal bank: degenerate levels as a whole
    assert on["sym_rdm"] is True
    for n in off["purity_pure"]:
        assert abs(off["purity_pure"][n] - on["purity_pure"][n]) < 1e-10
    for n in off["T_results"]["purity"]:
        assert np.max(np.abs(np.asarray(off["T_results"]["purity"][n]) - np.asarray(on["T_results"]["purity"][n]))) < 1e-10
    assert any(k == "norm" for k, _, _ in a)                               # the thermal part was compared too
    same_inputs(a, b)


def test_E10_tT_surface(folders, solver_inputs):
    (off, a), (on, b) = run_both(solver_inputs, lambda sym: tp.run_tT_surface_gme(
        4, np.array([0.5, 1.0]), np.array([0.0, 0.5]), 4.0, n_electrons=4, output_dir=str(folders), n_workers=1,
        spin_preserving=False, save_pkl=False, thermal_energy_cutoff=None, sym_rdm=sym, **GCG))
    assert on["sym_rdm"] is True and off["sym_rdm"] is False
    for key in ("purity_surface", "entropy_surface", "rank_surface"):
        for n in off[key]:
            assert np.max(np.abs(np.asarray(off[key][n], float) - np.asarray(on[key][n], float))) < 1e-10, key
    same_inputs([x for x in a if x[0] == "norm"], [x for x in b if x[0] == "norm"])


# ═══════════════════════════════ F. main.py ═══════════════════════════════

@pytest.fixture
def cli(folders, monkeypatch):
    spec = importlib.util.spec_from_file_location("hubbard_main_sym_test", REPO / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["main.py", *argv, "--output-dir", str(folders)])
        module.main()
    return run


COMMON = ["--L", "4", "--n-restarts", "2", "--max-iter", "30", "--k-max-extra", "0", "--workers", "1", "--use-pkl-cache",
          "--full-hamiltonian"]


@pytest.mark.parametrize("argv,pattern", [
    (["--u-scan", "--Ne", "4", "--u-scan-min", "1", "--u-scan-max", "4", "--u-scan-steps", "2"], "uscan_*"),
    (["--u-scan", "--Ne", "4", "--u-scan-min", "1", "--u-scan-max", "4", "--u-scan-steps", "2", "--project-doublons"], "uscan_*"),
    (["--v-scan", "--Ne", "4", "--U", "4", "--v-scan-min", "1", "--v-scan-max", "3", "--v-scan-steps", "2"], "vscan_*"),
    (["--n-scan", "--U", "4", "--n-scan-values", "3", "4"], "nscan_*"),
    (["--t-scan", "--Ne", "4", "--U", "4", "--t-scan-min", "0.5", "--t-scan-max", "1", "--t-scan-steps", "2"], "tscan_*"),
    (["--thermal", "--Ne", "4", "--U", "4", "--T-min", "0", "--T-max", "0.5", "--T-steps", "2"], "thermal_*"),
])
def test_F1_command_line_records_the_flag_and_gives_the_same_numbers(folders, cli, solver_inputs, argv, pattern):
    _, a = solver_inputs(lambda: cli(*argv, *COMMON, "--force-thermal-recompute"))
    assert solver_inputs.sym_calls == 0
    off = {f.name: pickle.load(open(f, "rb")) for f in (folders / "runs").glob(pattern)}
    _, b = solver_inputs(lambda: cli(*argv, *COMMON, "--force-thermal-recompute", "--sym-rdm"))
    assert solver_inputs.sym_calls > 0                                      # the flag reached the RDM code
    same_inputs([x for x in a if x[0] == "norm"] or a, [x for x in b if x[0] == "norm"] or b)
    files = list((folders / "runs").glob(pattern))
    assert len(files) == 1                                                # same name: result PKLs are not tagged
    on = pickle.load(open(files[0], "rb"))
    off = next(iter(off.values()))
    assert on["result"]["sym_rdm"] is True and on["meta"]["sym_rdm"] is True
    assert off["result"].get("sym_rdm", False) is False
    same_results(off["result"], on["result"])


def test_F2_build_bank_command(folders, cli):
    cli("--build-bank", "--Ne", "4", "--U", "4", *COMMON, "--sym-rdm")
    banks = list((folders / "bank").glob("rdmbank_*"))
    assert len(banks) == 1 and banks[0].name.endswith("_sym1.pkl")
    assert pickle.load(open(banks[0], "rb"))["sym_rdm"] is True


@pytest.mark.parametrize("argv", [["--sym-rdm"], ["--scan-U", "--sym-rdm"], ["--scan-Ut", "--sym-rdm"],
                                  ["--u-scan", "--scan-U", "--sym-rdm"]])
def test_F3_unsupported_modes_are_refused(cli, argv):
    with pytest.raises(SystemExit) as stop:
        cli(*argv)
    assert stop.value.code == 2


def test_F4_every_route_passes_the_flag():
    """Static check: every route call in main.py that reaches the RDM code passes sym_rdm."""
    text = (REPO / "main.py").read_text()
    for call in ("build_or_load_rdm_bank(", "run_thermal_gme_analysis(", "run_excited_singlet_gme(", "run_t_dependence_gme(",
                 "run_tT_surface_gme(", "run_U_dependence_gme(", "run_V_dependence_gme(", "run_Ne_dependence_gme("):
        start = text.index(call, text.index("def main("))
        depth, i = 0, start + len(call) - 1
        while True:
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            if depth == 0:
                break
            i += 1
        assert "sym_rdm=args.sym_rdm" in text[start:i], call


# ═══════════════════════════════ G. larger ring ═══════════════════════════════

def test_G1_L6_full_hamiltonian_all_cluster_sizes():
    L, Ne = 6, 6
    E, W, nso = compute_all_eigenstates(L, 1.0, 10.0, "uniform", Ne, spin_preserving=False, V=0.0, classify_spin=False)
    for n in (2, 3, 4, 5, 6):
        sites = tuple(range(n))
        ops = tp._precompute_ops(sites, nso, Ne, spin_preserving=False, sym_rdm=True)
        for level in levels(E)[:4]:
            assert_same(tp._sym_rdm_sum(ops, W[:, level], n), reference_sum(W[:, level], sites, L, Ne, False))
        assert all(c[1] and c[2] for c in ops.sym.calls)                  # every level used both symmetries


def test_E11_flag_off_degenerate_scan_is_bit_identical_to_the_per_vector_loop(folders, solver_inputs):
    """With the flag off, the degenerate-level average must be the original loop, bit for bit (no change at all)."""
    (res, got) = solver_inputs(lambda: tp.run_Ne_dependence_gme(4, [3], 1.0, 4.0, output_dir=str(folders), save_pkl=False,
                                                                spin_preserving=False, **GCG))
    assert solver_inputs.sym_calls == 0
    E, W, nso = compute_all_eigenstates(4, 1.0, 4.0, "uniform", 3, spin_preserving=False, V=0.0, classify_spin=False)
    degen = np.where(np.abs(E - E[0]) < 1e-10)[0]
    assert len(degen) > 1
    rhos = [r for k, n, r in got if k == "gme"]
    for (_, n, rho) in [x for x in got if x[0] == "gme"]:
        ops = tp._precompute_ops(tuple(range(n)), nso, 3, spin_preserving=False)
        ref = np.zeros((2 ** n, 2 ** n), dtype=complex)
        for d in degen:
            ref += tp._rdm_from_ops(ops, W[:, d], n)
        ref /= len(degen)
        ref = ref / np.real(np.trace(ref))
        assert np.array_equal(rho, ref), n
    assert len(rhos) == 2                                                  # n = 2, 3 (n <= Ne)
