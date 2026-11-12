"""
Tests for the projected nBRDM (--project-doublons): the configurations with a doublon on the n sites of the cluster
are removed from each eigenstate, without renormalising, before the nBRDM is built.

Strictness: comparisons are element by element with an absolute tolerance of 1e-12 on O(1) values. Reference values
never come from the code under test: occupations are read from number operators, the mask is rebuilt here, and the
operator form n_{i,s} -> n_{i,s}(1 - n_{i,-s}), an exact U = 0 result and the density-matrix form P rho P are
independent routes to the same numbers. Negative controls check that the tests can fail. The route tests run every
production function, and main.py itself, on a small ring with the result folders redirected to a temporary directory.
"""
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from openfermion import FermionOperator
from openfermion.linalg import get_number_preserving_sparse_operator as number_preserving
from scipy.sparse import diags

import hubbard_gme.thermal_pipeline as tp
from hubbard_gme.hamiltonian import build_hubbard_hamiltonian, compute_all_eigenstates
from hubbard_gme.rdm import compute_general_nRDM, generate_site_clusters
from hubbard_gme.thermal_pipeline import (
    _basis_occupations,
    _build_rdm_bank,
    _cluster_doublon_mask,
    _doublon_mask,
    _precompute_ops,
    _rdm_from_ops,
    _wootters_concurrence,
    compute_gme,
    find_first_excited_singlet,
)

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "audit_reports" / "table_tests"
sys.path.insert(0, str(TABLES))

TOL = 1e-12
RUNS = Path("/mnt/d/Dropbox/PhDWork/hubbard_gme/results/rdm_pkl_results/thermal_runs")


# ── independent references ───────────────────────────────────────────────────

def occupations(L, Ne, spin_preserving=True):
    """Occupation table read from number operators, independent of _basis_occupations."""
    return np.column_stack([
        number_preserving(FermionOperator(((k, 1), (k, 0))), 2 * L, Ne,
                          spin_preserving=spin_preserving).diagonal().real.round().astype(int)
        for k in range(2 * L)
    ])


def mask_of(sites, L, Ne, spin_preserving=True):
    occ = occupations(L, Ne, spin_preserving)
    return np.all([occ[:, 2 * s] + occ[:, 2 * s + 1] < 2 for s in sites], axis=0).astype(float)


def single_probability(psi, sites, L, Ne, spin_preserving=True):
    """Probability of exactly one electron on every site of the cluster."""
    occ = occupations(L, Ne, spin_preserving)
    single = np.all([occ[:, 2 * s] + occ[:, 2 * s + 1] == 1 for s in sites], axis=0)
    return float((np.abs(psi) ** 2)[single].sum())


def product_occupation(psi, sites, L, Ne, spin_preserving=True):
    """<prod_i n_i>, the trace of the unprojected nBRDM."""
    occ = occupations(L, Ne, spin_preserving)
    return float((np.abs(psi) ** 2 * np.prod([occ[:, 2 * s] + occ[:, 2 * s + 1] for s in sites], axis=0)).sum())


def spectrum(L, Ne, U, spin_preserving=True, t=1.0, V=0.0):
    H = number_preserving(build_hubbard_hamiltonian(L, t, U, "uniform", V=V), 2 * L, Ne,
                          spin_preserving=spin_preserving)
    return np.linalg.eigh(H.toarray())


def ground(L, Ne, U):
    E, V = spectrum(L, Ne, U)
    assert E[1] - E[0] > 1e-8, "this test needs a non-degenerate ground state"
    return V[:, 0]


def rdm_old(psi, sites, L, Ne, spin_preserving=True):
    return _rdm_from_ops(_precompute_ops(tuple(sites), 2 * L, Ne, spin_preserving=spin_preserving), psi, len(sites))


def rdm_new(psi, sites, L, Ne, spin_preserving=True):
    """The code under test."""
    ops = _precompute_ops(tuple(sites), 2 * L, Ne, spin_preserving=spin_preserving, project_doublons=True)
    return _rdm_from_ops(ops, psi, len(sites))


def assert_same(A, B, tol=TOL):
    assert np.abs(np.asarray(A) - np.asarray(B)).max() < tol


# ── A. the mask ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("L,Ne,flag", [(4, 4, True), (4, 4, False), (6, 6, True), (4, 3, False), (5, 2, True)])
def test_A1_basis_order_matches_the_state_vector(L, Ne, flag):
    assert np.array_equal(_basis_occupations(2 * L, Ne, flag).astype(int), occupations(L, Ne, flag))


def test_A2_counts_for_L6_cluster_012():
    occ = occupations(6, 6)
    m = _doublon_mask((0, 1, 2), occ.astype(bool))
    per_site = np.stack([occ[:, 2 * s] + occ[:, 2 * s + 1] for s in (0, 1, 2)], axis=1)
    single = (per_site == 1).all(1)
    assert (len(occ), int(single.sum()), int((m == 0).sum()), int(((m == 1) & ~single).sum())) == (400, 56, 253, 91)


@pytest.mark.parametrize("flag", [True, False])
def test_A3_mask_is_exactly_the_doublon_rule(flag):
    occ = occupations(5, 4, flag)
    assert_same(_doublon_mask((1, 3), occ.astype(bool)), mask_of((1, 3), 5, 4, flag), 0.5)


def test_A4_doublon_outside_the_cluster_is_kept():
    occ = occupations(6, 6).astype(bool)
    m = _doublon_mask((0, 1, 2), occ)
    row = lambda bits: int(np.flatnonzero((occ == np.array(bits, bool)).all(1))[0])
    assert m[row([1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1])] == 1    # 11 on site 3, outside the cluster
    assert m[row([1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1])] == 0    # 11 on site 0, inside the cluster


def test_A5_mask_is_zero_or_one():
    m = _cluster_doublon_mask((0, 1), 8, 4)
    assert set(np.unique(m)) == {0.0, 1.0}
    assert np.array_equal(m * m, m)


def test_A6_a_wrong_basis_order_is_refused(monkeypatch):
    real = _basis_occupations
    monkeypatch.setattr(tp, "_basis_occupations", lambda *a, **k: real(*a, **k)[::-1])
    with pytest.raises(RuntimeError, match="Basis order mismatch"):
        _precompute_ops((0, 1), 8, 4, project_doublons=True)


@pytest.mark.parametrize("flag,L,Ne,sites", [(True, 4, 4, (0, 1, 2)), (False, 4, 3, (1, 3))])
def test_A8_operators_commute_with_the_projector(flag, L, Ne, sites):
    """c†_{i,s} c_{i,s'} keeps the occupation of site i, so masking the state is the same as D O D."""
    D = diags(_cluster_doublon_mask(sites, 2 * L, Ne, flag), format="csc")
    for op in _precompute_ops(sites, 2 * L, Ne, spin_preserving=flag).values():
        assert np.abs((D @ op - op @ D).toarray()).max() == 0


@pytest.mark.parametrize("flag,L,Ne,sites", [(True, 4, 4, (0, 1, 2)), (False, 4, 3, (1, 3))])
def test_A9_the_flag_changes_the_state_and_not_the_operators(flag, L, Ne, sites):
    plain = _precompute_ops(sites, 2 * L, Ne, spin_preserving=flag)
    table = _precompute_ops(sites, 2 * L, Ne, spin_preserving=flag, project_doublons=True)
    assert plain.mask is None
    assert np.array_equal(table.mask, mask_of(sites, L, Ne, flag))
    assert table.keys() == plain.keys()
    for key, op in plain.items():
        assert (op != table[key]).nnz == 0


def test_A7_state_vector_is_not_modified():
    psi = ground(4, 4, 4.0)
    before = psi.copy()
    rdm_new(psi, (0, 1), 4, 4)
    assert np.array_equal(psi, before)


# ── B. the projected nBRDM ───────────────────────────────────────────────────

def site_factor(site, a, b):
    """c†_a c_b on one site, with n_{i,s} -> n_{i,s}(1 - n_{i,-s}) on the diagonal."""
    p, q = 2 * site + a, 2 * site + b
    factor = FermionOperator(((p, 1), (q, 0)))
    if a == b:
        other = 2 * site + (1 - a)
        factor = factor * (FermionOperator(()) - FermionOperator(((other, 1), (other, 0))))
    return factor


@pytest.mark.parametrize("L,Ne,U,state,sites", [(4, 4, 1.0, 0, (0, 1)), (4, 4, 4.0, 3, (1, 2, 3)),
                                               (6, 6, 1.0, 0, (0, 1, 2)), (6, 6, 10.0, 0, (0, 1)),
                                               (6, 5, 2.0, 0, (0, 1, 2)), (5, 3, 3.0, 1, (4, 0))])
def test_B1_equals_an_independent_operator_form(L, Ne, U, state, sites):
    psi = spectrum(L, Ne, U)[1][:, state]
    n = len(sites)
    reference = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for a in range(2 ** n):
        for b in range(2 ** n):
            op = FermionOperator(())
            for k, s in enumerate(sites):
                op = op * site_factor(s, (a >> k) & 1, (b >> k) & 1)
            reference[a, b] = np.vdot(psi, number_preserving(op, 2 * L, Ne, spin_preserving=True) @ psi)
    assert_same(rdm_new(psi, sites, L, Ne), reference)


@pytest.mark.parametrize("flag,L,Ne,sites", [(True, 6, 6, (0, 1, 2)), (False, 4, 4, (0, 1, 2)), (False, 4, 3, (1, 2)), (True, 6, 4, (5, 0))])
def test_B2_equals_masking_the_state(flag, L, Ne, sites):
    psi = spectrum(L, Ne, 2.0, flag)[1][:, 0]
    assert_same(rdm_new(psi, sites, L, Ne, flag), rdm_old(psi * mask_of(sites, L, Ne, flag), sites, L, Ne, flag))


@pytest.mark.parametrize("L,Ne,U,sites", [(6, 6, 1.0, (0, 1)), (6, 6, 1.0, (0, 1, 2)), (6, 6, 10.0, (0, 1, 2, 3)), (6, 4, 2.0, (0, 1, 2))])
def test_B3_trace_is_the_probability_of_single_occupancy(L, Ne, U, sites):
    psi = ground(L, Ne, U)
    assert abs(np.trace(rdm_new(psi, sites, L, Ne)).real - single_probability(psi, sites, L, Ne)) < TOL


def test_B4_trace_is_not_the_norm_of_the_masked_vector():
    psi = ground(6, 6, 1.0)
    masked = psi * mask_of((0, 1, 2), 6, 6)
    assert abs(np.linalg.norm(masked) ** 2 - 0.4225) < 1e-3
    assert abs(np.trace(rdm_new(psi, (0, 1, 2), 6, 6)).real - 0.2473) < 1e-3


def test_B5_reference_numbers_L6():
    psi1, psi10 = ground(6, 6, 1.0), ground(6, 6, 10.0)
    assert abs(np.trace(rdm_old(psi1, (0, 1), 6, 6)).real - 0.8156) < 1e-3      # omega_2 of the paper table
    assert abs(np.trace(rdm_new(psi1, (0, 1), 6, 6)).real - 0.3737) < 1e-3
    assert abs(np.trace(rdm_new(psi10, (0, 1), 6, 6)).real - 0.9214) < 1e-3


def test_B6_flag_off_is_the_old_nbrdm():
    psi = ground(6, 6, 4.0)
    assert_same(rdm_old(psi, (0, 1, 2), 6, 6), compute_general_nRDM([0, 1, 2], psi, 12, 6))


@pytest.mark.parametrize("L,Ne,U,sites,flip", [(6, 6, 1.0, (0, 1, 2), True), (6, 6, 10.0, (0, 1, 2, 3), True),
                                              (6, 4, 2.0, (0, 1, 2), True), (5, 3, 4.0, (0, 1, 2), False)])
def test_B7_structure_survives(L, Ne, U, sites, flip):
    E, V = spectrum(L, Ne, U)
    G = rdm_new(V[:, 0], sites, L, Ne)
    n = len(sites)
    full = 2 ** n - 1
    assert_same(G, G.conj().T)
    assert np.linalg.eigvalsh(G).min() > -TOL
    down = [bin(a).count("1") for a in range(2 ** n)]
    assert all(abs(G[a, b]) < TOL for a in range(2 ** n) for b in range(2 ** n) if down[a] != down[b])
    if flip:
        assert all(abs(G[a, b] - G[a ^ full, b ^ full]) < 1e-10 for a in range(2 ** n) for b in range(2 ** n))


def test_B8_which_elements_change():
    psi = ground(6, 6, 1.0)
    old, new = rdm_old(psi, (0, 1), 6, 6), rdm_new(psi, (0, 1), 6, 6)
    off = ~np.eye(4, dtype=bool)
    assert np.abs((old - new)[off]).max() < TOL                       # n = 2: off-diagonal unchanged
    assert np.all(np.diag(new).real < np.diag(old).real - 1e-3)       # diagonal shrinks, does not vanish
    assert np.all(np.diag(new).real > 1e-3)
    old, new = rdm_old(psi, (0, 1, 2), 6, 6), rdm_new(psi, (0, 1, 2), 6, 6)
    partial = 0.0
    for a in range(8):
        for b in range(8):
            if a ^ b == 7:                                            # every site flips: nothing to remove
                assert abs(old[a, b] - new[a, b]) < TOL
            elif a != b:
                partial = max(partial, abs(old[a, b] - new[a, b]))
    assert partial > 1e-3                                             # one site flips, another keeps its spin


@pytest.mark.parametrize("L,Ne,U", [(6, 6, 1.0), (6, 6, 10.0), (6, 4, 2.0), (4, 4, 0.5)])
def test_B9_n_equal_Ne_is_unchanged(L, Ne, U):
    psi = ground(L, Ne, U)
    sites = tuple(range(Ne))
    assert_same(rdm_old(psi, sites, L, Ne), rdm_new(psi, sites, L, Ne))


def test_B10_exact_free_fermion_values():
    """U = 0, L = 6, half filling: <c†_0 c_1> = 1/3, so Wick gives omega_P = 2(1/4-1/9)^2 + 2(1/4+1/9)^2 = 97/324."""
    psi = ground(6, 6, 0.0)
    assert abs(np.trace(rdm_new(psi, (0, 1), 6, 6)).real - 97 / 324) < 1e-10
    assert abs(np.trace(rdm_new(psi, (0,), 6, 6)).real - 0.5) < 1e-10


def test_B11_large_U_limit():
    ratio = {}
    for U in (1.0, 4.0, 10.0, 40.0, 80.0):
        psi = ground(6, 6, U)
        ratio[U] = np.trace(rdm_new(psi, (0, 1), 6, 6)).real / np.trace(rdm_old(psi, (0, 1), 6, 6)).real
    assert ratio[1.0] < ratio[4.0] < ratio[10.0] < ratio[40.0] < ratio[80.0] < 1.0
    assert 3.5 < (1 - ratio[40.0]) / (1 - ratio[80.0]) < 4.5           # doublon weight ~ (t/U)^2


# ── C. clusters ──────────────────────────────────────────────────────────────

def test_C1_translation_with_each_cluster_mask():
    psi = ground(6, 6, 4.0)
    reference = rdm_new(psi, (0, 1, 2), 6, 6)
    for sites in generate_site_clusters(6, 3)[1:]:
        assert_same(rdm_new(psi, sites, 6, 6), reference, 1e-10)


def test_C2_negative_controls():
    psi = ground(6, 6, 1.0)
    sites = (1, 2, 3)
    good = rdm_new(psi, sites, 6, 6)
    wrong_cluster = rdm_old(psi * mask_of((0, 1, 2), 6, 6), sites, 6, 6)
    whole_ring = rdm_old(psi * mask_of(range(6), 6, 6), sites, 6, 6)
    renormalised = psi * mask_of(sites, 6, 6)
    renormalised = rdm_old(renormalised / np.linalg.norm(renormalised), sites, 6, 6)
    assert np.abs(good - wrong_cluster).max() > 1e-3
    assert np.abs(good - whole_ring).max() > 1e-3
    assert abs(np.trace(good - renormalised)) > 1e-3
    assert np.abs(good - rdm_old(psi, sites, 6, 6)).max() > 1e-3


# ── D. thermal and degenerate ────────────────────────────────────────────────

def boltzmann(E, T):
    p = np.exp(-(E - E[0]) / T)
    return p / p.sum()


@pytest.mark.parametrize("flag", [True, False])
def test_D1_each_state_then_sum_equals_P_rho_P(flag):
    L, Ne, sites = 4, 4, (0, 1, 2)
    E, V = spectrum(L, Ne, 4.0, flag)
    p = boltzmann(E, 0.7)
    ops = _precompute_ops(sites, 2 * L, Ne, spin_preserving=flag, project_doublons=True)
    summed = sum(pk * _rdm_from_ops(ops, V[:, k], 3) for k, pk in enumerate(p))
    m = mask_of(sites, L, Ne, flag)
    projected_rho = m[:, None] * ((V * p) @ V.conj().T) * m[None, :]
    plain = _precompute_ops(sites, 2 * L, Ne, spin_preserving=flag)
    direct = np.array([[(plain[(a, b)] @ projected_rho).trace() for b in range(8)] for a in range(8)])
    assert_same(summed, direct)


def test_D2_thermal_trace_is_the_weighted_trace():
    E, V = spectrum(4, 4, 4.0)
    p = boltzmann(E, 0.5)
    each = [rdm_new(V[:, k], (0, 1), 4, 4) for k in range(len(E))]
    weighted = sum(pk * np.trace(G).real for pk, G in zip(p, each))
    assert abs(np.trace(sum(pk * G for pk, G in zip(p, each))).real - weighted) < TOL


def test_D3_low_temperature_limit():
    E, V = spectrum(4, 4, 4.0)
    p = boltzmann(E, 0.002)
    G = sum(pk * rdm_new(V[:, k], (0, 1), 4, 4) for k, pk in enumerate(p) if pk > 1e-14)
    assert_same(G, rdm_new(V[:, 0], (0, 1), 4, 4), 1e-8)


def test_D4_degenerate_average_does_not_depend_on_the_basis():
    L, Ne, sites = 4, 3, (0, 1, 2)
    E, V = spectrum(L, Ne, 4.0, False)
    idx = np.flatnonzero(np.abs(E - E[0]) < 1e-10)
    g = len(idx)
    assert g >= 2
    rng = np.random.default_rng(7)
    Q = np.linalg.qr(rng.normal(size=(g, g)) + 1j * rng.normal(size=(g, g)))[0]
    W = V[:, idx] @ Q
    ops = _precompute_ops(sites, 2 * L, Ne, spin_preserving=False, project_doublons=True)
    A = sum(_rdm_from_ops(ops, V[:, i], 3) for i in idx) / g
    B = sum(_rdm_from_ops(ops, W[:, i], 3) for i in range(g)) / g
    assert_same(A, B, 1e-10)
    assert np.abs(_rdm_from_ops(ops, V[:, idx[0]], 3) - _rdm_from_ops(ops, W[:, 0], 3)).max() > 1e-3   # a single vector does depend


def test_D5_bank_is_the_same_with_one_and_two_workers():
    E, V = spectrum(4, 4, 4.0)
    serial = _build_rdm_bank([2, 3], V, 8, 4, L=4, n_workers=1, verbose=False, project_doublons=True)
    parallel = _build_rdm_bank([2, 3], V, 8, 4, L=4, n_workers=2, verbose=False, project_doublons=True)
    for n in (2, 3):
        assert_same(serial[n], parallel[n])
    plain = _build_rdm_bank([2], V, 8, 4, L=4, n_workers=2, verbose=False)
    assert np.abs(plain[2] - serial[2]).max() > 1e-3                   # the flag reaches the workers


# ── E. GME end to end ────────────────────────────────────────────────────────

@pytest.mark.parametrize("U,C_old,C_new", [(1.0, 0.0, 0.5713), (10.0, 0.3922, 0.4440)])
def test_E1_two_sites_gcg_equals_wootters(U, C_old, C_new):
    psi = ground(6, 6, U)
    for G, C in ((rdm_old(psi, (0, 1), 6, 6), C_old), (rdm_new(psi, (0, 1), 6, 6), C_new)):
        rho = G / np.trace(G).real
        assert abs(_wootters_concurrence(rho) - C) < 1e-3
        value = compute_gme(rho, 2, solver="gcg", n_restarts=4, max_iter=200, tol=1e-9, k_max_extra=2)[0]
        assert value == pytest.approx(C, abs=2e-3)


def test_E2_n_equal_L_stays_pure_and_unchanged():
    psi = ground(6, 6, 4.0)
    sites = tuple(range(6))
    old, new = rdm_old(psi, sites, 6, 6), rdm_new(psi, sites, 6, 6)
    a, b = old / np.trace(old).real, new / np.trace(new).real
    assert abs(np.trace(b @ b).real - 1) < 1e-10
    assert compute_gme(a, 6)[0] == pytest.approx(compute_gme(b, 6)[0], abs=1e-12)


# ── F. the removed part is biseparable, so projection cannot lower C_GME ─────

@pytest.mark.parametrize("L,Ne,U,state,sites", [(6, 6, 1.0, 0, (0, 1)), (6, 6, 1.0, 0, (0, 1, 2)),
                                               (6, 6, 4.0, 2, (1, 2, 3)), (6, 4, 2.0, 0, (0, 1, 2))])
def test_F1_doublon_pieces_are_identity_on_the_doublon_sites(L, Ne, U, state, sites):
    psi = spectrum(L, Ne, U)[1][:, state]
    occ = occupations(L, Ne)
    n = len(sites)
    doublon = np.stack([occ[:, 2 * s] & occ[:, 2 * s + 1] for s in sites], axis=1).astype(bool)
    total = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for pattern_id in range(2 ** n):
        pattern = np.array([(pattern_id >> k) & 1 for k in range(n)], bool)
        G = rdm_old(psi * (doublon == pattern).all(1), sites, L, Ne)
        total += G
        assert np.linalg.eigvalsh(G).min() > -TOL
        for k in np.flatnonzero(pattern):
            bit = 1 << int(k)
            for a in range(2 ** n):
                for b in range(2 ** n):
                    if (a ^ b) & bit:
                        assert abs(G[a, b]) < TOL
                    else:
                        assert abs(G[a, b] - G[a ^ bit, b ^ bit]) < TOL
    assert_same(total, rdm_old(psi, sites, L, Ne))


@pytest.mark.parametrize("U", [0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 20.0])
@pytest.mark.parametrize("state", [0, 1, 3])
def test_F2_bound_on_the_two_site_concurrence(U, state):
    psi = spectrum(6, 6, U)[1][:, state]
    old, new = rdm_old(psi, (0, 1), 6, 6), rdm_new(psi, (0, 1), 6, 6)
    w, wP = np.trace(old).real, np.trace(new).real
    assert _wootters_concurrence(old / w) <= wP / w * _wootters_concurrence(new / wP) + 1e-10


# ── G. the real routes ───────────────────────────────────────────────────────

GCG = dict(solver="gcg", n_restarts=2, max_iter=30, tol=1e-9, k_max_extra=0, verbose=False)


@pytest.fixture
def folders(tmp_path, monkeypatch):
    """Every result folder goes to tmp_path: the tests must never touch results/."""
    for name, sub in (("_thermal_runs_dir", "runs"), ("_rdm_bank_dir", "bank"), ("_thermal_job_cache_dir", "jobs")):
        d = tmp_path / sub
        d.mkdir()
        monkeypatch.setattr(tp, name, lambda d=d: d)
    monkeypatch.setenv("HUBBARD_GME_NO_LATEX", "1")
    return tmp_path


def path_args(**over):
    args = dict(L=6, n_electrons=6, n_min=2, solver="gcg", n_restarts=10, max_iter=300, tol=1e-9,
                k_max_extra=5, cache_version="v1", spin_preserving=False)
    args.update(over)
    return args


@pytest.mark.parametrize("build,expected", [
    (lambda p: tp._uscan_cache_path(t=1.0, mode="uniform", U_values=np.linspace(1, 20, 20),
                                    **path_args(max_iter=200, k_max_extra=4, spin_preserving=True), **p),
     "uscan_v1_L6_Ne6_t1_mode-uniform_nmin2_solver-gcg_sz0_restarts10_iter200_kmax4_tol1em09_U1to20_N20-d03a74da789e.pkl"),
    (lambda p: tp._uscan_cache_path(t=0.1, mode="uniform", U_values=np.linspace(0.1, 5, 21), excited_singlet=True, **path_args(), **p),
     "uscan_v1_L6_Ne6_t0p1_mode-uniform_nmin2_solver-gcg_full_es_restarts10_iter300_kmax5_tol1em09_U0.1to5_N21-8081b758e7cc.pkl"),
    (lambda p: tp._vscan_cache_path(U=10.0, t=1.0, mode="uniform", V_values=np.linspace(0, 12, 25), **path_args(), **p),
     "vscan_v1_L6_Ne6_U10_t1_mode-uniform_nmin2_solver-gcg_full_restarts10_iter300_kmax5_tol1em09_V0to12_N25-ebabade1a2a1.pkl"),
    (lambda p: tp._nscan_cache_path(L=8, U=10.0, t=1.0, V=0.0, mode="uniform", n_min=2, solver="gcg", n_restarts=10, max_iter=300,
                                    tol=1e-9, k_max_extra=2, N_values=[3, 4, 5, 6, 7, 8], cache_version="v1", spin_preserving=False, **p),
     "nscan_v1_L8_U10_t1_mode-uniform_nmin2_solver-gcg_full_restarts10_iter300_kmax2_tol1em09_N3to8_n6-d2c9afae4595.pkl"),
    (lambda p: tp._thermal_cache_path(analysis_type="thermal", L=6, t=1.0, U=10.0, t_prime=None, t2=None, t_perp=None, mode="uniform",
                                      n_electrons=6, n_min=2, solver="gcg", n_restarts=10, max_iter=300, tol=1e-9, k_max_extra=5, grid_name="T",
                                      grid_values=np.linspace(0, 0.2894737, 12), cache_version="v1", thermal_energy_cutoff=10.0,
                                      average_site_clusters=True, adaptive_cutoff=True, cutoff_increment=5.0, spin_preserving=False, **p),
     "thermal_v1_L6_Ne6_U10_t1_mode-uniform_tpall_t2all_tperpall_nmin2_solver-gcg_full_restarts10_iter300_kmax5_tol1em09_cut10_adaptive1_step5_clusteravg1_T0to0.289474_N12-f3dd7936eb54_knone_diag0.pkl"),
    (lambda p: tp._rdm_bank_cache_path(L=6, t=1.0, U=10.0, t_prime=None, t2=None, t_perp=None, mode="uniform", n_electrons=6, n_min=2,
                                       average_site_clusters=True, cache_version="v1", spin_preserving=False, **p),
     "rdmbank_v1_L6_Ne6_U10_t1_mode-uniform_tpall_t2all_tperpall_nmin2_full_clusteravg1.pkl"),
])
def test_G1_G2_cache_names(build, expected):
    assert build({}).name == expected                                  # existing PKLs are still found
    assert build(dict(project_doublons=False)).name == expected
    tagged = build(dict(project_doublons=True)).name
    assert tagged != expected and "_proj1" in tagged


def expected_T0(states, L, Ne, n, flag, project):
    """omega and purity of the nBRDM on sites 0..n-1, from the references above."""
    sites = tuple(range(n))
    build = (lambda p: rdm_old(p * mask_of(sites, L, Ne, flag), sites, L, Ne, flag)) if project else (lambda p: rdm_old(p, sites, L, Ne, flag))
    G = sum(build(p) for p in states) / len(states)
    w = np.trace(G).real
    prob = single_probability if project else product_occupation
    assert abs(w - np.mean([prob(p, sites, L, Ne, flag) for p in states])) < 1e-10
    return w, np.trace((G / w) @ (G / w)).real, G / w


def check_T0_result(res, i, states, L, Ne, flag, project):
    assert res["project_doublons"] is project
    for n in res["n_values"]:
        if n > min(L, Ne):
            continue
        w, purity, rho = expected_T0(states, L, Ne, n, flag, project)
        assert res["omega"][n][i] == pytest.approx(w, abs=1e-10)
        assert res["purity"][n][i] == pytest.approx(purity, abs=1e-10)
        if n == 2:
            assert res["C_wootters_n2"][i] == pytest.approx(_wootters_concurrence(rho), abs=1e-10)


def degenerate_level(E, V, E0):
    return [V[:, j] for j in np.flatnonzero(np.abs(E - E0) < 1e-10)]


@pytest.mark.parametrize("project", [False, True])
@pytest.mark.parametrize("flag", [True, False])
def test_G3_U_scan_route(folders, project, flag):
    res = tp.run_U_dependence_gme(4, np.array([1.0, 4.0]), 1.0, n_electrons=4, output_dir=str(folders), save_pkl=False,
                                  spin_preserving=flag, project_doublons=project, **GCG)
    for i, U in enumerate((1.0, 4.0)):
        E, V = spectrum(4, 4, U, flag)
        check_T0_result(res, i, degenerate_level(E, V, E[0]), 4, 4, flag, project)


@pytest.mark.parametrize("project", [False, True])
def test_G3_excited_singlet_route(folders, project):
    res = tp.run_U_dependence_gme(4, np.array([4.0]), 1.0, n_electrons=4, output_dir=str(folders), save_pkl=False,
                                  spin_preserving=False, excited_singlet=True, project_doublons=project, **GCG)
    info = find_first_excited_singlet(L=4, t=1.0, U=4.0, V=0.0, mode="uniform", n_electrons=4, k=12, spin_preserving=False)
    assert res["singlet_idx"][0] == info["idx"]
    E, V, _ = compute_all_eigenstates(4, 1.0, 4.0, "uniform", 4, spin_preserving=False, V=0.0, classify_spin=False)
    level = np.flatnonzero(np.abs(E - info["energy"]) < 1e-10)
    states = [np.asarray(info["psi"])] if len(level) <= 1 else [V[:, j] for j in level]
    check_T0_result(res, 0, states, 4, 4, False, project)


@pytest.mark.parametrize("project", [False, True])
def test_G3_V_scan_route(folders, project):
    res = tp.run_V_dependence_gme(4, np.array([1.0, 3.0]), 1.0, 4.0, n_electrons=4, output_dir=str(folders), save_pkl=False,
                                  spin_preserving=True, project_doublons=project, **GCG)
    for i, V_nn in enumerate((1.0, 3.0)):
        E, V = spectrum(4, 4, 4.0, True, V=V_nn)
        check_T0_result(res, i, degenerate_level(E, V, E[0]), 4, 4, True, project)


@pytest.mark.parametrize("project", [False, True])
def test_G3_N_scan_route_with_degenerate_levels(folders, project):
    res = tp.run_Ne_dependence_gme(4, [3, 4], 1.0, 4.0, output_dir=str(folders), save_pkl=False,
                                   spin_preserving=False, project_doublons=project, **GCG)
    for i, Ne in enumerate((3, 4)):
        E, V = spectrum(4, Ne, 4.0, False)
        states = degenerate_level(E, V, E[0])
        if Ne == 3:
            assert len(states) >= 2
        check_T0_result(res, i, states, 4, Ne, False, project)
    assert np.isnan(res["omega"][4][0])                                # n > Ne stays undefined


@pytest.mark.parametrize("project", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
def test_G3_thermal_route(folders, project, workers):
    T = np.array([0.0, 0.5])
    res = tp.run_thermal_gme_analysis(4, 1.0, 4.0, T, n_electrons=4, output_dir=str(folders), save_pkl=False, thermal_energy_cutoff=None,
                                      n_workers=workers, spin_preserving=True, project_doublons=project, **GCG)
    assert res["project_doublons"] is project
    E, V = spectrum(4, 4, 4.0, True)
    weights = [np.array([1.0 if k == 0 else 0.0 for k in range(len(E))]), boltzmann(E, 0.5)]
    for i, p in enumerate(weights):
        for n in res["n_values"]:
            clusters = generate_site_clusters(4, n) if n < 4 else [tuple(range(4))]
            G = np.zeros((2 ** n, 2 ** n), dtype=complex)
            for sites in clusters:
                build = lambda v: rdm_old(v * mask_of(sites, 4, 4), sites, 4, 4) if project else rdm_old(v, sites, 4, 4)
                G += sum(pk * build(V[:, k]) for k, pk in enumerate(p) if pk > 0) / len(clusters)
            w = np.trace(G).real
            assert res["omega"][n][i] == pytest.approx(w, abs=1e-10)
            assert res["purity"][n][i] == pytest.approx(np.trace((G / w) @ (G / w)).real, abs=1e-10)


def test_G3_caches_are_kept_apart_end_to_end(folders):
    run = lambda project: tp.run_U_dependence_gme(4, np.array([2.0]), 1.0, n_electrons=4, output_dir=str(folders), save_pkl=True, use_pkl_cache=True,
                                                  spin_preserving=True, project_doublons=project, **GCG)
    plain, projected = run(False), run(True)
    assert not projected.get("loaded_from_cache")
    assert projected["omega"][2][0] < plain["omega"][2][0] - 1e-3
    again = run(True)
    assert again["loaded_from_cache"] and "_proj1" in again["cache_path"]
    assert again["omega"][2][0] == projected["omega"][2][0]
    assert run(False)["omega"][2][0] == plain["omega"][2][0]


def test_G3_thermal_banks_are_kept_apart_end_to_end(folders):
    T = np.array([0.5])
    run = lambda project: tp.run_thermal_gme_analysis(4, 1.0, 4.0, T, n_electrons=4, output_dir=str(folders), save_pkl=True, use_pkl_cache=True,
                                                      spin_preserving=True, project_doublons=project, **GCG)
    plain, projected = run(False), run(True)
    assert not projected["loaded_rdm_bank_cache"]
    assert "_proj1" in projected["rdm_bank_cache_path"] and "_proj1" not in plain["rdm_bank_cache_path"]
    assert projected["omega"][2][0] < plain["omega"][2][0] - 1e-3
    assert run(True)["loaded_from_cache"]


def test_G3_a_bank_with_the_wrong_flag_is_not_loaded(folders):
    kwargs = dict(L=4, t=1.0, U=4.0, n_electrons=4, spin_preserving=True, use_pkl_cache=True, verbose=False)
    built = tp.build_or_load_rdm_bank(project_doublons=True, **kwargs)
    payload = pickle.load(open(built["cache_path"], "rb"))
    payload["project_doublons"] = False
    pickle.dump(payload, open(built["cache_path"], "wb"))
    assert not tp.build_or_load_rdm_bank(project_doublons=True, **kwargs)["loaded_from_cache"]


# ── main.py ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cli(folders, monkeypatch):
    spec = importlib.util.spec_from_file_location("hubbard_main_under_test", REPO / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["main.py", *argv, "--output-dir", str(folders)])
        module.main()
    return run


COMMON = ["--L", "4", "--n-restarts", "2", "--max-iter", "30", "--k-max-extra", "0", "--workers", "1", "--use-pkl-cache"]


@pytest.mark.parametrize("argv,pattern,key", [
    (["--u-scan", "--Ne", "4", "--u-scan-min", "1", "--u-scan-max", "4", "--u-scan-steps", "2"], "uscan_*", "U_values"),
    (["--v-scan", "--Ne", "4", "--U", "4", "--v-scan-min", "1", "--v-scan-max", "3", "--v-scan-steps", "2"], "vscan_*", "V_values"),
    (["--n-scan", "--U", "4", "--n-scan-values", "3", "4"], "nscan_*", "N_values"),
    (["--thermal", "--Ne", "4", "--U", "4", "--T-min", "0", "--T-max", "0.5", "--T-steps", "2"], "thermal_*", "T_values"),
])
def test_G3_command_line(folders, cli, argv, pattern, key):
    cli(*argv, *COMMON)
    cli(*argv, *COMMON, "--project-doublons")
    files = sorted((folders / "runs").glob(pattern))
    assert len(files) == 2
    old, new = [f for f in files if "_proj1" not in f.name], [f for f in files if "_proj1" in f.name]
    assert len(old) == len(new) == 1
    old, new = pickle.load(open(old[0], "rb")), pickle.load(open(new[0], "rb"))
    assert old["result"]["project_doublons"] is False and new["result"]["project_doublons"] is True
    assert new["meta"]["project_doublons"] is True
    i = len(old["result"][key]) - 1
    assert new["result"]["omega"][2][i] < old["result"]["omega"][2][i] - 1e-3


@pytest.mark.parametrize("argv", [
    ["--project-doublons"],
    ["--t-scan", "--project-doublons"],
    ["--tT-surface", "--project-doublons"],
    ["--build-bank", "--project-doublons"],
    ["--excited-singlet", "--project-doublons"],
    ["--scan-U", "--project-doublons"],
    ["--u-scan", "--t-scan", "--project-doublons"],
    ["--u-scan", "--v-scan", "--project-doublons"],
])
def test_G3_unsupported_modes_are_refused(cli, argv):
    with pytest.raises(SystemExit) as stop:
        cli(*argv)
    assert stop.value.code == 2


# ── tables and production data ───────────────────────────────────────────────

def test_G4_tables_lib_matches_the_stored_omega(folders):
    from tables_lib import omega, rho_s
    res = tp.run_U_dependence_gme(4, np.array([3.0]), 1.0, n_electrons=4, output_dir=str(folders), save_pkl=False,
                                  spin_preserving=True, project_doublons=True, **GCG)
    E, V = spectrum(4, 4, 3.0, True)
    psis = degenerate_level(E, V, E[0])
    for n in res["n_values"]:
        assert omega(psis, 4, 4, n, True, project_doublons=True) == pytest.approx(res["omega"][n][0], abs=1e-10)
        rho = rho_s(psis, 4, 4, n, True, project_doublons=True)
        assert np.trace(rho @ rho).real == pytest.approx(res["purity"][n][0], abs=1e-10)
    plain = omega(psis, 4, 4, 2, True)
    assert plain == pytest.approx(product_occupation(psis[0], (0, 1), 4, 4), abs=1e-10)


def test_G5_flag_off_reproduces_the_paper_rdms():
    """Deterministic quantities of Fig. 1(a); the GCG values are checked at production level with check_projection_bound.py."""
    path = RUNS / "uscan_v1_L6_Ne6_t1_mode-uniform_nmin2_solver-gcg_sz0_restarts10_iter200_kmax4_tol1em09_U1to20_N20-d03a74da789e.pkl"
    if not path.exists():
        pytest.skip("Fig. 1 PKL not available here")
    r = pickle.load(open(path, "rb"))["result"]
    i = int(np.argmin(np.abs(np.asarray(r["U_values"]) - 10.0)))
    psi = ground(6, 6, 10.0)
    for n in range(2, 7):
        G = rdm_old(psi, tuple(range(n)), 6, 6)
        rho = G / np.trace(G).real
        assert np.trace(rho @ rho).real == pytest.approx(float(r["purity"][n][i]), abs=1e-9)
    assert float(r["C_GME"][6][i]) == pytest.approx(0.958, abs=1e-3)


def test_G6_bound_checker_passes_and_detects_errors(folders):
    from check_projection_bound import check
    run = lambda project: tp.run_U_dependence_gme(4, np.array([1.0, 4.0]), 1.0, n_electrons=4, output_dir=str(folders), save_pkl=True, use_pkl_cache=False,
                                                  solver="gcg", n_restarts=6, max_iter=200, tol=1e-9, k_max_extra=2, verbose=False,
                                                  spin_preserving=True, project_doublons=project)
    old, new = run(False)["pkl_path"], run(True)["pkl_path"]
    rows, bad = check(old, new, tol=2e-3)
    assert bad == 0 and len(rows) == 2 * 3
    payload = pickle.load(open(new, "rb"))
    payload["result"]["omega"][2][0] += 1e-3
    pickle.dump(payload, open(new, "wb"))
    assert check(old, new, tol=2e-3)[1] == 1                           # a wrong stored omega is caught
    payload["result"]["omega"][2][0] -= 1e-3
    payload["result"]["C_GME"][3][1] = 1e-4
    pickle.dump(payload, open(new, "wb"))
    assert check(old, new, tol=2e-3)[1] >= 1                           # a violated bound is caught
    with pytest.raises(ValueError):
        check(new, new)
