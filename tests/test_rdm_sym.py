"""
Tests for rdm_sym.compute_nRDM_symmetric: the n-spin RDM from its independent elements
(S_z blocks, hermiticity, spin flip), against the full construction rdm.compute_general_nRDM.

Strictness: every comparison is element by element with an absolute tolerance of 1e-12
(the values are O(1)); structural properties (zeros outside the S_z blocks, hermiticity,
spin-flip copies) are checked exactly. States are chosen to satisfy and to violate each
assumption, and the fallbacks are checked to give the full result.
"""
import itertools
import pickle
from math import comb
from pathlib import Path

import numpy as np
import pytest
from openfermion import FermionOperator
from openfermion.linalg import get_number_preserving_sparse_operator

import hubbard_gme.rdm_sym as rs
from hubbard_gme.hamiltonian import compute_k_eigenstates
from hubbard_gme.rdm import compute_general_nRDM, compute_thermal_nRDM, generate_site_clusters

TOL = 1e-12


def assert_same(G, R, tol=TOL):
    err = np.max(np.abs(G - R))
    assert err < tol, f"max |G - ref| = {err:.2e}"


# ═══════════════════════════════ helpers: states ═══════════════════════════════

def eigenstates(L, U, mode="uniform", Ne=None, k=3, spin_preserving=True, **kw):
    Ne = L if Ne is None else Ne
    E, V, nso = compute_k_eigenstates(L, 1.0, U, mode, Ne, k=k, spin_preserving=spin_preserving, **kw)
    return E, V, nso, Ne


def random_hamiltonian(L, seed, zeeman=0.0, flux=0.0, U_scale=4.0):
    """Spin-independent random hoppings (all pairs), on-site energies and U_i; optional
    site-dependent Zeeman field h_i (n_up - n_dn), which breaks the up/down exchange, and
    optional complex hopping phases (a flux), which make the eigenstates complex."""
    rng = np.random.default_rng(seed)
    H = FermionOperator()
    for i, j in itertools.combinations(range(L), 2):
        t = rng.uniform(0.3, 1.2) * np.exp(1j * flux * rng.uniform(-np.pi, np.pi))
        for s in (0, 1):
            H += FermionOperator(((2 * i + s, 1), (2 * j + s, 0)), -t)
            H += FermionOperator(((2 * j + s, 1), (2 * i + s, 0)), -np.conj(t))
    for i in range(L):
        eps, Ui, h = rng.uniform(-1, 1), rng.uniform(0.5, 1.5) * U_scale, zeeman * rng.uniform(0.5, 1.5)
        H += FermionOperator(((2 * i, 1), (2 * i, 0)), eps + h)
        H += FermionOperator(((2 * i + 1, 1), (2 * i + 1, 0)), eps - h)
        H += FermionOperator(((2 * i, 1), (2 * i, 0), (2 * i + 1, 1), (2 * i + 1, 0)), Ui)
    return H


def dense_eigenstates(H, L, Ne, spin_preserving=True):
    M = get_number_preserving_sparse_operator(H, 2 * L, Ne, spin_preserving=spin_preserving).toarray()
    E, V = np.linalg.eigh(M)
    return E, V


# ═══════════════════════════════ combinatorics ═══════════════════════════════

KNOWN = {1: (4, 2, 2, 1), 2: (16, 6, 5, 3), 3: (64, 20, 14, 7), 4: (256, 70, 43, 23),
         5: (1024, 252, 142, 71), 6: (4096, 924, 494, 252), 7: (16384, 3432, 1780, 890),
         8: (65536, 12870, 6563, 3299), 9: (262144, 48620, 24566, 12283)}


@pytest.mark.parametrize("n", sorted(KNOWN))
def test_element_counts(n):
    c = rs.count_elements(n)
    assert (c["full"], c["sz_blocks"], c["sz_hermitian"], c["sz_hermitian_flip"]) == KNOWN[n]
    assert c["sz_blocks"] == comb(2 * n, n) == sum(comb(n, d) ** 2 for d in range(n + 1))
    assert c["sz_hermitian"] == sum(comb(n, d) * (comb(n, d) + 1) // 2 for d in range(n + 1))


@pytest.mark.parametrize("n", range(1, 9))
@pytest.mark.parametrize("use_flip", [False, True])
def test_independent_pairs_generate_exactly_the_blocks(n, use_flip):
    """Closing the computed set under hermiticity (and spin flip) gives every element inside the
    S_z blocks and nothing outside; no pair is listed twice; bra <= ket."""
    pairs = rs.independent_pairs(n, use_sz=True, use_spin_flip=use_flip)
    assert len(pairs) == len(set(pairs))
    assert all(a <= b for a, b in pairs)
    assert all(rs.n_down(a) == rs.n_down(b) for a, b in pairs)
    reached = set()
    for a, b in pairs:
        orbit = {(a, b), (b, a)}
        if use_flip:
            orbit |= {(rs.flip(x, n), rs.flip(y, n)) for x, y in orbit}
        reached |= orbit
    blocks = {(a, b) for a in range(1 << n) for b in range(1 << n) if rs.n_down(a) == rs.n_down(b)}
    assert reached == blocks
    if use_flip:                                       # no two computed pairs in the same orbit
        reps = {}
        for a, b in pairs:
            key = frozenset({(a, b), (b, a), (rs.flip(a, n), rs.flip(b, n)), (rs.flip(b, n), rs.flip(a, n))})
            assert key not in reps, f"{(a, b)} and {reps.get(key)} are copies of each other"
            reps[key] = (a, b)


def test_without_sz_all_upper_pairs():
    for n in range(1, 6):
        assert len(rs.independent_pairs(n, use_sz=False)) == (4 ** n + 2 ** n) // 2


def test_flip_and_blocks_helpers():
    assert rs.flip(0b0110, 4) == 0b1001 and rs.flip(rs.flip(13, 5), 5) == 13
    for n in range(1, 8):
        blocks = rs.sz_blocks(n)
        assert [len(b) for b in blocks] == [comb(n, d) for d in range(n + 1)]
        assert sorted(sum(blocks, [])) == list(range(1 << n))


# ═══════════════════════════ agreement with the full RDM ═══════════════════════════

TOPOLOGIES = [("uniform", {}), ("open", {}), ("nnn", dict(t_prime=0.3)),
              ("ssh", dict(t2=0.6)), ("ladder", dict(t_perp=0.8))]


@pytest.mark.parametrize("mode,kw", TOPOLOGIES)
@pytest.mark.parametrize("U", [0.7, 4.0, 10.0])
def test_topologies_L4_all_clusters(mode, kw, U):
    """Every cluster of every size (contiguous or not), ground and two excited states."""
    E, V, nso, Ne = eigenstates(4, U, mode, k=3, **kw)
    for state in range(3):
        psi = V[:, state]
        for n in range(1, 5):
            for sites in itertools.combinations(range(4), n):
                G = rs.compute_nRDM_symmetric(sites, psi, nso, Ne)
                assert_same(G, compute_general_nRDM(list(sites), psi, nso, Ne))


@pytest.mark.parametrize("U", [1.0, 4.0, 10.0])
@pytest.mark.parametrize("state", [0, 1, 2])
def test_ring_L6_contiguous_and_scattered(U, state):
    E, V, nso, Ne = eigenstates(6, U, k=3)
    psi = V[:, state]
    for sites in [(0,), (0, 1), (2, 3, 4), (0, 1, 2, 3), (1, 2, 3, 4, 5), (0, 2), (5, 1, 3), (4, 0, 2, 1)]:
        G, info = rs.compute_nRDM_symmetric(sites, psi, nso, Ne, return_info=True)
        assert_same(G, compute_general_nRDM(list(sites), psi, nso, Ne))
        assert info["use_sz"]


def test_ring_L6_n_equals_L():
    E, V, nso, Ne = eigenstates(6, 4.0, k=1)
    sites = list(range(6))
    assert_same(rs.compute_nRDM_symmetric(sites, V[:, 0], nso, Ne), compute_general_nRDM(sites, V[:, 0], nso, Ne))


def test_doped_and_extended():
    """Away from half filling (K = 4 on L = 6) and with nearest-neighbour V."""
    for Ne, V_ in [(4, 0.0), (6, 3.0)]:
        E, V, nso = compute_k_eigenstates(6, 1.0, 6.0, "uniform", Ne, k=2, V=V_)
        for state in range(2):
            for sites in [(0, 1, 2), (0, 1, 2, 3)]:
                assert_same(rs.compute_nRDM_symmetric(sites, V[:, state], nso, Ne),
                            compute_general_nRDM(list(sites), V[:, state], nso, Ne))


@pytest.mark.parametrize("seed", range(6))
def test_random_spin_symmetric_hamiltonian(seed):
    """No translation symmetry, generic couplings: spin flip must be detected as valid."""
    L, Ne = 5, 4
    E, V = dense_eigenstates(random_hamiltonian(L, seed), L, Ne)
    assert E[1] - E[0] > 1e-6
    for state in (0, 1, 2):
        psi = V[:, state]
        for sites in [(0, 1, 2), (4, 1, 3, 0), (0, 1, 2, 3, 4)]:
            G, info = rs.compute_nRDM_symmetric(sites, psi, 2 * L, Ne, return_info=True)
            assert_same(G, compute_general_nRDM(list(sites), psi, 2 * L, Ne))
            assert info["use_spin_flip"] is True


@pytest.mark.parametrize("seed", range(4))
def test_complex_states_with_flux(seed):
    """Complex hopping phases: complex state; the 3- and 4-spin RDMs get imaginary parts
    (scalar spin chirality), so hermiticity must use the conjugate. The 2-spin RDM stays real."""
    L, Ne = 4, 4
    E, V = dense_eigenstates(random_hamiltonian(L, seed, flux=0.8), L, Ne)
    psi = V[:, 0]
    assert np.max(np.abs(psi.imag)) > 1e-3
    for sites in [(0, 1), (0, 1, 2), (3, 0, 2, 1)]:
        G, info = rs.compute_nRDM_symmetric(sites, psi, 2 * L, Ne, return_info=True)
        R = compute_general_nRDM(list(sites), psi, 2 * L, Ne)
        if len(sites) >= 3:
            assert np.max(np.abs(R.imag)) > 1e-3
        assert_same(G, R)
        assert info["use_spin_flip"] is True


# ═══════════════════════════ assumptions violated: fallbacks ═══════════════════════════

@pytest.mark.parametrize("seed", range(4))
def test_zeeman_field_breaks_spin_flip(seed):
    """A site-dependent Zeeman field keeps S_z but breaks up/down exchange: 'auto' must
    detect it and still give the full result; forcing spin_flip=True must be wrong."""
    L, Ne = 5, 4
    E, V = dense_eigenstates(random_hamiltonian(L, seed, zeeman=0.4), L, Ne)
    psi = V[:, 0]
    sites = (0, 1, 2, 3)
    R = compute_general_nRDM(list(sites), psi, 2 * L, Ne)
    G, info = rs.compute_nRDM_symmetric(sites, psi, 2 * L, Ne, return_info=True)
    assert info["use_spin_flip"] is False and info["use_sz"] is True
    assert_same(G, R)
    assert_same(rs.compute_nRDM_symmetric(sites, psi, 2 * L, Ne, spin_flip=False), R)
    forced = rs.compute_nRDM_symmetric(sites, psi, 2 * L, Ne, spin_flip=True)
    assert np.max(np.abs(forced - R)) > 1e-6


@pytest.mark.parametrize("seed", range(4))
def test_random_vector_in_sector(seed):
    """Not an eigenstate of anything: S_z blocks still hold (sector), spin flip does not."""
    E, V, nso, Ne = eigenstates(4, 4.0, k=1)
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=V.shape[0]) + 1j * rng.normal(size=V.shape[0])
    psi /= np.linalg.norm(psi)
    for sites in [(0, 1), (0, 1, 2), (0, 1, 2, 3)]:
        G, info = rs.compute_nRDM_symmetric(sites, psi, nso, Ne, return_info=True)
        assert_same(G, compute_general_nRDM(list(sites), psi, nso, Ne))
        assert info["use_spin_flip"] is False


def test_odd_electron_number():
    """L = 3, Ne = 3: S_z = +1/2 sector, the spin flip maps it to another sector."""
    E, V, nso = compute_k_eigenstates(3, 1.0, 4.0, "uniform", 3, k=2)
    for state in range(2):
        for sites in [(0, 1), (0, 1, 2)]:
            G, info = rs.compute_nRDM_symmetric(sites, V[:, state], nso, 3, return_info=True)
            assert_same(G, compute_general_nRDM(list(sites), V[:, state], nso, 3))
            assert info["use_spin_flip"] is False


def test_full_space_sz_definite_and_mixed():
    """spin_preserving=False (all S_z): a mixed-S_z vector has coherences between S_z blocks,
    which must be kept (no S_z shortcut); an S_z = 0 singlet ground state may use both."""
    L, Ne = 4, 4
    E, V, nso = compute_k_eigenstates(L, 1.0, 4.0, "uniform", Ne, k=8, spin_preserving=False)
    rng = np.random.default_rng(3)
    mixed = rng.normal(size=V.shape[0]); mixed /= np.linalg.norm(mixed)
    sites = (0, 1, 2)
    R = compute_general_nRDM(list(sites), mixed, nso, Ne, spin_preserving=False)
    same = np.array([[rs.n_down(a) == rs.n_down(b) for b in range(8)] for a in range(8)])
    assert np.max(np.abs(R[~same])) > 1e-3                     # the test case has cross-block terms
    G, info = rs.compute_nRDM_symmetric(sites, mixed, nso, Ne, spin_preserving=False, return_info=True)
    assert info["use_sz"] is False and info["sz_variance"] > 1e-3
    assert_same(G, R)
    for state in range(8):                                     # eigenstates, incl. degenerate multiplets
        psi = V[:, state]
        G, info = rs.compute_nRDM_symmetric(sites, psi, nso, Ne, spin_preserving=False, return_info=True)
        assert_same(G, compute_general_nRDM(list(sites), psi, nso, Ne, spin_preserving=False))


def test_degenerate_multiplet_mixture_in_sector():
    """An arbitrary combination inside a degenerate S_z = 0 manifold need not be flip-symmetric."""
    L, Ne = 4, 4
    E, V, nso = compute_k_eigenstates(L, 1.0, 0.0, "uniform", Ne, k=12)      # U = 0: degenerate
    E0 = E[0]
    deg = np.where(np.abs(E - E0) < 1e-8)[0]
    assert len(deg) > 1
    rng = np.random.default_rng(1)
    c = rng.normal(size=len(deg)) + 1j * rng.normal(size=len(deg))
    psi = V[:, deg] @ (c / np.linalg.norm(c))
    for sites in [(0, 1), (0, 1, 2), (0, 1, 2, 3)]:
        assert_same(rs.compute_nRDM_symmetric(sites, psi, nso, Ne),
                    compute_general_nRDM(list(sites), psi, nso, Ne))


# ═══════════════════════════ thermal sums and the RDM bank ═══════════════════════════

@pytest.mark.parametrize("T", [0.0, 0.3, 2.0])
def test_thermal_sum_matches_compute_thermal_nRDM(T):
    """Gamma(T) = sum_m p_m Gamma[psi_m] over the whole S_z = 0 spectrum (degenerate states
    included), normalised, against rdm.compute_thermal_nRDM."""
    L, Ne = 4, 4
    E, V, nso = compute_k_eigenstates(L, 1.0, 2.0, "uniform", Ne, k=35)
    sites = [0, 1, 2]
    ref = compute_thermal_nRDM(sites, E, V, T, nso, Ne) if T > 0 else compute_thermal_nRDM(sites, E, V, 1e-15, nso, Ne)
    if T > 0:
        w = np.exp(-(E - E[0]) / T); w /= w.sum()
    else:
        g = np.abs(E - E[0]) < 1e-10; w = g / g.sum()
    G = sum(p * rs.compute_nRDM_symmetric(sites, V[:, m], nso, Ne) for m, p in enumerate(w) if p > 1e-14)
    G = G / np.trace(G).real
    assert_same(G, ref)


BANK = Path(__file__).resolve().parents[1] / "results/rdm_pkl_results/rdm_bank"
BANK_DROPBOX = Path("/mnt/d/Dropbox/PhDWork/hubbard_gme/results/rdm_pkl_results/rdm_bank")


def _bank(U):
    name = f"rdmbank_v1_L6_Ne6_U{U}_t1_mode-uniform_tpall_t2all_tperpall_nmin2_sz0_clusteravg1.pkl"
    for d in (BANK, BANK_DROPBOX):
        if (d / name).exists():
            return pickle.load(open(d / name, "rb"))
    return None


@pytest.mark.parametrize("U", [4, 10])
def test_cluster_average_equals_rdm_bank(U):
    bank = _bank(U)
    if bank is None:
        pytest.skip("RDM bank not available")
    E, V, nso, Ne = eigenstates(6, float(U), k=2)
    for state in (0, 1):
        assert abs(E[state] - bank["energies"][state]) < 1e-9
        for n in (2, 3, 4):
            G = sum(rs.compute_nRDM_symmetric(c, V[:, state], nso, Ne) for c in generate_site_clusters(6, n)) / 6
            assert_same(G, bank["bank"][n][state], tol=1e-10)


# ═══════════════════════════ structure of the output ═══════════════════════════

def test_exact_structure():
    """Off-block zeros are exact, hermiticity is exact (copies), spin-flip copies are exact."""
    E, V, nso, Ne = eigenstates(6, 10.0, k=1)
    for n in (2, 3, 4, 5):
        G, info = rs.compute_nRDM_symmetric(list(range(n)), V[:, 0], nso, Ne, return_info=True)
        assert info["use_spin_flip"]
        d = np.array([rs.n_down(a) for a in range(1 << n)])
        assert np.all(G[d[:, None] != d[None, :]] == 0)
        assert np.array_equal(G, G.conj().T)
        f = np.array([rs.flip(a, n) for a in range(1 << n)])
        assert np.array_equal(G[np.ix_(f, f)], G)
        assert abs(np.trace(G).real - np.trace(compute_general_nRDM(list(range(n)), V[:, 0], nso, Ne)).real) < TOL


def test_number_of_evaluations(monkeypatch):
    """The element evaluator is called exactly once per independent element (plus the flip probes)."""
    calls = []
    real = rs.gamma_element
    monkeypatch.setattr(rs, "gamma_element", lambda *a, **k: (calls.append(a[1:3]), real(*a, **k))[1])
    E, V, nso, Ne = eigenstates(6, 4.0, k=1)
    for n in (3, 4, 5):
        for mode, expected in ((True, rs.count_elements(n)["sz_hermitian_flip"]),
                               (False, rs.count_elements(n)["sz_hermitian"])):
            calls.clear()
            G, info = rs.compute_nRDM_symmetric(list(range(n)), V[:, 0], nso, Ne, spin_flip=mode, return_info=True)
            assert len(calls) == expected == info["evaluations"]
            assert len(set(calls)) == len(calls)
        calls.clear()
        G, info = rs.compute_nRDM_symmetric(list(range(n)), V[:, 0], nso, Ne, spin_flip="auto", return_info=True)
        assert len(calls) == info["evaluations"] == len(set(calls))
        assert len(calls) <= rs.count_elements(n)["sz_hermitian_flip"] + 2 * (n // 2 + 1) + 2 * 12


# ═══════════════════════════ input handling ═══════════════════════════

def test_input_validation():
    E, V, nso, Ne = eigenstates(4, 4.0, k=1)
    psi = V[:, 0]
    with pytest.raises(ValueError):
        rs.compute_nRDM_symmetric([0, 0], psi, nso, Ne)
    with pytest.raises(ValueError):
        rs.compute_nRDM_symmetric([0, 4], psi, nso, Ne)
    with pytest.raises(ValueError):
        rs.compute_nRDM_symmetric([-1, 0], psi, nso, Ne)
    with pytest.raises(ValueError):
        rs.compute_nRDM_symmetric([0, 1], psi, nso, Ne, spin_flip="yes")
    assert np.array_equal(rs.compute_nRDM_symmetric([], psi, nso, Ne), np.array([[1.0 + 0j]]))
    assert np.array_equal(rs.compute_nRDM_symmetric([], psi, nso, Ne), compute_general_nRDM([], psi, nso, Ne))


def test_n1_two_elements():
    """n = 1: Gamma = diag(<n_up>, <n_dn>), equal in an S_z = 0 flip-symmetric state."""
    E, V, nso, Ne = eigenstates(4, 4.0, k=1)
    G = rs.compute_nRDM_symmetric([2], V[:, 0], nso, Ne)
    assert G[0, 1] == 0 and G[1, 0] == 0 and G[0, 0] == G[1, 1]
    assert abs(G[0, 0] - 0.5) < 1e-10                              # half filling: <n_up> = 1/2


# ═══════════════════════════ speed (slow) ═══════════════════════════

@pytest.mark.slow
def test_speedup_L8():
    import time
    E, V, nso, Ne = eigenstates(8, 10.0, k=1)
    sites = list(range(6))
    t0 = time.time(); R = compute_general_nRDM(sites, V[:, 0], nso, Ne); t_ref = time.time() - t0
    t0 = time.time(); G = rs.compute_nRDM_symmetric(sites, V[:, 0], nso, Ne); t_sym = time.time() - t0
    assert_same(G, R)
    assert t_ref / t_sym > 8, f"speed-up only {t_ref / t_sym:.1f}x"
