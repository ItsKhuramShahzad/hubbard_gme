"""
rdm_sym.py
----------
The n-spin RDM of rdm.compute_general_nRDM, built from its independent elements only.

    Gamma^(n)[a, b] = < Psi | prod_k c^dag_{2 s_k + a_k} c_{2 s_k + b_k} | Psi >,   a_k = (a >> k) & 1

Three exact properties (docs/sz_block_rdm.tex):

  1. S_z blocks   Gamma[a, b] = 0 unless d(a) = d(b), d = number of down spins
                  (valid when |Psi> has a definite total S_z)
  2. Hermiticity  Gamma[b, a] = conj(Gamma[a, b])
  3. Spin flip    Gamma[abar, bbar] = Gamma[a, b], abar = a with all spins flipped
                  (valid when F|Psi> = e^{i phi}|Psi>, F = exchange of up and down on every site,
                  e.g. a non-degenerate S_z = 0 eigenstate of a spin-symmetric Hamiltonian)

Each computed element is evaluated exactly as in compute_general_nRDM (same operator string, same
OpenFermion sparse operator), so the result agrees with the full construction to rounding.

Property 1 is used only if the state has a definite S_z: automatic in the S_z sector
(spin_preserving=True), checked from <S_z^2> - <S_z>^2 otherwise. Property 3 is applied with
spin_flip="auto" (default) only after spot checks on the state itself; if any check fails, the
blocks d > n/2 are computed as well, so a degenerate multiplet or a spin-dependent field never
gives a wrong RDM. spin_flip=True skips the checks, spin_flip=False never uses the symmetry.
"""
from __future__ import annotations

from math import comb
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from openfermion import FermionOperator, sz_operator
from openfermion.linalg import get_number_preserving_sparse_operator

Pair = Tuple[int, int]


# ── combinatorics (no physics) ───────────────────────────────────────────────

def n_down(idx: int) -> int:
    """Number of down spins in a cluster configuration (bit = 1 means down)."""
    return bin(idx).count("1")


def flip(idx: int, n: int) -> int:
    """Configuration with every spin flipped."""
    return idx ^ ((1 << n) - 1)


def sz_blocks(n: int) -> List[List[int]]:
    """Configurations grouped by number of down spins, blocks[d] sorted ascending."""
    blocks: List[List[int]] = [[] for _ in range(n + 1)]
    for idx in range(1 << n):
        blocks[n_down(idx)].append(idx)
    return blocks


def independent_pairs(n: int, use_sz: bool = True, use_spin_flip: bool = True) -> List[Pair]:
    """The (bra, ket) elements that are actually evaluated, each with bra <= ket.

    use_sz=False: all pairs a <= b (hermiticity only).
    use_spin_flip: blocks d < n/2 fully (upper triangle); for even n, the middle block keeps one
    representative of each orbit {(a, b), (abar, bbar)} (up to hermiticity)."""
    if not use_sz:
        return [(a, b) for a in range(1 << n) for b in range(a, 1 << n)]
    pairs: List[Pair] = []
    for d, block in enumerate(sz_blocks(n)):
        if use_spin_flip and 2 * d > n:
            continue
        for i, a in enumerate(block):
            for b in block[i:]:
                if use_spin_flip and 2 * d == n:
                    fa, fb = flip(a, n), flip(b, n)
                    if (min(fa, fb), max(fa, fb)) < (a, b):
                        continue                    # its mirror is the representative
                pairs.append((a, b))
    return pairs


def count_elements(n: int) -> Dict[str, int]:
    """Number of elements per level of reduction (full, S_z blocks, + hermiticity, + spin flip)."""
    return dict(full=4 ** n,
                sz_blocks=comb(2 * n, n),
                sz_hermitian=len(independent_pairs(n, use_sz=True, use_spin_flip=False)),
                sz_hermitian_flip=len(independent_pairs(n, use_sz=True, use_spin_flip=True)))


# ── one element, exactly as in rdm.compute_general_nRDM ──────────────────────

def gamma_element(sites: Sequence[int], bra_idx: int, ket_idx: int, psi: np.ndarray,
                  n_spin_orbitals: int, n_electrons: int, spin_preserving: bool = True) -> complex:
    op_terms: list[tuple[int, int]] = []
    for body, site in enumerate(sites):
        op_terms.append((2 * site + ((bra_idx >> body) & 1), 1))     # c^dag_p
        op_terms.append((2 * site + ((ket_idx >> body) & 1), 0))     # c_q
    op = get_number_preserving_sparse_operator(FermionOperator(tuple(op_terms)), n_spin_orbitals,
                                               n_electrons, spin_preserving=spin_preserving)
    return complex(np.vdot(psi, op @ psi))


# ── checks on the state ──────────────────────────────────────────────────────

def sz_moments(psi: np.ndarray, n_spin_orbitals: int, n_electrons: int,
               spin_preserving: bool = True) -> Tuple[float, float]:
    """<S_z> and <S_z^2> - <S_z>^2 of the state (orbital 2j = up, 2j+1 = down)."""
    Sz = get_number_preserving_sparse_operator(sz_operator(n_spin_orbitals // 2), n_spin_orbitals,
                                               n_electrons, spin_preserving=spin_preserving)
    v = Sz @ psi
    norm = float(np.real(np.vdot(psi, psi)))
    mean = float(np.real(np.vdot(psi, v))) / norm
    var = float(np.real(np.vdot(v, v))) / norm - mean ** 2
    return mean, var


def _flip_probe_pairs(n: int, n_random: int, seed: int) -> List[Pair]:
    """Pairs (a, b) with d <= n/2 used to test Gamma[abar, bbar] = Gamma[a, b]: from every block
    the first diagonal element and, where it exists, the first off-diagonal one; plus random ones."""
    blocks = sz_blocks(n)
    probe: List[Pair] = []
    for d in range(n // 2 + 1):
        blk = blocks[d]
        probe.append((blk[0], blk[0]))
        if len(blk) > 1:
            probe.append((blk[0], blk[1]))
    rng = np.random.default_rng(seed)
    cand = [(a, b) for d in range(n // 2 + 1) for a in blocks[d] for b in blocks[d] if a <= b]
    for k in rng.choice(len(cand), size=min(n_random, len(cand)), replace=False):
        probe.append(cand[int(k)])
    return [p for p in dict.fromkeys(probe) if (flip(p[0], n), flip(p[1], n)) != p]


# ── the RDM ──────────────────────────────────────────────────────────────────

def compute_nRDM_symmetric(
    orbital_indices: Sequence[int],
    psi: np.ndarray,
    n_spin_orbitals: int,
    n_electrons: int,
    spin_preserving: bool = True,
    spin_flip: Union[bool, str] = "auto",
    n_flip_checks: int = 12,
    tol: float = 1e-10,
    seed: int = 0,
    return_info: bool = False,
):
    """Unnormalised Gamma^(n), same convention and values as rdm.compute_general_nRDM.

    spin_flip: "auto" (spot-check the relation on the state, fall back if it fails), True
    (assume it), False (never use it). tol: absolute tolerance of the S_z and spin-flip checks."""
    sites = [int(s) for s in orbital_indices]
    n = len(sites)
    if n == 0:
        G = np.array([[1.0 + 0j]])
        return (G, dict(evaluations=0, use_sz=False, use_spin_flip=False)) if return_info else G
    if len(set(sites)) != n:
        raise ValueError(f"orbital_indices must be distinct, got {sites}")
    if min(sites) < 0 or max(sites) >= n_spin_orbitals // 2:
        raise ValueError(f"sites {sites} outside 0 .. {n_spin_orbitals // 2 - 1}")
    if spin_flip not in (True, False, "auto"):
        raise ValueError("spin_flip must be True, False or 'auto'")
    psi = np.asarray(psi)

    def elem(a: int, b: int) -> complex:
        return gamma_element(sites, a, b, psi, n_spin_orbitals, n_electrons, spin_preserving)

    info: Dict[str, object] = dict(n=n, evaluations=0)

    # property 1: definite S_z ?
    if spin_preserving:
        use_sz, sz_mean, sz_var = True, None, 0.0
    else:
        sz_mean, sz_var = sz_moments(psi, n_spin_orbitals, n_electrons, spin_preserving=False)
        use_sz = sz_var < tol
    info.update(use_sz=use_sz, sz_variance=sz_var)

    # property 3: spin flip ?
    cache: Dict[Pair, complex] = {}

    def cached(a: int, b: int) -> complex:
        if (a, b) not in cache:
            cache[(a, b)] = elem(a, b)
        return cache[(a, b)]
    if not use_sz or spin_flip is False or n == 0:
        use_flip = False
    elif spin_flip is True:
        use_flip = True
    else:
        if sz_mean is not None and abs(sz_mean) > tol:
            use_flip = False                         # S_z != 0: no flip symmetry
        else:
            use_flip = True
            for a, b in _flip_probe_pairs(n, n_flip_checks, seed):
                g = cached(a, b)
                fa, fb = flip(a, n), flip(b, n)
                key = (min(fa, fb), max(fa, fb))
                gf = cached(*key)
                gf = gf if (fa, fb) == key else np.conj(gf)
                if abs(gf - g) > tol * max(1.0, abs(g)):
                    use_flip = False
                    break
            info["flip_probe_evaluations"] = len(cache)
    info["use_spin_flip"] = use_flip

    pairs = independent_pairs(n, use_sz=use_sz, use_spin_flip=use_flip)
    values = {(a, b): (cache[(a, b)] if (a, b) in cache else elem(a, b)) for a, b in pairs}
    info["evaluations"] = len(set(pairs) | set(cache))
    G = fill_from_independent(values, n, use_sz=use_sz, use_spin_flip=use_flip)
    return (G, info) if return_info else G


def fill_from_independent(values: Dict[Pair, complex], n: int, use_sz: bool = True,
                          use_spin_flip: bool = True) -> np.ndarray:
    """The full 2^n x 2^n matrix from the values of independent_pairs(n, use_sz, use_spin_flip).

    values[(a, b)] = Gamma[a, b] for every pair of independent_pairs(n, use_sz, use_spin_flip), in that
    order. The lower triangle is the complex conjugate (hermiticity); with use_spin_flip the blocks
    d >= n/2 are copied by Gamma[a, b] = Gamma[abar, bbar]; with use_sz everything outside the S_z blocks
    is zero. A missing value raises, so an element can never be silently left at zero."""
    dim = 1 << n
    G = np.zeros((dim, dim), dtype=complex)
    filled = np.zeros((dim, dim), dtype=bool)
    for (a, b), g in values.items():                 # computed elements and their hermitian copies
        G[a, b], G[b, a] = g, np.conj(g)
        filled[a, b] = filled[b, a] = True

    if use_spin_flip:                                # spin-flip copies: middle block first, then d > n/2
        blocks = sz_blocks(n)
        for d in sorted(range(n + 1), key=lambda x: (2 * x != n, x)):
            if 2 * d < n:
                continue
            for a in blocks[d]:
                for b in blocks[d]:
                    if not filled[a, b]:
                        fa, fb = flip(a, n), flip(b, n)
                        if not filled[fa, fb]:
                            raise RuntimeError(f"spin-flip source ({fa}, {fb}) of ({a}, {b}) not filled")
                        G[a, b] = G[fa, fb]
                        filled[a, b] = True

    if use_sz:                                       # everything outside the S_z blocks is zero
        same = np.array([[n_down(a) == n_down(b) for b in range(dim)] for a in range(dim)])
        if not filled[same].all():
            raise RuntimeError("an element inside the S_z blocks was not filled")
    elif not filled.all():
        raise RuntimeError("an element was not filled")
    return G
