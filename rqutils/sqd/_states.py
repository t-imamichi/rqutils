"""Packed state lists: shape checks, filler padding, uniquification and the source-index search."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec, get_abstract_mesh
from numpy.typing import NDArray

# Subspace positions are int32 throughout, so the size is capped and enforced to raise on overflow
# (NOTES.md, "The `N ≤ 2^31 - 1` ceiling: why it is enforced where it is").
_MAX_STATES = 2**31 - 1


type StateList = np.ndarray[tuple[int, int], np.dtype[np.uint8]]


def _check_states_shape(states: Any, num_qubits: int, packed: bool = False) -> StateList:
    """Raise unless ``states`` is ``(subspace_dim, num_qubits)``, or packed-width when ``packed``.

    One ``O(1)`` look at a shape, shared by :func:`sqd` and :func:`hproj` so the two cannot drift. It
    catches re-fed *packed* states (``pack_states`` is not idempotent), a transposed array, and a
    mismatched Hamiltonian, all of which used to return a plausible finite answer (``NOTES.md``,
    "sqd._check_states_shape: three mistakes, and the packed declaration").

    ``packed=True`` expects ``ceil((num_qubits + 1) / 8)`` columns instead. It is a **declaration, not
    an inference**: at ``num_qubits == 1`` both widths are 1, so unpacked states under the flag
    silently return a different eigenvalue. The reverse (packed, flag omitted) and a 1-byte re-feed
    are caught by :meth:`PauliSumXZ.pack_states`' binary check.

    Coerces with :func:`numpy.asarray` and returns the result, so both entry points accept a list of
    lists, and a malformed one gets the documented ``ValueError`` rather than an ``AttributeError``.

    Args:
        states: The caller's states, as anything :func:`numpy.asarray` accepts.
        num_qubits: The Hamiltonian's qubit count.
        packed: Whether ``states`` is already bit-packed, so the expected width is
            ``ceil((num_qubits + 1) / 8)`` rather than ``num_qubits``.

    Returns:
        ``states`` as an array, for the caller to use in place of its argument.

    Raises:
        ValueError: If ``states`` is not 2-D, or its second axis is not the expected width.
    """
    states = np.asarray(states)
    if states.ndim != 2:
        raise ValueError(
            f"`states` must be 2-D with shape (subspace_dim, num_qubits), got shape {states.shape}"
        )
    if packed:
        width = -(-(num_qubits + 1) // 8)
        if states.shape[1] != width:
            raise ValueError(
                f"`states` has {states.shape[1]} columns but `packed=True` with "
                f"{num_qubits} qubits expects {width} (= ceil(({num_qubits} + 1) / 8)). Pass the "
                "output of `PauliSumXZ.pack_states`, or drop `packed=True` for unpacked states."
            )
        if states.dtype != np.uint8:
            raise ValueError(
                f"packed `states` must be uint8, got {states.dtype}. `PauliSumXZ.pack_states` "
                "returns uint8; a wider dtype means the array is not its output."
            )
        return states
    if states.shape[1] != num_qubits:
        raise ValueError(
            f"`states` has {states.shape[1]} columns but the Hamiltonian has {num_qubits} qubits; "
            "`states` must be (subspace_dim, num_qubits) and *unpacked*. Note "
            "`PauliSumXZ.pack_states` is not idempotent, so a packed array kept from a previous call "
            "(or from `uniquify_states`) cannot be fed back in -- it would re-pack into a different "
            "subspace. Also check for a transposed array or a mismatched Hamiltonian."
        )
    return states


def _is_filler(states_u: StateList) -> jax.Array:
    """Return the 0/1 fill-in marker bit of each row of a uniquified state list.

    Filler slots are all-ones rows (``255``); genuine states have byte 0 ``< 128`` because
    :meth:`PauliSumXZ.pack_states` inserts a leading zero pad bit. So the high bit of byte 0 identifies fillers,
    and testing it is equivalent to testing ``states_u[:, 0] == 255`` -- one spelling for all three
    consumers, rather than two that a reader has to re-derive as equal.

    Returned as the bit itself, not a bool, because the fillers sort to the end: the result is a
    non-decreasing 0/1 array, which is what lets ``run_sqd`` locate the subspace boundary with a
    ``searchsorted`` instead of a count.
    """
    return states_u[:, 0] >> 7


def _pad_states(states_p: StateList, states_size: int) -> StateList:
    """``states_p`` padded to ``states_size`` rows with the ``255`` filler :func:`run_sqd` expects.

    An all-ones row sorts last and sets byte 0's high bit, which ``_is_filler`` tests; a genuine
    state never collides, since the pad bit keeps its byte 0 below 128.
    """
    deficit = states_size - states_p.shape[0]
    if deficit < 0:
        raise ValueError(f"{states_p.shape[0]} states exceed states_size={states_size}")
    filler = np.full((deficit, states_p.shape[1]), 255, dtype=np.uint8)
    return np.append(states_p, filler, axis=0)


@jax.jit(static_argnames=["states_size"])
def uniquify_states(states_p: StateList, states_size: int) -> StateList:
    """A stripped-down implementation of jnp.unique.

    The returned array will have shape (states_size, states_p.shape[1]). If states_size is greater
    than the number of unique states, the residual entries at the end are filled with 255.

    Raises:
        ValueError: If ``states_size`` exceeds :data:`_MAX_STATES`, the ceiling imposed by the int32
            iota below.
    """
    # Also checked here, where the int32 iota is created: poc/ calls this directly, and it is free
    # at trace time (NOTES.md, "The `N ≤ 2^31 - 1` ceiling: why it is enforced where it is").
    if states_size > _MAX_STATES:
        raise ValueError(
            f"states_size {states_size} exceeds the {_MAX_STATES} limit imposed by the int32 index "
            "below; beyond it the iota wraps negative and the subspace is silently permuted"
        )
    # Lexsort on uint64 words, not uint8 columns: lax.sort pays per key and the order is identical;
    # it trades memory for speed (NOTES.md, "sqd.uniquify_states: lexsort on uint64 words").
    words = _pack_state_words(states_p)
    iota = jax.lax.broadcasted_iota(np.int32, (states_p.shape[0],), 0)
    perm = jax.lax.sort((*words.T, iota), dimension=0, num_keys=words.shape[1])[-1]
    states_srt = states_p[perm]
    # Uniqueness on the packed words, not the rows: same answer (the packing is injective -- the pad
    # bytes are constant) over ceil(B/8) columns instead of B.
    words_srt = words[perm]
    is_unique = jnp.any(jax.lax.ne(words_srt[1:], words_srt[:-1]), axis=1)
    # Element 0 is always considered unique -> add 1
    total_unique = jnp.sum(is_unique, dtype=np.int32) + 1
    # This cumsum(bincount(cumsum)) accounts for the uniqueness of the 0th element
    idx_unique = jnp.cumsum(
        jnp.bincount(jnp.cumsum(is_unique, dtype=np.int32), length=states_size), dtype=np.int32
    )
    # Finally flag out filler slots by setting total_unique: to -1
    if states_size != states_p.shape[0]:
        iota = jax.lax.broadcasted_iota(np.int32, (states_size,), 0)
    idx_unique = jnp.where(iota < total_unique, idx_unique, -1)
    # With wrap_negative_indices=False we'll have 255 for filler slots
    return states_srt.at[idx_unique].get(mode="fill", fill_value=255, wrap_negative_indices=False)


def _pack_state_keys(states: StateList) -> jax.Array:
    """Pack `[N, B]` uint8 state rows into `[N]` uint64 scalar keys, preserving lex order.

    Byte 0 becomes the most significant, so integer order on the keys is identical to row lex order.
    That equivalence is the whole point: it lets a scalar binary search stand in for a lexicographic
    one.

    Args:
        states: Packed state rows, shape ``[N, B]`` with ``B <= 8``.

    Returns:
        One ``uint64`` key per row, ordered identically to the rows.

    Raises:
        ValueError: If ``B > 8`` (``n >= 64``), where byte 0's shift reaches 64 bits and distinct
            states alias onto one key. Defence-in-depth: :func:`get_xsource` routes wide input to the
            lexicographic path (``NOTES.md``, "sqd._pack_state_keys: why B > 8 raises").
    """
    nbytes = states.shape[1]
    if nbytes > 8:
        raise ValueError(
            f"`_pack_state_keys` needs at most 8 bytes per row to fit a uint64 key, got {nbytes}. "
            "Byte 0 is the most significant, so a wider row shifts it out entirely and distinct "
            "states alias onto one key. `get_xsource` routes B > 8 to the lexicographic search."
        )
    shifts = jnp.asarray([8 * (nbytes - 1 - i) for i in range(nbytes)], dtype=jnp.uint64)
    return jnp.sum(states.astype(jnp.uint64) << shifts, axis=1)


def _is_lex_sorted(states: NDArray[np.uint8]) -> bool:
    """Return whether `[N, B]` uint8 rows are in strictly increasing lexicographic order.

    Host-side numpy, deliberately: the one caller (:func:`hproj`) is eager, and the point is to
    ``raise`` before any tracing. Compares adjacent rows at their first differing byte; equal rows
    count as *unsorted*, since a duplicate row makes the projection ambiguous. One vectorized pass,
    12-14% of `hproj` (``NOTES.md``, "sqd._is_lex_sorted: cost").

    **Rejects a padded :func:`uniquify_states` result, by design**, for any number of filler slots:
    an explicit high-bit test on the *packed* byte 0, since a lone filler is strictly increasing and
    `hproj` does not mask fillers (``NOTES.md``, "sqd._is_lex_sorted: the filler test"). Slice to the
    real rows first (`~_is_filler(states)`) if you hold a padded array.
    """
    # Any filler disqualifies, tested apart from sortedness (a lone filler is strictly increasing);
    # fillers sort last, so the last row decides (NOTES.md, "sqd._is_lex_sorted: the filler test").
    if states.shape[0] and states[-1, 0] >= 128:
        return False
    if states.shape[0] < 2:
        return True
    lhs, rhs = states[:-1], states[1:]
    differs = lhs != rhs
    # A row pair with no differing byte is a duplicate -> not strictly increasing.
    if not bool(np.all(np.any(differs, axis=1))):
        return False
    first = np.argmax(differs, axis=1)
    rows = np.arange(lhs.shape[0])
    return bool(np.all(lhs[rows, first] < rhs[rows, first]))


def _pack_state_words(states: StateList) -> jax.Array:
    """Pack `[N, B]` uint8 rows into `[N, ceil(B/8)]` uint64 words, preserving lex order.

    The wide counterpart to :func:`_pack_state_keys`, which is limited to a single word. Words are
    big-endian, MSW-first, and the most significant word is left-padded with zero bytes, so word-wise
    lexicographic order matches byte-wise row order exactly.

    Leading-padding makes ``nwords == 1`` reproduce :func:`_pack_state_keys` bit for bit, so the two
    paths compare directly. It is *not* load-bearing: trailing-padding also preserves order
    (``NOTES.md``, "sqd._pack_state_words: either padding end").

    Returns:
        Shape ``[N, ceil(B/8)]`` uint64. ``B <= 8`` yields one column, identical in value to
        :func:`_pack_state_keys`' output.
    """
    nbytes = states.shape[1]
    nwords = -(-nbytes // 8)
    pad = nwords * 8 - nbytes
    padded = (
        states
        if not pad
        else jnp.concatenate([jnp.zeros((states.shape[0], pad), dtype=jnp.uint8), states], axis=1)
    )
    shifts = jnp.asarray([8 * (7 - i) for i in range(8)], dtype=jnp.uint64)
    words = [
        jnp.sum(padded[:, w * 8 : (w + 1) * 8].astype(jnp.uint64) << shifts, axis=1)
        for w in range(nwords)
    ]
    return jnp.stack(words, axis=1)


def _word_less_than(rows: jax.Array, targets: jax.Array) -> jax.Array:
    """Elementwise lexicographic `rows[i] < targets[i]` over uint64 words, MSW-first.

    Lexicographic row order, but on packed words: a search level costs ``ceil(B/8)`` comparisons
    instead of ``B`` -- 2 rather than 13 at n=100 (``NOTES.md``, "sqd.get_xsource: search on uint64
    words").

    The unrolled Python loop is deliberate: ``nwords`` is static, and a `lax` loop would need a
    traced index into a static shape. Do not "vectorize" it with a ``jnp.cumprod`` prefix, which
    promotes the mask to int64 (``NOTES.md``, "sqd._word_less_than: no cumprod prefix").
    """
    lt = jnp.zeros(targets.shape[0], dtype=bool)
    eq = jnp.ones(targets.shape[0], dtype=bool)
    for w in range(rows.shape[1]):
        a, b = rows[:, w], targets[:, w]
        lt = jnp.logical_or(lt, jnp.logical_and(eq, a < b))
        eq = jnp.logical_and(eq, a == b)
    return lt


@jax.jit
def get_xsource(xsignature: NDArray[np.uint8], states: StateList) -> jax.Array:
    """Return an index array into the source of an X operation.

    Let `V` be a vector of complex or float values with shape `[N]`, `S` be a lex-sorted 2-d array
    of uint8 with shape `[N, B]` where `B = ceil(Q/8)`, and `X` be a vector of uint8 with shape
    `[B]`. An unpacked (truncated to `Q` bits) `X` is a bitstring that represents the location of X
    being applied to the states in `S`; X (I) is applied to qubit `q` if `Q-q-1`th bit is 1 (0). Let
    `P` be the projector of the shape `[2 ** Q]` state vector `W` onto `V`.

    We want a vector of indices `A` where `(PXW)[i] = V[A[i]]`. Without a projection (`S` holding
    every bitstring) `A[i] = i ^ X`. With one, the source `S[i] ^ X` need not be in `S`; then
    `A[i] = -1`, so that `V[A[i]]` defaults to a `fill_value` of 0.0 through `at[].get()`.

    **`states` must be lex-sorted**, since finding `A` is a binary search of `S ^ X` into `S`. Both
    in-tree callers satisfy it: `run_sqd` passes `uniquify_states`' output, and `hproj` passes
    `np.unique`'s or validates `unique_states=True` input. It replaced a `[2N, B]` sort (``NOTES.md``,
    "sqd.get_xsource: why a search, not a sort"; ``markdown/scaling-pocs.md``).

    Two paths, selected statically on width. `B <= 8` packs each row into a `uint64` and uses
    `jnp.searchsorted` directly; wider inputs fall back to an explicit binary search over the rows
    packed into `ceil(B/8)` `uint64` words (:func:`_pack_state_words`), MSW-first. The boundary is a
    correctness limit, not a tuning parameter: at `B > 8` a *single* `uint64` key would alias
    distinct states (``NOTES.md``, "sqd.get_xsource: search on uint64 words").

    Returns `-1` at every position whose source is absent. Tests comparing against a stored index
    array should compare only valid rows, or the gathered result, since `apply_xgrp` treats any
    negative index as absent.
    """
    size, nbytes = states.shape
    targets = jnp.bitwise_xor(states, xsignature)  # S^X
    invalid = np.array(-1, dtype=np.int32)

    if nbytes <= 8:
        keys = _pack_state_keys(states)
        # One name for the search key, used by both the search and the hit test: that they are the
        # same quantity is the whole correctness argument for the branch.
        target_keys = _pack_state_keys(targets)
        pos = jnp.searchsorted(keys, target_keys, side="left")
        # searchsorted returns N for a target above every key; clamp before gathering so the
        # equality test below stays in bounds.
        pos = jnp.minimum(pos, size - 1)
        found = keys[pos] == target_keys
    else:
        # Binary search on uint64 words, not bytes: ceil(B/8) comparisons per level instead of B
        # (NOTES.md, "sqd.get_xsource: search on uint64 words"). lo counts rows below the target.
        swords = _pack_state_words(states)
        twords = _pack_state_words(targets)

        def step(carry, _):
            lo, hi = carry
            mid = (lo + hi) // 2
            go_right = _word_less_than(swords[jnp.minimum(mid, size - 1)], twords)
            return (jnp.where(go_right, mid + 1, lo), jnp.where(go_right, hi, mid)), None

        nsteps = int(np.ceil(np.log2(max(size, 2)))) + 1
        lo = jnp.zeros(size, dtype=jnp.int32)
        hi = jnp.full(size, size, dtype=jnp.int32)
        (lo, _), _ = jax.lax.scan(step, (lo, hi), None, length=nsteps)
        pos = jnp.minimum(lo, size - 1)
        found = jnp.all(swords[pos] == twords, axis=1)

    xsource = jnp.where(found, pos, invalid).astype(np.int32)
    if not (mesh := get_abstract_mesh()).empty:
        xsource = jax.reshard(xsource, PartitionSpec(mesh.axis_names))
    return xsource
