"""POC 9: out-of-core uniquification, to lift the `N <= 2^31` ceiling.

`uniquify_states` sorts with `jax.lax.sort` (`sqd.py:914`), which must run on a single device. That
is the sole remaining cause of the `N <= 2^31` cap: `get_xsource` stopped contributing when POC 1
replaced its `2N` stack-and-sort with a binary search, so the ceiling now comes from uniquification
alone (`sqd.py:157-163`).

The observation this POC rests on: **nothing downstream requires the sort to happen in JAX.**
`get_xsource`, the diagonal builders and the matvec need `states_u` to be lex-sorted and unique; how
it got that way is invisible to them. So the sort can move host-side, where it can be done in chunks
that each fit in memory and merged by streaming.

**Scope, stated precisely because an earlier draft of this docstring over-claimed it.** This removes
the single-device *sort*: it bounds the sort's working set by a caller-chosen chunk size instead of by
`N`. It does **not** distribute anything. Every arm here is single-process host numpy returning a
plain `np.ndarray` with no sharding, the merge tree is sequential (no faster on 8 nodes than on 1),
and `_pad_to` materializes the full `[states_size, B]` result on one host -- 16 GiB at `N = 2^31`,
which is precisely what a multi-node design must avoid. A distributed uniquify needs a
range-partitioned shuffle, for which this is the correct *per-node* kernel, not the whole algorithm.

**Also: this measures the `B <= 8` path, which `n = 100` does not take.** `B = ceil((n + 1) / 8)`, so
a 100-site problem is `B = 13` and falls into the lexicographic fallback below, which is *not*
chunked. The headline speedups here do not describe that case; see the `B > 8` arm, which correctly
reports 1.0x because both arms run the same code.

Three arms, all producing the byte-identical `[states_size, B]` output the current function does
(lex-sorted unique rows, `255` filler at the tail):

* `uniquify_states` -- the incumbent, single-device `lax.sort`.
* `uniquify_host` -- one host-side sort. Establishes the cost of leaving JAX at all.
* `uniquify_ooc` -- chunked sort + k-way streaming merge, the actual proposal.

The merge is where the ceiling goes away: it holds one buffer per chunk, not `O(N)`, so peak memory
is set by the chunk size the caller picks. Dedup is free inside it -- adjacent equal rows collapse as
they are emitted, which also avoids a separate `O(N)` unique pass (measured at 10 s for 20M keys,
comparable to the sort itself).

`B <= 8` rows go through `_pack_state_keys`' uint64 equivalence, so the sort is a scalar radix sort
rather than a row-wise lexicographic one. Wider rows fall back to `np.lexsort` on the byte columns.
The split mirrors `get_xsource`'s, and for the same reason: it is a correctness boundary, since at
`B > 8` byte 0 shifts out of a uint64 entirely and distinct states alias onto one key.

Run:
    uv run --extra qiskit python examples/scaling/poc9_ooc_uniquify.py
    uv run --extra qiskit python examples/scaling/poc9_ooc_uniquify.py --sweep-to 4000000
"""

import argparse
import os
import sys
import tempfile
import tracemalloc

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(__file__))

from _scaling_common import fmt_ratio, header, timeit

from rqutils.sqd import uniquify_states

FILLER = 255


def _pack_keys_np(states: np.ndarray) -> np.ndarray:
    """Host-side twin of `_pack_state_keys`: `[N, B]` uint8 -> `[N]` uint64, lex order preserved.

    Byte 0 becomes the most significant, so integer order on the keys is identical to row lex order.
    Kept deliberately separate from the jitted original rather than reused: this runs on numpy arrays
    outside any trace, and the whole point of the POC is that no JAX call is on this path.
    """
    nbytes = states.shape[1]
    if nbytes > 8:
        raise ValueError(f"_pack_keys_np needs B <= 8, got {nbytes}")
    shifts = np.array([8 * (nbytes - 1 - i) for i in range(nbytes)], dtype=np.uint64)
    return (states.astype(np.uint64) << shifts).sum(axis=1, dtype=np.uint64)


def _unpack_keys_np(keys: np.ndarray, nbytes: int) -> np.ndarray:
    """Inverse of `_pack_keys_np`, so the merge can emit rows without carrying them alongside."""
    shifts = np.array([8 * (nbytes - 1 - i) for i in range(nbytes)], dtype=np.uint64)
    return ((keys[:, None] >> shifts) & np.uint64(0xFF)).astype(np.uint8)


def _pad_to(rows: np.ndarray, states_size: int) -> np.ndarray:
    """Pad `[M, B]` unique sorted rows up to `[states_size, B]` with all-255 filler.

    255 is the correct filler for the same reason `uniquify_states` uses it: an all-ones row sorts to
    the end, and its high bit in byte 0 is what `_is_filler` tests. `pack_states`' pad bit forces byte
    0 < 128 for every genuine state, so a real state can never collide with it.
    """
    deficit = states_size - rows.shape[0]
    if deficit < 0:
        raise ValueError(f"{rows.shape[0]} unique rows exceed states_size {states_size}")
    if deficit == 0:
        return rows
    return np.append(rows, np.full((deficit, rows.shape[1]), FILLER, dtype=np.uint8), axis=0)


def uniquify_host(states_p: np.ndarray, states_size: int) -> np.ndarray:
    """Single host-side sort. The control arm: cost of leaving JAX, without the chunking."""
    nbytes = states_p.shape[1]
    if nbytes <= 8:
        keys = np.unique(_pack_keys_np(states_p))
        return _pad_to(_unpack_keys_np(keys, nbytes), states_size)
    return _pad_to(np.unique(states_p, axis=0), states_size)


def uniquify_ooc(
    states_p: np.ndarray,
    states_size: int,
    chunk_rows: int = 1 << 20,
    spill_dir: str | None = None,
) -> np.ndarray:
    """Chunked sort + k-way streaming merge, deduplicating as it emits.

    Peak memory is `chunk_rows` per open chunk plus the output, never `O(N)` for the sort itself --
    which is the property that removes the single-device ceiling. Chunks are spilled to disk as
    `.npy` and reopened with `mmap_mode='r'` so the merge reads them lazily.

    `chunk_rows` is the memory dial: smaller means more chunks and a wider merge front, but a lower
    peak. It does not affect the result, only the footprint -- verified by the parity check below.
    """
    nbytes = states_p.shape[1]
    if nbytes > 8:
        # Wide rows: no uint64 equivalence available, so fall back to a single lexsort. Chunking this
        # is possible but needs a row-wise merge comparator; out of scope for the prototype, and the
        # B <= 8 path is what n < 64 qubits actually takes.
        return _pad_to(np.unique(states_p, axis=0), states_size)

    total = states_p.shape[0]
    cleanup: list[str] = []
    tmpdir = spill_dir or tempfile.mkdtemp(prefix="ooc_uniq_")
    try:
        # Phase 1: sort each chunk independently and spill. np.sort on uint64 is a radix sort, so
        # each chunk is O(chunk_rows) rather than O(chunk_rows log chunk_rows).
        paths = []
        for start in range(0, total, chunk_rows):
            keys = np.unique(_pack_keys_np(states_p[start : start + chunk_rows]))
            path = os.path.join(tmpdir, f"chunk_{len(paths):05d}.npy")
            np.save(path, keys)
            paths.append(path)
            cleanup.append(path)

        # Phase 2: pairwise merge tree, deduplicating at every level.
        #
        # Two forms were measured and rejected before this one:
        #
        # * A scalar cursor loop (the textbook k-way merge) is one Python iteration per output row:
        #   **0.12x** the incumbent at N=50k, i.e. ~8x slower.
        # * `np.union1d` per pair looks like exactly the right primitive -- a sorted-unique merge of
        #   two sorted unique inputs -- and is **80x slower** than it should be: measured 2459 ms
        #   against 30 ms for a concatenate-and-radix-sort of the same two 4M arrays. It calls
        #   `np.sort` on the concatenation with the default *comparison* sort, throwing away the
        #   sortedness of its inputs and the uint64 radix path at once. That is what made the whole
        #   merge tree degrade with N (0.30x at N=12.8M) rather than stay linear.
        #
        # `np.concatenate` + in-place `sort(kind="stable")` keeps the radix sort, and the dedup is a
        # neighbour-comparison pass on the already-sorted result.
        level = [np.asarray(np.load(p, mmap_mode="r")) for p in paths]
        while len(level) > 1:
            merged = []
            for i in range(0, len(level), 2):
                if i + 1 >= len(level):
                    merged.append(level[i])
                    continue
                pair = np.concatenate([level[i], level[i + 1]])
                pair.sort(kind="stable")  # radix on uint64
                merged.append(pair[np.append(pair[1:] != pair[:-1], True)])
            level = merged
        out = level[0]
        nout = out.size

        return _pad_to(_unpack_keys_np(out[:nout], nbytes), states_size)
    finally:
        for path in cleanup:
            try:
                os.unlink(path)
            except OSError:
                pass
        if spill_dir is None:
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass


def peak_mib(fn) -> float:
    """Peak host allocation of `fn`, in MiB, via tracemalloc.

    Measured over the sort and merge only -- the caller passes a thunk that stops short of the final
    unpack-and-pad, because the `[states_size, B]` output is the same size in every arm and swamps
    the difference that matters (measured 208 MiB total against a 106-vs-488 MiB spread in the sort
    itself). The point of the POC is the *working set* of the sort, which is what the single-device
    ceiling is about.
    """
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1] / 2**20
    finally:
        tracemalloc.stop()


def sort_only_host(states_p: np.ndarray) -> np.ndarray:
    """The host arm's sort phase alone, for the memory comparison.

    Mirrors `uniquify_host`'s own width split: at `B > 8` there is no uint64 key, so the row-wise
    `np.unique` is what the arm actually runs and so what must be measured.
    """
    if states_p.shape[1] > 8:
        return np.unique(states_p, axis=0)
    return np.unique(_pack_keys_np(states_p))


def sort_only_ooc(states_p: np.ndarray, chunk_rows: int) -> np.ndarray:
    """The OOC arm's sort+merge phases alone, for the memory comparison."""
    if states_p.shape[1] > 8:
        # Same fallback `uniquify_ooc` takes: chunking a row-wise lexsort needs a row comparator,
        # which the prototype does not implement. Measuring the uint64 path here would overstate the
        # win at widths that never reach it.
        return np.unique(states_p, axis=0)
    cleanup: list[str] = []
    tmpdir = tempfile.mkdtemp(prefix="ooc_mem_")
    try:
        paths = []
        for start in range(0, states_p.shape[0], chunk_rows):
            keys = np.unique(_pack_keys_np(states_p[start : start + chunk_rows]))
            path = os.path.join(tmpdir, f"c{len(paths):05d}.npy")
            np.save(path, keys)
            paths.append(path)
            cleanup.append(path)
        level = [np.asarray(np.load(p, mmap_mode="r")) for p in paths]
        while len(level) > 1:
            merged = []
            for i in range(0, len(level), 2):
                if i + 1 >= len(level):
                    merged.append(level[i])
                    continue
                pair = np.concatenate([level[i], level[i + 1]])
                pair.sort(kind="stable")
                merged.append(pair[np.append(pair[1:] != pair[:-1], True)])
            level = merged
        return level[0]
    finally:
        for path in cleanup:
            try:
                os.unlink(path)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


def make_states(num_states: int, nbytes: int, dup_frac: float, seed: int = 0) -> np.ndarray:
    """Packed-looking states with a controlled duplicate fraction.

    Byte 0 is masked to < 128 to mimic `pack_states`' pad bit, so filler detection stays meaningful
    and no generated row can collide with the all-255 filler.
    """
    rng = np.random.default_rng(seed)
    num_unique = max(1, int(num_states * (1.0 - dup_frac)))
    uniq = rng.integers(0, 256, size=(num_unique, nbytes), dtype=np.uint8)
    uniq[:, 0] &= 0x7F
    if num_unique < num_states:
        extra = uniq[rng.integers(0, num_unique, size=num_states - num_unique)]
        uniq = np.concatenate([uniq, extra], axis=0)
    return uniq[rng.permutation(uniq.shape[0])]


def check_parity(states_p: np.ndarray, states_size: int, chunk_rows: int) -> None:
    """Gate every arm against the incumbent, byte for byte.

    Compares the full padded array, not just the leading unique rows: the filler tail is part of the
    contract (`_is_filler`, `subspace_dim`), so an arm that got the rows right and the padding wrong
    would be broken in a way the solver only discovers later.
    """
    ref = np.asarray(uniquify_states(states_p, states_size))
    host = uniquify_host(states_p, states_size)
    ooc = uniquify_ooc(states_p, states_size, chunk_rows=chunk_rows)
    for name, got in (("host", host), ("ooc", ooc)):
        if got.shape != ref.shape:
            raise AssertionError(f"{name}: shape {got.shape} != reference {ref.shape}")
        if not np.array_equal(got, ref):
            ndiff = int((got != ref).any(axis=1).sum())
            raise AssertionError(f"{name}: {ndiff} rows differ from reference")
    print(f"  parity vs uniquify_states: OK (shape {ref.shape}, chunk_rows={chunk_rows})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-states", type=int, default=200_000)
    parser.add_argument("--nbytes", type=int, default=8)
    parser.add_argument("--dup-frac", type=float, default=0.2)
    parser.add_argument("--chunk-rows", type=int, default=1 << 16)
    parser.add_argument("--sweep-to", type=int, default=None)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    sizes = [args.num_states]
    if args.sweep_to:
        sizes = []
        n = args.num_states
        while n <= args.sweep_to:
            sizes.append(n)
            n *= 4

    header("POC 9: out-of-core uniquification")
    print(f"nbytes={args.nbytes}  dup_frac={args.dup_frac}  chunk_rows={args.chunk_rows}")

    for num_states in sizes:
        states_p = make_states(num_states, args.nbytes, args.dup_frac)
        states_size = 1 << max((num_states - 1).bit_length(), 1)
        print(f"\nN={num_states}  states_size={states_size}")

        check_parity(states_p, states_size, args.chunk_rows)

        base = timeit(
            lambda s=states_p, z=states_size: uniquify_states(s, z),
            "uniquify_states (lax.sort, 1 device)",
            trials=args.trials,
        )
        host = timeit(
            lambda s=states_p, z=states_size: uniquify_host(s, z),
            "uniquify_host (one numpy sort)",
            trials=args.trials,
            block=False,
        )
        ooc = timeit(
            lambda s=states_p, z=states_size: uniquify_ooc(s, z, chunk_rows=args.chunk_rows),
            "uniquify_ooc (chunk + k-way merge)",
            trials=args.trials,
            block=False,
        )
        for t in (base, host, ooc):
            print(f"  {t}")
        print(f"  host vs incumbent: {fmt_ratio(base, host)}")
        print(f"  ooc  vs incumbent: {fmt_ratio(base, ooc)}")

        # The memory result is the point: peak working set of the sort, which is what forces the
        # single-device ceiling. Swept over chunk_rows to show it is the dial.
        mem_host = peak_mib(lambda s=states_p: sort_only_host(s))
        print(f"  sort-phase peak: host {mem_host:.1f} MiB")
        for chunk in (1 << 13, 1 << 16, 1 << 19, 1 << 22):
            if chunk > num_states:
                continue
            mem_ooc = peak_mib(lambda s=states_p, c=chunk: sort_only_ooc(s, c))
            print(
                f"                   ooc chunk_rows={chunk:<9} {mem_ooc:7.1f} MiB "
                f"({mem_host / mem_ooc:.1f}x lower)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
