# Out-of-core uniquification

Moved from `NOTES.md` ("Replacing that sort out-of-core: prototyped and rejected", 2026-08-29); host-side
numpy, one machine. Rejected. The script is `poc/ooc_uniquify.py` (§4).

## 1. The proposal

`uniquify_states`' single-device `jax.lax.sort` is still the ceiling on `N ≤ 2^31`, and the obvious move is
`poc/ooc_uniquify.py`'s chunk-sort-and-merge, which bounds the working set by a chosen chunk size rather
than by `N`. That POC bails out at `B > 8` ("no uint64 equivalence available"), so it never covered
`n = 100`. `_pack_state_words` removes that obstacle — wide rows pack into `ceil(B/8)` uint64 columns, and
a structured-dtype view makes `np.unique` / `np.union1d` lexicographic over them with no row comparator —
so the wide case was built and measured.

## 2. It loses on both axes it exists to win

At n=100, B=13, N=8M host-side:

| approach | time | peak RSS |
| --- | --- | --- |
| `np.unique(rows, axis=0)` (incumbent shape) | 7.3 s | **365 MB** |
| word-packed `np.unique` | **4.9 s** | 1655 MB |
| chunked sort + merge tree on words | 31.3 s | 1048 MB |

4.3× slower and 2.9× more peak memory than plain `np.unique`. Output verified identical in all arms.

## 3. What it means

- **Packing widens the data** — the same fact that makes the *in-JAX* fix a good trade and this one a
  bad trade. `8*ceil(B/8) - B` bytes per row: +7 at n=64 (B=9→16), +3 at n=100, free at n=127. Speed
  bought with memory is right for the JAX sort, whose own working set dominates, and exactly wrong for a
  design whose entire purpose is bounding memory. Do not read the shipped `uniquify_states` change as a
  step toward an out-of-core one; they pull opposite ways.
- **Chunking distributes nothing** (the script's docstring says so; confirmed): it removes the
  single-device *sort*, but the merge is sequential and the full result lands on one host. A real
  multi-node uniquify needs a range-partitioned shuffle, for which chunk-local sorting is the per-node
  kernel and not the algorithm.

## 4. The script

`poc/ooc_uniquify.py` ("POC 9") covers only the `B ≤ 8` path: at `B > 8` both `uniquify_ooc` and its
memory twin fall back to one `np.unique(axis=0)`, so the wide arms in §2 (`_pack_state_words`) are not in
it. Three arms, each checked byte-for-byte (filler tail included) against `uniquify_states`:
`uniquify_states` (incumbent), `uniquify_host` (one numpy sort), `uniquify_ooc` (chunked sort, spilled
to `.npy`, pairwise merge tree deduplicating as it goes). Per size it prints timings, ratios against the
incumbent, and tracemalloc sort-phase peaks for `chunk_rows` `2^13`, `2^16`, `2^19`, `2^22`.

| argument | default | meaning |
| --- | --- | --- |
| `--num-states` | 200000 | `N` (first size of a sweep) |
| `--nbytes` | 8 | `B`, bytes per packed row |
| `--dup-frac` | 0.2 | duplicate fraction of the random rows |
| `--chunk-rows` | `2^16` | chunk size for the timed OOC arm and the parity check |
| `--sweep-to` | none | sweep `N` ×4 from `--num-states` up to this |
| `--trials` | 5 | timing trials |
