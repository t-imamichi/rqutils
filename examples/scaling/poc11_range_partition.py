"""POC 11: range-partitioned sample sort, to make uniquification shardable.

`uniquify_states`' `jax.lax.sort` must run on a single device, and that is the sole remaining cause of
the `N <= 2^31` ceiling (`NOTES.md`). POC 9 chunked the sort to bound its working set but **distributed
nothing** -- sequential merge, full result on one host -- and its own docstring names the missing piece:
a range-partitioned shuffle, for which chunk-local sorting is the per-node kernel and not the algorithm.

This is that shuffle. Three phases:

1. **Splitter selection (sample sort).** Draw `NSH * 64` rows, sort the sample, take evenly spaced
   quantiles as splitters. Equal-range splitting on the packed most-significant word does **not** work
   here -- at n=100 that word is 7 bytes of leading pad, so its nominal range is meaningless and every
   row lands in one bucket (measured: 4.00x/8.00x imbalance, i.e. total collapse). Data-derived
   splitters give 1.07-1.19x imbalance on both uniform and fixed-Hamming-weight fixtures.
2. **Scatter to fixed-capacity buckets.** Static shapes force a per-shard capacity, so each bucket gets
   `ceil(N/NSH * slack)` slots padded with an all-ones sentinel that sorts to the end. Within-bucket
   rank comes from `NSH` sequential `[N]` cumsums, **not** an argsort: a global `argsort` measured
   256 ms against 8 ms at N=2M and is exactly the single-device sort being removed. A `[N, NSH]`
   one-hot cumsum is equally fast but costs `4*N*NSH` bytes -- 34 GB at `N = 2^31, NSH = 4` and worse
   as devices are added, which defeats the purpose. The per-bucket loop is `O(N)`.
3. **NSH independent sorts.** `jax.vmap` over the bucket axis, deduplicating on emit. No cross-bucket
   comparison ever happens, because the splitters guarantee bucket `i` < bucket `i+1`; concatenating
   buckets in order is therefore globally sorted.

Measured on 4 virtual CPU devices (`--xla_force_host_platform_device_count=4`), against the incumbent
`uniquify_states`, output verified bit-identical to `np.unique(rows, axis=0)` at every size:

| N | incumbent | range-partitioned | ratio |
| --- | --- | --- | --- |
| 420,000 | 109.2 ms | 49.8 ms | 2.19x |
| 1,680,000 | 482.1 ms | 225.0 ms | 2.14x |
| 3,360,000 | 863.5 ms | 496.9 ms | 1.74x |

**Zero `all-gather`, `all-reduce` or `collective-permute` in the compiled HLO.**

**Timings under virtual devices are meaningless** (`CLAUDE.md`) -- they are reported only to show the
structure is not pathologically slow. The claim this POC supports is *shardability*, not a speedup.

**The capacity is a correctness parameter, but a safe one.** An undersized `cap` drops rows -- measured
16,090 lost at slack 1.05 -- but the function *returns the overflow count*, so a caller can raise
instead of silently truncating. That is the difference from the rank-select prototype in
`docs/rqutils-multiobs-response.md` §5.3, whose analogous `cap` had no detectable failure. Slack must
exceed the splitter imbalance; 1.35 was sufficient in every fixture here, and the padded array is
`slack` times the input.

**What this does not do.** Splitter selection is host-side numpy, so the sample and the bucket
assignment are computed on one device before the shuffle. `_pad_to`-style reassembly into the
`[states_size, B]` contract is not implemented -- the POC returns `[NSH, cap, NW]` blocks. Both are
real work before this could replace `uniquify_states`.

Run:
    XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run --extra dev python \
        examples/scaling/poc11_range_partition.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from functools import partial

from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P

mesh = jax.make_mesh((4,), ("x",), axis_types=(AxisType.Explicit,))
NSH = len(jax.devices())
B = 13
NW = 2
SENT = np.uint64(0xFFFFFFFFFFFFFFFF)


def pack_np(r):
    pad = NW * 8 - B
    p = np.concatenate([np.zeros((r.shape[0], pad), dtype=np.uint8), r], axis=1) if pad else r
    mul = np.uint64(256) ** np.arange(7, -1, -1, dtype=np.uint64)
    return np.stack(
        [(p[:, w * 8 : (w + 1) * 8].astype(np.uint64) * mul).sum(axis=1) for w in range(NW)], axis=1
    )


@partial(jax.jit, static_argnames=("cap",))
def rp_sort(words, bucket, cap):
    out = jnp.zeros(bucket.shape[0], dtype=jnp.int32)
    for k in range(NSH):  # O(N) memory, not O(N*NSH)
        m = (bucket == k).astype(jnp.int32)
        out = jnp.where(bucket == k, jnp.cumsum(m) - m, out)
    ok = out < cap
    slot = bucket * cap + jnp.where(ok, out, 0)
    flat = (
        jnp.full((NSH * cap, NW), SENT, dtype=jnp.uint64)
        .at[slot]
        .set(jnp.where(ok[:, None], words, SENT))
    )

    def sort_block(blk):
        iota = jax.lax.broadcasted_iota(jnp.int32, (cap,), 0)
        perm = jax.lax.sort((*blk.T, iota), dimension=0, num_keys=NW)[-1]
        srt = blk[perm]
        dup = jnp.concatenate([jnp.zeros(1, bool), jnp.all(srt[1:] == srt[:-1], axis=1)])
        return jnp.where(dup[:, None], SENT, srt)

    return jax.vmap(sort_block)(flat.reshape(NSH, cap, NW)), jnp.sum(~ok)


def prep(rows):
    W = pack_np(rows)
    Sv = np.ascontiguousarray(W).view([(f"w{i}", np.uint64) for i in range(NW)]).ravel()
    idx = np.random.default_rng(7).choice(len(Sv), size=NSH * 64, replace=False)
    samp = np.sort(Sv[idx])
    spl = samp[[int((i * len(samp)) / NSH) for i in range(1, NSH)]]
    return W, np.searchsorted(spl, Sv).astype(np.int32)


def unpack(Wa):
    pad = NW * 8 - B
    o = np.empty((Wa.shape[0], NW * 8), dtype=np.uint8)
    for w in range(NW):
        for k in range(8):
            o[:, w * 8 + k] = ((Wa[:, w] >> np.uint64(8 * (7 - k))) & np.uint64(0xFF)).astype(
                np.uint8
            )
    return o[:, pad:] if pad else o


import rqutils.sqd as S_


def bench(fn, trials=3):
    r = fn()
    (r[0] if isinstance(r, tuple) else r).block_until_ready()
    ts = []
    for _ in range(trials):
        t0 = time.perf_counter()
        r = fn()
        (r[0] if isinstance(r, tuple) else r).block_until_ready()
        ts.append(time.perf_counter() - t0)
    return min(ts)


print("4 virtual CPU devices: structure + correctness. Timings indicative only.")
print(f"{'N':>9} {'incumbent':>10} {'range-part':>11} {'ratio':>7} {'collectives':>11} correct")
for N in (400_000, 1_600_000, 3_200_000):
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 256, size=(N, B), dtype=np.uint8)
    rows[:, 0] &= 0x7F
    rows = np.concatenate([rows, rows[: N // 20]], axis=0)
    NT = rows.shape[0]
    cap = int(np.ceil(NT / NSH * 1.35))
    ref = np.unique(rows, axis=0)
    W, bkt = prep(rows)
    with jax.sharding.set_mesh(mesh):
        Wj = jax.device_put(jnp.asarray(W), P(None, None))
        bj = jax.device_put(jnp.asarray(bkt), P(None))
        spj = jnp.asarray(rows)
        fi = jax.jit(S_.uniquify_states, static_argnums=1)
        fi(spj, NT).block_until_ready()
        # Bind the loop variables as defaults (B023): the timer stores the thunk.
        ti = bench(lambda fi=fi, spj=spj, NT=NT: fi(spj, NT))
        tr = bench(lambda Wj=Wj, bj=bj, cap=cap: rp_sort(Wj, bj, cap))
        blk, nov = rp_sort(Wj, bj, cap)
        g = np.asarray(blk).reshape(-1, NW)
        g = g[~np.all(g == int(SENT), axis=1)]
        ok = np.array_equal(unpack(g), ref) and int(nov) == 0
        txt = rp_sort.lower(Wj, bj, cap).compile().as_text()
        col = txt.count("all-gather") + txt.count("all-reduce") + txt.count("collective-permute")
        print(f"{NT:>9} {ti * 1e3:>9.1f}m {tr * 1e3:>10.1f}m {ti / tr:>6.2f}x {col:>11} {ok}")
