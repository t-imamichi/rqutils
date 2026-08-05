r"""
=========================================
Single-vector LOBPCG on MLX (Apple Metal)
=========================================

.. currentmodule:: rqutils.ground_locg_mlx

Overview
========

An MLX port of :mod:`rqutils.ground_locg` and of ``rqutils.sqd.apply_h`` at
``cache_level=(1, 2)``, for running
the SQD eigensolver loop on Apple GPUs. The algorithm is the same single-vector LOBPCG -- see
:mod:`rqutils.ground_locg` for the derivation, the Rayleigh-Ritz specialization, and the
numerical analysis behind every guard reproduced here.

``mlx`` is an optional, darwin-only extra (``uv run --extra mlx ...``); importing this module
without it raises ``ImportError``. Nothing else in the package imports it.

Relationship to the JAX implementation
======================================

The numerics are kept deliberately in step with :mod:`rqutils.ground_locg`: both eigenpair
kernels balance before forming their characteristic polynomials, :func:`eigenpair_3x3` takes its
null vector from the rank-aware :func:`_nullvec_3x3`, both close with a Rayleigh-quotient polish,
the iteration re-orthogonalizes ``t`` against the new ``x``, renormalizes the search direction,
masks a zeroed direction out of Rayleigh-Ritz contention, and reports that condition as
convergence. Those are items I1-I7 of ``docs/locg.md``; each was measured to fail *silently* --
a plausible wrong number rather than a raise or a ``NaN``. **When editing either file, change
both.**

Five structural differences, all forced by MLX:

* ``jax.lax.scan`` over X groups becomes a Python loop. J is static and small (tens), so
  unrolling into MLX's lazy graph is fine.
* ``jax.lax.while_loop`` becomes a Python loop. MLX has no while_loop/cond/scan, and reading
  the convergence flag forces ``mx.eval`` -- a device sync per iteration. That cost is real
  for an MLX user, not a measurement artifact. Pass ``tol=0.`` to skip the check entirely
  and get a sync-free fixed-iteration run.
* All ``out_sharding=`` arguments are dropped; MLX has unified memory and no sharding.
* The zero-residual guard at the seed step branches in Python on a synced ``float()`` rather
  than staying in the graph as a ``jnp.where``, since there is no ``lax.cond`` to fold it into.
* **Everything is real-valued.** MLX has no complex128 anywhere and Metal has no float64 at
  all, so :func:`eigenpair_2x2` and :func:`eigenpair_3x3` drop the ``.conjugate()`` calls and
  real/imag norm splits their JAX counterparts need for the general complex-Hermitian case.
  :func:`ground_locg_mlx` therefore **rejects complex input** rather than silently returning
  wrong eigenvalues: an earlier version of this port relied on an upstream even-Y constraint to
  keep coefficients real and admitted in its own docstring that nothing in the repo would catch
  a regression there.

The analytic 2x2/3x3 Rayleigh-Ritz step carries over directly, which is what makes this port
feasible at all -- MLX has no ``eigh``.

A sixth difference is not structural but a performance consequence of the same asymmetry, and it
is why several expressions here look less like their JAX counterparts than a straight
transcription would: **in JAX an unrolled Python loop over scalars is free, and in MLX it is
not.** XLA fuses ``jnp.stack([normalize(c) for c in cands])`` into the surrounding ``jit``, so
:mod:`rqutils.ground_locg` pays nothing for building its seven null-vector candidates one at a
time; MLX constructs one lazily-evaluated op per call, so the identical source shape became 18%
of this port's entire per-iteration op count. The Rayleigh-Ritz step here therefore uses batched
whole-array forms -- one broadcast cross product for all three orthogonal-complement candidates,
one ``(7, 3)`` normalization, ``mx.diagonal``/``mx.roll`` instead of element-by-element
``mx.stack`` -- and hoists its dtype-dependent constants into :data:`_CONST_CACHE` instead of
rebuilding them every iteration. Each of those was verified *bit-identical* to the form it
replaced, not merely close: they change only how many ops are launched, never the arithmetic.
Measured with ``examples/count_mlx_ops.py``, the LOBPCG body went from 116 to 76.3 op
constructions per iteration (-34%) with the eigenvalue and iteration count unchanged. This is the
one respect in which "when you change one, change both" should *not* be applied mechanically:
porting these batched forms back to JAX would be churn, since XLA already fuses what they
hand-fuse. The algebra is what must stay in step, not the op granularity.

Beyond the port itself, this module also adds three custom Metal kernels via
``mx.fast.metal_kernel``, selected by :func:`ground_locg_mlx`'s ``sas`` and ``eig`` parameters and
by the caller's choice of matvec: :func:`apply_h_xz_mlx_metal` fuses the matvec's
gather-multiply-accumulate into one GPU launch (the op-graph path,
:func:`apply_h_xz_mlx`/:func:`apply_h_xz_mlx_chunked`, needs several), the private
``_compute_sas_metal`` fuses the Rayleigh-Ritz inner products the same way -- 16 op launches
(measured) collapsing to 1 -- and :func:`eigenpair_3x3_metal` fuses the whole 3x3 eigensolve. All
three are float32-only, since Metal has no float64; the default ``sas="ops"``/``eig="ops"``
op-graph paths are unchanged from before those parameters existed and remain the only route for an
f64 solve. This is additional surface area, not a structural difference forced by the port -- it
introduces no new numerics, only fused execution strategies for quantities the op-graph functions
already compute, so it does not belong in the list above.

**The kernels landed on opposite sides of the same argument, and the discriminator is output
parallelism -- not launch count.** Fusing the matvec was a large win; fusing the Rayleigh-Ritz
inner products measured *slower* than the op-graph path (0.697 vs 0.593 ms/iter at N~800) and
``sas="ops"`` therefore remains the default. The matvec has N outputs and launches one thread
each, while the Rayleigh-Ritz reduction has nine, so its kernel launches a fixed six threadgroups
regardless of N and under-parallelizes a reduction that MLX's own kernels spread across the whole
GPU. Fewer launches did not compensate. ``_compute_sas_metal``'s docstring carries the numbers;
the full sweep, including why N above ~4000 is unmeasurable under fp32, is in
``docs/superpowers/specs/2026-08-04-metal-sas-kernel-design.md``.

A third kernel, :func:`eigenpair_3x3_metal` (``eig="metal"``), fuses the entire 3x3 eigensolve --
balancing, Cardano, the rank-aware seven-candidate null-vector search, and the closing Rayleigh
quotient -- into one launch. It sits on the *winning* side of that argument for a reason the
``_compute_sas`` result makes precise: the eigensolve does not scale with N at all, so it has no
reduction to under-parallelize. It is ~34 launches whose entire content is scalar arithmetic on
nine numbers, which one thread does in registers. Measured with ``examples/count_mlx_ops.py``, it
takes the LOBPCG body from 77.3 to 44.3 op constructions per iteration (-43%), the eigensolve
itself going from 34.5 ops to exactly 1 launch. It is fp32-only like the others, and inherits --
does not introduce -- Cardano's near-degeneracy fragility.

This is also the one place this port deliberately diverges from :mod:`rqutils.ground_locg`
without a JAX-side counterpart, and thus an exception to the "when you change one, change both"
rule above: a fused Metal kernel adds no algebraic content over the op-graph computation it
replaces -- it is a fused execution strategy for an existing quantity, not new numerics -- and
Metal's float64-less hardware is exactly what this port targets, so there is no float64 JAX path
this fusion could be mirrored onto.

Verification
============

MLX cannot initialize without a Metal device, which rules out headless testing. Two checkers
cover the gap: ``examples/check_ground_locg_mlx_static.py`` re-executes this module's source
against a numpy shim bound to the name ``mx`` (no MLX, no GPU -- this is what validates the
algorithm and the caller-facing contract), and ``examples/check_ground_locg_mlx_mlx.py`` runs
the real thing on both devices and both precisions, which requires a real Metal device (see the
device status below -- it has now been run).

Neither checker exercises the Metal kernels' actual ``source`` strings. The numpy shim
reimplements the intended per-thread indexing in Python/numpy rather than compiling and running
the Metal C++ text, so it is blind to a bug in that text itself.

**Device status, as of 2026-08-05.** ``apply_h_xz_mlx_metal`` and ``_compute_sas_metal`` have now
been executed on a real Metal device (Apple M1, 7-core GPU) via ``examples/bench_mlx.py --arm
mlx-gpu-f32 --matvec metal --compile-body``: the run passed every correctness gate, with
``matvec_err`` 1.69e-06 (the documented fp32 floor, matching the JAX f32 reference) and the
eigenvalue agreeing with the recorded reference. So the MSL text of those two is validated, and
the GPU measured 2.13x faster than MLX's CPU backend (0.452 vs 0.961 ms/iter).

:func:`eigenpair_3x3_metal` has **also now been validated on the same M1**, via
``examples/check_ground_locg_mlx_mlx.py`` (all 12 arms pass, ``FAILURES: none``) and
``examples/bench_mlx.py --eig metal``. Its ``metal::sqrt``/``cos``/``sin``/``atan2`` calls compile,
and on-device it agrees with ``numpy.linalg.eigh`` to 2.98e-07 (GPU) and 4.00e-07 (CPU) over 40
random symmetric matrices. It is the first kernel here to call math functions at all, so two static
guards were added to cover what the shim cannot: ``check_ground_locg_mlx_static.py`` asserts that
every kernel source qualifies its math calls with ``metal::`` (MLX emits no
``using namespace metal;``, so an unqualified call fails to compile) alongside the pre-existing
MSL-reserved-identifier check that the ``half`` incident motivated. Both guards were verified to
fire by breaking them deliberately.

**Measured, controlled (M1, n=12/p=100/s=1000, ``--matvec metal --compile-body``, only ``--eig``
differing):** 0.465 -> 0.265 ms/iter (**1.75x**) and 0.0465 -> 0.0275 s to converge (**1.69x**),
with the eigenvalue **bit-identical** (-5.3960399628) and the iteration count unchanged at 70. The
end-to-end and per-iteration gains agree because ``iters`` does not move, and the 1.75x exceeds
what the -43% launch reduction alone predicts -- the fused kernel also removes ~34 intermediate
allocations per iteration, not just the launches. Note this win is *on top of* ``compile_body``,
which already amortizes graph construction.

MLX LOBPCG API
==============

.. autofunction:: ground_locg_mlx

.. autofunction:: eigenpair_2x2

.. autofunction:: eigenpair_3x3

.. autofunction:: apply_h_xz_mlx

.. autofunction:: eigenpair_3x3_metal
"""

import math

import mlx.core as mx
import numpy as np

_SQRT3 = math.sqrt(3.0)

# Per-dtype constant cache.
#
# Every `mx.array(...)` call constructs a fresh array -- a host-to-device allocation and, in the
# lazy graph, another node -- so the scalar and small-vector constants the eigenpair kernels need
# were being rebuilt on every iteration. `eigenpair_3x3` alone constructed 7 of them per call
# (measured with examples/count_mlx_ops.py --by-op), which made it the single largest op-count
# contributor in the LOBPCG body. They depend only on the dtype, which is fixed for a whole solve,
# so they are built once per dtype here and reused.
#
# Keyed by the dtype object rather than its name so this works identically under real MLX and under
# the numpy shim in examples/check_ground_locg_mlx_static.py, whose dtypes are numpy types.
# Unbounded in principle, bounded by {float32, float64} in practice -- this module rejects complex
# input and Metal has no float64, so at most two entries are ever created.
_CONST_CACHE = {}


def _consts(dtype):
    """Return the cached constants for ``dtype``, building them on first use."""
    entry = _CONST_CACHE.get(dtype)
    if entry is not None:
        return entry
    entry = _CONST_CACHE[dtype] = {
        "zero": mx.array(0.0, dtype),
        "one": mx.array(1.0, dtype),
        # eigenpair_3x3's root coefficients: the three Cardano roots are the same linear
        # combination a*cos(phi) + b*sin(phi) with these (a, b) pairs.
        "cphi_coeff": mx.array([2.0, -1.0, -1.0], dtype),
        "sphi_coeff": mx.array([0.0, -_SQRT3, _SQRT3], dtype),
        "eye3": mx.eye(3, dtype=dtype),
        # eigenpair_2x2's fallback eigenvector for a multiple of the identity.
        "e0_2": mx.array([1.0, 0.0], dtype),
        # iter_body's selector for the p direction's diagonal entry, used to lift a zeroed
        # search direction out of Rayleigh-Ritz contention (docs/locg.md item I7). A fixed
        # constant, so it does not need rebuilding per iteration.
        "p_mask": mx.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype),
    }
    return entry


# Integer index arrays, cached the same way but keyed separately from the float constants above
# because they are dtype-independent: the same indices serve an f32 and an f64 solve.
_INDEX_CACHE = {}


def _indices():
    """Return the cached index arrays, building them on first use.

    ``lower_rows``/``lower_cols`` address a 3x3's strict lower triangle in the order (1,0), (2,0),
    (2,1) -- ``eigenpair_3x3``'s off-diagonal triple in one gather. ``col_pair`` rolls the columns
    by one so that a single batched cross product over ``(matT, matT[col_pair])`` yields
    ``_nullvec_3x3``'s three rank-2 candidates (c0xc1, c1xc2, c2xc0).
    """
    if not _INDEX_CACHE:
        _INDEX_CACHE["lower_rows"] = mx.array([1, 2, 2])
        _INDEX_CACHE["lower_cols"] = mx.array([0, 0, 1])
        _INDEX_CACHE["col_pair"] = mx.array([1, 2, 0])
    return _INDEX_CACHE


def apply_h_xz_mlx(vec, xsources, diagonals):
    """Return Hv from precomputed X sources and diagonals.

    Mirrors ``rqutils.sqd.apply_h`` at ``cache_level=(1, 2)``. ``xsources`` must already be
    sanitized (no
    negative entries, with the corresponding diagonals zeroed) -- see
    ``_bench_common.build_solver_inputs``.
    """
    out = mx.zeros_like(vec)
    for igroup in range(xsources.shape[0]):
        out = out + mx.take(vec, xsources[igroup]) * diagonals[igroup]
    return out


def apply_h_xz_mlx_chunked(vec, xsources, diagonals, chunk=16):
    """Return Hv via chunked batched gather -- the MLX counterpart of ``apply_h_xz_mlx``.

    ``apply_h_xz_mlx`` above loops over the J X-groups doing one ``take``+multiply+add per
    group -- 3J Python-level MLX op constructions per matvec. Since ``xsources``/``diagonals``
    are dense ``(J, N)`` arrays, groups can instead be processed in chunks: gather a whole chunk
    with one flat ``take``, reshape, and reduce with a single weighted sum, cutting the op count
    from ``3*J`` to roughly ``3*ceil(J/chunk)``.

    ``chunk`` bounds the size of the gathered temporary to ``chunk * N`` elements, rather than
    the full ``(J, N)`` a fully-batched (``chunk=J``) version would materialize -- at the large N
    this benchmark is ultimately meant to probe (SQD's matrix-free design targets N up to
    ~10**7), a full ``(J, N)`` gather would cost ~8 GB versus ~80 MB for the unchunked
    group-at-a-time loop. This is a deliberate tradeoff, not an oversight: do not "simplify"
    this to full batching (``chunk=J``).

    Op-count / temporary-size tradeoff measured at J=100:

    ========  ========================  ===================
    chunk     ops per matvec (3*ceil)   temporary (chunk*N)
    ========  ========================  ===================
    1 (loop)  300                       N        (baseline)
    8         39   (7.7x fewer)         8*N
    16        21   (14.3x fewer)        16*N   <- default
    32        12   (25x fewer)          32*N
    ========  ========================  ===================

    See ``examples/_bench_common.apply_h_xz_chunked`` for the JAX equivalent used by the JAX
    arms of the benchmark -- batching the gather is applied symmetrically to both frameworks so
    the comparison stays about the solver loop, not about who has the better matvec.

    Verified (design doc, Optimization 1): max abs diff vs the unchunked loop matvec is
    <= 2.7e-15 for chunk in {1, 4, 8, 16, 32, 50, 100, 128}, and matches ``H @ v`` to 1.8e-15.

    Args:
        vec: The vector to multiply, shape ``(N,)``.
        xsources: Sanitized X-source indices, shape ``(J, N)``.
        diagonals: Sanitized diagonals, shape ``(J, N)``, matching ``vec``'s dtype.
        chunk: Number of X-groups to gather per flat ``take``. J is static and small, so this
            is a plain Python int controlling how many groups are unrolled per flat gather.

    Returns:
        ``H @ vec``, algebraically identical to ``apply_h_xz_mlx``.
    """
    num_groups = xsources.shape[0]
    out = mx.zeros_like(vec)
    for start in range(0, num_groups, chunk):
        xc = xsources[start : start + chunk]
        dc = diagonals[start : start + chunk]
        gathered = mx.take(vec, xc.reshape(-1)).reshape(xc.shape)
        out = out + mx.sum(gathered * dc, axis=0)
    return out


_METAL_MATVEC_SOURCE = """
    uint i = thread_position_in_grid.x;
    if (i >= n_states) {
        return;
    }
    T acc = 0;
    for (uint j = 0; j < n_groups; ++j) {
        // xsources/diagonals are row-major (J, N), so column i of group j lives at j*N + i.
        // Adjacent threads read adjacent addresses, so these loads coalesce; only the
        // vec[] gather is irregular, and vec is small enough to sit in cache.
        uint off = j * n_states + i;
        acc += vec[xsources[off]] * diagonals[off];
    }
    out[i] = acc;
"""

_METAL_MATVEC_KERNEL = None


def _get_metal_matvec_kernel():
    """Build (once) the fused gather-multiply-accumulate Metal kernel."""
    global _METAL_MATVEC_KERNEL
    if _METAL_MATVEC_KERNEL is None:
        _METAL_MATVEC_KERNEL = mx.fast.metal_kernel(
            name="sqd_apply_h_xz",
            input_names=["vec", "xsources", "diagonals", "n_groups", "n_states"],
            output_names=["out"],
            source=_METAL_MATVEC_SOURCE,
        )
    return _METAL_MATVEC_KERNEL


def apply_h_xz_mlx_metal(vec, xsources, diagonals, threadgroup=256):
    """Return Hv via a single fused custom Metal kernel.

    Both ``apply_h_xz_mlx`` and ``apply_h_xz_mlx_chunked`` express the matvec as a sequence of
    MLX ops, so each one launches its own kernel and materializes a full intermediate array.
    This version computes ``out[i] = sum_j vec[xsources[j, i]] * diagonals[j, i]`` in one launch
    with the accumulator held in a per-thread register, so there are no intermediates at all.

    One thread owns one output element, which means no atomics are needed: thread ``i`` is the
    only writer of ``out[i]``.

    At J=100, N=893 this replaces 21 op launches (chunk=16) plus ~0.4 MB of intermediate
    traffic per matvec with a single launch and none. The gather into ``vec`` stays irregular --
    that is inherent to SQD -- but ``vec`` is only ~3.5 KiB at this N and sits in cache.

    Metal is float32-only for this purpose (it has no float64 at all), so this path is usable
    only by the f32 arms. Callers must check ``vec.dtype``; passing float64 raises.

    The kernel's arithmetic was validated against ``apply_h`` at ``cache_level=(1, 2)``
    by simulating the exact
    per-thread indexing in numpy: max abs diff 2.7e-15, and 3.6e-15 against a dense ``H @ v``.

    Args:
        vec: The vector to multiply, shape ``(N,)``, float32.
        xsources: Sanitized X-source indices, shape ``(J, N)``, int32.
        diagonals: Sanitized diagonals, shape ``(J, N)``, float32.
        threadgroup: Threads per threadgroup. 256 is a reasonable default on Apple GPUs.

    Returns:
        ``H @ vec``, algebraically identical to ``apply_h_xz_mlx``.

    Raises:
        ValueError: If ``vec`` is not float32, since Metal has no float64.
    """
    if vec.dtype != mx.float32:
        raise ValueError(
            f"apply_h_xz_mlx_metal requires float32 (Metal has no float64), got {vec.dtype}. "
            "Use apply_h_xz_mlx_chunked for the f64 arms."
        )

    num_groups, num_states = xsources.shape
    kernel = _get_metal_matvec_kernel()
    outputs = kernel(
        inputs=[vec, xsources, diagonals, num_groups, num_states],
        template=[("T", mx.float32)],
        grid=(num_states, 1, 1),
        threadgroup=(min(threadgroup, num_states), 1, 1),
        output_shapes=[(num_states,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


# Threadgroup memory arrays need a compile-time-constant size in Metal -- they cannot be sized by
# the runtime `lanes` value, since MLX generates the kernel's signature from input_names/
# output_names plus a fixed table of attributes, with no way to thread a runtime size into a
# `threadgroup` declaration. So `partials` is declared with this literal size, and the Python
# wrapper below must raise rather than silently clamp if a caller's `threadgroup` would exceed it.
_METAL_SAS_MAX_THREADGROUP = 256

_METAL_SAS_SOURCE = """
    // One threadgroup per (i, j) pair with i <= j; one thread per stride-slice of the vectors.
    // Fixed-size threadgroup memory: see _METAL_SAS_MAX_THREADGROUP above for why this must be a
    // compile-time literal rather than `lanes`. Only the first `lanes` slots are ever touched.
    threadgroup T partials[_METAL_SAS_MAX_THREADGROUP_LITERAL_];

    uint pair = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint lanes = threads_per_threadgroup.x;

    // Unrank `pair` into (i, j) with i <= j. n_basis is 2 or 3, so a short scan is cheaper
    // than any closed form and avoids integer-sqrt rounding concerns entirely.
    // n_basis/n_states arrive as `const constant int32_t` (MLX passes Python ints as int32), so
    // the loop counters are int too: comparing a uint counter against them is a signedness
    // mismatch that Metal warns on (-Wsign-compare). Both are small positive counts.
    int nbasis = n_basis;
    int nstates = n_states;

    int i = 0;
    int j = 0;
    int seen = 0;
    for (int a = 0; a < nbasis; ++a) {
        for (int b = a; b < nbasis; ++b) {
            if (seen == (int)pair) {
                i = a;
                j = b;
            }
            seen += 1;
        }
    }

    // Strided partial sum in a register. Stride `lanes` keeps adjacent lanes on adjacent
    // addresses, so these loads coalesce.
    T acc = 0;
    for (int k = (int)lane; k < nstates; k += (int)lanes) {
        acc += vectors[i * nstates + k] * mvs[j * nstates + k];
    }
    partials[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction over the threadgroup. `lanes` is a power of two (the caller rounds down),
    // so the halving is exact and no lane reads past the written region.
    //
    // NOTE: the stride variable must NOT be named `half` -- that is a reserved built-in scalar
    // type in Metal Shading Language (16-bit float), so `uint half = ...` fails to compile with
    // "cannot combine with previous 'type-name' declaration specifier" and cascades into eight
    // further errors. Neither the numpy shim nor any static check can catch this, since Python
    // has no such reserved word; it was found only by a real-device run.
    for (uint stride = lanes / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            partials[lane] += partials[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        // Write BOTH triangles with the identical value, so the result is exactly symmetric by
        // construction and no separate symmetrization op is needed. For i == j this writes the
        // same slot twice, which is harmless.
        out[i * nbasis + j] = partials[0];
        out[j * nbasis + i] = partials[0];
    }
""".replace("_METAL_SAS_MAX_THREADGROUP_LITERAL_", str(_METAL_SAS_MAX_THREADGROUP))

_METAL_SAS_KERNEL = None


def _get_metal_sas_kernel():
    """Build (once) the fused Rayleigh-Ritz inner-product kernel."""
    global _METAL_SAS_KERNEL
    if _METAL_SAS_KERNEL is None:
        _METAL_SAS_KERNEL = mx.fast.metal_kernel(
            name="sqd_compute_sas",
            input_names=["vectors", "mvs", "n_basis", "n_states"],
            output_names=["out"],
            source=_METAL_SAS_SOURCE,
        )
    return _METAL_SAS_KERNEL


def _compute_sas_metal(vectors, mvs, threadgroup=256):
    """Return the (n x n) matrix of <v_i | A | v_j> in a single fused Metal launch.

    **Measured SLOWER than the op-graph :func:`_compute_sas` -- do not switch the default to
    this.** Retained because it is correct, tested, and a verified negative result worth not
    rediscovering. ``sas="ops"`` is the faster path and stays the default.

    The op-graph path costs 16 op launches per iteration (measured) to produce nine numbers, of
    which only six are distinct. This computes all six distinct inner products in one launch: one
    threadgroup per (i, j) pair with i <= j, a strided per-thread partial sum, then a
    threadgroup-memory tree reduction.

    Why the launch-count argument fails here, and why the same argument *succeeded* for
    :func:`apply_h_xz_mlx_metal`: this kernel launches **one threadgroup per pair -- six, for
    n=3 -- regardless of N**, so serial work per thread grows linearly with N (3.1 steps at
    N=800, 45.1 at N=11533) while parallelism stays pinned at 1536 threads. The op-graph path
    calls MLX's own reduction kernels, which spread a reduction across the whole GPU. Trading 15
    launches for a drastically under-parallelized reduction loses at every measurable N. The
    matvec kernel avoids this because it has N outputs and launches one thread per output
    element; the Rayleigh-Ritz step has nine outputs, so there is no output parallelism to
    exploit. Measured on an M-series GPU at N~800: 0.697 ms/iter versus 0.593 ms/iter for the
    op-graph path, against a launch-count model that predicted 0.419 ms. See
    ``docs/superpowers/specs/2026-08-04-metal-sas-kernel-design.md`` for the full sweep, and note
    that N above ~4000 is unmeasurable here because fp32 fails the solver's convergence gate
    while Metal offers no float64.

    Thread 0 of each threadgroup writes **both** ``out[i*n + j]`` and ``out[j*n + i]``, so the
    result is exactly symmetric by construction -- stronger than the op-graph path's
    ``(sas + sas.T) * 0.5``, which averages two values that differ by rounding -- and the
    symmetrization op disappears rather than merely being fused.

    The tree reduction changes summation order relative to ``mx.sum``. For dot products of
    unit-norm vectors this is benign and typically *more* accurate than sequential summation
    (error growing as log n rather than n), but it is a change: see
    ``examples/check_ground_locg_mlx_static.py`` case 3h, which pins agreement with both the
    op-graph path and a direct ``v @ m``.

    Metal has no float64, so this path is f32-only, exactly like :func:`apply_h_xz_mlx_metal`.
    The f64 arms must keep using :func:`_compute_sas`.

    Args:
        vectors: Basis vectors, a tuple of 2 or 3 arrays of shape ``(N,)``, float32.
        mvs: Their images under A, same length and shapes, float32.
        threadgroup: Maximum threads per threadgroup. Rounded down to a power of two so the
            tree reduction halves exactly, and must not exceed
            :data:`_METAL_SAS_MAX_THREADGROUP` -- the kernel's ``partials`` array is sized by
            that literal at compile time, not by this runtime value.

    Returns:
        The ``(n, n)`` matrix of ``<v_i | A | v_j>``, exactly symmetric.

    Raises:
        ValueError: If the inputs are not float32, since Metal has no float64.
        ValueError: If ``vectors`` and ``mvs`` differ in length, or the length is not 2 or 3.
        ValueError: If ``threadgroup`` exceeds :data:`_METAL_SAS_MAX_THREADGROUP`.
    """
    if len(vectors) != len(mvs):
        raise ValueError(f"vectors and mvs must have equal length, got {len(vectors)}/{len(mvs)}")
    num_basis = len(vectors)
    if num_basis not in (2, 3):
        raise ValueError(f"_compute_sas_metal supports a basis of 2 or 3, got {num_basis}")
    for name, arrays in (("vectors", vectors), ("mvs", mvs)):
        for array in arrays:
            if array.dtype != mx.float32:
                raise ValueError(
                    f"_compute_sas_metal requires float32 (Metal has no float64), got "
                    f"{array.dtype} in {name}. Use _compute_sas for the f64 arms."
                )
    # The kernel's `partials` threadgroup array is declared with a compile-time-constant size
    # (see _METAL_SAS_MAX_THREADGROUP above); raise loudly rather than silently clamping, since a
    # silent clamp here would mean the caller's requested threadgroup size is not what actually
    # ran, which is exactly the class of "plausible wrong number" bug this repo's guards exist to
    # prevent.
    if threadgroup > _METAL_SAS_MAX_THREADGROUP:
        raise ValueError(
            f"threadgroup={threadgroup} exceeds _METAL_SAS_MAX_THREADGROUP="
            f"{_METAL_SAS_MAX_THREADGROUP}, the compile-time size of the kernel's `partials` "
            "threadgroup array"
        )

    num_states = vectors[0].shape[0]
    stacked_v = mx.stack(vectors)
    stacked_m = mx.stack(mvs)
    num_pairs = num_basis * (num_basis + 1) // 2

    # Round the threadgroup size down to a power of two: the tree reduction halves `lanes` until
    # it reaches 1, which only visits every written lane exactly once if it starts as a power of
    # two. Also cap it at num_states so no lane sits idle with nothing to accumulate.
    lanes = 1
    while lanes * 2 <= min(threadgroup, num_states):
        lanes *= 2

    # No output initialization is needed: the pairs (i, j) with i <= j cover every slot of the
    # (n, n) output once the j > i writes are mirrored, so every element is written before it is
    # read. Do not "fix" this by zero-filling first; that would add back a launch.
    kernel = _get_metal_sas_kernel()
    outputs = kernel(
        inputs=[stacked_v, stacked_m, num_basis, num_states],
        template=[("T", mx.float32)],
        # grid is in THREADS, not threadgroups: one threadgroup of `lanes` threads per pair.
        grid=(num_pairs * lanes, 1, 1),
        threadgroup=(lanes, 1, 1),
        output_shapes=[(num_basis, num_basis)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


_METAL_EIG3_SOURCE = """
    // ONE thread does the whole 3x3 eigensolve. There is no output parallelism to exploit -- the
    // result is one eigenvalue plus a 3-vector -- so this kernel exists purely to collapse ~34 op
    // launches into 1, exactly the tradeoff that made the fused matvec a win and the fused
    // Rayleigh-Ritz reduction a loss (see _compute_sas_metal). Here there is no reduction to
    // under-parallelize: the op-graph version launches 34 kernels to do scalar arithmetic on nine
    // numbers, and every one of those launches is pure overhead.
    if (thread_position_in_grid.x > 0) {
        return;
    }

    // ---- load the matrix into registers -------------------------------------------------
    T m[3][3];
    for (uint i = 0; i < 3; ++i) {
        for (uint j = 0; j < 3; ++j) {
            m[i][j] = mat[i * 3 + j];
        }
    }

    // ---- balance: shift traceless, scale by the largest entry ----------------------------
    // Load-bearing (docs/locg.md items I1/I2): without it the characteristic polynomial's
    // coefficients lose all significance for a large trace and `disc` goes negative -> NaN.
    T shift = (m[0][0] + m[1][1] + m[2][2]) / (T)3.0;
    T scale = 0;
    for (uint i = 0; i < 3; ++i) {
        for (uint j = 0; j < 3; ++j) {
            scale = metal::max(scale, metal::abs(m[i][j]));
        }
    }
    if (!(scale > (T)0.0)) {
        scale = (T)1.0;
    }
    T b[3][3];
    for (uint i = 0; i < 3; ++i) {
        for (uint j = 0; j < 3; ++j) {
            b[i][j] = (m[i][j] - (i == j ? shift : (T)0.0)) / scale;
        }
    }

    // ---- characteristic polynomial of the traceless balanced matrix: x^3 + c1 x + c0 ------
    T bd0 = b[0][0], bd1 = b[1][1], bd2 = b[2][2];
    // modod in the op-graph order (1,0), (2,0), (2,1), squared.
    T od0 = b[1][0] * b[1][0];
    T od1 = b[2][0] * b[2][0];
    T od2 = b[2][1] * b[2][1];
    // sum(bd * roll(bd, 1)) = bd0*bd2 + bd1*bd0 + bd2*bd1, then minus sum(modod).
    T c1 = (bd0 * bd2 + bd1 * bd0 + bd2 * bd1) - (od0 + od1 + od2);
    // sum(bd * modod[::-1]) = bd0*od2 + bd1*od1 + bd2*od0.
    T c0 = (bd0 * od2 + bd1 * od1 + bd2 * od0)
           - (bd0 * bd1 * bd2)
           - (T)2.0 * (b[0][2] * b[1][0] * b[2][1]);

    // Both radicands are non-negative for a symmetric matrix; clamp against rounding.
    T p = metal::max((T)(-3.0) * c1, (T)0.0);
    T disc = metal::max((T)(-27.0) * c1 * c1 * c1 - (T)182.25 * c0 * c0, (T)0.0);
    T phi = metal::atan2(metal::sqrt(disc), (T)(-13.5) * c0) / (T)3.0;
    T cphi = metal::cos(phi);
    T sphi = metal::sin(phi);
    // Roots are (sqrt(p)/3) * {2cos(phi), -cos(phi) -+ sqrt(3) sin(phi)}; take the smallest.
    T r0 = (T)2.0 * cphi;
    T r1 = -cphi - (T)SQRT3_ * sphi;
    T r2 = -cphi + (T)SQRT3_ * sphi;
    T xmin = metal::min(r0, metal::min(r1, r2)) * metal::sqrt(p) / (T)3.0;

    // ---- rank-aware null vector of (balanced - xmin I) -----------------------------------
    // Item I3: seven candidates, winner chosen on the MEASURED residual |Mv|, not on a magnitude
    // threshold -- for a degenerate eigenvalue the cross products do not vanish but decay only to
    // O(eps |M|^2), which no fixed cutoff separates from a genuinely small rank-2 cross product.
    T s[3][3];
    for (uint i = 0; i < 3; ++i) {
        for (uint j = 0; j < 3; ++j) {
            s[i][j] = b[i][j] - (i == j ? xmin : (T)0.0);
        }
    }

    // Columns of s. cands rows 0..2 are col_i x col_j for (0,1), (1,2), (2,0) -- the rank-2 case.
    // The pairings are (k, (k+1) % 3), computed arithmetically rather than read from a local
    // lookup table: a runtime-indexed `const uint[3]` inside a kernel body would be forced out of
    // registers into memory, and the modulo of a loop counter is free by comparison.
    T cands[7][3];
    for (uint k = 0; k < 3; ++k) {
        uint a = k;
        uint bcol = (k + 1) % 3;
        // col_a = (s[0][a], s[1][a], s[2][a]); cross product of the two columns.
        cands[k][0] = s[1][a] * s[2][bcol] - s[2][a] * s[1][bcol];
        cands[k][1] = s[2][a] * s[0][bcol] - s[0][a] * s[2][bcol];
        cands[k][2] = s[0][a] * s[1][bcol] - s[1][a] * s[0][bcol];
    }

    // Rank 1: the null space is the orthogonal complement of the largest column; rows 3..5 are
    // col x e_k for the standard basis vectors.
    uint col_index = 0;
    T best_colnorm = (T)(-1.0);
    for (uint j = 0; j < 3; ++j) {
        T cn = s[0][j] * s[0][j] + s[1][j] * s[1][j] + s[2][j] * s[2][j];
        // Strict >, scanning j ascending, reproduces argmax's first-maximum tie-breaking.
        if (cn > best_colnorm) {
            best_colnorm = cn;
            col_index = j;
        }
    }
    T c[3] = {s[0][col_index], s[1][col_index], s[2][col_index]};
    // col x e_0 = [0, c2, -c1]; col x e_1 = [-c2, 0, c0]; col x e_2 = [c1, -c0, 0].
    cands[3][0] = (T)0.0;   cands[3][1] = c[2];     cands[3][2] = -c[1];
    cands[4][0] = -c[2];    cands[4][1] = (T)0.0;   cands[4][2] = c[0];
    cands[5][0] = c[1];     cands[5][1] = -c[0];    cands[5][2] = (T)0.0;
    // Rank 0 (a multiple of the identity): every candidate above is zero, so offer e_0. Its
    // residual is 0, so it wins by default.
    cands[6][0] = (T)1.0;   cands[6][1] = (T)0.0;   cands[6][2] = (T)0.0;

    // Normalize each candidate, leaving an exactly-zero row at zero rather than making NaN.
    T norms[7];
    for (uint k = 0; k < 7; ++k) {
        T n = metal::sqrt(cands[k][0] * cands[k][0]
                        + cands[k][1] * cands[k][1]
                        + cands[k][2] * cands[k][2]);
        norms[k] = n;
        T den = (n == (T)0.0) ? (T)1.0 : n;
        for (uint i = 0; i < 3; ++i) {
            cands[k][i] /= den;
        }
    }

    // Residual |s v| per candidate, matching the op-graph `cands @ mat.T` (mat here is `s`, which
    // is symmetric, but transpose it explicitly so the transcription is order-for-order).
    T resid[7];
    T max_resid = (T)0.0;
    for (uint k = 0; k < 7; ++k) {
        T rv[3];
        for (uint i = 0; i < 3; ++i) {
            rv[i] = s[i][0] * cands[k][0] + s[i][1] * cands[k][1] + s[i][2] * cands[k][2];
        }
        resid[k] = metal::sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
        max_resid = metal::max(max_resid, resid[k]);
    }
    // A candidate that collapsed to zero is not a valid eigenvector; penalize it out of contention
    // using the same finite penalty the op-graph path uses (MLX has no terse inf-fill idiom, and
    // the comparison only needs an ordering).
    for (uint k = 0; k < 7; ++k) {
        if (!(norms[k] > (T)0.0)) {
            resid[k] += max_resid + (T)1.0;
        }
    }
    uint best = 0;
    T best_resid = resid[0];
    for (uint k = 1; k < 7; ++k) {
        // Strict <, scanning k ascending, reproduces argmin's first-minimum tie-breaking.
        if (resid[k] < best_resid) {
            best_resid = resid[k];
            best = k;
        }
    }
    T v[3] = {cands[best][0], cands[best][1], cands[best][2]};

    // ---- closing Rayleigh quotient -------------------------------------------------------
    // Second order in the eigenvector error, so it recovers the precision Cardano alone loses
    // (sqrt(eps) for a near-degenerate lowest pair).
    T bv[3];
    for (uint i = 0; i < 3; ++i) {
        bv[i] = b[i][0] * v[0] + b[i][1] * v[1] + b[i][2] * v[2];
    }
    T rq = v[0] * bv[0] + v[1] * bv[1] + v[2] * bv[2];

    theta[0] = rq * scale + shift;
    for (uint i = 0; i < 3; ++i) {
        kappa[i] = v[i];
    }
""".replace("SQRT3_", repr(_SQRT3))

_METAL_EIG3_KERNEL = None


def _get_metal_eig3_kernel():
    """Build (once) the fused 3x3 eigensolve Metal kernel."""
    global _METAL_EIG3_KERNEL
    if _METAL_EIG3_KERNEL is None:
        _METAL_EIG3_KERNEL = mx.fast.metal_kernel(
            name="sqd_eigenpair_3x3",
            input_names=["mat"],
            output_names=["theta", "kappa"],
            source=_METAL_EIG3_SOURCE,
        )
    return _METAL_EIG3_KERNEL


def eigenpair_3x3_metal(mat):
    """Lowest eigenpair of a real symmetric 3x3 matrix in a single fused Metal launch.

    Algebraically the same computation as :func:`eigenpair_3x3` composed with
    :func:`_nullvec_3x3` -- balance, Cardano, the rank-aware seven-candidate null-vector search,
    and the closing Rayleigh quotient -- transcribed into one kernel so that ~34 op launches per
    LOBPCG iteration (measured: 44% of the whole body, ``examples/count_mlx_ops.py``) become one.

    **Why fusing wins here when it lost for the Rayleigh-Ritz reduction.**
    :func:`_compute_sas_metal` is a verified negative result: it under-parallelized an O(N)
    reduction that MLX's own kernels spread across the GPU, so trading 15 launches for that was a
    loss at every measurable N. This kernel has the opposite profile. Its work does not scale with
    N at all -- it is scalar arithmetic on nine numbers -- so there is nothing to parallelize and
    nothing to under-parallelize. A single thread doing ~34 ops' worth of register arithmetic
    replaces ~34 kernel launches whose *only* content is that same arithmetic. The launch-count
    argument applies cleanly precisely because there is no reduction to lose.

    Every guard from the op-graph version is reproduced deliberately, not incidentally:

    * **Balancing** before forming the characteristic polynomial (``docs/locg.md`` I1/I2) --
      without it a large-trace matrix's coefficients lose all significance and ``disc`` goes
      negative, yielding NaN.
    * **The rank-aware seven-candidate search** (I3), selecting on the measured residual rather
      than a magnitude threshold, with the zero-collapse penalty.
    * **The closing Rayleigh quotient**, which is what recovers the precision Cardano alone loses
      near degeneracy.
    * **argmax/argmin tie-breaking**: both scans use a strict comparison over ascending indices,
      which selects the *first* extremum exactly as MLX's reductions do. A non-strict comparison
      would pick the last and could return a different (still valid, but different) candidate.

    fp32 only, like every Metal kernel here -- Metal has no float64. The accuracy consequence is
    inherited, not introduced: Cardano's method loses ~8 digits when two eigenvalues are nearly
    degenerate (measured 3.6e-08 versus ``eigh``'s 1.8e-15), and ``rqutils.ground_locg``'s JAX
    original exhibits *bit-identical* error on the same matrices. This kernel is that same
    formulation, so an f32 solve using it is subject to the same fragility as an f32 solve without
    it -- see ``docs/superpowers/specs/2026-08-03-mlx-sqd-poc-design.md``, which traces
    ``mlx-cpu-f64``'s 652-iteration count to exactly this.

    Args:
        mat: A 3x3 real symmetric matrix, float32. Only the diagonal and lower triangle are read
            for the polynomial, matching the op-graph version.

    Returns:
        ``(theta, kappa)`` -- the smallest eigenvalue as a scalar array and its unit eigenvector,
        the same pair :func:`eigenpair_3x3` returns.

    Raises:
        ValueError: If ``mat`` is not float32, since Metal has no float64.
    """
    if mat.dtype != mx.float32:
        raise ValueError(
            f"eigenpair_3x3_metal requires float32 (Metal has no float64), got {mat.dtype}. "
            "Use eigenpair_3x3 for the f64 arms."
        )
    kernel = _get_metal_eig3_kernel()
    outputs = kernel(
        inputs=[mat],
        template=[("T", mx.float32)],
        # One thread total: there is no output parallelism in a 3x3 eigensolve. The grid must still
        # be at least one threadgroup, and the kernel returns early for any thread beyond the first.
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(1,), (3,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    # theta is shaped (1,) because MLX kernels need an explicit output shape; squeeze it to the
    # scalar the op-graph path returns so the two are drop-in interchangeable.
    return outputs[0][0], outputs[1]


def ground_locg_mlx(
    mat,
    xinit,
    args=(),
    maxiter=1000,
    tol=None,
    compile_body=False,
    compile_chunk=10,
    sas="ops",
    eig="ops",
):
    """Single-vector LOBPCG in MLX.

    Args:
        mat: Callable mapping ``(vec, *args)`` to ``A @ vec``.
        xinit: Initial vector. Must have nonvanishing overlap with the ground state.
        args: Extra arguments forwarded to ``mat``.
        maxiter: Maximum gradient-descent iterations.
        tol: Convergence tolerance. ``None`` uses the dtype epsilon. ``0.`` disables the
            check, running exactly ``maxiter`` iterations with no per-iteration device sync.
        compile_body: If True, wrap the per-iteration body in ``mx.compile`` so the ~1260
            MLX ops it constructs are traced once instead of on every call. Opt-in and OFF by
            default: with ``compile_body=False`` this function's behaviour, including its
            exact iteration count, is unchanged from before this parameter existed.
        compile_chunk: When ``compile_body`` is True, the number of raw iterations the compiled
            body runs before control returns to Python for a convergence check. MLX has no
            ``while_loop``/``cond``, so checking convergence at all forces a ``float()``
            device sync; checking after every single iteration (chunk=1) would sync exactly as
            often as the uncompiled path and defeat the point of compiling. Running a chunk of
            iterations per compiled call amortizes that sync over ``compile_chunk`` iterations
            instead of paying it every time. In fixed-iteration mode (``tol=0.``) no
            convergence check happens at all, so the compiled body runs start-to-finish with no
            per-iteration sync regardless of this value -- see the fixed-iteration branch below.
        sas: Which Rayleigh-Ritz inner-product implementation to use. ``"ops"`` (default) is
            the portable op-graph :func:`_compute_sas`, and reproduces this function's
            behaviour exactly as it was before this parameter existed. ``"metal"`` uses the
            fused single-launch :func:`_compute_sas_metal`, which requires float32 (Metal has
            no float64) and so raises on an f64 solve. The choice is made once here rather than
            per iteration, so it cannot introduce a per-iteration device sync.

            Note that ``"metal"`` is a **measured performance loss** and is retained only as a
            verified negative result -- see :func:`_compute_sas_metal`. Do not enable it
            expecting a speedup.
        eig: Which 3x3 eigensolve to use for the Rayleigh-Ritz step. ``"ops"`` (default) is the
            portable op-graph :func:`eigenpair_3x3`, and reproduces this function's behaviour
            exactly as it was before this parameter existed. ``"metal"`` uses the fused
            single-launch :func:`eigenpair_3x3_metal`, which requires float32 and so raises on
            an f64 solve. Unlike ``sas="metal"`` this is expected to *win*, because the
            eigensolve has no N-scaling reduction to under-parallelize -- it is ~34 launches of
            pure scalar arithmetic. Bound once here, for the same reason as ``sas``.

    Returns:
        ``(eigenvalue, eigenvector, iterations, converged)``. Check the fourth value rather than
        comparing the third against ``maxiter``, which is ambiguous when convergence happens on
        the final permitted iteration.

    Raises:
        ValueError: If ``xinit`` is complex. Every kernel here assumes real symmetric input.
        ValueError: If ``sas`` is not ``"ops"`` or ``"metal"``, or if ``sas="metal"`` is
            combined with a non-float32 ``xinit``.
        ValueError: If ``eig`` is not ``"ops"`` or ``"metal"``, or if ``eig="metal"`` is
            combined with a non-float32 ``xinit``.
        ValueError: If ``compile_chunk`` is less than 1, which would make the chunked
            convergence-checking loop fail to advance and spin forever.
    """
    xinit = mx.array(xinit)
    # Test the dtype's name rather than comparing against mx.complex64: this holds under real MLX
    # and under the numpy shim in examples/check_ground_locg_mlx_static.py, which defines only the
    # dtypes the port actually uses.
    if "complex" in str(xinit.dtype):
        # The kernels below drop the .conjugate() calls their JAX counterparts need, so complex
        # input would silently produce wrong eigenvalues. MLX has no complex128 at all and Metal
        # no float64, so this port is real-only by construction -- fail loudly instead.
        raise ValueError(
            f"ground_locg_mlx is real-symmetric only, got {xinit.dtype}. MLX has no complex128 "
            "and the eigenpair kernels here omit the conjugations the complex case requires; use "
            "rqutils.ground_locg (JAX) for a complex-Hermitian operator."
        )
    if sas not in ("ops", "metal"):
        raise ValueError(f"sas must be 'ops' or 'metal', got {sas!r}")
    if sas == "metal" and xinit.dtype != mx.float32:
        # Fail here rather than at the first iteration's kernel call, so the error names the
        # parameter the caller actually set. Metal has no float64.
        raise ValueError(
            f"sas='metal' requires float32 (Metal has no float64), got {xinit.dtype}. Use "
            "sas='ops' for an f64 solve."
        )
    if eig not in ("ops", "metal"):
        raise ValueError(f"eig must be 'ops' or 'metal', got {eig!r}")
    if eig == "metal" and xinit.dtype != mx.float32:
        raise ValueError(
            f"eig='metal' requires float32 (Metal has no float64), got {xinit.dtype}. Use "
            "eig='ops' for an f64 solve."
        )
    if compile_chunk < 1:
        # The chunked convergence-checking loop advances by `this_chunk = min(compile_chunk,
        # maxiter - niter)`, so a non-positive chunk makes `niter += this_chunk` a no-op and
        # `while niter < maxiter` spins forever -- a silent hang rather than a wrong answer, but
        # the same class of failure the rest of this module's guards exist to prevent. Reject it
        # here instead, where the message can name the parameter.
        raise ValueError(f"compile_chunk must be >= 1, got {compile_chunk}")
    # Bind the implementation once, before iterating: the dtype is fixed for the whole solve, so
    # a per-iteration dispatch would buy nothing and might force a device sync inside the
    # compiled body (see the design doc -- unverified, and avoided rather than risked).
    compute_sas = _compute_sas_metal if sas == "metal" else _compute_sas
    solve_eig3 = eigenpair_3x3_metal if eig == "metal" else eigenpair_3x3
    check_convergence = tol != 0.0
    if tol is None:
        # Compare the dtype object directly rather than parsing its repr: this works
        # identically under real MLX and under the numpy shim used by the static check.
        tol = float(np.finfo(np.float32 if xinit.dtype == mx.float32 else np.float64).eps)

    def matvec(vec):
        return mat(vec, *args)

    # Constants fetched once per solve rather than rebuilt on every normalize/iter_body call (see
    # _consts). The dtype is fixed for the whole solve, so this lookup cannot change mid-run.
    _solve_consts = _consts(xinit.dtype)
    one = _solve_consts["one"]
    p_mask = _solve_consts["p_mask"]

    def normalize(vector, norm=None):
        """Divide by the norm, leaving a zero vector untouched instead of producing NaN."""
        if norm is None:
            norm = mx.linalg.norm(vector)
        return vector / mx.where(norm == 0.0, one, norm)

    xinit = normalize(xinit)

    # Iteration 0: no previous direction yet. Keep ax -- iteration 1 reuses it (item S1).
    ax = matvec(xinit)
    rho = mx.sum(xinit * ax)
    xcurr = xinit
    rcurr = ax - rho * xcurr

    # Iteration 1: two-vector Rayleigh-Ritz over {x, r}.
    #
    # The zero-residual guard mirrors the JAX version's body_iter1: an exactly-zero residual means
    # xinit is already an eigenvector (sqd.py's diagonal-Hamiltonian path seeds exactly that), and
    # without the guard eigenpair_2x2 sees a sas_mat whose row/col 1 vanish and selects that null
    # direction, collapsing theta towards 0 instead of reporting rho -- the true answer.
    norm_r = mx.linalg.norm(rcurr)
    r_is_zero = float(norm_r) == 0.0
    tmp_p = normalize(rcurr, norm_r)
    # Reuse ax from iteration 0 rather than recomputing it inside compute_sas.
    sas_mat = compute_sas((xcurr, tmp_p), (ax, matvec(tmp_p)))
    if r_is_zero:
        # Lift the p diagonal out of contention so Rayleigh-Ritz cannot pick the null direction.
        # With p excluded the 2x2 solve collapses onto x alone, giving theta = rho and kappa =
        # [1, 0], so xcurr is unchanged and no new search direction is introduced.
        #
        # Assign the entry directly, mirroring ground_locg's `sas.at[1, 1].set(excluded)`. MLX
        # supports item assignment, so the nested mx.stack tower this used to build (four ops to
        # add `excluded - sas_mat[1, 1]` into one slot) was both longer and *less* exact -- the
        # subtract-then-re-add does not cancel, measured at 5.7e-14 versus 0 for a direct set.
        # This branch runs once per solve, not per iteration, so no launch count depends on it.
        excluded = mx.abs(rho) + mx.abs(rho) + 1.0
        sas_mat[1, 1] = excluded
    theta, kappa = eigenpair_2x2(sas_mat)
    tmp_t = tmp_p * kappa[0] - xcurr * kappa[1]
    tmp_u = xcurr * kappa[0] + tmp_p * kappa[1]
    xcurr = normalize(tmp_u)
    # Re-orthogonalize for the same reason as in iter_body below (item I5).
    for _ in range(2):
        tmp_t = tmp_t - xcurr * mx.sum(xcurr * tmp_t)
    ycurr = normalize(tmp_t)
    axnext = matvec(xcurr)
    rcurr = axnext - theta * xcurr
    # A zeroed post-seed residual means {x} already spans the relevant space, so no further
    # iteration can lower theta: report convergence without entering the loop.
    seed_converged = r_is_zero

    def iter_body(xcurr, ycurr, rcurr, axcurr):
        """One LOBPCG step. Returns (theta, xcurr, ycurr, rcurr, axnext, p_is_zero)."""
        tmp_p = _project_out((xcurr, ycurr), rcurr)
        # _project_out guarantees only |tmp_p| >= 0.99, but the Rayleigh-Ritz step below solves a
        # standard eigenproblem and so assumes an orthonormal basis: a short tmp_p scales
        # sas_mat[2, 2] by |tmp_p|^2, which for a large positive shift is a spuriously low diagonal
        # that gets selected in place of the true minimizer (item I6).
        norm_p = mx.linalg.norm(tmp_p)
        p_is_zero = norm_p == 0.0
        tmp_p = normalize(tmp_p, norm_p)
        # xcurr's image is already known from the previous iteration -- three matvecs, not four.
        sas_mat = compute_sas((xcurr, ycurr, tmp_p), (axcurr, matvec(ycurr), matvec(tmp_p)))
        # A zeroed tmp_p leaves sas_mat row/col 2 empty, and for a positive-definite A that zero
        # diagonal is the smallest eigenvalue, so Rayleigh-Ritz would pick the null direction and
        # the normalizations below would divide by zero. Lift it out of contention (item I7); the
        # p_is_zero case is reported as convergence by the caller.
        # The x/y diagonal via one strided read of the diagonal rather than two scalar gathers
        # plus a stack, and the [2, 2] selector mask from the per-dtype constant cache instead of
        # a fresh zeros_like plus an in-place write every iteration -- the mask is a fixed
        # constant, not a function of sas_mat.
        diag_xy = mx.diagonal(sas_mat)[:2]
        excluded = mx.max(diag_xy) + mx.sum(mx.abs(diag_xy)) + 1.0
        # ground_locg spells this `jnp.where(p_is_zero, sas.at[2, 2].set(excluded), sas)`. Here it
        # stays a masked blend rather than an item assignment: this runs every iteration inside
        # mx.compile, so the selection must remain part of the traced graph, and MLX has no
        # functional `.at[].set()` that composes with a traced predicate.
        sas_mat = mx.where(p_is_zero, sas_mat * (1.0 - p_mask) + excluded * p_mask, sas_mat)
        theta, kappa = solve_eig3(sas_mat)
        tmp_s = ycurr * kappa[1] + tmp_p * kappa[2]
        norm_s = mx.linalg.norm(tmp_s)
        tmp_t = tmp_s * (kappa[0] / mx.where(norm_s == 0.0, one, norm_s)) - xcurr * norm_s
        tmp_u = xcurr * kappa[0] + tmp_s
        xnext = normalize(tmp_u)
        # tmp_t is a difference of two quantities both nearly parallel to xcurr as norm_s -> 0, so
        # catastrophic cancellation lets ynext drift into xnext. Once <x|y> is O(1) the basis is no
        # longer orthonormal and the standard Rayleigh-Ritz above returns a theta *below* the true
        # minimum eigenvalue -- a silent wrong answer rather than a visible failure (item I5).
        for _ in range(2):
            tmp_t = tmp_t - xnext * mx.sum(xnext * tmp_t)
        ynext = normalize(tmp_t)
        axnext = matvec(xnext)
        rnext = axnext - xnext * theta
        return theta, xnext, ynext, rnext, axnext, p_is_zero

    def converged(xcurr, theta, rcurr, axnext, p_is_zero):
        # Compare the residual norm against the floating-point error we'd expect from forming the
        # residual at all. abs(theta) rather than +theta, and a sum rather than the difference the
        # first version of this port inherited: norm(Ax) - theta is a cancellation of two nearly
        # equal large positive numbers for a positive-definite operator, which was measured going
        # negative and made the test unsatisfiable, so the solver never converged and always burned
        # maxiter (item I4). A ground-state search is typically negative-definite, where +theta
        # would cancel in turn.
        reltol = (mx.linalg.norm(axnext) + mx.abs(theta)) * xcurr.shape[0] * 10
        # These float() calls force a device sync -- the price of MLX having no while_loop.
        if bool(p_is_zero):
            return True
        return float(mx.linalg.norm(rcurr)) < tol * float(reltol)

    niter = 0
    is_converged = False
    if seed_converged:
        # {x} already spans the residual; the loop cannot improve on the seed pair.
        return float(theta), xcurr, 0, True
    if not compile_body:
        for niter in range(1, maxiter + 1):
            theta, xcurr, ycurr, rcurr, axnext, p_is_zero = iter_body(xcurr, ycurr, rcurr, axnext)
            if check_convergence and converged(xcurr, theta, rcurr, axnext, p_is_zero):
                is_converged = True
                break
    elif not check_convergence:
        # Fixed-iteration mode: no convergence check at all, so the compiled body can run every
        # requested iteration with zero per-iteration device sync -- the clean per-iteration
        # speed measurement the design calls for. mx.compile traces iter_body's op graph once
        # (over the (xcurr, ycurr, rcurr, axcurr) array tree) instead of once per maxiter call.
        compiled_body = mx.compile(iter_body)
        for niter in range(1, maxiter + 1):
            theta, xcurr, ycurr, rcurr, axnext, _p_is_zero = compiled_body(
                xcurr, ycurr, rcurr, axnext
            )
    else:
        # Convergence checking mode with compilation: run compile_chunk raw iterations inside a
        # single compiled function, then sync once to check convergence, instead of syncing
        # after every single iteration. A compiled chunk-of-iterations body is what amortizes
        # the unavoidable float()-sync cost over compile_chunk iterations rather than paying it
        # every time -- compiling a body that itself contains a Python-level convergence branch
        # is not an option, since mx.compile traces a fixed array-in/array-out computation and
        # has no equivalent of jax.lax.while_loop/cond to make that branch part of the graph.
        def chunk_body(xcurr, ycurr, rcurr, axcurr):
            # No theta/p_is_zero initializers: compile_chunk >= 1 is validated above, so the loop
            # always runs at least once and any initial value would be dead. (They were not merely
            # redundant -- a zero-trip loop would have returned them, reporting theta = 0.)
            for _ in range(compile_chunk):
                theta, xcurr, ycurr, rcurr, axcurr, p_is_zero = iter_body(
                    xcurr, ycurr, rcurr, axcurr
                )
            return theta, xcurr, ycurr, rcurr, axcurr, p_is_zero

        compiled_chunk = mx.compile(chunk_body)
        while niter < maxiter:
            this_chunk = min(compile_chunk, maxiter - niter)
            if this_chunk == compile_chunk:
                theta, xcurr, ycurr, rcurr, axnext, p_is_zero = compiled_chunk(
                    xcurr, ycurr, rcurr, axnext
                )
            else:
                # Final partial chunk: fall back to the uncompiled body so niter lands exactly
                # on maxiter, matching the uncompiled path's semantics one iteration at a time.
                for _ in range(this_chunk):
                    theta, xcurr, ycurr, rcurr, axnext, p_is_zero = iter_body(
                        xcurr, ycurr, rcurr, axnext
                    )
            niter += this_chunk
            if converged(xcurr, theta, rcurr, axnext, p_is_zero):
                is_converged = True
                break

    return float(theta), xcurr, niter, is_converged


def _compute_sas(vectors, mvs):
    """Return the (n x n) matrix of <v_i | A | v_j> for n in {2, 3}.

    Takes the images ``mvs`` rather than computing them, so a caller holding an already-known
    ``A x`` can pass it in instead of paying for a fourth matrix-vector product per iteration
    (``docs/locg.md`` item S1).
    """
    rows = []
    for v1 in vectors:
        rows.append(mx.stack([mx.sum(v1 * mv) for mv in mvs]))
    sas = mx.stack(rows)
    # Symmetrize: the two triangles differ only by rounding for real symmetric A.
    return (sas + sas.T) * 0.5


def _project_out(basis, vector):
    """Orthogonalize ``vector`` against ``basis``, ending on a subtraction.

    The repeated passes and the final zeroing are load-bearing; see the comments in
    ``rqutils.ground_locg._project_out``, which this mirrors. Near convergence, ending on a
    normalization can reintroduce basis components through catastrophic cancellation and wreck the
    Rayleigh-Ritz conditioning.
    """
    one = _consts(vector.dtype)["one"]
    for _ in range(2):
        ips = [mx.sum(vb * vector) for vb in basis]
        for vb, ip in zip(basis, ips):
            vector = vector - vb * ip
        norm = mx.linalg.norm(vector)
        vector = vector / mx.where(norm == 0.0, one, norm)

    for _ in range(2):
        ips = [mx.sum(vb * vector) for vb in basis]
        for vb, ip in zip(basis, ips):
            vector = vector - vb * ip

    return vector * (mx.linalg.norm(vector) >= 0.99).astype(vector.dtype)


def eigenpair_2x2(mat):
    """Lowest eigenpair of a real symmetric 2x2 matrix.

    Mirrors ``rqutils.ground_locg.eigenpair_2x2``, including its balancing: the matrix is scaled
    by its largest entry and reduced to its traceless part before the quadratic is solved. Without
    that, ``tr*tr - 4*det`` cancels catastrophically for a large trace (relative error 5.8e-1 at
    shift 1e9) and the unbalanced intermediates overflow at extreme scale. The eigenvector row is
    selected on the sign of ``delta`` so ``delta`` is never cancelled against a nearly equal
    radius, and the closing Rayleigh quotient is second order in the eigenvector error. See
    ``docs/locg.md`` items I1/I2.

    Assumes ``mat`` is real symmetric, not just Hermitian: the JAX original's ``.conjugate()``
    calls are no-ops here and are dropped. ``ground_locg_mlx`` rejects complex input up front, so
    that assumption is enforced rather than merely documented.
    """
    consts = _consts(mat.dtype)
    scale = mx.max(mx.abs(mat))
    scale = mx.where(scale > 0.0, scale, consts["one"])
    balanced = mat / scale
    d = mx.diagonal(balanced)
    delta = (d[0] - d[1]) * 0.5
    offd = balanced[1, 0]
    rad = mx.sqrt(delta * delta + offd * offd)
    # Null vector of T + rad I = [[delta + rad, offd], [offd, rad - delta]], which is singular.
    # Row 1 gives [-offd, delta + rad] and row 2 gives [rad - delta, -offd]; the two are parallel,
    # but each cancels when its own pivot is small, so select on the sign of delta.
    vec = mx.where(
        delta >= 0.0,
        mx.stack([-offd, delta + rad]),
        mx.stack([rad - delta, -offd]),
    )
    # rad == 0 means a multiple of the identity, for which any unit vector is an eigenvector.
    norm = mx.linalg.norm(vec)
    vec = mx.where(
        norm > 0.0,
        vec / mx.where(norm > 0.0, norm, consts["one"]),
        consts["e0_2"],
    )
    # Rayleigh quotient: recovers full precision where the closed form alone reaches only
    # sqrt(eps), as for a near-degenerate lowest pair.
    return mx.sum(vec * (balanced @ vec)) * scale, vec


def _nullvec_3x3(mat):
    """Unit null vector of a singular real symmetric 3x3 matrix, robust to any rank.

    Mirrors ``rqutils.ground_locg._nullvec_3x3``. Seven candidates are offered and the one with
    the smallest residual ``|Mv|`` wins. Selecting on the measured residual rather than on a
    magnitude threshold matters because the rank-2 and rank-1 constructions fail in ways no fixed
    cutoff separates cleanly: for a degenerate eigenvalue the cross products do not vanish but
    decay only to O(eps |M|^2).
    """
    # Rank 2 (simple eigenvalue): the null vector is col_i x col_j. Any single pair can be rank
    # deficient, in which case its cross product points nowhere useful, so all three are offered.
    #
    # All three pairings go through ONE batched cross product over stacked (3, 3) operands rather
    # than three separate mx.linalg.cross calls. `matT[k]` is column k of `mat`, so the pairs are
    # (c0,c1), (c1,c2), (c2,c0) -- the same three, in the same order, as the unrolled version.
    matT = mat.T
    crosses = mx.linalg.cross(matT, matT[_indices()["col_pair"]])
    # Rank 1 (degenerate lowest eigenvalue): every cross product is numerical noise and the null
    # space is the orthogonal complement of the largest column; any member of it is an eigenvector.
    # mx.square alone, not mx.square(mx.abs(...)): |x|^2 == x^2 for real x, and this module rejects
    # complex input, so the abs the JAX original needs for the complex-Hermitian case is a pure
    # extra op launch here. Bit-identical, and argmax picks the same column -- verified at 0.0 max
    # difference over 5000 random matrices spanning scales 1e-8..1e8.
    col_index = mx.argmax(mx.sum(mx.square(mat), axis=0))
    col = mat[:, col_index]
    # The three complement vectors are exactly `col x e_k` for the standard basis vectors e_k:
    # col x e_0 = [0, col_2, -col_1], col x e_1 = [-col_2, 0, col_0], col x e_2 = [col_1, -col_0, 0],
    # which is what the unrolled version built by indexing out individual scalars and negating them
    # one at a time -- roughly a dozen launches for nine numbers. One broadcast cross replaces all
    # of it, bit-identically (verified at 0.0 max relative difference over 3000 random matrices
    # spanning scales 1e-8..1e8, including rank-deficient ones with a zeroed column).
    eye = _consts(mat.dtype)["eye3"]
    comps = mx.linalg.cross(mx.broadcast_to(col, (3, 3)), eye)
    # Rank 0 (a multiple of the identity): every candidate above is zero, so offer an arbitrary
    # unit vector as the last resort. It has residual 0 and wins by default.
    cands = [crosses, comps, eye[:1]]

    # Stack FIRST, then normalize the whole (7, 3) block in one pass. Normalizing candidate by
    # candidate -- what the JAX original does, and what this port copied -- costs 3 op launches per
    # candidate, 21 per iteration (measured: 18% of the whole LOBPCG body, see
    # examples/count_mlx_ops.py). In JAX that loop is free because XLA fuses it into the surrounding
    # jit; in MLX every op is its own launch, so the identical source shape has a completely
    # different cost. The batched form is *bit-identical* to the per-candidate one, not merely close
    # -- verified at 0.0 max abs difference over 2000 random triples spanning scales 1e-12..1e12
    # plus the exact-zero row the guard below exists for -- because each row's divisor is unchanged;
    # only the number of launches differs. Do not unroll this back into a Python loop for symmetry
    # with the JAX file.
    # concatenate, not stack: the three entries are already (3, 3), (3, 3) and (1, 3) blocks rather
    # than seven separate (3,) vectors, so this joins them into the same (7, 3) array the
    # per-candidate version produced -- in the same row order (crosses, complements, fallback).
    cands = mx.concatenate(cands)
    norms = mx.linalg.norm(cands, axis=1, keepdims=True)
    cands = cands / mx.where(norms == 0.0, _consts(norms.dtype)["one"], norms)
    # `cands @ mat`, not `cands @ mat.T`: this function's contract is a real *symmetric* matrix
    # (see the docstring, and eigenpair_3x3 only ever passes `balanced - xmin*eye`, symmetric by
    # construction), so the transpose is an exact no-op -- verified identical -- and costs an op
    # launch per iteration. The JAX original writes the einsum in the equivalent `mat @ cands_i`
    # order. If this ever needs to accept an asymmetric matrix, the transpose must come back.
    resid = mx.linalg.norm(cands @ mat, axis=1)
    # A candidate that collapsed to zero is not a valid eigenvector; disqualify it. MLX has no
    # inf-filling idiom as terse as jnp.where(..., jnp.inf), so add a large finite penalty
    # proportional to the residual scale instead -- the comparison only needs an ordering.
    #
    # Test the PRE-normalization norms rather than re-reducing the normalized rows: after the
    # division above every surviving row has norm exactly 1 and every zero row is still exactly 0,
    # so `norms > 0` and `norm(cands, axis=1) > 0.5` select the same rows while the former is
    # already computed. The threshold moves from 0.5 to 0 because it is now applied to the
    # unnormalized magnitudes, where 0.5 would be a scale-dependent cutoff on the candidates
    # themselves rather than the "did this collapse to zero" test intended -- the JAX version's 0.5
    # is only meaningful because it tests post-normalization rows, whose norms are exactly 1 or 0.
    alive = (norms[:, 0] > 0.0).astype(mat.dtype)
    resid = resid + (1.0 - alive) * (mx.max(resid) + 1.0)
    return cands[mx.argmin(resid)]


def eigenpair_3x3(mat):
    """Lowest eigenpair of a real symmetric 3x3 matrix via Cardano's method.

    Mirrors ``rqutils.ground_locg.eigenpair_3x3``. The matrix is balanced -- shifted traceless and
    scaled by its largest entry -- before the characteristic polynomial is formed: without that,
    the coefficients of a large-trace matrix lose all significance and the radicand of the square
    root goes negative, yielding NaN (``docs/locg.md`` items I1/I2). The null vector comes from
    :func:`_nullvec_3x3`, which is rank-aware rather than using one fixed column pair (item I3),
    and the closing Rayleigh quotient recovers the precision Cardano alone loses.

    Assumes ``mat`` is real symmetric, not just Hermitian: the JAX original's ``.conjugate()``
    calls and real/imag norm splits are no-ops here and are dropped. ``ground_locg_mlx`` rejects
    complex input up front, so that assumption is enforced rather than merely documented.

    Reference: J. Kopp, Int. J. Mod. Phys. C. 19, 523 (2008). MLX has no eigh, so the analytic
    route is the only route.
    """
    consts = _consts(mat.dtype)
    eye = consts["eye3"]
    # Whole-array extraction rather than element-by-element stacking. `mx.stack([mat[0, 0],
    # mat[1, 1], mat[2, 2]])` costs four launches (three scalar gathers plus the stack) to obtain
    # what mx.diagonal gets in one, and the two hand-permuted stacks below are mx.roll and a
    # reversed slice. All of it is index arithmetic, not algebra, so the reformulation is exact:
    # c1 and c0 agree with the element-wise version at 0.0 max relative difference over 5000
    # random symmetric matrices spanning scales 1e-8..1e8. This also brings the source closer to
    # rqutils.ground_locg.eigenpair_3x3, which already uses diagonal/roll/prod (the "change one,
    # change both" rule in CLAUDE.md).
    d = mx.diagonal(mat)
    shift = mx.sum(d) / 3.0
    scale = mx.max(mx.abs(mat))
    scale = mx.where(scale > 0.0, scale, consts["one"])
    balanced = (mat - shift * eye) / scale

    bd = mx.diagonal(balanced)
    # The strict lower triangle, in the order (1,0), (2,0), (2,1) -- one gather, not three.
    indices = _indices()
    modod = mx.square(balanced[indices["lower_rows"], indices["lower_cols"]])
    # Characteristic polynomial of the traceless balanced matrix: x^3 + c1 x + c0.
    # roll(bd, 1) is [bd2, bd0, bd1] and modod[::-1] is [modod2, modod1, modod0], exactly the
    # permutations the explicit stacks spelled out.
    c1 = mx.sum(bd * mx.roll(bd, 1)) - mx.sum(modod)
    c0 = (
        mx.sum(bd * modod[::-1])
        - mx.prod(bd)
        - 2.0 * (balanced[0, 2] * balanced[1, 0] * balanced[2, 1])
    )
    # Both radicands are non-negative for a symmetric matrix; clamp them against rounding.
    zero = consts["zero"]
    p = mx.maximum(-3.0 * c1, zero)
    disc = mx.maximum(-27.0 * c1 * c1 * c1 - 182.25 * c0 * c0, zero)
    phi = mx.arctan2(mx.sqrt(disc), -13.5 * c0) / 3.0
    cphi = mx.cos(phi)
    sphi = mx.sin(phi)
    # Roots are (sqrt(p) / 3) {2 cos(phi), 2 cos(phi -+ 2pi/3)}. Built as one length-3 expression
    # over stacked coefficients instead of three separately-computed scalars: the three roots are
    # the same linear combination a*cphi + b*sphi with different (a, b), so stacking the
    # coefficients turns six scalar ops plus a stack into two multiplies and an add. Exact -- the
    # same products in the same order, only batched. The coefficient vectors come from the
    # per-dtype cache so they are not rebuilt every iteration.
    xmin = mx.min(consts["cphi_coeff"] * cphi + consts["sphi_coeff"] * sphi)
    xmin = xmin * mx.sqrt(p) / 3.0

    vec = _nullvec_3x3(balanced - xmin * eye)
    # Rayleigh quotient: recovers full precision where Cardano alone reaches only sqrt(eps).
    return mx.sum(vec * (balanced @ vec)) * scale + shift, vec
