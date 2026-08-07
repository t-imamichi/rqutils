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

That document is a **stale historical audit** of the pre-rewrite JAX module: its line numbers and
severity rankings no longer apply, and it is cited here (and below) only for the I-numbers, which
remain a stable index into the measured failure modes. The binding statement of each invariant is
the JAX module's docstring next to the code.

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
feasible at all.

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
rebuilding them every iteration. Each was verified *bit-identical* to the form it replaced, not
merely close. Measured with ``examples/mlx/count_ops.py``, the LOBPCG body went from 116 to 65.0 op
constructions per iteration, and to 32.5 under ``device="gpu"``. This is the one respect in which
"when you change one, change both" should *not* be applied mechanically: porting these batched
forms back to JAX would be churn, since XLA already fuses what they hand-fuse. **The algebra is
what must stay in step, not the op granularity.**

Performance, and the measurements behind it
===========================================

:func:`ground_locg_mlx` takes a single ``device`` parameter. ``device="gpu"`` selects two fused
Metal kernels -- :func:`_apply_h_xz_metal` for the matvec and :func:`_eigenpair_3x3_metal` for the
Rayleigh-Ritz eigensolve -- and requires float32, since Metal has no float64. ``device="cpu"`` runs
the portable op-graph path, which is the **only** route for an f64 solve. ``mx.compile`` is applied
unconditionally in both cases; it was the single largest measured win (2.71x) and is no longer
selectable.

The measured wins, the two verified *negative* results (a fused Rayleigh-Ritz reduction that ran
slower, and a host-side per-iteration ``eigh`` that ran 1.37x slower), the full op-count history
with its backend split, and the cost model that explains all of them are in
``docs/mlx-metal-kernels.md``. **Read that before proposing a fusion, a host-side eigensolve, or an
"obvious" simplification here** -- several of the obvious ones were already tried and measured to
lose. The short version of the transferable lesson: whether fusing wins is decided by **output
parallelism, not launch count**.

Verification
============

MLX cannot initialize without a Metal device, which rules out headless testing.
``examples/mlx/check_solver_headless.py`` re-executes this module's source against a numpy shim
bound to the name ``mx`` (no MLX, no GPU -- this is what validates the algorithm and the
caller-facing contract), and ``examples/mlx/check_solver_device.py`` runs the real thing on both
devices and both precisions, which requires a real Metal device.

Neither exercises the Metal kernels' actual ``source`` strings: the shim reimplements the intended
per-thread indexing in numpy rather than compiling the Metal C++ text, so it is blind to a bug in
that text itself. Two static guards cover part of that gap -- every kernel source must qualify its
math calls with ``metal::``, and none may declare an MSL-reserved identifier -- and both were
verified to fire by breaking them deliberately. See ``docs/mlx-metal-kernels.md`` for the device
validation status.

MLX LOBPCG API
==============

.. autofunction:: ground_locg_mlx

.. autofunction:: apply_h_xz

.. autofunction:: eigenpair_2x2

.. autofunction:: eigenpair_3x3
"""

import math

import mlx.core as mx
import numpy as np

_SQRT3 = math.sqrt(3.0)

# How many LOBPCG iterations run inside one compiled call before control returns to Python for a
# convergence check. Only relevant when `tol != 0.`; a fixed-iteration solve never checks and so
# never syncs. Checking after every iteration (chunk of 1) would sync as often as an uncompiled loop
# and defeat the point of compiling, so this trades a slightly coarser iteration count for ~10x
# fewer syncs -- see ground_locg_mlx's Returns section for what "coarser" costs. A module constant
# rather than a parameter: 10 was the default nothing in the tree ever overrode.
_COMPILE_CHUNK = 10

# Per-dtype constant cache.
#
# Every `mx.array(...)` call constructs a fresh array -- a host-to-device allocation and, in the
# lazy graph, another node -- so the scalar and small-vector constants the eigenpair kernels need
# were being rebuilt on every iteration. `eigenpair_3x3` alone constructed 7 of them per call
# (measured with examples/mlx/count_ops.py --by-op), which made it the single largest op-count
# contributor in the LOBPCG body. They depend only on the dtype, which is fixed for a whole solve,
# so they are built once per dtype here and reused.
#
# Keyed by the dtype object rather than its name so this works identically under real MLX and under
# the numpy shim in examples/mlx/check_solver_headless.py, whose dtypes are numpy types.
# Unbounded in principle, bounded by {float32, float64} in practice -- this module rejects complex
# input and Metal has no float64, so at most two entries are ever created.
_CONST_CACHE = {}


def _consts(dtype):
    """Return the cached constants for ``dtype``, building them on first use."""
    entry = _CONST_CACHE.get(dtype)
    if entry is not None:
        return entry
    entry = _CONST_CACHE[dtype] = {
        # Only values that must be ARRAYS of the solve dtype belong here -- chiefly mx.where
        # operands, which have to match their branch's dtype. A bare Python float is promoted
        # in place by mx.maximum/mx.minimum and needs no entry (see eigenpair_3x3's clamps).
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
        # The complement of p_mask, cached rather than recomputed as `1.0 - p_mask` inside the
        # per-iteration blend: it is loop-invariant, so building it in the hot path was one op
        # construction per iteration for a constant.
        "p_keep": mx.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype),
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


def normalize(vector, norm=None):
    """Divide by the norm, leaving a zero vector untouched instead of producing NaN.

    Pass ``norm`` when the caller has already computed it -- several call sites need the norm
    itself for a separate zero test, and recomputing it here would cost an extra reduction per
    iteration.

    Module-level, mirroring ``rqutils.ground_locg.normalize``. It was previously a closure inside
    :func:`ground_locg_mlx` over that solve's cached ``one``, which meant the zero-norm guard --
    load-bearing against NaN, and one of the invariants ``docs/locg.md`` covers -- had to be
    re-inlined at every other call site rather than shared.
    """
    if norm is None:
        norm = mx.linalg.norm(vector)
    return vector / mx.where(norm == 0.0, _consts(norm.dtype)["one"], norm)


def apply_h_xz(vec, xsources, diagonals, chunk=16):
    """Return Hv from precomputed X sources and diagonals, via chunked batched gather.

    Mirrors ``rqutils.sqd.apply_h`` at ``cache_level=(1, 2)``. ``xsources`` must already be
    sanitized (no negative entries, with the corresponding diagonals zeroed) -- see
    ``examples/mlx/_bench_common.build_solver_inputs``.

    The obvious formulation loops over the J X-groups doing one ``take``+multiply+add per group,
    which is 3J Python-level MLX op constructions per matvec. Since ``xsources``/``diagonals`` are
    dense ``(J, N)`` arrays, groups are instead processed in chunks: gather a whole chunk with one
    flat ``take``, reshape, and reduce with a single weighted sum, cutting the op count from
    ``3*J`` to roughly ``3*ceil(J/chunk)``. The group-at-a-time version is gone -- it measured
    8.106 ms/iter against 4.393 for this one, the slowest of every configuration tried
    (``docs/mlx-metal-kernels.md``).

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

    See ``examples/mlx/_bench_common.apply_h_xz_chunked`` for the JAX equivalent used by the JAX
    arms of the benchmark -- batching the gather is applied symmetrically to both frameworks so
    the comparison stays about the solver loop, not about who has the better matvec.

    Verified (design doc, Optimization 1): max abs diff across chunk in
    {1, 4, 8, 16, 32, 50, 100, 128} is <= 2.7e-15, and every chunk matches ``H @ v`` to 1.8e-15.

    Args:
        vec: The vector to multiply, shape ``(N,)``.
        xsources: Sanitized X-source indices, shape ``(J, N)``.
        diagonals: Sanitized diagonals, shape ``(J, N)``, matching ``vec``'s dtype.
        chunk: Number of X-groups to gather per flat ``take``. J is static and small, so this
            is a plain Python int controlling how many groups are unrolled per flat gather.

    Returns:
        ``H @ vec``.
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

# Compiled-kernel cache, keyed by kernel name. `mx.fast.metal_kernel` compiles MSL, so each kernel
# must be built exactly once and reused; this replaces what used to be three near-identical
# `global _METAL_*_KERNEL` / `if ... is None` memos, one per kernel, differing only in the four
# arguments below. Same "build once into a module dict" shape as _CONST_CACHE above.
_KERNEL_CACHE = {}


def _get_kernel(name, input_names, output_names, source):
    """Build (once) and return the Metal kernel called ``name``."""
    kernel = _KERNEL_CACHE.get(name)
    if kernel is None:
        kernel = _KERNEL_CACHE[name] = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=source,
        )
    return kernel


def _get_metal_matvec_kernel():
    """Build (once) the fused gather-multiply-accumulate Metal kernel."""
    return _get_kernel(
        "sqd_apply_h_xz",
        ["vec", "xsources", "diagonals", "n_groups", "n_states"],
        ["out"],
        _METAL_MATVEC_SOURCE,
    )


def _apply_h_xz_metal(vec, xsources, diagonals, threadgroup=256):
    """Return Hv via a single fused custom Metal kernel.

    :func:`apply_h_xz` expresses the matvec as a sequence of MLX ops, so each one launches its own
    kernel and materializes a full intermediate array. This version computes
    ``out[i] = sum_j vec[xsources[j, i]] * diagonals[j, i]`` in one launch with the accumulator
    held in a per-thread register, so there are no intermediates at all.

    Selected by ``ground_locg_mlx(..., device="gpu")``; private because ``device`` is the supported
    way to reach it.

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
        ``H @ vec``, algebraically identical to :func:`apply_h_xz`.

    Raises:
        ValueError: If ``vec`` is not float32, since Metal has no float64.
    """
    if vec.dtype != mx.float32:
        raise ValueError(
            f"_apply_h_xz_metal requires float32 (Metal has no float64), got {vec.dtype}. "
            "Use apply_h_xz with device='cpu' for the f64 arms."
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


_METAL_EIG3_SOURCE = """
    // ONE thread does the whole 3x3 eigensolve. There is no output parallelism to exploit -- the
    // result is one eigenvalue plus a 3-vector -- so this kernel exists purely to collapse ~34 op
    // launches into 1, exactly the tradeoff that made the fused matvec a win and the fused
    // Rayleigh-Ritz reduction a loss (docs/mlx-metal-kernels.md). Here there is no reduction to
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


def _get_metal_eig3_kernel():
    """Build (once) the fused 3x3 eigensolve Metal kernel."""
    return _get_kernel(
        "sqd_eigenpair_3x3",
        ["mat"],
        ["theta", "kappa"],
        _METAL_EIG3_SOURCE,
    )


def _eigenpair_3x3_metal(mat):
    """Lowest eigenpair of a real symmetric 3x3 matrix in a single fused Metal launch.

    Algebraically the same computation as :func:`eigenpair_3x3` composed with
    :func:`_nullvec_3x3` -- balance, Cardano, the rank-aware seven-candidate null-vector search,
    and the closing Rayleigh quotient -- transcribed into one kernel so that ~34 op launches per
    LOBPCG iteration (measured: 44% of the whole body, ``examples/mlx/count_ops.py``) become one.
    Measured 1.75x per-iteration and 1.69x end-to-end with a bit-identical eigenvalue.

    Selected by ``ground_locg_mlx(..., device="gpu")``; private because ``device`` is the supported
    way to reach it.

    **Why fusing wins here when it lost for the Rayleigh-Ritz reduction.** Its work does not scale
    with N at all -- it is scalar arithmetic on nine numbers -- so there is nothing to parallelize
    and nothing to *under*-parallelize. A single thread doing ~34 ops' worth of register arithmetic
    replaces ~34 kernel launches whose only content is that same arithmetic. The fused
    Rayleigh-Ritz reduction had the opposite profile and measured slower; see
    ``docs/mlx-metal-kernels.md`` before fusing anything else here.

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
    it. Note this fragility has NOT been observed to cost iterations: in a controlled run the f64
    arms of both frameworks converged in 220 (MLX) versus 217 (JAX) iterations to the same
    eigenvalue. See ``docs/mlx-metal-kernels.md``.

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
            f"_eigenpair_3x3_metal requires float32 (Metal has no float64), got {mat.dtype}. "
            "Use device='cpu' for the f64 arms."
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


def ground_locg_mlx(mat, xinit, args=(), maxiter=1000, tol=None, device="cpu"):
    """Single-vector LOBPCG in MLX.

    Args:
        mat: Callable mapping ``(vec, *args)`` to ``A @ vec``. Pair it with ``device``:
            :func:`apply_h_xz` for ``device="cpu"``, :func:`_apply_h_xz_metal` for
            ``device="gpu"``. This function never substitutes a matvec of its own.
        xinit: Initial vector. Must have nonvanishing overlap with the ground state.
        args: Extra arguments forwarded to ``mat``.
        maxiter: Maximum gradient-descent iterations.
        tol: Convergence tolerance. ``None`` uses the dtype epsilon. ``0.`` disables the
            check, running exactly ``maxiter`` iterations with no device sync at all.
        device: Which implementation of the Rayleigh-Ritz eigensolve to use. ``"cpu"`` (default)
            is the portable op-graph :func:`eigenpair_3x3`, and is the **only** route for a
            float64 solve. ``"gpu"`` is the fused single-launch :func:`_eigenpair_3x3_metal`,
            which requires float32 since Metal has no float64. Bound once here rather than per
            iteration, so it cannot introduce a per-iteration device sync.

            Two things this parameter deliberately does *not* do. It does not inspect or replace
            ``mat`` -- see above. And it does not call ``mx.set_default_device``: array placement
            is the caller's job, and silently mutating process-global state the caller already
            manages would be worse than making them do it. ``device`` names which kernels this
            solve uses, not where its arrays live.

    Returns:
        ``(eigenvalue, eigenvector, iterations, converged)``. Check the fourth value rather than
        comparing the third against ``maxiter``, which is ambiguous when convergence happens on
        the final permitted iteration.

        One caveat on ``iterations`` when ``tol != 0.``: the convergence test runs at
        ``_COMPILE_CHUNK`` boundaries, so a solve that terminates because its search direction
        zeroed out mid-chunk reports the boundary count rather than the exact iteration. Never
        wrong -- a zeroed direction means no further iteration can lower the eigenvalue, so the
        remainder of the chunk is wasted work, not incorrect work -- but ``iterations`` can round
        up to a multiple of ``_COMPILE_CHUNK`` in that case.

    Raises:
        ValueError: If ``xinit`` is complex. Every kernel here assumes real symmetric input.
        ValueError: If ``device`` is not ``"cpu"`` or ``"gpu"``, or if ``device="gpu"`` is
            combined with a non-float32 ``xinit``.
    """
    xinit = mx.array(xinit)
    # Test the dtype's name rather than comparing against mx.complex64: this holds under real MLX
    # and under the numpy shim in examples/mlx/check_solver_headless.py, which defines only the
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
    if device not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")
    if device == "gpu" and xinit.dtype != mx.float32:
        # Validate HERE, not inside _eigenpair_3x3_metal, for three reasons. The message can name
        # the parameter the caller actually set. It fires before the seed step spends three
        # matvecs. And decisively: the seed_converged early return below exits without entering
        # the loop at all, so a guard pushed down into the kernel would never fire on a
        # diagonal-operator seed and an f64 device='gpu' call would silently succeed -- exactly the
        # "plausible wrong configuration accepted quietly" class this module's guards exist for.
        raise ValueError(
            f"device='gpu' requires float32 (Metal has no float64), got {xinit.dtype}. Use "
            "device='cpu' for an f64 solve."
        )
    # Bind the implementation once, before iterating: the dtype is fixed for the whole solve, so
    # a per-iteration dispatch would buy nothing and might force a device sync inside the
    # compiled body (see the design doc -- unverified, and avoided rather than risked).
    solve_eig3 = _eigenpair_3x3_metal if device == "gpu" else eigenpair_3x3
    check_convergence = tol != 0.0
    if tol is None:
        # Compare the dtype object directly rather than parsing its repr: this works
        # identically under real MLX and under the numpy shim used by the static check.
        tol = float(np.finfo(np.float32 if xinit.dtype == mx.float32 else np.float64).eps)

    def matvec(vec):
        return mat(vec, *args)

    # Constants fetched once per solve rather than rebuilt on every iter_body call (see _consts).
    # The dtype is fixed for the whole solve, so this lookup cannot change mid-run.
    _solve_consts = _consts(xinit.dtype)
    one = _solve_consts["one"]
    p_mask = _solve_consts["p_mask"]
    p_keep = _solve_consts["p_keep"]
    # Static Python int, fixed for the whole solve, so converged() does not need a vector argument
    # just to read a shape off it.
    num_states = xinit.shape[0]

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
    sas_mat = _compute_sas((xcurr, tmp_p), (ax, matvec(tmp_p)))
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
        sas_mat = _compute_sas((xcurr, ycurr, tmp_p), (axcurr, matvec(ycurr), matvec(tmp_p)))
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
        sas_mat = mx.where(p_is_zero, sas_mat * p_keep + excluded * p_mask, sas_mat)
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

    def converged(theta, rcurr, axnext, p_is_zero):
        # A zeroed search direction means {x, y} already spans the residual: we are at a stationary
        # point and no further iteration can lower theta. Test it FIRST, before building reltol --
        # that early return discards reltol's two op constructions, so computing them above the
        # branch was pure waste on the terminating iteration.
        #
        # This bool() forces a device sync -- the price of MLX having no while_loop.
        if bool(p_is_zero):
            return True
        # Compare the residual norm against the floating-point error we'd expect from forming the
        # residual at all. abs(theta) rather than +theta, and a sum rather than the difference the
        # first version of this port inherited: norm(Ax) - theta is a cancellation of two nearly
        # equal large positive numbers for a positive-definite operator, which was measured going
        # negative and made the test unsatisfiable, so the solver never converged and always burned
        # maxiter (item I4). A ground-state search is typically negative-definite, where +theta
        # would cancel in turn.
        reltol = (mx.linalg.norm(axnext) + mx.abs(theta)) * num_states * 10
        # ONE float() sync, not two: the comparison is folded into a single on-device subtraction
        # and only its sign is read back. `norm(r) < tol * reltol` as two separate float() calls
        # drains the MLX pipeline twice per iteration, and syncs are the top item in this repo's
        # cost model (see docs/mlx-metal-kernels.md). Exactly
        # equivalent -- 0 disagreements over 400k random magnitude pairs spanning 1e-30..1e10 in
        # both f32 and f64, since tol is a Python float and both operands are non-negative.
        return float(mx.linalg.norm(rcurr) - tol * reltol) < 0.0

    niter = 0
    is_converged = False
    if seed_converged:
        # {x} already spans the residual; the loop cannot improve on the seed pair.
        return float(theta), xcurr, 0, True
    # Compilation is unconditional. mx.compile was the single largest measured win here (2.71x on
    # the MLX CPU backend) and its trajectory was pinned bit-for-bit against the uncompiled one
    # before that path was removed, so there was no configuration in which not compiling was right.
    # What remains is the ONE distinction that is not a performance knob: whether a convergence
    # check happens at all. MLX has no while_loop/cond, so a convergence check is necessarily a
    # Python branch on a synced value, and mx.compile traces a fixed array-in/array-out computation
    # that cannot contain one. Hence two loop shapes rather than one.
    if not check_convergence:
        # tol=0.: no convergence check, so the compiled body runs every requested iteration with
        # ZERO device syncs -- the clean per-iteration measurement the benchmark reads. Compile
        # iter_body directly rather than a chunk: maxiter need not be a multiple of _COMPILE_CHUNK,
        # and with no sync to amortize there is nothing for chunking to buy.
        compiled_body = mx.compile(iter_body)
        for niter in range(1, maxiter + 1):
            theta, xcurr, ycurr, rcurr, axnext, _p_is_zero = compiled_body(
                xcurr, ycurr, rcurr, axnext
            )
    else:
        # Run _COMPILE_CHUNK iterations inside one compiled function, then sync once to check
        # convergence. This amortizes the unavoidable float()-sync over _COMPILE_CHUNK iterations
        # rather than paying it every iteration; syncs are the top item in this repo's cost model
        # (~389 us, ~53x a kernel launch -- see docs/mlx-metal-kernels.md).
        def chunk_body(xcurr, ycurr, rcurr, axcurr):
            # No theta/p_is_zero initializers: _COMPILE_CHUNK is a module constant of 10, so the
            # loop always runs at least once and any initial value would be dead. (They were not
            # merely redundant -- a zero-trip loop would have returned them, reporting theta = 0.)
            #
            # NOTE, a real behaviour consequence: only the LAST iteration's p_is_zero survives, so
            # a zeroed search direction arising mid-chunk is not reported until the next boundary.
            # Still correct, never wrong -- p_is_zero means no further iteration can lower theta, so
            # the rest of the chunk is wasted work rather than wrong work, is_converged is still
            # True, and only the reported niter can round up (see this function's Returns section).
            # Widening the compiled signature to an OR-accumulated flag would make niter exact at
            # the cost of an extra array in the traced carry; that trade was declined deliberately.
            for _ in range(_COMPILE_CHUNK):
                theta, xcurr, ycurr, rcurr, axcurr, p_is_zero = iter_body(
                    xcurr, ycurr, rcurr, axcurr
                )
            return theta, xcurr, ycurr, rcurr, axcurr, p_is_zero

        compiled_chunk = mx.compile(chunk_body)
        # `range`, not a hand-advanced `while niter < maxiter`: the old shape derived its progress
        # from `this_chunk = min(chunk, maxiter - niter)` and advanced by it three lines later, so a
        # non-positive chunk made the advance a no-op and the loop spun forever. Python's own range
        # semantics guarantee termination instead. That hang is now unconstructible anyway --
        # _COMPILE_CHUNK is a module constant -- which is why the guard that used to validate it is
        # gone, but keep this form rather than reintroducing the hand-advanced one.
        for start in range(0, maxiter, _COMPILE_CHUNK):
            this_chunk = min(_COMPILE_CHUNK, maxiter - start)
            if this_chunk == _COMPILE_CHUNK:
                theta, xcurr, ycurr, rcurr, axnext, p_is_zero = compiled_chunk(
                    xcurr, ycurr, rcurr, axnext
                )
            else:
                # Final partial chunk: run the body uncompiled so niter lands exactly on maxiter
                # rather than overshooting it.
                for _ in range(this_chunk):
                    theta, xcurr, ycurr, rcurr, axnext, p_is_zero = iter_body(
                        xcurr, ycurr, rcurr, axnext
                    )
            niter = start + this_chunk
            if converged(theta, rcurr, axnext, p_is_zero):
                is_converged = True
                break

    return float(theta), xcurr, niter, is_converged


def _compute_sas(vectors, mvs):
    """Return the (n x n) matrix of <v_i | A | v_j> for n in {2, 3}.

    Takes the images ``mvs`` rather than computing them, so a caller holding an already-known
    ``A x`` can pass it in instead of paying for a fourth matrix-vector product per iteration
    (``docs/locg.md`` item S1).

    All n^2 inner products come from ONE stacked ``(n, N) @ (N, n)`` matmul rather than n^2
    separate ``mx.sum(v1 * mv)`` reductions plus n+1 ``mx.stack`` calls -- 9 sums and 4.5 stacks
    per iteration at n=3 (measured, ``examples/mlx/count_ops.py --by-op``) collapsing to 3 ops.

    This is *more* accurate than the form it replaces, not merely faster: against a longdouble
    reference over 3000 random unit-norm triples at N=900, the matmul is exact (0.0 max error)
    while the n^2-sums version carries up to 7.1e-15, because BLAS accumulates in blocks
    (pairwise, error growing as log N) where ``mx.sum`` over a product array accumulates
    sequentially. The symmetrization is kept: it is what makes the two triangles agree exactly,
    and it costs 2 ops against the 3x3 result rather than anything O(N).

    Unlike :func:`_project_out`, this function carries no accumulation-order contract -- its own
    symmetrization already averages away the rounding difference between the two triangles -- so
    reassociating the sum here is safe in a way it would *not* be there. Do not apply the same
    transformation to ``_project_out``: those passes guard against catastrophic cancellation
    (``docs/locg.md`` items I5/I6) and a matmul reassociates their summation order, measured to
    shift results by ~1e-14.

    One visible consequence, expected and benign: because the accumulation order changes, **f32
    eigenvalues shift in their last one or two digits** relative to runs recorded before this
    change (e.g. -5.3960409164 versus -5.3960399628, ~1.8e-7 relative, against an f32 eps of
    1.19e-7), and the f32 arms no longer agree digit-for-digit across device settings.
    Iteration counts are unchanged and the correctness gate passes with a >500x margin at
    ``rtol=1e-4``. f64 is unaffected at the printed precision. Do not treat an f32 last-digit
    difference against a pre-matmul recorded value as a regression.
    """
    stacked_v = mx.stack(vectors)
    stacked_m = mx.stack(mvs)
    sas = stacked_v @ stacked_m.T
    # Symmetrize: the two triangles differ only by rounding for real symmetric A.
    return (sas + sas.T) * 0.5


def _project_out(basis, vector):
    """Orthogonalize ``vector`` against ``basis``, ending on a subtraction.

    The repeated passes and the final zeroing are load-bearing; see the comments in
    ``rqutils.ground_locg._project_out``, which this mirrors. Near convergence, ending on a
    normalization can reintroduce basis components through catastrophic cancellation and wreck the
    Rayleigh-Ritz conditioning.

    **Deliberately NOT batched into a matmul, unlike :func:`_compute_sas`.** The eight ``mx.sum``
    reductions here (four passes x two basis vectors, 13 ops/iteration total) could become one
    stacked ``B @ v`` per pass, which is what made ``_compute_sas`` 14.2 -> 2.3 ops. It is not
    done here because a matmul reassociates the summation order, and this function's whole purpose
    is resisting catastrophic cancellation (``docs/locg.md`` items I5/I6, both measured to fail
    silently). Tested rather than assumed: over 4000 adversarial cases with ``r`` placed almost
    entirely inside ``span(x, y)`` plus an orthogonal part of size 1e-14..1e-6, both forms hold
    residual orthogonality at machine epsilon, but the matmul is consistently *worse* -- worst
    ``|<b|p>|`` of 8.3e-17 versus 6.2e-17. That margin is small and neither form is broken, so this
    is a judgement call, not a measured failure: ~4 ops/iteration is not worth a 33% erosion of the
    quantity these guards exist to protect. Do not "optimize" this without re-running that
    comparison.
    """
    for _ in range(2):
        ips = [mx.sum(vb * vector) for vb in basis]
        for vb, ip in zip(basis, ips):
            vector = vector - vb * ip
        vector = normalize(vector)

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
    consts = _consts(mat.dtype)
    eye = consts["eye3"]
    comps = mx.linalg.cross(mx.broadcast_to(col, (3, 3)), eye)
    # Rank 0 (a multiple of the identity): every candidate above is zero, so offer an arbitrary
    # unit vector as the last resort. It has residual 0 and wins by default.
    cands = [crosses, comps, eye[:1]]

    # Stack FIRST, then normalize the whole (7, 3) block in one pass. Normalizing candidate by
    # candidate -- what the JAX original does, and what this port copied -- costs 3 op launches per
    # candidate, 21 per iteration (measured: 18% of the whole LOBPCG body, see
    # examples/mlx/count_ops.py). In JAX that loop is free because XLA fuses it into the surrounding
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
    # `consts` from above, not a second `_consts(norms.dtype)` lookup: norms is the norm of a real
    # array, so its dtype is mat.dtype by construction, and re-deriving it invited the reader to
    # wonder whether the two could differ.
    cands = cands / mx.where(norms == 0.0, consts["one"], norms)
    # `cands @ mat`, not `cands @ mat.T`: this function's contract is a real *symmetric* matrix
    # (see the docstring, and eigenpair_3x3 only ever passes `balanced - xmin*eye`, symmetric by
    # construction), so the transpose is an exact no-op -- verified identical -- and costs an op
    # launch per iteration. The JAX original writes the einsum in the equivalent `mat @ cands_i`
    # order. If this ever needs to accept an asymmetric matrix, the transpose must come back.
    resid = mx.linalg.norm(cands @ mat, axis=1)
    # A candidate that collapsed to zero is not a valid eigenvector; disqualify it with a large
    # FINITE penalty proportional to the residual scale -- the comparison only needs an ordering.
    #
    # `mx.inf` does exist, so `mx.where(alive, resid, mx.inf)` would mirror the JAX original more
    # closely and in one fewer op. It is avoided deliberately: the rank-0 case (a multiple of the
    # identity) zeroes the first six candidates, and the surviving e_0 fallback has residual exactly
    # 0, so an inf-filled vector would be six infs and one zero. That works, but it puts non-finite
    # values into an array this function then reduces with argmin, and every other guard in this
    # module is written to keep NaN/inf out of the graph entirely rather than rely on comparison
    # semantics. A finite penalty is the same ordering with no non-finite intermediates.
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

    Reference: J. Kopp, Int. J. Mod. Phys. C. 19, 523 (2008).

    ``mx.linalg.eigh`` does exist and would accept this input directly; the analytic route is kept
    because a per-iteration eigensolve that crosses the device boundary measured 1.37x slower, not
    because MLX lacks an eigensolver. An earlier note here claimed Cardano's near-degeneracy error
    costs the f64 CPU arm 217 iterations against JAX's 89, which would have made an ``eigh``-backed
    f64 path a large win; a controlled run retired that claim (220 vs 217 -- both frameworks use the
    same formulation, so they converge alike). ``docs/mlx-metal-kernels.md`` has the numbers.
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
    # Bare 0.0, as ground_locg does: mx.maximum promotes a Python scalar without constructing an
    # array, so this needs no cached constant (unlike the mx.where operands, which must be arrays
    # to match dtype).
    p = mx.maximum(-3.0 * c1, 0.0)
    disc = mx.maximum(-27.0 * c1 * c1 * c1 - 182.25 * c0 * c0, 0.0)
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
