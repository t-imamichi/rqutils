"""Count MLX op constructions per LOBPCG iteration, without a Metal device.

Re-executes ``rqutils/ground_locg_mlx.py`` against a numpy shim (the same technique as
``examples/mlx/check_solver_headless.py``) with every shim entry point wrapped in a counter,
then attributes the counts to the function that constructed them by walking the Python stack.

This measures **op-construction count**, which is what MLX's lazy graph builder turns into kernel
launches. It is not a timing measurement and cannot be one: there is no GPU here. Its purpose is
to say *where* the launches are, so an optimization targets the real hot spot -- the cost model in
``docs/mlx-metal-kernels.md`` ("sync count dominates, launch count is secondary") is what turns a
launch count into a prediction.

Run with:
    uv run python examples/mlx/count_ops.py
"""

import collections
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# HERE is examples/mlx/, so the package root is two levels up.
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "rqutils", "ground_locg_mlx.py")

# Functions in the module under test that we attribute ops to. Anything constructed outside these
# (e.g. directly in ground_locg_mlx's body) is attributed to its own frame name.
_TRACKED = (
    "apply_h_xz",
    "_apply_h_xz_metal",
    "_compute_sas",
    "_project_out",
    "eigenpair_2x2",
    "eigenpair_3x3",
    "_nullvec_3x3",
    "_eigenpair_3x3_metal",
    "iter_body",
    "chunk_body",
    "normalize",
    "matvec",
    "converged",
)

COUNTS = collections.Counter()
_ENABLED = [False]


def _attribute():
    """Return the innermost tracked function name on the stack."""
    frame = sys._getframe(2)
    while frame is not None:
        name = frame.f_code.co_name
        if name in _TRACKED:
            return name
        frame = frame.f_back
    return "<other>"


def _counting(fn, label):
    def wrapper(*args, **kwargs):
        if _ENABLED[0]:
            COUNTS[(_attribute(), label)] += 1
        return fn(*args, **kwargs)

    return wrapper


def build_shim():
    """A numpy shim for mlx.core with every op wrapped in a counter."""
    shim = types.ModuleType("mx")
    for fn in (
        "sum",
        "sqrt",
        "where",
        "stack",
        "zeros_like",
        "take",
        "array",
        "abs",
        "minimum",
        "maximum",
        "argmin",
        "real",
        "imag",
        "conjugate",
        "zeros",
        "arange",
        "cos",
        "sin",
        "arctan2",
        "diagonal",
        "roll",
        "prod",
        "square",
        "insert",
        "concatenate",
        "min",
        "max",
        "eye",
        "sign",
        "argmax",
        "logical_or",
        "matmul",
        "broadcast_to",
    ):
        if hasattr(np, fn):
            setattr(shim, fn, _counting(getattr(np, fn), fn))
    shim.linalg = types.SimpleNamespace(
        norm=_counting(np.linalg.norm, "linalg.norm"),
        cross=_counting(np.cross, "linalg.cross"),
    )
    shim.float32, shim.float64 = np.float32, np.float64
    shim.eval = lambda *a, **k: None
    shim.compile = lambda f: f
    shim.Dtype = type(np.dtype("float64"))
    shim.finfo = np.finfo

    def _shim_metal_kernel(name, input_names, output_names, source, **kwargs):
        def call_matvec(inputs, output_dtypes=None, **kw):
            vec, xsources, diagonals, num_groups, num_states = inputs
            xs_flat = np.asarray(xsources).reshape(-1)
            dg_flat = np.asarray(diagonals).reshape(-1)
            vec_np = np.asarray(vec)
            out = np.zeros(num_states, dtype=np.dtype(output_dtypes[0]))
            for j in range(num_groups):
                off = j * num_states
                out = (
                    out + vec_np[xs_flat[off : off + num_states]] * dg_flat[off : off + num_states]
                )
            return [out]

        def call_eig3(inputs, output_dtypes=None, **kw):
            # Only the op COUNT matters here: one launch replaces the whole eigensolve. The
            # arithmetic is validated in check_solver_headless.py, so this returns a
            # cheap-but-valid eigenpair via numpy rather than re-deriving Cardano.
            (mat,) = inputs
            m = np.asarray(mat, dtype=np.float64)
            vals, vecs = np.linalg.eigh((m + m.T) * 0.5)
            return [
                np.array([vals[0]], dtype=np.dtype(output_dtypes[0])),
                np.asarray(vecs[:, 0], dtype=np.dtype(output_dtypes[1])),
            ]

        # One launch per kernel call, attributed to the calling function.
        if name == "sqd_apply_h_xz":
            return _counting(call_matvec, "metal_kernel:matvec")
        if name == "sqd_eigenpair_3x3":
            return _counting(call_eig3, "metal_kernel:eig3")
        raise AssertionError(f"no shim for kernel {name!r}")

    shim.fast = types.SimpleNamespace(metal_kernel=_shim_metal_kernel)
    return shim


def load_module(shim):
    module = types.ModuleType("ground_locg_mlx_counted")
    module.__dict__["mx"] = shim
    with open(SRC) as handle:
        src = handle.read()
    # Strip the `import mlx.core as mx` so the shim binding survives.
    src = src.replace("import mlx.core as mx", "mx = mx  # shimmed")
    # exec is the point: the module's own source text is re-executed against the counting shim, so
    # the counts describe the real ground_locg_mlx.py rather than a copy that could drift from it.
    # Same technique, and same suppression, as examples/mlx/check_solver_headless.py.
    exec(compile(src, SRC, "exec"), module.__dict__)  # noqa: S102
    return module


def make_problem(num_states=400, num_groups=12, seed=3):
    """A small symmetric problem in the xsources/diagonals form the matvec consumes."""
    rng = np.random.default_rng(seed)
    xsources = np.zeros((num_groups, num_states), dtype=np.int64)
    diagonals = np.zeros((num_groups, num_states))
    dense = np.zeros((num_states, num_states))
    for group in range(num_groups):
        perm = rng.permutation(num_states) if group else np.arange(num_states)
        xsources[group] = perm
        vals = rng.normal(size=num_states)
        diagonals[group] = vals
        dense[np.arange(num_states), perm] += vals
    dense = (dense + dense.T) * 0.5
    # Recover xsources/diagonals consistent with the symmetrized dense matrix is not needed:
    # the op COUNT does not depend on the numbers being a faithful Hamiltonian, only on shapes.
    return xsources, diagonals, dense


def main():
    shim = build_shim()
    module = load_module(shim)
    xsources, diagonals, _ = make_problem()
    num_states = xsources.shape[1]
    vinit = np.random.default_rng(0).normal(size=num_states)

    # The two configurations ground_locg_mlx now offers. device="gpu" is f32-only, since Metal has
    # no float64; device="cpu" is the only route for an f64 solve and so is measured at f64.
    for label, matvec, dtype, device in (
        ("cpu (op-graph eig, f64)", module.apply_h_xz, np.float64, "cpu"),
        ("gpu (fused matvec + eig, f32)", module._apply_h_xz_metal, np.float32, "gpu"),
    ):
        COUNTS.clear()
        iters = 6
        _ENABLED[0] = True
        module.ground_locg_mlx(
            matvec,
            vinit.astype(dtype),
            args=(xsources, diagonals.astype(dtype)),
            maxiter=iters,
            tol=0.0,
            device=device,
        )
        _ENABLED[0] = False

        per_func = collections.Counter()
        for (func, _op), n in COUNTS.items():
            per_func[func] += n
        # Attribute only the steady-state loop body: seed-iteration ops are paid once, not
        # per iteration, so dividing them by `iters` would understate the per-iteration cost of
        # the body and overstate the seed's.
        #
        # NOTE what this total does NOT include: the matvec implementation's own ops. `matvec` below
        # is ground_locg_mlx's internal closure, not `apply_h_xz`/`_apply_h_xz_metal`, so the gather
        # ops are attributed to those frames and fall outside this filter. That is why the cpu arm
        # reports the same 65.0 as every previous op-graph measurement even though the matvec it uses
        # changed: this number measures the Rayleigh-Ritz-and-below body, and comparing matvecs is
        # bench.py's job (they differ in launches AND in memory traffic, which no op count sees).
        body = {
            f: n
            for f, n in per_func.items()
            if f
            in (
                "iter_body",
                "_compute_sas",
                "_project_out",
                "eigenpair_3x3",
                "_eigenpair_3x3_metal",
                "_nullvec_3x3",
                "matvec",
                # converged() runs once per iteration whenever tol != 0, so its ops are part of the
                # per-iteration cost. It was previously listed in _TRACKED but omitted here, which
                # understated the body total for any convergence-checking solve. Note the arms below
                # run with tol=0.0 (fixed-iteration), so it contributes 0 there -- include it so a
                # future tol!=0 measurement is not silently short.
                "converged",
                "normalize",
            )
        }
        total_body = sum(body.values())
        print(f"\n=== {label}: op constructions, {iters} iterations ===")
        print(f"{'function':<22}{'total':>8}{'per_iter':>10}{'share':>8}")
        print("-" * 48)
        for func, n in sorted(body.items(), key=lambda kv: -kv[1]):
            print(f"{func:<22}{n:>8}{n / iters:>10.1f}{n / total_body * 100:>7.1f}%")
        print("-" * 48)
        print(f"{'TOTAL (body)':<22}{total_body:>8}{total_body / iters:>10.1f}")

        if "--by-op" in sys.argv:
            # Per-op breakdown within each function, so a regression can be traced to the
            # individual construction that caused it rather than just the function total.
            for func in sorted(body, key=lambda f: -per_func[f]):
                ops = collections.Counter()
                for (owner, op), n in COUNTS.items():
                    if owner == func:
                        ops[op] += n
                detail = "  ".join(f"{op}:{n / iters:g}" for op, n in ops.most_common())
                print(f"    {func:<20}{detail}")


if __name__ == "__main__":
    main()
