# MLX SQD Solver PoC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the SQD eigensolver loop (`ground_locg` + `apply_h_xz_cached`) to MLX and benchmark it against JAX across six CPU/GPU/precision arms.

**Architecture:** Three files under `examples/`, nothing under `rqutils/` changes. Setup (state uniquification, X-source lookup, diagonal composition) stays in JAX on CPU and runs once, untimed; only the solver loop is ported and timed. Every arm consumes byte-identical `xsources`/`diagonals`/`vinit` arrays, so timing differences are attributable to the solver alone.

**Tech Stack:** Python 3.14, numpy, JAX 0.11.0, MLX 0.32.0, scipy (hard deps already present). No pytest — this repo has no test suite; every check is a `uv run python -c` or `uv run python <script>` invocation.

**Spec:** `docs/superpowers/specs/2026-08-03-mlx-sqd-poc-design.md` (commit `e2bea1c`)

## Global Constraints

- Always use `uv run python`, never bare `python`. The venv at `.venv` is uv-managed.
- **Nothing under `rqutils/` may be modified.** This is a PoC; all new code lives in `examples/`.
- **MLX cannot be executed in the authoring session.** `mlx.core` loads a Metal device even for `mx.cpu` arrays; a headless session fails with `RuntimeError: [metal::load_device] No Metal device available`. Every task below is split so its non-MLX parts are verified by the implementer and its MLX parts are verified by the user. **Never claim an MLX arm works without user-supplied output.**
- **Do not report any benchmark number you have not seen output for.** There is no pytest suite; "tests pass" is never a valid claim here.
- Line width 100 chars. Pre-commit runs whitespace/EOF/YAML hooks only — no linter, no type-checker.
- Hamiltonians must be generated with an **even number of `Y`s per Pauli string**, so `PauliSumXZ.from_paulisum(..., force_real=True)` yields `c.dtype == float64`. With odd-Y strings the `(-i)^{x·z}` phase makes coefficients complex128, which MLX cannot represent at all.
- MLX has no `float64` on Metal and no `complex128` anywhere. `mlx-gpu` arms are fp32-only.
- `hproj` is broken (raises `TypeError` on shape mismatch) and must not be used as a reference.
- Fixed-iteration mode is `tol=0.` (verified: forces exactly `maxiter` iterations in both precisions).
- MLX is lazy — any timed MLX region must end with `mx.eval(...)` or the measurement is meaningless.

## File Structure

| file | responsibility |
|---|---|
| `examples/_bench_common.py` | Seeded even-Y problem generation; JAX setup producing sanitized `xsources`/`diagonals`/`vinit`; dense reference matrix; brute-force reference; `timeit` for both frameworks. No MLX import at module level. |
| `examples/ground_locg_mlx.py` | The port: `apply_h_xz_mlx` + `ground_locg_mlx`. Imports `mlx.core` at module top (it is the MLX-only file). |
| `examples/bench_mlx.py` | Arm registry, correctness gate, timing, reporting, `--arm`/`--all`/`--json`. Imports MLX lazily, per arm. |

Task order matters: Task 1 is fully verifiable without MLX, Task 2 is verifiable without MLX only in part, Task 3 depends on both.

---

### Task 1: Problem generation, JAX setup, and references

**Files:**
- Create: `examples/_bench_common.py`
- Check: `/tmp/check_task1.py` (throwaway, not committed)

**Interfaces:**
- Produces, consumed by Tasks 2 and 3:
  - `generate_problem(num_qubits: int, num_paulis: int, num_states: int, seed: int) -> tuple[list[str], np.ndarray, np.ndarray]` returning `(pauli_strings, coeffs, states)`; every Pauli string has an even number of `Y`s; `states` is `uint8` shape `(num_states, num_qubits)`.
  - `SolverInputs` dataclass with fields `xsources: np.ndarray` (int32, `(J, N)`), `diagonals: np.ndarray` (float64, `(J, N)`), `vinit: np.ndarray` (float64, `(N,)`), `subspace_dim: int` (= N), `num_xgroups: int` (= J).
  - `build_solver_inputs(pauli_strings, coeffs, states) -> SolverInputs` — runs the JAX setup and returns **sanitized** arrays (no `-1` entries).
  - `dense_reference(inputs: SolverInputs) -> tuple[np.ndarray, float]` returning `(H, ground_energy)` built from the solver inputs.
  - `brute_force_reference(pauli_strings, coeffs, states) -> float` — independent full-2^n construction.
  - `timeit(fn, repeat, sync) -> tuple[float, float]` returning `(compile_seconds, mean_steady_seconds)`; `sync` is a callable applied to `fn()`'s result to force completion.

- [ ] **Step 1: Write the check script that must fail**

Create `/tmp/check_task1.py`:

```python
"""Verify _bench_common: even-Y generation, sanitized setup, two agreeing references."""
import sys
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
sys.path.insert(0, 'examples')
from _bench_common import (generate_problem, build_solver_inputs, dense_reference,
                           brute_force_reference, timeit)

ps, cs, states = generate_problem(num_qubits=10, num_paulis=20, num_states=200, seed=1)
assert len(ps) == 20 and len(cs) == 20
assert all(p.count('Y') % 2 == 0 for p in ps), 'generated an odd-Y Pauli string'
assert states.shape == (200, 10) and states.dtype == np.uint8

# The even-Y property is what makes the coefficients real; assert it explicitly.
from rqutils.paulis.symplectic import PauliSumXZ
ham = PauliSumXZ.from_paulisum((ps, cs), force_real=True, add_padding=True)
assert ham.c.dtype == np.float64, f'coeffs not real: {ham.c.dtype}'

inputs = build_solver_inputs(ps, cs, states)
assert inputs.xsources.dtype == np.int32, inputs.xsources.dtype
assert inputs.diagonals.dtype == np.float64, inputs.diagonals.dtype
assert inputs.xsources.min() >= 0, 'xsources not sanitized: still contains -1'
assert inputs.xsources.shape == inputs.diagonals.shape
assert inputs.vinit.shape == (inputs.subspace_dim,)
assert np.isclose(np.linalg.norm(inputs.vinit), 1.0), 'vinit not normalized'

H, ref = dense_reference(inputs)
assert np.abs(H - H.T).max() == 0.0, f'reference not symmetric: {np.abs(H - H.T).max()}'

bf = brute_force_reference(ps, cs, states)
assert abs(ref - bf) < 1e-9 * max(1.0, abs(bf)), f'references disagree: {ref} vs {bf}'

# timeit must return two positive floats and actually call fn
calls = []
c, s = timeit(lambda: calls.append(1) or np.zeros(10), repeat=3, sync=lambda r: r)
assert len(calls) == 4, f'expected 1 warmup + 3 timed, got {len(calls)}'
assert c > 0 and s >= 0

print(f'OK  N={inputs.subspace_dim} J={inputs.num_xgroups}')
print(f'OK  dense_reference={ref:.12f}  brute_force={bf:.12f}  diff={abs(ref-bf):.2e}')
```

- [ ] **Step 2: Run it to make sure it fails**

```bash
cd /Users/ima/tasks/quantum/rqutils && uv run python /tmp/check_task1.py
```

Expected: `ModuleNotFoundError: No module named '_bench_common'`

- [ ] **Step 3: Write `examples/_bench_common.py`**

Note on `vinit`: it mirrors `sqd`'s `vinit_from_min_diag` — a one-hot vector at the minimum of the first X-group's diagonal when that group is the identity X signature (`all(ham.x[0] == 0)`), else index 0. It is normalized here so all arms start from an identical unit vector.

```python
"""Shared problem generation, setup, and timing for the JAX-vs-MLX SQD benchmarks.

The setup stage (uniquification, X-source lookup, diagonal composition) always runs in JAX on
CPU and is deliberately *not* timed: every benchmark arm consumes the identical arrays this
module produces, so timing differences are attributable to the solver alone.

This module must not import mlx at module level -- the JAX-only arms run in processes where
mlx may be unavailable.
"""
from dataclasses import dataclass
import time
from collections.abc import Callable
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import uniquify_states, get_xsource, get_diagonal

_PAULI_MATRICES = {
    'I': np.eye(2, dtype=np.complex128),
    'X': np.array([[0., 1.], [1., 0.]], dtype=np.complex128),
    'Y': np.array([[0., -1.j], [1.j, 0.]], dtype=np.complex128),
    'Z': np.diag([1., -1.]).astype(np.complex128)
}


@dataclass
class SolverInputs:
    """Everything the solver loop needs, with invalid X sources already neutralized."""
    xsources: np.ndarray
    diagonals: np.ndarray
    vinit: np.ndarray
    subspace_dim: int
    num_xgroups: int


def generate_problem(
    num_qubits: int,
    num_paulis: int,
    num_states: int,
    seed: int = 0
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Generate a random Hamiltonian with real coefficients, plus a subspace of states.

    Every Pauli string carries an even number of Ys. This makes the phased coefficients
    ``alpha * (-i)^{x.z}`` real, which matters because MLX has no complex128: a general
    (odd-Y) Hamiltonian cannot be represented in MLX at any precision.
    """
    rng = np.random.default_rng(seed)
    pauli_strings = []
    while len(pauli_strings) < num_paulis:
        chars = rng.choice(list('IXYZ'), size=num_qubits)
        if np.count_nonzero(chars == 'Y') % 2:
            continue
        pauli_strings.append(''.join(chars))

    coeffs = rng.uniform(-1., 1., num_paulis)
    states = rng.choice(2, size=(num_states, num_qubits)).astype(np.uint8)
    return pauli_strings, coeffs, states


def build_solver_inputs(
    pauli_strings: list[str],
    coeffs: np.ndarray,
    states: np.ndarray
) -> SolverInputs:
    """Run the JAX setup stage and return sanitized solver inputs.

    Sanitization: JAX's ``.at[].get(mode='fill', fill_value=0.)`` maps ``xsource == -1``
    (no source state in the subspace) to zero. MLX's ``take`` has no fill mode and its
    out-of-bounds behavior is undocumented, so instead of relying on it we clamp the index
    to 0 and zero the matching diagonal. The gathered value is then arbitrary but multiplied
    by 0., which is algebraically identical and costs nothing inside the solver loop.
    Both frameworks receive these same arrays, so neither is advantaged.
    """
    hamiltonian = PauliSumXZ.from_paulisum((pauli_strings, coeffs), force_real=True,
                                           add_padding=True)
    if hamiltonian.c.dtype != np.float64:
        raise ValueError(f'Hamiltonian coefficients are {hamiltonian.c.dtype}, expected float64.'
                         ' Pauli strings must have an even number of Ys.')

    states_p = np.packbits(np.pad(states.astype(np.uint8), {1: (1, 0)}), axis=1)
    subspace_dim = int(np.unique(states_p, axis=0).shape[0])
    states_u = uniquify_states(states_p, subspace_dim)

    xsources = np.stack([np.asarray(get_xsource(x, states_u)) for x in hamiltonian.x])
    diagonals = np.stack([np.asarray(get_diagonal(z, c, states_u))
                          for z, c in zip(hamiltonian.z, hamiltonian.c)])

    valid = xsources >= 0
    xsources = np.where(valid, xsources, 0).astype(np.int32)
    diagonals = np.where(valid, diagonals, 0.).real.astype(np.float64)

    # Mirror sqd's vinit_from_min_diag: one-hot at the minimum diagonal entry.
    if np.all(hamiltonian.x[0] == 0):
        start = int(np.argmin(diagonals[0]))
    else:
        start = 0
    vinit = np.zeros(subspace_dim, dtype=np.float64)
    vinit[start] = 1.

    return SolverInputs(xsources, diagonals, vinit, subspace_dim, xsources.shape[0])


def dense_reference(inputs: SolverInputs) -> tuple[np.ndarray, float]:
    """Build the projected Hamiltonian densely from the solver inputs and diagonalize it.

    Used instead of ``rqutils.sqd.hproj``, which raises a shape-mismatch TypeError: it builds
    the Hamiltonian with add_padding=True but packs the states without the pad bit.
    """
    dim = inputs.subspace_dim
    matrix = np.zeros((dim, dim), dtype=np.float64)
    rows = np.arange(dim)
    for xsource, diagonal in zip(inputs.xsources, inputs.diagonals):
        np.add.at(matrix, (rows, xsource), diagonal)
    return matrix, float(np.linalg.eigvalsh(matrix)[0])


def brute_force_reference(
    pauli_strings: list[str],
    coeffs: np.ndarray,
    states: np.ndarray
) -> float:
    """Ground energy via the full 2^n matrix, projected onto the unique states.

    Independent of the whole packing/padding/uniquification chain, so agreement with
    dense_reference validates that chain. Costs ~0.13 s at n=10; do not call it for large n.
    """
    num_qubits = states.shape[1]
    full = np.zeros((2 ** num_qubits,) * 2, dtype=np.complex128)
    for string, coeff in zip(pauli_strings, coeffs):
        operator = np.array([[1.]], dtype=np.complex128)
        for char in string:
            operator = np.kron(operator, _PAULI_MATRICES[char])
        full += coeff * operator

    unique = np.unique(states, axis=0)
    # Row bits are most-significant-first, matching the Pauli string character order.
    indices = unique.dot(1 << np.arange(num_qubits)[::-1])
    projected = full[np.ix_(indices, indices)]
    return float(np.linalg.eigvalsh(projected)[0].real)


def timeit(
    fn: Callable[[], Any],
    repeat: int,
    sync: Callable[[Any], Any]
) -> tuple[float, float]:
    """Return (compile_seconds, mean_steady_seconds).

    ``sync`` must force the computation to complete: ``jax.block_until_ready`` for JAX,
    ``mx.eval`` for MLX. Without it both frameworks return before the work is done -- MLX is
    lazy and JAX is async -- and the measurement is meaningless.
    """
    start = time.perf_counter()
    sync(fn())
    compile_time = time.perf_counter() - start

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        sync(fn())
        times.append(time.perf_counter() - start)
    return compile_time, float(np.mean(times))
```

- [ ] **Step 4: Run the check and confirm it passes**

```bash
cd /Users/ima/tasks/quantum/rqutils && uv run python /tmp/check_task1.py
```

Expected: two `OK` lines. `N=181 J=20`, and the two references agreeing to ~1e-15.

If `references disagree` fires, the bug is almost certainly a dropped imaginary
part or a bit-order mismatch in `brute_force_reference` — the row bits of
`states` are most-significant-first, matching the Pauli string character order.

- [ ] **Step 5: Commit**

```bash
git add examples/_bench_common.py
git commit -m "Add shared problem generation and references for MLX SQD benchmark

Setup runs in JAX and is untimed; all arms consume identical sanitized
arrays. Invalid X sources (~48% of entries) are clamped to index 0 with
their diagonals zeroed, so MLX's take needs no fill mode.

Two independent references (dense-from-solver-inputs and brute-force
2^n) cross-validate the packing/padding/uniquification chain, replacing
hproj which raises on a shape mismatch."
```

---

### Task 2: The MLX port

**Files:**
- Create: `examples/ground_locg_mlx.py`
- Check: `/tmp/check_task2_static.py` (implementer-runnable), `/tmp/check_task2_mlx.py` (**user-run only**)

**Interfaces:**
- Consumes from Task 1: `SolverInputs` (fields `xsources`, `diagonals`, `vinit`, `subspace_dim`, `num_xgroups`).
- Produces, consumed by Task 3:
  - `apply_h_xz_mlx(vec, xsources, diagonals)` → `mx.array`. Signature mirrors `rqutils.sqd.apply_h_xz_cached`.
  - `ground_locg_mlx(mat, xinit, args=(), maxiter=1000, tol=None) -> tuple[float, mx.array, int]`. Signature and return order mirror `rqutils.ground_locg.ground_locg`. `tol=0.` disables convergence checking (fixed-iteration mode, no per-iteration sync); `tol=None` defaults to `float(np.finfo(dtype).eps)`.

- [ ] **Step 1: Write the static check (no MLX execution)**

This check verifies everything checkable without a Metal device: that the module parses, that its public names exist with the right signatures, and that the numerical algorithm is right — by executing the *same source* with a numpy shim standing in for `mx`. The shim works because the port only uses `take`/`sum`/`sqrt`/`where`/`stack`/`zeros_like`/`linalg.norm`/`linalg.cross`, all of which numpy provides with identical semantics for our real-valued, in-bounds case.

Create `/tmp/check_task2_static.py`:

```python
"""Verify ground_locg_mlx without a Metal device.

Parses the module, checks its API, then re-executes its source with a numpy shim bound to
the name `mx` to validate the numerics. This catches algorithm transcription errors -- the
kind that matter most -- without needing MLX to initialize.
"""
import ast
import inspect
import sys
import types
import numpy as np

SRC = 'examples/ground_locg_mlx.py'
source = open(SRC).read()

# 1. It must parse, and must define the two public functions.
tree = ast.parse(source)
defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
for name in ('apply_h_xz_mlx', 'ground_locg_mlx'):
    assert name in defined, f'{name} not defined in {SRC}'
print('OK  module parses and defines both public functions')

# 2. Build a numpy shim exposing the mlx.core surface the port uses.
shim = types.ModuleType('mx')
for fn in ('sum', 'sqrt', 'where', 'stack', 'zeros_like', 'take', 'array', 'abs',
           'minimum', 'maximum', 'argmin', 'real', 'imag', 'conjugate', 'zeros',
           'arange', 'cos', 'sin', 'arctan2', 'diagonal', 'roll', 'prod', 'square',
           'insert', 'concatenate', 'min', 'max'):
    if hasattr(np, fn):
        setattr(shim, fn, getattr(np, fn))
shim.linalg = types.SimpleNamespace(norm=np.linalg.norm, cross=np.cross)
shim.float32, shim.float64 = np.float32, np.float64
shim.eval = lambda *a, **k: None
shim.compile = lambda f: f
shim.Dtype = type(np.dtype('float64'))
shim.finfo = np.finfo


class _CPU:
    pass


shim.cpu = _CPU()
shim.gpu = _CPU()
shim.set_default_device = lambda d: None
shim.default_device = lambda: shim.cpu

module = types.ModuleType('ground_locg_mlx_shimmed')
module.__dict__['np'] = np
sys.modules['mlx'] = types.ModuleType('mlx')
sys.modules['mlx.core'] = shim
exec(compile(source, SRC, 'exec'), module.__dict__)
print('OK  module executes against the numpy shim')

apply_h_xz_mlx = module.apply_h_xz_mlx
ground_locg_mlx = module.ground_locg_mlx

sig = inspect.signature(ground_locg_mlx)
for param in ('mat', 'xinit', 'args', 'maxiter', 'tol'):
    assert param in sig.parameters, f'ground_locg_mlx missing parameter {param}'
print('OK  ground_locg_mlx signature matches ground_locg')

# 3. Numerics, against the same problem Task 1 verified.
import jax
jax.config.update('jax_enable_x64', True)
sys.path.insert(0, 'examples')
from _bench_common import generate_problem, build_solver_inputs, dense_reference

ps, cs, states = generate_problem(10, 20, 200, seed=1)
inputs = build_solver_inputs(ps, cs, states)
H, ref = dense_reference(inputs)

# 3a. matvec must equal H @ v
rng = np.random.default_rng(7)
v = rng.normal(size=inputs.subspace_dim)
got = np.asarray(apply_h_xz_mlx(v, inputs.xsources, inputs.diagonals))
err = np.abs(got - H @ v).max()
assert err < 1e-12, f'apply_h_xz_mlx disagrees with H @ v by {err}'
print(f'OK  matvec matches H @ v (max err {err:.2e})')

# 3b. full solve must reach the reference eigenvalue
eigval, eigvec, iters = ground_locg_mlx(apply_h_xz_mlx, inputs.vinit,
                                        args=(inputs.xsources, inputs.diagonals))
assert abs(eigval - ref) < 1e-9 * max(1., abs(ref)), \
    f'solver got {eigval}, reference {ref}'
assert 0 < iters <= 1000, f'implausible iteration count {iters}'
print(f'OK  solve: eig={eigval:.12f} ref={ref:.12f} iters={iters}')

# 3c. tol=0. must run exactly maxiter iterations (fixed-iteration mode)
_, _, fixed_iters = ground_locg_mlx(apply_h_xz_mlx, inputs.vinit,
                                    args=(inputs.xsources, inputs.diagonals),
                                    maxiter=100, tol=0.)
assert fixed_iters == 100, f'tol=0. ran {fixed_iters} iterations, expected exactly 100'
print('OK  tol=0. gives fixed-iteration mode')
print('\nALL STATIC CHECKS PASSED (numpy shim; MLX itself still unverified)')
```

- [ ] **Step 2: Run it to make sure it fails**

```bash
cd /Users/ima/tasks/quantum/rqutils && uv run python /tmp/check_task2_static.py
```

Expected: `FileNotFoundError: examples/ground_locg_mlx.py`

- [ ] **Step 3: Write `examples/ground_locg_mlx.py`**

Transcribe `rqutils/ground_locg.py` structure. Keep `_project_out`, `eigenpair_2x2`, `eigenpair_3x3`, and the `body_iter0`/`body_iter1`/`body` decomposition. Read `rqutils/ground_locg.py:191-421` alongside this — the algorithm is unchanged and the comments there explain *why* each step is shaped as it is.

```python
"""MLX port of the single-vector LOBPCG solver and the cached SQD matvec.

A transcription of ``rqutils.ground_locg`` and ``rqutils.sqd.apply_h_xz_cached`` onto MLX,
for benchmarking against the JAX originals. See
``docs/superpowers/specs/2026-08-03-mlx-sqd-poc-design.md``.

Three structural differences from the JAX version, all forced by MLX:

* ``jax.lax.scan`` over X groups becomes a Python loop. J is static and small (tens), so
  unrolling into MLX's lazy graph is fine.
* ``jax.lax.while_loop`` becomes a Python loop. MLX has no while_loop/cond/scan, and reading
  the convergence flag forces ``mx.eval`` -- a device sync per iteration. That cost is real
  for an MLX user, not a measurement artifact. Pass ``tol=0.`` to skip the check entirely
  and get a sync-free fixed-iteration run.
* All ``out_sharding=`` arguments are dropped; MLX has unified memory and no sharding.

The analytic 2x2/3x3 Rayleigh-Ritz step carries over directly, which is what makes this port
feasible at all -- MLX has no ``eigh``.
"""
import numpy as np
import mlx.core as mx


def apply_h_xz_mlx(vec, xsources, diagonals):
    """Return Hv from precomputed X sources and diagonals.

    Mirrors ``rqutils.sqd.apply_h_xz_cached``. ``xsources`` must already be sanitized (no
    negative entries, with the corresponding diagonals zeroed) -- see
    ``_bench_common.build_solver_inputs``.
    """
    out = mx.zeros_like(vec)
    for igroup in range(xsources.shape[0]):
        out = out + mx.take(vec, xsources[igroup]) * diagonals[igroup]
    return out


def ground_locg_mlx(mat, xinit, args=(), maxiter=1000, tol=None):
    """Single-vector LOBPCG in MLX.

    Args:
        mat: Callable mapping ``(vec, *args)`` to ``A @ vec``.
        xinit: Initial vector. Must have nonvanishing overlap with the ground state.
        args: Extra arguments forwarded to ``mat``.
        maxiter: Maximum gradient-descent iterations.
        tol: Convergence tolerance. ``None`` uses the dtype epsilon. ``0.`` disables the
            check, running exactly ``maxiter`` iterations with no per-iteration device sync.

    Returns:
        (eigenvalue, eigenvector, iterations).
    """
    xinit = mx.array(xinit)
    check_convergence = tol != 0.
    if tol is None:
        # Compare the dtype object directly rather than parsing its repr: this works
        # identically under real MLX and under the numpy shim used by the static check.
        tol = float(np.finfo(np.float32 if xinit.dtype == mx.float32 else np.float64).eps)

    def matvec(vec):
        return mat(vec, *args)

    def rayleigh_ritz(*vectors):
        sas = _compute_sas(matvec, *vectors)
        if len(vectors) == 2:
            return eigenpair_2x2(sas)
        return eigenpair_3x3(sas)

    xinit = xinit / mx.linalg.norm(xinit)

    # Iteration 0: no previous direction yet.
    ax = matvec(xinit)
    rho = mx.sum(xinit * ax)
    xcurr = xinit
    rcurr = ax - rho * xcurr

    # Iteration 1: two-vector Rayleigh-Ritz over {x, r}.
    norm_r = mx.linalg.norm(rcurr)
    tmp_p = rcurr / mx.where(norm_r == 0., mx.array(1., norm_r.dtype), norm_r)
    theta, kappa = rayleigh_ritz(xcurr, tmp_p)
    tmp_t = tmp_p * kappa[0] - xcurr * kappa[1]
    tmp_u = xcurr * kappa[0] + tmp_p * kappa[1]
    xcurr = tmp_u / mx.linalg.norm(tmp_u)
    ycurr = tmp_t / mx.linalg.norm(tmp_t)
    rcurr = matvec(xcurr) - theta * xcurr

    niter = 0
    for niter in range(1, maxiter + 1):
        tmp_p = _project_out((xcurr, ycurr), rcurr)
        theta, kappa = rayleigh_ritz(xcurr, ycurr, tmp_p)
        tmp_s = ycurr * kappa[1] + tmp_p * kappa[2]
        norm_s = mx.linalg.norm(tmp_s)
        tmp_t = tmp_s * (kappa[0] / norm_s) - xcurr * norm_s
        tmp_u = xcurr * kappa[0] + tmp_s
        xcurr = tmp_u / mx.linalg.norm(tmp_u)
        ycurr = tmp_t / mx.linalg.norm(tmp_t)
        axnext = matvec(xcurr)
        rcurr = axnext - xcurr * theta

        if check_convergence:
            # Same heuristic as the JAX version: compare the residual norm against the
            # floating-point error we'd expect from forming the residual at all.
            reltol = (mx.linalg.norm(axnext) - theta) * xcurr.shape[0] * 10
            # This float() forces a device sync -- the price of MLX having no while_loop.
            if float(mx.linalg.norm(rcurr)) < tol * float(reltol):
                break

    return float(theta), xcurr, niter


def _compute_sas(matvec, *vectors):
    """Return the (n x n) matrix of <v_i | A | v_j> for n in {2, 3}."""
    mvs = [matvec(v) for v in vectors]
    rows = []
    for iv1, v1 in enumerate(vectors):
        rows.append(mx.stack([mx.sum(v1 * mvs[iv2]) for iv2 in range(len(vectors))]))
    sas = mx.stack(rows)
    # Symmetrize: the two triangles differ only by rounding for real symmetric A.
    return (sas + sas.T) * 0.5


def _project_out(basis, vector):
    """Orthogonalize ``vector`` against ``basis``, ending on a subtraction.

    The repeated passes and the final zeroing are load-bearing; see the comments in
    ``rqutils/ground_locg.py:390-410``. Near convergence, ending on a normalization can
    reintroduce basis components through catastrophic cancellation and wreck the
    Rayleigh-Ritz conditioning.
    """
    for _ in range(2):
        ips = [mx.sum(vb * vector) for vb in basis]
        for vb, ip in zip(basis, ips):
            vector = vector - vb * ip
        norm = mx.linalg.norm(vector)
        vector = vector / mx.where(norm == 0., mx.array(1., norm.dtype), norm)

    for _ in range(2):
        ips = [mx.sum(vb * vector) for vb in basis]
        for vb, ip in zip(basis, ips):
            vector = vector - vb * ip

    return vector * (mx.linalg.norm(vector) >= 0.99).astype(vector.dtype)


def eigenpair_2x2(mat):
    """Lowest eigenpair of a real symmetric 2x2 matrix."""
    d = mx.stack([mat[0, 0], mat[1, 1]])
    off = mat[1, 0]
    det = d[0] * d[1] - off * off
    tr = d[0] + d[1]
    eigval = (tr - mx.sqrt(tr * tr - 4. * det)) * 0.5
    first = (d[1] - eigval + off) / (d[0] - eigval + off)
    eigvec = mx.stack([first, mx.array(-1., first.dtype)])
    return eigval, eigvec / mx.sqrt(first * first + 1.)


def eigenpair_3x3(mat):
    """Lowest eigenpair of a real symmetric 3x3 matrix via Cardano's method.

    Reference: J. Kopp, Int. J. Mod. Phys. C. 19, 523 (2008). Ported from
    ``rqutils.ground_locg.eigenpair_3x3``; MLX has no eigh, so the analytic route is the
    only route.
    """
    d = mx.stack([mat[0, 0], mat[1, 1], mat[2, 2]])
    modod = mx.stack([mat[1, 0], mat[2, 0], mat[2, 1]]) ** 2
    c2 = -mx.sum(d)
    c1 = mx.sum(d * mx.stack([d[2], d[0], d[1]])) - mx.sum(modod)
    c0 = mx.sum(d * mx.stack([modod[2], modod[1], modod[0]]))
    c0 = c0 - d[0] * d[1] * d[2]
    c0 = c0 - 2. * (mat[0, 2] * mat[1, 0] * mat[2, 1])
    p = c2 * c2 - 3. * c1
    q = -13.5 * c0 - c2 * c2 * c2 + 4.5 * c2 * c1
    phi = mx.arctan2(
        mx.sqrt(27. * (0.25 * c1 * c1 * (p - c1) + c0 * (q + 6.75 * c0))),
        q
    ) / 3.
    cphi = mx.cos(phi)
    sphi = mx.sin(phi)
    root3 = float(np.sqrt(3.))
    xmin = mx.min(mx.stack([2. * cphi, -cphi - root3 * sphi, -cphi + root3 * sphi]))
    eigval = mx.sqrt(p) / 3. * xmin - c2 / 3.
    v0 = mx.stack([mat[0, 1], mat[1, 1] - eigval, mat[2, 1]])
    v1 = mx.stack([mat[0, 2], mat[1, 2], mat[2, 2] - eigval])
    eigvec = mx.linalg.cross(v0, v1)
    return eigval, eigvec / mx.linalg.norm(eigvec)
```

Two transcription notes the implementer must not "simplify" away:

1. `_compute_sas` symmetrizes with `(sas + sas.T) * 0.5`. The JAX original builds only the lower triangle and mirrors it, because `.at[].set()` on a 3×3 is cheaper there. Computing all 9 entries costs 3 extra `mx.sum` calls on scalars — negligible — and avoids scatter, whose MLX out-of-bounds behavior is explicitly documented as undefined.
2. `eigenpair_3x3` drops the `.conjugate()` and the real/imag norm split from the JAX version. That is correct **only because** the even-Y constraint makes everything real. If someone later removes that constraint, this function is wrong. The module docstring and the `build_solver_inputs` dtype guard both say so.

- [ ] **Step 4: Run the static check and confirm it passes**

```bash
cd /Users/ima/tasks/quantum/rqutils && uv run python /tmp/check_task2_static.py
```

Expected: all `OK` lines, ending in `ALL STATIC CHECKS PASSED`. In particular
`solve:` must show an eigenvalue matching the reference to ~1e-12 and an
iteration count near 89 (that is what the JAX f64 arm needs on this problem).

If the eigenvalue is wrong but the matvec check passed, the bug is in the
LOBPCG transcription — compare against `rqutils/ground_locg.py:264-341`
step by step. If iterations hit 1000, convergence is broken: check
`_project_out` (the final zeroing at `>= 0.99` is easy to get wrong) and
the `eigenpair_3x3` sign conventions.

- [ ] **Step 5: Write the MLX check script for the user to run**

The implementer cannot run this. Create `/tmp/check_task2_mlx.py`:

```python
"""Verify ground_locg_mlx under real MLX, on both devices and both precisions.

Must be run by the user: mlx.core loads a Metal device even for mx.cpu arrays, so this
fails in a headless session with
  RuntimeError: [metal::load_device] No Metal device available
"""
import sys
import numpy as np
import mlx.core as mx
import jax
jax.config.update('jax_enable_x64', True)
sys.path.insert(0, 'examples')
from _bench_common import generate_problem, build_solver_inputs, dense_reference
from ground_locg_mlx import apply_h_xz_mlx, ground_locg_mlx

ps, cs, states = generate_problem(10, 20, 200, seed=1)
inputs = build_solver_inputs(ps, cs, states)
H, ref = dense_reference(inputs)
print(f'reference ground energy = {ref:.12f}')

failures = []
for device, name in ((mx.cpu, 'cpu'), (mx.gpu, 'gpu')):
    for dtype, dtname, rtol in ((mx.float64, 'f64', 1e-9), (mx.float32, 'f32', 1e-4)):
        arm = f'mlx-{name}-{dtname}'
        if name == 'gpu' and dtname == 'f64':
            print(f'{arm}: skipped (Metal has no float64)')
            continue
        try:
            mx.set_default_device(device)
            xs = mx.array(inputs.xsources)
            dg = mx.array(inputs.diagonals).astype(dtype)
            v0 = mx.array(inputs.vinit).astype(dtype)

            mv = np.asarray(apply_h_xz_mlx(v0, xs, dg), dtype=np.float64)
            mverr = np.abs(mv - H @ inputs.vinit).max()

            eig, _, iters = ground_locg_mlx(apply_h_xz_mlx, v0, args=(xs, dg))
            ok = abs(eig - ref) < rtol * max(1., abs(ref))
            print(f'{arm}: eig={eig:.10f} iters={iters} matvec_err={mverr:.2e} '
                  f'{"OK" if ok else "FAIL"}')
            if not ok:
                failures.append(arm)
        except Exception as exc:
            print(f'{arm}: ERROR {type(exc).__name__}: {exc}')
            failures.append(arm)

print('\nFAILURES:', failures if failures else 'none')
```

- [ ] **Step 6: Ask the user to run it, and wait**

Ask the user to run, from the repo root:

```bash
uv run python /tmp/check_task2_mlx.py
```

**Do not proceed to Task 3 until the user reports output.** Expect `mlx-cpu-f64`
to match the reference closely, `*-f32` arms to land within ~1e-6, and
`mlx-gpu-f64` to be skipped. If an arm errors, fix it and ask again — this is
the debug round-trip the spec predicted.

- [ ] **Step 7: Commit (only after the user confirms the MLX check passes)**

```bash
git add examples/ground_locg_mlx.py
git commit -m "Add MLX port of single-vector LOBPCG and the cached SQD matvec

scan and while_loop become Python loops; the convergence check forces a
device sync per iteration, which tol=0. skips for a clean fixed-iteration
measurement. The analytic 2x2/3x3 Rayleigh-Ritz step carries over
directly, which is what makes the port feasible: MLX has no eigh.

Real-valued only, relying on the even-Y constraint from _bench_common."
```

---

### Task 3: Benchmark harness

**Files:**
- Create: `examples/bench_mlx.py`
- Check: `/tmp/check_task3.py` (implementer-runnable, JAX arms only)

**Interfaces:**
- Consumes from Task 1: `generate_problem`, `build_solver_inputs`, `dense_reference`, `brute_force_reference`, `timeit`, `SolverInputs`.
- Consumes from Task 2: `apply_h_xz_mlx`, `ground_locg_mlx`.
- Produces: a CLI. `run_arm(arm: str, options) -> dict` returning keys `arm`, `status`, `eigval`, `iters`, `compile_s`, `fixed_s`, `per_it_ms`, `solve_s`, and `reason` when `status == 'skipped'`.

Arms: `jax-cpu-f64`, `jax-cpu-f32`, `jax-metal-f32`, `mlx-cpu-f64`, `mlx-cpu-f32`, `mlx-gpu-f32`.

- [ ] **Step 1: Write the check script**

Create `/tmp/check_task3.py`:

```python
"""Verify bench_mlx's JAX arms and its gate/reporting logic end to end."""
import json
import subprocess
import sys

BASE = ['uv', 'run', 'python', 'examples/bench_mlx.py',
        '--num-qubits', '10', '--num-paulis', '20', '--num-states', '200',
        '--repeat', '2', '--fixed-iters', '50']

# 1. A JAX arm must pass its gate and emit parseable JSON.
out = subprocess.run(BASE + ['--arm', 'jax-cpu-f64', '--json'],
                     capture_output=True, text=True)
assert out.returncode == 0, f'jax-cpu-f64 failed:\n{out.stdout}\n{out.stderr}'
record = json.loads(out.stdout)
row = record['results'][0] if 'results' in record else record
assert row['status'] == 'ok', row
assert row['iters'] > 0 and row['per_it_ms'] > 0, row
print(f"OK  jax-cpu-f64 eig={row['eigval']:.10f} per_it={row['per_it_ms']:.3f}ms")

# 2. The f32 arm must also pass, at its looser tolerance.
out = subprocess.run(BASE + ['--arm', 'jax-cpu-f32', '--json'],
                     capture_output=True, text=True)
assert out.returncode == 0, f'jax-cpu-f32 failed:\n{out.stdout}\n{out.stderr}'
row = json.loads(out.stdout)
row = row['results'][0] if 'results' in row else row
assert row['status'] == 'ok', row
print(f"OK  jax-cpu-f32 eig={row['eigval']:.10f}")

# 3. An unknown arm must be rejected, not silently ignored.
out = subprocess.run(BASE + ['--arm', 'nonsense'], capture_output=True, text=True)
assert out.returncode != 0, 'unknown arm was accepted'
print('OK  unknown arm rejected')

# 4. A deliberately corrupted problem must be caught by the gate rather than timed.
out = subprocess.run(BASE + ['--arm', 'jax-cpu-f64', '--self-test-break-gate'],
                     capture_output=True, text=True)
assert out.returncode != 0, 'gate did not fail on a corrupted Hamiltonian'
assert 'gate' in (out.stdout + out.stderr).lower()
print('OK  correctness gate rejects a corrupted problem')

# 5. Text (non-JSON) output must contain a header and the arm row.
out = subprocess.run(BASE + ['--arm', 'jax-cpu-f64'], capture_output=True, text=True)
assert out.returncode == 0, out.stderr
assert 'per_it_ms' in out.stdout and 'jax-cpu-f64' in out.stdout, out.stdout
print('OK  text report renders')
print('\nJAX-SIDE CHECKS PASSED (MLX arms still unverified)')
```

- [ ] **Step 2: Run it to make sure it fails**

```bash
cd /Users/ima/tasks/quantum/rqutils && uv run python /tmp/check_task3.py
```

Expected: failure — `examples/bench_mlx.py` does not exist, so the first `subprocess.run` returns nonzero and the assert fires.

- [ ] **Step 3: Write `examples/bench_mlx.py`**

```python
"""Benchmark the SQD eigensolver loop across JAX and MLX, CPU and GPU.

Only the solver loop is compared. Setup (uniquification, X-source lookup, diagonal
composition) always runs in JAX on CPU and is not timed, so every arm consumes identical
arrays -- see docs/superpowers/specs/2026-08-03-mlx-sqd-poc-design.md.

JAX's platform and x64 flag are process-global and must be set before importing jax, so each
JAX arm needs its own process. --all re-executes this script once per arm and collates.

.. code-block:: sh

    uv run python examples/bench_mlx.py --arm mlx-gpu-f32
    uv run python examples/bench_mlx.py --all
    uv run python examples/bench_mlx.py --all --json > results.json

Two metrics are reported per arm: per-iteration cost at a fixed iteration count (identical
work per arm, so a clean speed comparison) and time-to-convergence with its iteration count
(what production actually pays). Reporting both makes it visible when fp32 is faster per
iteration but needs more iterations.
"""
import argparse
import json
import os
import subprocess
import sys

ARMS = ('jax-cpu-f64', 'jax-cpu-f32', 'jax-metal-f32',
        'mlx-cpu-f64', 'mlx-cpu-f32', 'mlx-gpu-f32')

# Relative tolerance for the correctness gate, by precision.
RTOL = {'f64': 1e-9, 'f32': 1e-4}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--arm', choices=ARMS, help='Single arm to run.')
    parser.add_argument('--all', action='store_true',
                        help='Run every arm, one subprocess each, and collate.')
    parser.add_argument('--num-qubits', type=int, default=14)
    parser.add_argument('--num-paulis', type=int, default=100)
    parser.add_argument('--num-states', type=int, default=4000)
    parser.add_argument('--repeat', type=int, default=3,
                        help='Timed iterations after warmup.')
    parser.add_argument('--fixed-iters', type=int, default=100,
                        help='Iteration count for the fixed-work measurement.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--json', action='store_true', help='Emit JSON instead of a table.')
    parser.add_argument('--skip-brute-force', action='store_true',
                        help='Skip the 2^n reference (needed above ~n=14).')
    parser.add_argument('--self-test-break-gate', action='store_true',
                        help=argparse.SUPPRESS)  # corrupts the problem to prove the gate bites
    options = parser.parse_args(argv)
    if not options.arm and not options.all:
        parser.error('specify --arm or --all')
    return options


def run_arm(arm, options):
    """Run one arm in this process. Returns a result dict."""
    framework, device, precision = arm.split('-')

    # JAX must be configured before import, so do it here rather than at module scope.
    if framework == 'jax':
        os.environ['JAX_PLATFORMS'] = 'cpu' if device == 'cpu' else device
    import jax
    jax.config.update('jax_enable_x64', precision == 'f64')

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _bench_common import (generate_problem, build_solver_inputs, dense_reference,
                               brute_force_reference, timeit)

    if framework == 'jax' and device == 'metal' and jax.default_backend() != 'metal':
        return {'arm': arm, 'status': 'skipped',
                'reason': f'jax backend is {jax.default_backend()}, not metal'}

    pauli_strings, coeffs, states = generate_problem(
        options.num_qubits, options.num_paulis, options.num_states, options.seed
    )
    inputs = build_solver_inputs(pauli_strings, coeffs, states)

    if options.self_test_break_gate:
        # Corrupt the diagonals so the solver cannot reach the reference eigenvalue.
        inputs.diagonals = inputs.diagonals * 2.5 + 1.0

    matrix, reference = dense_reference(inputs)
    if not options.skip_brute_force:
        brute = brute_force_reference(pauli_strings, coeffs, states)
        if abs(reference - brute) > 1e-9 * max(1., abs(brute)):
            raise SystemExit(f'gate failed: dense reference {reference} disagrees with '
                             f'brute force {brute} -- the setup chain is wrong')

    rtol = RTOL[precision]
    if framework == 'jax':
        result = _time_jax(arm, inputs, precision, options)
    else:
        result = _time_mlx(arm, inputs, device, precision, options)

    if abs(result['eigval'] - reference) > rtol * max(1., abs(reference)):
        raise SystemExit(f'gate failed for {arm}: eigenvalue {result["eigval"]} differs from '
                         f'reference {reference} by more than rtol={rtol}')

    result['reference'] = reference
    result['status'] = 'ok'
    return result


def _time_jax(arm, inputs, precision, options):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from rqutils.ground_locg import ground_locg
    from rqutils.sqd import apply_h_xz_cached
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _bench_common import timeit

    dtype = np.float64 if precision == 'f64' else np.float32
    xsources = jnp.asarray(inputs.xsources)
    diagonals = jnp.asarray(inputs.diagonals, dtype=dtype)
    vinit = jnp.asarray(inputs.vinit, dtype=dtype)

    # Gate: the ported matvec and the original must agree on the same input.
    matvec_out = np.asarray(apply_h_xz_cached(vinit, xsources, diagonals), dtype=np.float64)
    matrix, _ = _reference_matrix(inputs)
    matvec_err = float(np.abs(matvec_out - matrix @ inputs.vinit).max())

    def fixed():
        return ground_locg(apply_h_xz_cached, vinit, args=(xsources, diagonals),
                           maxiter=options.fixed_iters, tol=0.)

    compile_s, fixed_s = timeit(fixed, options.repeat, jax.block_until_ready)

    def solve():
        return ground_locg(apply_h_xz_cached, vinit, args=(xsources, diagonals))

    _, solve_s = timeit(solve, options.repeat, jax.block_until_ready)
    eigval, _, iters = solve()

    return {'arm': arm, 'compile_s': compile_s, 'fixed_s': fixed_s,
            'per_it_ms': fixed_s / options.fixed_iters * 1e3,
            'solve_s': solve_s, 'iters': int(iters), 'eigval': float(eigval),
            'matvec_err': matvec_err}


def _time_mlx(arm, inputs, device, precision, options):
    import numpy as np
    import mlx.core as mx
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _bench_common import timeit
    from ground_locg_mlx import apply_h_xz_mlx, ground_locg_mlx

    mx.set_default_device(mx.cpu if device == 'cpu' else mx.gpu)
    dtype = mx.float64 if precision == 'f64' else mx.float32
    xsources = mx.array(inputs.xsources)
    diagonals = mx.array(inputs.diagonals).astype(dtype)
    vinit = mx.array(inputs.vinit).astype(dtype)

    matvec_out = np.asarray(apply_h_xz_mlx(vinit, xsources, diagonals), dtype=np.float64)
    matrix, _ = _reference_matrix(inputs)
    matvec_err = float(np.abs(matvec_out - matrix @ inputs.vinit).max())

    def sync(result):
        # MLX is lazy: without this we would time graph construction, not computation.
        mx.eval(result[1])
        return result

    def fixed():
        return ground_locg_mlx(apply_h_xz_mlx, vinit, args=(xsources, diagonals),
                               maxiter=options.fixed_iters, tol=0.)

    compile_s, fixed_s = timeit(fixed, options.repeat, sync)

    def solve():
        return ground_locg_mlx(apply_h_xz_mlx, vinit, args=(xsources, diagonals))

    _, solve_s = timeit(solve, options.repeat, sync)
    eigval, _, iters = solve()

    return {'arm': arm, 'compile_s': compile_s, 'fixed_s': fixed_s,
            'per_it_ms': fixed_s / options.fixed_iters * 1e3,
            'solve_s': solve_s, 'iters': int(iters), 'eigval': float(eigval),
            'matvec_err': matvec_err}


def _reference_matrix(inputs):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _bench_common import dense_reference
    return dense_reference(inputs)


def run_all(options):
    """Run every arm in its own subprocess and collate the results."""
    results = []
    for arm in ARMS:
        argv = [sys.executable, os.path.abspath(__file__), '--arm', arm, '--json',
                '--num-qubits', str(options.num_qubits),
                '--num-paulis', str(options.num_paulis),
                '--num-states', str(options.num_states),
                '--repeat', str(options.repeat),
                '--fixed-iters', str(options.fixed_iters),
                '--seed', str(options.seed)]
        if options.skip_brute_force:
            argv.append('--skip-brute-force')
        proc = subprocess.run(argv, capture_output=True, text=True)
        if proc.returncode != 0:
            results.append({'arm': arm, 'status': 'failed',
                            'reason': (proc.stderr or proc.stdout).strip().split('\n')[-1]})
            continue
        try:
            results.append(json.loads(proc.stdout))
        except json.JSONDecodeError:
            results.append({'arm': arm, 'status': 'failed',
                            'reason': f'unparseable output: {proc.stdout[:200]}'})
    return results


def report(results, as_json):
    if as_json:
        print(json.dumps({'results': results}, indent=2))
        return

    header = (f'{"arm":<15}{"compile_s":>10}{"fixed_s":>10}{"per_it_ms":>11}'
              f'{"solve_s":>10}{"iters":>7}  eigval')
    print(header)
    print('-' * len(header))
    for row in results:
        if row.get('status') != 'ok':
            print(f'{row["arm"]:<15}{row.get("status", "?"):>10}  {row.get("reason", "")}')
            continue
        print(f'{row["arm"]:<15}{row["compile_s"]:>10.4f}{row["fixed_s"]:>10.4f}'
              f'{row["per_it_ms"]:>11.3f}{row["solve_s"]:>10.4f}{row["iters"]:>7d}'
              f'  {row["eigval"]:.10f}')


def main():
    options = parse_args()
    if options.all:
        report(run_all(options), options.json)
    else:
        result = run_arm(options.arm, options)
        if options.json:
            print(json.dumps(result, indent=2))
        else:
            report([result], False)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the check and confirm the JAX arms pass**

```bash
cd /Users/ima/tasks/quantum/rqutils && uv run python /tmp/check_task3.py
```

Expected: all five `OK` lines, ending in `JAX-SIDE CHECKS PASSED`.

If check 4 (`--self-test-break-gate`) does not fail, the gate is not actually
guarding the timing — that is the single most important behavior in this file,
because a gate that does not bite means every number reported downstream could
be from a broken solver. Fix it before continuing.

- [ ] **Step 5: Run the JAX arms for real and read the output**

```bash
cd /Users/ima/tasks/quantum/rqutils && \
  uv run python examples/bench_mlx.py --arm jax-cpu-f64 --num-qubits 12 --num-states 1000 && \
  uv run python examples/bench_mlx.py --arm jax-cpu-f32 --num-qubits 12 --num-states 1000
```

Expected: two tables. Both eigenvalues should agree to ~1e-6; f32 will
typically converge in fewer iterations at a looser effective tolerance.
Do not report these as final results — they are a smoke test of the harness.

- [ ] **Step 6: Commit the harness**

```bash
git add examples/bench_mlx.py
git commit -m "Add six-arm JAX-vs-MLX benchmark harness for the SQD solver

Each arm passes a correctness gate before any timing: the dense
reference is cross-checked against a brute-force 2^n construction, the
ported matvec against H @ v, and each arm's eigenvalue against the
reference at a per-precision tolerance. --self-test-break-gate proves
the gate bites.

JAX arms run in separate processes because the platform and x64 flag
are process-global. MLX timings sync with mx.eval, without which the
lazy graph would make MLX look artificially fast."
```

- [ ] **Step 7: Ask the user to run the full comparison**

Ask the user to run, from the repo root:

```bash
uv run python examples/bench_mlx.py --all --num-qubits 14 --num-states 4000
```

and, for a larger case where the GPU has a chance to matter:

```bash
uv run python examples/bench_mlx.py --all --num-qubits 18 --num-states 200000 \
    --skip-brute-force
```

(`--skip-brute-force` is required above ~n=14: the 2^n reference becomes
intractable. The dense-from-solver-inputs gate still runs.)

**Wait for their output.** Then, and only then, write up the comparison —
reporting the numbers they provide, noting which arms were skipped and why,
and restating the four limitations from the spec (solver loop only, even-Y
Hamiltonians only, MLX-GPU is fp32-only, no sharding).

- [ ] **Step 8: Record the measured results**

Append a "Results" section to
`docs/superpowers/specs/2026-08-03-mlx-sqd-poc-design.md` containing the
user's actual table, the machine it ran on, and a one-paragraph verdict on
whether MLX is worth pursuing for this workload. Commit it.

Do not fabricate or extrapolate any figure. If an arm was skipped or failed,
say so in the table rather than omitting the row.

---

## Self-Review

**Spec coverage:**

| spec section | task |
|---|---|
| Scope: solver loop only, `cache_level=(1,2)` | 1 (setup), 2 (`apply_h_xz_mlx`) |
| Verified fact 1–2 (complex128 / even-Y) | 1 Step 3 dtype guard + Step 1 assertion |
| Verified fact 3 (no complex128, no Metal f64) | 3 (`mlx-gpu-f64` absent from `ARMS`) |
| Verified fact 4 (`hproj` unusable) | 1 (`dense_reference` replaces it) |
| Verified facts 5–6 (references, matvec equality) | 1 Step 4, 2 Step 1 check 3a |
| Architecture: three files, shared generation | File Structure, Tasks 1–3 |
| Invalid-source sanitization | 1 Step 3 `build_solver_inputs` |
| Port substitution table | 2 Step 3 |
| Loop strategy A + `tol=0.` fixed mode | 2 Step 3, verified 2 Step 1 check 3c |
| Six arms, separate processes, metal skipped | 3 Step 3 (`ARMS`, `run_all`, backend check) |
| Correctness gate before timing | 3 Step 3 `run_arm`, proven by check 4 |
| Both metrics, `compile_s` split, `--json` | 3 Step 3 `_time_*` and `report` |
| Limitations restated with results | 3 Steps 7–8 |

**Placeholder scan:** No TBD/TODO/"handle errors appropriately". Every code step
has complete runnable content. No step says "similar to Task N".

**Type consistency:** `SolverInputs` field names (`xsources`, `diagonals`,
`vinit`, `subspace_dim`, `num_xgroups`) are identical in Tasks 1, 2, and 3.
`apply_h_xz_mlx(vec, xsources, diagonals)` matches
`rqutils.sqd.apply_h_xz_cached`'s argument order. `ground_locg_mlx` returns
`(eigval, eigvec, niter)` matching `ground_locg`, and both `_time_jax` and
`_time_mlx` unpack it in that order. `timeit(fn, repeat, sync)` is called with
three arguments everywhere. `RTOL` keys (`f64`/`f32`) match the precision
suffix produced by `arm.split('-')`.

One known wart, left deliberately: `_reference_matrix` recomputes the dense
matrix that `run_arm` already built, because `_time_jax`/`_time_mlx` do not
receive it. At gate sizes this costs milliseconds and keeps the timing
functions independently callable. It is outside every timed region.
