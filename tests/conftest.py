"""Shared configuration and helpers for the rqutils test suite.

``jax_enable_x64`` is set here, at conftest import time, so that it takes effect before any test
module imports ``rqutils``. This is load-bearing rather than incidental: without x64 JAX silently
produces float32/complex64 and every tolerance in this suite is wrong by nine orders of magnitude.
``conftest.py`` is the only file pytest guarantees to import before the test modules, which is why
this cannot move to a plain helper module.

Two caches are also pointed at writable directories below, purely for speed -- measured 53.3 s ->
31.5 s -> 10.4 s on this suite (5.1x). Unlike the x64 flag these are **not** load-bearing: nothing
here depends on them, both defer to a value the caller already set, and a cache miss only costs
time. See the comments at each for what they buy and why the default is wrong for this repo.

Seeds are constructed *inside each test body*, deliberately, and there are no ``@pytest.fixture``
state generators in this suite. Several tests pick a specific seed to produce a specific pathology
(a decoupled seed state, a subspace that splits into two blocks, 13 Z terms in one X group) and
assert that the fixture still has that property. Moving the draws into fixtures would make RNG
stream position depend on fixture ordering, which is invisible at the call site -- see
:func:`collapsing_states`, whose docstring records a measured instance of exactly that hazard.
Please keep new fixtures as plain functions taking ``rng``.
"""

import os
import shutil
import subprocess
import sys
import tempfile

# Both of the following must be set BEFORE jax/matplotlib are imported, which is what puts them
# above the `import jax` line rather than in a fixture.
#
# matplotlib rebuilds its font cache from scratch on every interpreter start when MPLCONFIGDIR is
# unwritable -- ~24 s, paid by `import rqutils.qprint` (which imports pyplot eagerly) and so by
# tests/test_qprint.py. `~/.matplotlib` is not writable in a sandboxed session, and matplotlib's
# own fallback is a fresh temp dir per process, which never warms. A stable temp dir does.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "rqutils-mplconfig"))
# JAX recompiles all ~250 XLA kernels every run without a persistent cache (~23 s of the suite).
# MIN_COMPILE_TIME_SECS is required alongside the directory: the default 1.0 s threshold excludes
# nearly every kernel here, the largest single compile being ~0.44 s, so the cache would stay empty.
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR", os.path.join(tempfile.gettempdir(), "rqutils-jaxcache")
)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest


def herm(n, rng, complex_=True):
    """Return a random ``(n, n)`` Hermitian matrix drawn from ``rng``."""
    mat = rng.normal(size=(n, n))
    if complex_:
        mat = mat + 1.0j * rng.normal(size=(n, n))
    return mat + mat.conjugate().T


def symmetrize(mat):
    """Mirror the lower triangle of ``mat`` over the diagonal.

    ``eigenpair_2x2`` and ``eigenpair_3x3`` read only the diagonal and the lower triangle, so a
    reference eigendecomposition must be taken of *this* matrix, not of the raw input. Comparing
    against ``eigvalsh`` of an unsymmetrized input compares against a different matrix.
    """
    lower = np.tril(mat)
    return lower + np.tril(mat, -1).conjugate().T


def lowest(mat):
    """Reference lowest eigenvalue of ``mat``, via LAPACK, after symmetrization."""
    return float(np.linalg.eigvalsh(symmetrize(mat))[0])


def rel_resid(mat, val, vec):
    """Eigenpair residual ``|Av - λv|``, scaled by ``max|A|``.

    The scaling is what makes a single tolerance usable across the shifted and extreme-scale cases:
    the 1e9-shifted 2x2 input has an absolute residual of 6e-8 but a relative residual of 6e-17.
    """
    mat = symmetrize(mat)
    vec = np.asarray(vec)
    return float(np.linalg.norm(mat @ vec - val * vec) / np.abs(mat).max())


_PAULI_MATRICES = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


def dense_pauli_sum(pauli_strings, coeffs):
    """Return the full ``2**n``-by-``2**n`` matrix of ``sum(c * Q)``, by Kronecker products.

    Character order is most-significant-qubit-first, matching Qiskit's ``SparsePauliOp`` (verified:
    ``ZI``, ``IZ``, and ``XY`` all agree with ``SparsePauliOp(s).to_matrix()``). Only for small
    ``n`` -- this is ``4**n`` work.
    """
    num_qubits = len(pauli_strings[0])
    full = np.zeros((2**num_qubits,) * 2, dtype=np.complex128)
    for string, coeff in zip(pauli_strings, coeffs):
        operator = np.array([[1.0]], dtype=np.complex128)
        for char in string:
            operator = np.kron(operator, _PAULI_MATRICES[char])
        full += coeff * operator
    return full


def project_dense(pauli_strings, coeffs, states):
    """Return the Pauli sum projected onto the subspace spanned by ``states``, densely.

    Independent of ``rqutils.sqd``'s entire packing/padding/uniquification/matvec chain, so
    agreement with it is real evidence rather than a tautology. A benchmark under ``examples/`` keeps
    an equivalent brute-force reference, gated at a 1e-9 relative tolerance against a second
    independent dense construction (observed agreement ~3.6e-15); this one is written out again
    rather than imported so the test suite does not depend on ``examples/``, which is script
    territory carrying its own optional-dependency imports.
    """
    num_qubits = states.shape[1]
    full = dense_pauli_sum(pauli_strings, coeffs)
    unique = np.unique(np.asarray(states, dtype=np.uint8), axis=0)
    # Row bits are most-significant-first, matching the Pauli string character order.
    indices = unique.dot(1 << np.arange(num_qubits)[::-1])
    return full[np.ix_(indices, indices)]


def lowest_projected(pauli_strings, coeffs, states):
    """Reference lowest eigenvalue of the projected Pauli sum. See :func:`project_dense`."""
    return float(np.linalg.eigvalsh(project_dense(pauli_strings, coeffs, states))[0].real)


def gate_unitary(name, qubits, num_qubits, angle=None):
    """Return the ``2**num_qubits`` unitary for one ``svsim`` gate spec.

    Built from Kronecker products of the 2x2 Paulis, independent of both ``rqutils.svsim`` and
    qiskit. ``qubits`` are indices in ``svsim``'s convention, where qubit ``q`` is bit ``q`` of the
    statevector index (bit 0 = least significant), i.e. the reverse of the Pauli-string character
    order used by :func:`dense_pauli_sum`.

    ``x``/``y``/``z`` are the bare Pauli gates. The rotations are ``exp(-i * angle * P / 2)``,
    matching Qiskit's ``rx``/``ry``/``rz``/``rzz``.
    """
    letters = {"x": "X", "y": "Y", "z": "Z", "rx": "X", "ry": "Y", "rz": "Z", "rzz": "Z"}
    if name not in letters:
        raise ValueError(f"unsupported gate {name}")
    qubits = np.atleast_1d(np.asarray(qubits))
    # Build the Pauli operator as a tensor product over all qubits, identity except on `qubits`.
    # Index bit q is qubit q, and np.kron's first factor is the MOST significant bit, so the
    # per-qubit factors go in reverse qubit order.
    factors = []
    for qubit in reversed(range(num_qubits)):
        factors.append(_PAULI_MATRICES[letters[name] if qubit in qubits else "I"])
    operator = factors[0]
    for factor in factors[1:]:
        operator = np.kron(operator, factor)
    if name in ("x", "y", "z"):
        return operator
    identity = np.eye(2**num_qubits, dtype=np.complex128)
    return np.cos(angle / 2.0) * identity - 1.0j * np.sin(angle / 2.0) * operator


def simulate_dense(gate_specs, num_qubits, initial_state=0):
    """Apply ``gate_specs`` to a statevector by dense matrix multiplication.

    The independent reference for :func:`rqutils.svsim.svsim`: it shares no code with the
    symplectic ``CircuitXZ`` representation or the ``lax.scan`` kernel under test.
    """
    state = np.zeros(2**num_qubits, dtype=np.complex128)
    if np.ndim(initial_state) == 0:
        state[int(initial_state)] = 1.0
    else:
        state = np.asarray(initial_state, dtype=np.complex128).copy()
    for spec in gate_specs:
        name, qubits, *rest = spec
        state = gate_unitary(name, qubits, num_qubits, *rest) @ state
    return state


def phaseless_distance(first, second):
    """Return the distance between two state vectors, minimized over a global phase.

    ``1 - |<a|b>| / (|a| |b|)`` -- zero exactly when the two agree up to a global phase. Used where
    a global phase is genuinely unobservable; assert exact equality instead wherever it is not.
    """
    first = np.asarray(first).ravel()
    second = np.asarray(second).ravel()
    norms = np.linalg.norm(first) * np.linalg.norm(second)
    if norms == 0.0:
        return 0.0 if np.linalg.norm(first) == np.linalg.norm(second) else 1.0
    return float(1.0 - abs(np.vdot(first, second)) / norms)


def real_pauli_strings(num_qubits, count, rng, letters="IXYZ"):
    """Return ``count`` distinct Pauli strings with an even number of Ys.

    ``PauliSumXZ`` narrows ``.c`` to float64 exactly when the folded ``(-i)^{x.z}`` phase is real,
    which holds precisely when each string has an even Y count. An odd-Y string leaves the
    coefficients complex128 -- correct, but not the real-arithmetic path these tests want.
    """
    strings, seen = [], set()
    while len(strings) < count:
        candidate = "".join(rng.choice(list(letters), size=num_qubits))
        if candidate.count("Y") % 2 or candidate in seen:
            continue
        seen.add(candidate)
        strings.append(candidate)
    return strings


def unique_states(num_draws, num_qubits, rng):
    """Return a lex-sorted, duplicate-free state array drawn from ``rng``.

    ``sqd`` pads a state list up to ``states_size`` with all-ones (255) *filler* rows, and
    uniquification produces the same fillers wherever the input collapsed. So whether a fixture's
    rows are already distinct decides whether that fixture exercises the filler path at all -- which
    makes it a *precondition* of any test whose reference is another arm of itself, not an incidental
    property. Naming it here is what stops the seven call sites from each looking like an unexplained
    ``np.unique``.

    Note the returned row count is ``<= num_draws`` and varies with the seed: 7 draws over 4 qubits
    measured anywhere from 3 to 7 distinct rows across 200 seeds. Callers needing a floor should
    assert it; callers needing an exact count should not use random draws at all.

    See :func:`collapsing_states` for the deliberate opposite.
    """
    draws = rng.integers(0, 2, size=(num_draws, num_qubits)).astype(np.uint8)
    return np.unique(draws, axis=0)


def collapsing_states(num_draws, num_qubits, rng):
    """Return a state array that is guaranteed to contain duplicates, and assert it does.

    The counterpart to :func:`unique_states`, for tests that need filler slots present in *every*
    arm. Draw enough rows that collision is overwhelmingly likely, then check rather than assume --
    an unrelated edit upstream of the draw shifts the RNG stream and can silently turn a collapsing
    fixture into a distinct one (measured: changing a preceding ``real_pauli_strings`` count from 5
    to 6 moved the collapse from 7 uniques to 9). A test whose blind spot depends on the collapse
    would then quietly start testing something else.
    """
    draws = rng.integers(0, 2, size=(num_draws, num_qubits)).astype(np.uint8)
    assert len(np.unique(draws, axis=0)) < num_draws, (
        f"{num_draws} draws over {num_qubits} qubits did not collide -- this fixture must collapse; "
        "raise num_draws or lower num_qubits"
    )
    return draws


def run_sharded_child(script_name, subject, num_devices=4):
    """Run a ``tests/_sharded_*.py`` script under virtual devices and return its stdout.

    Multi-device coverage has to go through a subprocess: the virtual device count comes from
    ``XLA_FLAGS=--xla_force_host_platform_device_count``, which XLA reads at backend initialization,
    and this module has already imported jax by collection time. The child scripts live in files
    rather than ``textwrap.dedent`` blobs so ruff and ty check them -- as a blob, an ``ImportError``
    from a rename would surface as a nonzero exit, indistinguishable from the regression under test.

    A plain function, not a ``@pytest.fixture``: the prohibition at the top of this module is about
    RNG stream position depending on fixture ordering, and this draws no RNG in the parent (each
    child seeds itself).

    ``check=False`` is deliberate -- the caller-facing assertion here reports the child's stderr,
    which is far more useful than ``CalledProcessError``'s bare exit code for a jax sharding raise.

    Args:
        script_name: Basename of the script in this directory, e.g. ``"_sharded_svsim.py"``.
        subject: Named in the failure message, e.g. ``"svsim"``.
        num_devices: Virtual device count to request.

    Returns:
        The child's stdout. Callers parse their own line protocol and **must** assert their case set
        is complete before checking values, or a child that dies partway passes on what it printed.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, script_name)
    assert os.path.exists(script), f"missing sharding harness at {script}"
    env = {**os.environ, "XLA_FLAGS": f"--xla_force_host_platform_device_count={num_devices}"}
    proc = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=os.path.dirname(here),
    )
    assert proc.returncode == 0, f"sharded {subject} raised:\n{proc.stderr[-3000:]}"
    return proc.stdout


def assert_type_checks(probe_source, subject, rules=("invalid-argument-type",)):
    """Assert ``ty`` accepts ``probe_source``, with ``rules`` forced to error.

    For the optional-dependency type-alias defect: ``type X = A | B`` followed by ``X |= C`` under a
    ``HAS_*`` guard leaves the ``C`` arm invisible to a checker, because a ``type`` statement is
    evaluated statically and the augmented assignment only mutates the runtime object. Three modules
    had it (``sqd``, ``svsim``, ``qprint``).

    Two reasons this shells out rather than asserting at runtime, both of which silently turn the test
    into a no-op if missed:

    - The defect is invisible to the interpreter. A runtime assertion on the alias object passes
      against the broken and fixed shapes alike.
    - This repo sets the relevant rules to ``ignore`` in ``pyproject.toml`` (numpy/JAX stub noise), so
      they must be re-enabled per-invocation via ``-c`` or the check passes against the bug.

    The probe is written inside the project tree because ``ty`` resolves configuration and first-party
    imports from the project root and silently checks **nothing** for a file outside it -- a probe in
    ``/tmp`` reports "All checks passed!" even for a blatant type error.

    Args:
        probe_source: Python source for the probe module.
        subject: Named in the failure message, e.g. ``"svsim.CircuitInput"``.
        rules: ``ty`` rule names to force to ``error`` for this invocation.

    Returns:
        None. Skips if ``ty`` is not installed (it is in the ``dev`` extra).
    """
    ty = shutil.which("ty")
    if ty is None:
        pytest.skip("ty is not installed (it is in the `dev` extra)")
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", prefix="_probe_", dir=here, delete=False
    ) as handle:
        handle.write(probe_source)
        probe = handle.name
    args = [ty, "check"]
    for rule in rules:
        args += ["-c", f'rules.{rule}="error"']
    try:
        proc = subprocess.run(
            args + [probe],
            capture_output=True,
            text=True,
            # A non-zero exit is the thing under test, so never raise on it.
            check=False,
            cwd=os.path.dirname(here),
        )
    finally:
        os.unlink(probe)
    assert proc.returncode == 0, (
        f"{subject} must accept its optional-dependency arm:\n{proc.stdout}\n{proc.stderr}"
    )


def assert_imports_without(module, blocked, extra_source=""):
    """Assert ``module`` imports in a subprocess where ``blocked`` top-level packages are unavailable.

    The companion risk to :func:`assert_type_checks`: naming an optional type in a ``type`` statement
    is only safe because the statement is lazy, so the annotation-only import is never resolved.
    "Nothing reads ``__value__``" is the kind of claim that rots, so it is pinned.

    Subprocessed with an import hook rather than monkeypatched, since these packages are installed in
    this venv and cannot be hidden in-process once the module under test has been imported.

    Args:
        module: Dotted module name to import, e.g. ``"rqutils.svsim"``.
        blocked: Top-level package names to make unimportable.
        extra_source: Optional extra statements run after the import, to exercise a runtime path.
    """
    script = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        f"        if name.split('.')[0] in {tuple(blocked)!r}:\n"
        "            raise ImportError('blocked for test')\n"
        "sys.meta_path.insert(0, Block())\n"
        f"import {module} as m\n" + extra_source + "\nprint('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, (
        f"{module} must import without {', '.join(blocked)}:\n{proc.stdout}\n{proc.stderr}"
    )
