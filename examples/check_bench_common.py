"""Standalone verification script for _bench_common.py.

This repo has no pytest suite by design. Run with: uv run python examples/check_bench_common.py
"""

import os
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bench_common import (
    brute_force_reference,
    build_solver_inputs,
    dense_reference,
    generate_problem,
    timeit,
)

ps, cs, states = generate_problem(num_qubits=10, num_paulis=20, num_states=200, seed=1)
assert len(ps) == 20 and len(cs) == 20
assert all(p.count("Y") % 2 == 0 for p in ps), "generated an odd-Y Pauli string"
assert states.shape == (200, 10) and states.dtype == np.uint8

# The even-Y property is what makes the coefficients real; assert it explicitly.
from rqutils.paulis.symplectic import PauliSumXZ

ham = PauliSumXZ.from_paulisum((ps, cs), force_real=True)
assert ham.c.dtype == np.float64, f"coeffs not real: {ham.c.dtype}"

inputs = build_solver_inputs(ps, cs, states)
assert inputs.xsources.dtype == np.int32, inputs.xsources.dtype
assert inputs.diagonals.dtype == np.float64, inputs.diagonals.dtype
assert inputs.xsources.min() >= 0, "xsources not sanitized: still contains -1"
assert inputs.xsources.shape == inputs.diagonals.shape
assert inputs.vinit.shape == (inputs.subspace_dim,)
assert np.isclose(np.linalg.norm(inputs.vinit), 1.0), "vinit not normalized"

H, ref = dense_reference(inputs)
assert np.abs(H - H.T).max() == 0.0, f"reference not symmetric: {np.abs(H - H.T).max()}"

bf = brute_force_reference(ps, cs, states)
assert abs(ref - bf) < 1e-9 * max(1.0, abs(bf)), f"references disagree: {ref} vs {bf}"

# timeit must return two positive floats and actually call fn
calls = []
c, s = timeit(lambda: calls.append(1) or np.zeros(10), repeat=3, sync=lambda r: r)
assert len(calls) == 4, f"expected 1 warmup + 3 timed, got {len(calls)}"
assert c > 0 and s >= 0

print(f"OK  N={inputs.subspace_dim} J={inputs.num_xgroups}")
print(f"OK  dense_reference={ref:.12f}  brute_force={bf:.12f}  diff={abs(ref - bf):.2e}")

# Sparse gate path: eigsh on the same operator must agree with the dense eigvalsh it replaces.
# Validated against the dense path rather than trusted on arrival -- CLAUDE.md prefers an
# independent reference, and eigsh is a different algorithm in a different library.
from _bench_common import DENSE_REFERENCE_MAX_DIM, sparse_reference

ps_s, cs_s, states_s = generate_problem(10, 20, 200, seed=1)
inputs_s = build_solver_inputs(ps_s, cs_s, states_s)
_, dense_eig = dense_reference(inputs_s)
sparse_eig = sparse_reference(inputs_s)
assert np.isfinite(sparse_eig), f"sparse_reference returned non-finite {sparse_eig}"
err_sparse = abs(sparse_eig - dense_eig)
assert err_sparse < 1e-9 * max(1.0, abs(dense_eig)), (
    f"sparse_reference {sparse_eig} disagrees with dense_reference {dense_eig} by {err_sparse}"
)
print(f"OK  sparse_reference matches dense_reference (err {err_sparse:.2e})")
assert DENSE_REFERENCE_MAX_DIM >= 4000, (
    f"DENSE_REFERENCE_MAX_DIM={DENSE_REFERENCE_MAX_DIM} must not be below the bench default "
    "--num-states 4000, or existing invocations would silently change reference path"
)
print(f"OK  DENSE_REFERENCE_MAX_DIM={DENSE_REFERENCE_MAX_DIM} keeps the bench default dense")
