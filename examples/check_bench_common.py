"""Standalone verification script for _bench_common.py.

This repo has no pytest suite by design. Run with: uv run python examples/check_bench_common.py
"""
import sys
import os
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
