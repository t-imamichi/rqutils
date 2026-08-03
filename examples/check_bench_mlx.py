"""Verify bench_mlx's JAX arms and its gate/reporting logic end to end.

This covers the JAX arms only (jax-cpu-f64, jax-cpu-f32) plus arm-name validation, the
correctness gate's self-test hook, and text-report rendering. It does not exercise any of
the three mlx-* arms: mlx.core loads a Metal device on import even for mx.cpu arrays, so
those arms can only be verified on the human partner's hardware.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, 'bench_mlx.py')
BASE = ['uv', 'run', 'python', BENCH,
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
