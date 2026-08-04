"""Verify bench_mlx's JAX arms and its gate/reporting logic end to end.

This covers the JAX arms only (jax-cpu-f64, jax-cpu-f32) plus arm-name validation, the
correctness gate's self-test hook, and text-report rendering. It does not exercise any of
the three mlx-* arms: mlx.core loads a Metal device on import even for mx.cpu arrays, so
those arms can only be verified on the human partner's hardware.
"""

import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench_mlx.py")
BASE = [
    "uv",
    "run",
    "python",
    BENCH,
    "--num-qubits",
    "10",
    "--num-paulis",
    "20",
    "--num-states",
    "200",
    "--repeat",
    "2",
    "--fixed-iters",
    "50",
]

# 1. A JAX arm must pass its gate and emit parseable JSON.
out = subprocess.run(
    BASE + ["--arm", "jax-cpu-f64", "--json"], capture_output=True, text=True, check=False
)
assert out.returncode == 0, f"jax-cpu-f64 failed:\n{out.stdout}\n{out.stderr}"
record = json.loads(out.stdout)
row = record["results"][0] if "results" in record else record
assert row["status"] == "ok", row
assert row["iters"] > 0 and row["per_it_ms"] > 0, row
print(f"OK  jax-cpu-f64 eig={row['eigval']:.10f} per_it={row['per_it_ms']:.3f}ms")

# 2. The f32 arm must also pass, at its looser tolerance.
out = subprocess.run(
    BASE + ["--arm", "jax-cpu-f32", "--json"], capture_output=True, text=True, check=False
)
assert out.returncode == 0, f"jax-cpu-f32 failed:\n{out.stdout}\n{out.stderr}"
row = json.loads(out.stdout)
row = row["results"][0] if "results" in row else row
assert row["status"] == "ok", row
print(f"OK  jax-cpu-f32 eig={row['eigval']:.10f}")

# 3. An unknown arm must be rejected, not silently ignored.
out = subprocess.run(BASE + ["--arm", "nonsense"], capture_output=True, text=True, check=False)
assert out.returncode != 0, "unknown arm was accepted"
print("OK  unknown arm rejected")

# 4. A deliberately corrupted problem must be caught by the gate rather than timed.
out = subprocess.run(
    BASE + ["--arm", "jax-cpu-f64", "--self-test-break-gate"],
    capture_output=True,
    text=True,
    check=False,
)
assert out.returncode != 0, "gate did not fail on a corrupted Hamiltonian"
assert "gate" in (out.stdout + out.stderr).lower()
print("OK  correctness gate rejects a corrupted problem")

# 5. Text (non-JSON) output must contain a header and the arm row.
out = subprocess.run(BASE + ["--arm", "jax-cpu-f64"], capture_output=True, text=True, check=False)
assert out.returncode == 0, out.stderr
assert "per_it_ms" in out.stdout and "jax-cpu-f64" in out.stdout, out.stdout
print("OK  text report renders")

# 6. --sas: a JAX arm must refuse it (MLX-only kernel), and the flag must be accepted by the
# parser. Driven through subprocess like every other check here -- importing bench_mlx would
# pull in mlx.core and need a Metal device, which is exactly what this file avoids.
out = subprocess.run(
    BASE + ["--arm", "jax-cpu-f32", "--sas", "metal"], capture_output=True, text=True, check=False
)
assert out.returncode != 0, f"jax-cpu-f32 accepted --sas metal:\n{out.stdout}"
assert "sas" in (out.stdout + out.stderr), (
    f"rejection did not mention sas:\n{out.stdout}\n{out.stderr}"
)
print("OK  --sas metal refused for a jax arm")

# --sas ops must be an explicit no-op: same eigenvalue as omitting the flag entirely, proving
# the default path is untouched.
out_default = subprocess.run(
    BASE + ["--arm", "jax-cpu-f64", "--json"], capture_output=True, text=True, check=False
)
out_ops = subprocess.run(
    BASE + ["--arm", "jax-cpu-f64", "--sas", "ops", "--json"],
    capture_output=True,
    text=True,
    check=False,
)
assert out_ops.returncode == 0, f"--sas ops rejected for a jax arm:\n{out_ops.stderr}"
row_default = json.loads(out_default.stdout)
row_default = row_default["results"][0] if "results" in row_default else row_default
row_ops = json.loads(out_ops.stdout)
row_ops = row_ops["results"][0] if "results" in row_ops else row_ops
assert row_ops["eigval"] == row_default["eigval"], (
    f"--sas ops changed a jax arm's eigenvalue: {row_ops['eigval']} vs {row_default['eigval']}"
)
print("OK  --sas ops is a no-op for a jax arm (identical eigenvalue)")

# An unknown --sas value must be rejected by argparse rather than falling through to a default.
out = subprocess.run(
    BASE + ["--arm", "jax-cpu-f64", "--sas", "bogus"], capture_output=True, text=True, check=False
)
assert out.returncode != 0, "--sas bogus was accepted"
print("OK  --sas bogus rejected")

print("\nJAX-SIDE CHECKS PASSED (MLX arms still unverified)")
