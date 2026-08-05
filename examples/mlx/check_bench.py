"""Verify bench.py's JAX arms and its gate/reporting logic end to end.

This covers the JAX arms only (jax-cpu-f64, jax-cpu-f32) plus arm-name validation, the
correctness gate's self-test hook, and text-report rendering. It does not exercise any of
the three mlx-* arms: mlx.core loads a Metal device on import even for mx.cpu arrays, so
those arms can only be verified on the human partner's hardware.
"""

import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench.py")
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

# 6. --matvec: a JAX arm must refuse "metal" (MLX-only kernel). Driven through subprocess like
# every other check here -- importing bench.py would pull in mlx.core and need a Metal device,
# which is exactly what this file avoids.
out = subprocess.run(
    BASE + ["--arm", "jax-cpu-f32", "--matvec", "metal"],
    capture_output=True,
    text=True,
    check=False,
)
assert out.returncode != 0, f"jax-cpu-f32 accepted --matvec metal:\n{out.stdout}"
assert "metal" in (out.stdout + out.stderr), (
    f"rejection did not mention metal:\n{out.stdout}\n{out.stderr}"
)
print("OK  --matvec metal refused for a jax arm")

# --chunk must be honoured and must not change the answer: the chunked gather is now the default
# matvec for both frameworks, so a chunk-size-dependent eigenvalue would mean an indexing bug in
# the one code path every arm goes through. Nothing else here exercises --chunk.
out_default = subprocess.run(
    BASE + ["--arm", "jax-cpu-f64", "--json"], capture_output=True, text=True, check=False
)
out_chunk8 = subprocess.run(
    BASE + ["--arm", "jax-cpu-f64", "--matvec", "chunked", "--chunk", "8", "--json"],
    capture_output=True,
    text=True,
    check=False,
)
assert out_chunk8.returncode == 0, f"--chunk 8 rejected for a jax arm:\n{out_chunk8.stderr}"
row_default = json.loads(out_default.stdout)
row_default = row_default["results"][0] if "results" in row_default else row_default
row_chunk8 = json.loads(out_chunk8.stdout)
row_chunk8 = row_chunk8["results"][0] if "results" in row_chunk8 else row_chunk8
assert row_chunk8["status"] == "ok", row_chunk8
assert row_chunk8["chunk"] == 8, f"--chunk 8 not recorded in the result row: {row_chunk8}"
assert abs(row_chunk8["eigval"] - row_default["eigval"]) < 1e-9, (
    f"--chunk 8 changed the eigenvalue: {row_chunk8['eigval']} vs {row_default['eigval']}"
)
print("OK  --chunk 8 passes its gate and leaves the eigenvalue unchanged")

# An unknown --matvec value must be rejected by argparse rather than falling through to a default.
out = subprocess.run(
    BASE + ["--arm", "jax-cpu-f64", "--matvec", "bogus"],
    capture_output=True,
    text=True,
    check=False,
)
assert out.returncode != 0, "--matvec bogus was accepted"
print("OK  --matvec bogus rejected")

# The removed flags must be gone from the parser, not silently ignored: a stale script passing
# --sas/--eig/--compile-body should fail loudly rather than appear to have configured something.
for dead_flag in ("--sas", "--eig", "--compile-body"):
    out = subprocess.run(
        BASE + ["--arm", "jax-cpu-f64", dead_flag, "metal"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode != 0, f"{dead_flag} is still accepted by the parser"
print("OK  --sas/--eig/--compile-body are rejected (collapsed into --matvec and the arm name)")

# 7. The sparse reference path: above DENSE_REFERENCE_MAX_DIM (5000), dense_reference's (N, N)
# array stops being allocatable at large N, so the gate must fall back to sparse_reference and
# say so. --num-qubits 13 keeps 2**13=8192 comfortably above --num-states 8000 (states are drawn
# with replacement, so the realized subspace_dim is a bit below 8000) while staying above
# BRUTE_FORCE_MAX_QUBITS (12), which auto-skips the slow 2^n cross-check -- --skip-brute-force
# also makes that skip explicit. --num-paulis 20 (rather than the default 100) keeps the sparse
# operator small enough (J*N nonzeros) to build and eigsh quickly.
out = subprocess.run(
    [
        "uv",
        "run",
        "python",
        BENCH,
        "--arm",
        "jax-cpu-f64",
        "--num-qubits",
        "13",
        "--num-paulis",
        "20",
        "--num-states",
        "8000",
        "--repeat",
        "1",
        "--fixed-iters",
        "20",
        "--skip-brute-force",
        "--json",
    ],
    capture_output=True,
    text=True,
    check=False,
)
assert out.returncode == 0, f"sparse-reference arm failed its gate:\n{out.stdout}\n{out.stderr}"
row = json.loads(out.stdout)
row = row["results"][0] if "results" in row else row
assert row["status"] == "ok", row
assert row["reference_path"] == "sparse", (
    f"expected the sparse reference path above DENSE_REFERENCE_MAX_DIM, got "
    f"{row.get('reference_path')!r}: {row}"
)
print(f"OK  sparse reference path used and passed its gate (dim>5000, eig={row['eigval']:.10f})")

print("\nJAX-SIDE CHECKS PASSED (MLX arms still unverified)")
