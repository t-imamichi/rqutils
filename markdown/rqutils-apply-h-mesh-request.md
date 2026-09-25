# rqutils change request: `apply_h` under a sharded mesh

Against `rqutils` on branch `dev` (installed rev `a09aa03`, tag `dev-0.1.4`, version 0.2.0), from the
`spinchain` side. Independent of the asks in `markdown/rqutils-requests.md` and the tol/precond/multiobs
files.

> **Status: OPEN.** Not sent upstream. `spinchain` works around it, so this is about deleting our code.

## The ask

Under a non-empty mesh, `apply_h` should do both of these:

1. accept a plain numpy `vec`, placing it internally, and
2. round its working length up to `mesh.size`, as `sqd()` does.

**Both or neither is what helps.** They are one mechanism: the vector's pad has to match the states
array's length, so a caller that rounds must also pad, and a caller that pads must know the rounded
size. `apply_h` doing only the placement still leaves every caller computing the size and padding to
it. Doing both lets a call site be `uniquify_states(packed, packed.shape[0])` and a bare vector.

Failing either, the docstring should state both requirements.

`apply_h` is public and documented as the contraction entry point, but its docstring does not mention a
mesh or sharding, so both facts below are learnable only from a traceback.

**`sqd()` already implements the fix, one function over.** `sqd.py:878` rounds `states_size` up to
`mesh.size` when the abstract mesh is non-empty, and `sqd.py:899` pads the input with 255 filler for the
same reason `uniquify_states` does. So this is not a request to choose a new policy — the library already
committed to one, and `apply_h` is the entry point that skips it. Our `_mesh_size`/`_place` re-implement
that committed policy verbatim outside the library, which is the code we are asking to delete.

## What happens

Measured on 4 virtual CPU devices (`XLA_FLAGS=--xla_force_host_platform_device_count=4`), n=14,
dim 3744–3746, on the eigenvector `sqd()` returned.

**A numpy `vec` raises**, even with correctly padded states:

```text
ValueError: Resource axis: x of P('x',) is not found in mesh: ().
```

`apply_xgrp` takes `out_sharding=jax.typeof(vec).sharding`; numpy carries no sharding, so the spec
resolves against an empty mesh inside the jit. `sqd()` never hits this because it builds `vec` inside
its own jit — but a caller passing `sqd()`'s **return** does, because that return is numpy.
`jax.device_put(vec, NamedSharding(mesh, PartitionSpec()))` fixes it.

**An indivisible length raises:**

```text
ValueError: Sharding spec ('x',) implies that array axis 0 is partitioned 4 times,
but does not evenly divide the dimension size 3746
```

`uniquify_states(states, states_size)` pads to any requested size with 255 filler — its documented
contract — and its return arrives already replicated, so asking it for the rounded length covers the
states array entirely. The error message names neither `uniquify_states` nor `states_size`, so the fix
is not discoverable from it.

Two further constraints we found by trial. Neither is mesh-specific — both bind any direct `apply_h` or
`run_sqd` caller — and the second is a silent-wrong-answer path, which is the class this module documents
at the call site everywhere else:

- `states` must be **replicated**, not sharded. Sharding it hits `apply_h`'s vmap: "Unmapped values
  passed to vmap cannot be sharded along the mesh axis you are vmapping over."
- The **states** filler must be `255`. Zeros are a reachable state that real rows map onto through
  `xsource`, so they steal amplitude — **7.9e-03** against 0.0e+00, with nothing raised. Our end-to-end
  energy test passed with a zero-fill bug in place, since `sqd()` never sees the pad and a residual guard
  tolerates 7.9e-03. `_is_filler`, `uniquify_states` and `sqd()`'s own padding all state this convention;
  nothing reachable from `apply_h` does.

  This is the **states** array. The vector's pad is zeros, which is right for it — zero amplitude
  contributes nothing to a norm or an inner product. Two arrays, two fillers.

## What we do instead

Two functions, 14 lines of body, two call sites each (`_pack_subspace`, `_pack_subspace_packed`).
`spinchain` works today, so nothing here is blocked — this lands after an rqutils release, not before,
and the call sites sit on a measured hot path (their lexsort is ~47% of a warm js sweep), so removing
them wants `test/test_mesh.py` run rather than treating it as free cleanup.

`sqd_backend._mesh_size` rounds the dimension up to a device multiple and passes it to
`uniquify_states` as `states_size`; `_place` pads and `device_put`s the vector. Both are no-ops without
a mesh. `jax_config.active_mesh()` normalizes `jax.sharding.get_mesh()`, since `get_abstract_mesh()`
returns an `AbstractMesh` that `device_put` rejects (`is_fully_addressable is not implemented`) — worth
mentioning because the abstract one is what `apply_h`'s own code reads, so it is the first thing found.

## Not asking for

- **Nothing on performance.** `cache_level=(0,0)` (`xsignatures`/`zsignatures`) already removed a
  46.4 GiB allocation at dim=67M/n=60 and is 3.2–4.4x faster than `(1,1)` on our shapes
  (`poc/residual-oom.md`). The six-strategy grid was sufficient.
- **Not the 2^31 ceiling.** `uniquify_states` sorting on one device and int32 subspace positions cap N;
  the module docstring says so. Sharding buys memory, not dimension.

## Weakest link

Everything is CPU-only — this repo has no GPU, so the evidence is virtual devices, which exercise the
sharding logic but not real interconnect, HBM, or multi-process behaviour. The failures are shape- and
dtype-level and should be device-independent, but we would not defend 7.9e-03 or 3.2–4.4x as GPU
numbers.

## Reproducer

Both errors, run and confirmed as written:

```bash
XLA_FLAGS="--xla_force_host_platform_device_count=4" python - <<'PY'
import numpy as np, jax
from jax.sharding import AxisType, NamedSharding, PartitionSpec
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import apply_h, uniquify_states
from qiskit.quantum_info import SparsePauliOp

jax.config.update("jax_enable_x64", True)
mesh = jax.make_mesh((4,), ("x",), (AxisType.Explicit,))
jax.set_mesh(mesh)

n = 6
ham = SparsePauliOp.from_sparse_list(
    [("XX", [i, i + 1], 1.0) for i in range(n - 1)], num_qubits=n
)
x, z, c = PauliSumXZ.from_paulisum(ham).arrays
bits = np.array(
    [[(i >> k) & 1 for k in reversed(range(n))] for i in range(1, 24)], dtype=np.uint8
)
packed = PauliSumXZ.pack_states(bits)          # 23 states, and 23 % 4 != 0

# Indivisible length.
try:
    apply_h(
        np.ones(23, dtype=np.complex128) / np.sqrt(23),
        xsignatures=x, zsignatures=z, coeffs=c,
        states=uniquify_states(packed, 23),
    )
except ValueError as exc:
    print("indivisible:", exc)

# Padded by uniquify_states, but the vector is numpy.
states = uniquify_states(packed, 24)
vec = np.ones(24, dtype=np.complex128) / np.sqrt(24)
try:
    apply_h(vec, xsignatures=x, zsignatures=z, coeffs=c, states=states)
except ValueError as exc:
    print("numpy vec  :", exc)

# Placed: passes.
placed = jax.device_put(vec, NamedSharding(mesh, PartitionSpec()))
out = apply_h(placed, xsignatures=x, zsignatures=z, coeffs=c, states=states)
print("placed     :", np.asarray(out)[:3])
PY
```

The 7.9e-03 filler figure does **not** reproduce at this size: it needs a real row whose `code ^ x_mask`
lands on the pad, which a 23-state dense subspace does not provide. It is measured at n=14, dim=3746
over an XXZ Hamiltonian, and `test/test_mesh.py::test_pad_filler_is_inert` in `spinchain` is the
standing check.
