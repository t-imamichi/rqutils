# Response: `apply_h` places `vec` for you; it will not round the length

Reply to `markdown/rqutils-apply-h-mesh-request.md`, from the `rqutils` side. Branch `dev`, version still
`0.2.0` (unreleased). Measured on 4 virtual CPU devices, the same harness the request used.

> **Status: half shipped, deliberately.** Ask (1) — accept a plain numpy `vec`, placing it internally —
> is done, and does slightly more than asked. Ask (2) — round the working length up to `mesh.size` — is
> **declined**, and this document is mostly about why, because the request anticipated the refusal and
> argued against it in advance. Your `_place` goes away; `_mesh_size` stays.
>
> Everything factual in the request verified. Both errors reproduce verbatim, the reproducer runs as
> written, and `sqd.py:878`/`:899` do already implement the policy you point at. One number is wrong
> (§5) and one mechanism sentence needs a footnote (§4); neither changes the ask.

## 1. What shipped

`apply_h` now places `vec` on the live mesh, replicated, before the kernel sees it. Your reproducer's
second case — the one that raised `Resource axis: x of P('x',) is not found in mesh: ()` — returns the
same values as the hand-placed arm, bit-for-bit.

It covers one case beyond the ask. The discriminator is **mesh identity**, not `isinstance(vec,
jax.Array)`, because a `jax.Array` *committed to a single device* carries an empty mesh exactly as a
host array does:

```python
jax.device_put(v, jax.devices()[0])          # a real jax.Array
jax.typeof(_).sharding                       # NamedSharding(AbstractMesh(()), P(None,))
```

An `isinstance` guard passes that straight through to the identical error. I shipped the `isinstance`
form first and mutation-testing killed it, so the test suite now pins the committed-array arm
(`test/sharded/apply_h_vec.py`, `committed` arm, asserted to agree with the host arm at exactly 0.0).

Nine lines, one call site in the public wrapper. `run_sqd`/`ground_locg` call the private
`_apply_h_kernel` directly and never reach it, which is correct — they build their vector on-device via
`_spread_seed`. `apply_h` used *as* `ground_locg`'s `matvec` callable traces inside the solver's jit,
where `jax.sharding.get_mesh()` raises outright, so the placement is skipped under tracing. That last
point cost a red test before it was handled; `test/sharded/sqd_grid.py` is the arm that caught it.

## 2. Why ask (2) is declined

The request's §"Both or neither is what helps" is **correct on its own terms**, and I verified it against
my own first implementation rather than arguing from principle. With placement only, your line-19 target
call site still raises:

```python
apply_h(vec, xsignatures=x, zsignatures=z, coeffs=c,
        states=uniquify_states(packed, packed.shape[0]))
# ValueError: apply_h: 23 states is not a multiple of the 4 mesh devices; ...
```

So `_mesh_size` survives and you delete 5 lines of 7, not 7 of 7. I am declining anyway, for a reason
the request could not have known: **the rounding cannot be applied consistently.**

`apply_h` has three diagonal strategies. Rounding `vec` and `states` requires rounding every per-state
array to match — and the two precomputed forms disagree about which axis that is:

| array | shape | state axis |
|---|---|---|
| `diagonals` | `(n_groups, n_states)` | trailing |
| `diag_signs` | `(n_states, n_zbytes)` | **leading** |

No single pad serves both. I built the rounding version anyway (commit `1a339e8`, kept on the branch so
it is recoverable) and it worked for `zsignatures=` — the strategy your request exercises — while
breaking the other two with `TypeError: mul got incompatible shapes for broadcasting: (24,), (23,)`
raised from inside the jitted scan. That is **worse than the error it replaced**, which at least named
the dimension.

The honest version of "both" is therefore: *"`apply_h` rounds for you, unless you use two of its three
diagonal strategies, in which case round it yourself."* A policy with that carve-out is worse than no
policy plus a good message, so `82c204b` cut it back.

## 3. What you get instead of rounding

The request's real complaint about ask (2) is discoverability — "the error message names neither
`uniquify_states` nor `states_size`, which is what makes it cost an afternoon". That is fixed without
rounding anything:

```text
apply_h: 23 states is not a multiple of the 4 mesh devices;
         size every per-state array to 24, e.g. uniquify_states(states, 24)
```

Identical across all three diagonal strategies. Three further notes on the constraint itself, all
measured:

- **It binds `states`, not `vec`.** `get_xsource` reshards one entry per *state*. My first version
  checked `vec`'s length and a divisible `vec` against an indivisible `states` sailed through to the raw
  jax error — the exact failure the check exists to prevent. Now checked on `states`, with a
  disagreement between the two named rather than reaching the kernel as a broadcast error.
- **It binds only `xsignatures=`.** An `xsources=` call does no search, so nothing reshards and any
  length is fine — verified, a 23-long `(1, 2)` call on 4 devices.
- **`vec` may be batched.** The kernel broadcasts over a leading axis, so the length check reads
  `shape[-1]`. `(2, 24)` and `(5, 24)` work and agree with the unbatched call at exactly 0.0.

## 4. One footnote on the mechanism

The request says numpy "carries no sharding, so the spec resolves against an empty mesh inside the jit".
Right, but the attribution is worth sharpening, because the `P('x',)` in the message does not come from
`vec`. A host `vec` resolves to `P(None,)` on an **empty** mesh. The `P('x',)` is derived from the
partitioned `xsource` index array, and then validated against the *operand's* empty mesh — a mismatch
between two arrays, not one malformed spec. Traceback, at `1a339e8^` (the revision the request was
written against): `sqd.py:2097 → :1876` in `apply_xgrp` → jax `named_sharding.py:579`.

Also worth stating plainly, since the request's opening frames it as the motivating case: **`sqd`'s
return does not round-trip even with placement fixed.** It is trimmed to the genuine uniques (length 23
on a 4-device mesh), so it is the one input guaranteed to fail the divisibility check. Build the subspace
at a divisible `states_size` instead; `apply_h`'s docstring now says so.

## 5. The one wrong number

§"What we do instead" says `_mesh_size` and `_place` are "14 lines of body". Measured by AST, they are
**7** — `_mesh_size` 2, `_place` 5. Line 19's derived "8 lines of 14" inherits the error. The inflated
figure came from an earlier estimate of mine that counted docstrings and blank lines, so this is my
error propagated into your document rather than yours; worth fixing before it goes anywhere, since a
reviewer checks that first.

Everything else in the document verified as written:

| Claim | Result |
|---|---|
| numpy `vec` → `Resource axis: x of P('x',) …` | verbatim |
| indivisible length → partition error naming neither helper | verbatim |
| `apply_h` docstring mentioned no mesh or sharding | zero hits |
| `uniquify_states` return arrives replicated | `P(None, None)` |
| `sqd.py:878` rounds, `:899` pads with 255 | both confirmed |
| `device_put` rejects `AbstractMesh` | verbatim |
| `apply_xgrp` uses `out_sharding=jax.typeof(vec).sharding` | `sqd.py:1876` |
| states filler 255 vs vector filler zeros are different fillers | correct, and `_place` is right to use zeros |
| reproducer runs as written | all three arms |

The 7.9e-03 zero-filler figure did not reproduce at the reproducer's size, as the document says. It is
reproduced at this branch's fixture scale, though: filling `states`' pad with zeros instead of 255 raises
the pad amplitude to **7.0e-02** and breaks the agreement check. So the finding stands, and
`test/test_mesh.py::test_pad_filler_is_inert` on your side remains the right guard.

## 6. `hproj` — not asked for, and now explicitly refused

Not in the request, but adjacent and worth reporting: `hproj` failed under a mesh at **every** subspace
size, divisible or not. `columns[valid]` is a boolean-mask gather on the partitioned array `get_xsource`
returns, which raises `ShardingTypeError`. Pre-existing, confirmed against `1a339e8^`.

I first made it work, then withdrew that in favour of an explicit `ValueError`, because nothing wants it:
`hproj` returns a host scipy matrix, `spinchain` never calls it, and `poc/sharding.py`
deliberately calls it *outside* its `with jax.set_mesh(...)` block as the unsharded oracle. Rejecting
outright removed 17 lines and retired two whole bug classes, including one I could document but never
test (the host transfer is single-process only by construction). Calling `hproj` outside a mesh context
is unaffected and stays bit-identical.

## 7. What to do on the `spinchain` side

- **Delete `_place`.** Pass the bare vector; `apply_h` places it, including the `sqd`-return and
  device-committed cases.
- **Keep `_mesh_size`.** Two lines, and its `uniquify_states(packed, size)` call site is unchanged.
- **Keep `jax_config.active_mesh()`** if anything else uses it. `apply_h` no longer needs you to
  normalize the mesh, but the `get_abstract_mesh()`/`device_put` wart it documents is a jax property, not
  an `apply_h` one.
- **Nothing is urgent.** This is unreleased, and `spinchain` works today. The call sites sit on the
  measured hot path the request flags, so run `test/test_mesh.py` rather than treating the deletion as
  free cleanup.

## 8. If you want ask (2) after all

`git diff ad61e27 1a339e8` on branch `fix/apply-h-mesh-vec` is the complete rounding implementation, with
its tests. Reinstating it means accepting the three-strategy carve-out in §2, or solving the pad-axis
problem — the only clean route I see there is for `apply_h` to take the state count as an explicit
argument rather than infer it, which is a wider API change than this request asked for. Say the word and
I will reopen it; I would rather not ship the carve-out silently.
