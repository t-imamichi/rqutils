# Response: D1 landed as proposed; D2 answered with documentation, not a solver change

Reply to `docs/rqutils-requests-2.md`, from the `rqutils` side. Branch `dev`, version still `0.2.0`
(unreleased).

| # | Ask | Outcome |
| --- | --- | --- |
| D1 | Make `HamiltonianInput` visible to type checkers | **Done, exactly as you proposed.** Delete your two `ty: ignore` comments. |
| D2 | Report ground-eigenvalue multiplicity, or return an eigenspace basis | **Declined as a solver change; the third option is done.** Determinism is now a documented guarantee, and the degeneracy caveat is stated at the API boundary. |

Both of your reproductions ran as written. Thank you for the standalone probes — D1's in particular,
because this repo cannot see that error itself (see §1.2), so without your probe the natural conclusion
here would have been "works for me".

---

## 1. D1: fixed as proposed

### 1.1 What landed

We took your fix but collapsed the branch — **one definition, no `else` arm**:

```python
if TYPE_CHECKING:
    from qiskit.quantum_info import SparsePauliOp

type HamiltonianInput = PauliSumXZ | tuple[Sequence[str], Sequence[Number]] | SparsePauliOp
```

This is closer to your "shorter and equivalent for checkers" alternative than to the two-branch form,
and it works for a reason worth stating: **a `type` statement is lazy.** Its value is not computed until
something reads `__value__`, so the `TYPE_CHECKING`-only import is never resolved at runtime. Verified
by blocking `qiskit` with a `meta_path` hook: the module imports, `sqd` runs a full solve, and
`HamiltonianInput` stays a lazy `TypeAliasType` whose `__value__` would raise `NameError` if anything
forced it. Nothing does — the alias appears only in `sqd`'s and `hproj`'s annotations, and no module
calls `get_type_hints`. Both facts are now pinned by tests.

We dropped the runtime `|=` rather than preserving it, which departs from your "keep the runtime alias
exactly as narrow as it is today" note. Two reasons. First, the runtime alias is unobservable: nothing
reads it, so "narrow" had no behavioural content. Second, the `|=` was quietly *lossy* — with qiskit
installed it **replaced** the `TypeAliasType` with a plain `typing.Union`, so `HamiltonianInput` was a
different kind of object depending on whether qiskit was present. The unified form is a `TypeAliasType`
either way. If you were relying on the old runtime shape anywhere, that is the one behavioural
difference, and we would want to know.

Worth noting for your side: `svsim.CircuitInput` and `qprint.PrintReturnType` have the identical latent
bug (`|=` under a `HAS_*` guard). They are unfixed — no reported caller — but if you annotate against
either, expect the same symptom.

Your diagnosis of the cause was correct in full, including the part that matters most: the arm was
invisible **whether or not qiskit is installed**, so this was never optional-dependency behaviour.

### 1.2 One thing worth knowing: we cannot reproduce your error, by configuration

`rqutils`' own `pyproject.toml` sets `invalid-argument-type = "ignore"` in `[tool.ty.rules]` (numpy
stubs vs. this library's dtype-generic `npmod` convention). That is precisely the rule your error is
reported under, so on this repo's settings the bug is invisible — `ty check` passed against the broken
alias, and a naive probe here passes for the wrong reason.

Worth flagging because it cuts both ways: a future regression of this exact kind will not be caught by
`rqutils`' default `ty` run. The regression test added for it therefore re-enables the single rule:

```
ty check -c 'rules.invalid-argument-type="error"' <probe>
```

`tests/test_sqd.py::TestHamiltonianInputIsCheckable` runs that on a generated probe inside the project
tree, and is verified to fail against the pre-fix alias and pass after. Two details cost us a cycle and
may save you one: `ty` silently checks **nothing** for a file outside the project root (a probe in
`/tmp` reports `All checks passed!` even for `take(3.5)`), and `ty` resolves `HamiltonianInput` to an
opaque `TypeAliasType` without evaluating the alias value, so a runtime assertion on the alias object
cannot pin this defect at all.

### 1.3 Not changed

We left `apply_h`'s numpy annotations alone, per your parenthetical — agreed that loosening them to
accommodate a traced `jax.Array` at a `jit` call site would misdescribe the runtime type. Your third
suppression stays yours.

## 2. D2: the multiplicity is not available cheaply, and here is why

**Answer: no to (1) and (2), yes to the third option.** Taking your two shapes in turn.

### 2.1 Option 1 (report multiplicity) is *not* nearly free — measured

Your hypothesis was that the Rayleigh–Ritz step may already have the information. It does not, and the
reason is structural rather than a matter of plumbing it out.

The projected matrix is 3×3 over the search basis `{x, y, p}` — not a subspace of eigenvectors, but the
current iterate, the previous direction, and the residual direction. Its eigenvalue spacing describes
that basis, not the operator's spectrum. On a constructed 40-dimensional operator with an exactly 2-fold
degenerate ground eigenvalue (dense, non-diagonal, `-2.0` twice):

| quantity | value |
| --- | --- |
| true lowest eigenvalues | `-2.0, -2.0, -0.5` (multiplicity 2) |
| `ground_locg` eigenvalue | `-2.0000000000000004` ✓ |
| projected 3×3 spectrum at convergence | `[-2.0, 2.2e-16, 1.85]` |
| projected gap `ev1 - ev0` | **2.0** — would need ≈0 to signal multiplicity |

So the second-lowest Ritz value is off by 1.5 from the true second eigenvalue, and a multiplicity test
built on it would report "non-degenerate" on an operator that is 2-fold degenerate. There is no
threshold that rescues this: the quantity simply is not an approximation to the operator's second
eigenvalue. Block-size-1 LOBPCG converges the *lowest* Ritz pair and nothing else.

A genuine multiplicity count needs a second eigenpair, which is option 2.

### 2.2 Option 2 (eigenspace basis) declined

As you anticipated, and for the reason you named: the whole memory argument for this specialization is
that it holds three vectors regardless of iteration count, and `ground_locg`'s callers run it on vectors
where that is the binding constraint. Blocking it up is a different solver, not a change to this one.
If you later need it, deflate-and-resolve outside `ground_locg` (solve, project the found vector out of
the operator, solve again) gets you a second eigenpair without touching the single-vector core, at the
cost of a second full solve — that is the route we would suggest, and it belongs on the `spinchain`
side where the tolerance policy for "same eigenvalue" lives.

### 2.3 The third option: done, and the determinism is now guaranteed

You asked for a documented answer either way. **It is intended to be deterministic, and you may rely on
it at a fixed `rqutils` version.** `sqd`'s `Returns:` section now says so, along with the caveat itself.
The substance:

- **Deterministic in the arguments.** The LOBPCG start is a fixed bit-mixing hash of the subspace index
  (`_spread_seed`), with no PRNG key threaded through the public signature and no host-order
  dependence; the rest of the solve is deterministic. Your observation of 4/4 overlap 1.0 is the
  designed behaviour, not luck. `_spread_seed`'s own docstring already committed to reproducibility —
  what was missing was any statement at the API boundary, which is the real gap you identified.
- **Not stable across versions.** A change to the seed constants, the prefilter default, or the
  iteration would move which member is returned, and we do **not** treat that as a breaking change.
  Your recovery fingerprint test should pin the `rqutils` version, or canonicalize the member itself
  (e.g. fix a global phase and sign convention on the returned vector).

That is the honest form of the guarantee: strong enough for reproducibility within a pinned
environment, explicitly not a cross-version contract. Given that `recover_configurations`'
reproducibility currently rests on this, the version-pinning half is the part we would act on.

- **The degeneracy caveat is now stated** where a caller reading about `eigvec` will meet it: the vector
  is one arbitrary eigenspace member, the eigenvalue is still correct, and a quantity that is not
  basis-independent — your `_site_occupancy` reading `|v_i|^2` is named as the motivating example — gets
  an arbitrary member's value. Detection is a caller-side second opinion (`eigvalsh` on `hproj` for
  small subspaces, or deflate-and-resolve), with the §2.1 reason the solver cannot do it.

### 2.4 Agreed on framing

Your "this is a reporting gap at the API boundary, not a correctness bug" is right, and it is why the
fix is documentation rather than code. The eigenvalue — the thing `sqd` is asked for — is correct under
degeneracy; we reproduced your 4-qubit case and got `-1.5` against a true `-1.5` with a 2-fold
degenerate eigenspace.

## 3. Note on your reproduction script

Your D2 snippet is correct as written. One caution if you extend it: we briefly mis-ran a variant of it
with `ground_locg(A, A.shape[0])`, reading the second argument as a dimension. It is `xinit`, and an
integer there selects a **one-hot** vector, so `A.shape[0]` on a dimension-16 operator is an
out-of-range index that yields the zero vector — from which `ground_locg` returns `0.0` with
`niter=0, converged=True`. That looked briefly like a serious wrong-answer bug and is not one; your
`sqd`-level script never touches the path. **That is now guarded**: an out-of-range (or negative)
integer `xinit` raises `ValueError` from the public entry point, on both the array and callable paths.
Host-side, because inside `jit` the index is traced and cannot raise. Found while investigating your D2
report, unrelated to either request, and nothing you need to act on.

Also worth knowing for any future degeneracy fixture: on a **purely diagonal** projected operator
(which your `IIZZ/IZZI/ZZII` example is) every one-hot start is an exact eigenvector, so an integer
`xinit` returns `niter=0` and whichever eigenvalue that index carries. Use a vector start, as your
script effectively does via `sqd`, when you want the fixture to exercise the iteration.

## 4. What you should change

1. **Delete the two `ty: ignore[invalid-argument-type]`** at `exact.py:112` and
   `skqd/sqd_backend.py:302`. Arrives with your next lockfile bump.
2. **Pin the `rqutils` version** wherever `recover_configurations`' fingerprint reproducibility is
   asserted, and reference §2.3 rather than `_spread_seed` in that docstring — the guarantee is now
   documented at the API boundary, so the limitation note can cite something stable.
3. Nothing else. No API moved, no default changed, and `prefilter=(32, 2)` remains correct at both call
   sites.
