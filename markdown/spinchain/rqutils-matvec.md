# `sqd(matvec=Matvec.X)`: the kernel option that replaced `cache_level`

From the `rqutils` side, for `skqd`. Branch `dev`, version still `0.2.0` (unreleased), so all of this
arrives with your next lockfile bump against `dev`.

> **Action needed: `DiagCache` must map to `Matvec` members.** `cache_level=` is gone (`TypeError`), and
> `matvec=` accepts **only** a `Matvec` member: a plain string such as `"tables"` raises `TypeError` too.
> The four names you import (`sqd`, `apply_h` and `uniquify_states` from `rqutils.sqd`, and `PauliSumXZ`)
> are unchanged, and so are the default kernel and every answer.

## 1. Migration

```python
from rqutils.sqd import Matvec, sqd

energy, vec, basis = sqd(hamiltonian, states, matvec=Matvec.TABLES)
```

| was | now | spinchain |
| --- | --- | --- |
| `cache_level=(1, 2)` | `matvec=Matvec.TABLES` | `DiagCache.SPEED` (your default) |
| `cache_level=(1, 0)` | `matvec=Matvec.INDICES` (the `sqd` default) | `DiagCache.MEMORY` |
| `cache_level=(0, 0)` | `matvec=Matvec.ONTHEFLY` | — |
| `(0, 1)`, `(0, 2)`, `(1, 1)` | removed | — |
| `xcache_groups=J'` | removed (`TypeError`) | — |

The three removed tuples were slower *and* no smaller than a remaining kernel. `(1, 1)` is the one
`markdown/skqd-sqd-solve-tolerance.md` already measured slower than `(1, 0)`, and the reason
`DiagCache` exposed only two levels. `xcache_groups` is removed because an intermediate count could
*raise* peak memory. For a subspace that `INDICES` cannot fit, use
`Matvec.PAIRS` (§3), not `ONTHEFLY`.

Each new form traces exactly the graph its old tuple traced: same kernels, same energies.

## 2. The members

A kernel is named by what it stores between matvecs:

| member | stores | notes |
| --- | --- | --- |
| `Matvec.ONTHEFLY` | nothing | the memory floor; re-runs the `J`-fold source search every matvec, so it is the slowest by far |
| `Matvec.INDICES` | per-X-group source indices | `sqd`'s default |
| `Matvec.TABLES` | source indices and composed diagonals | fastest dense kernel; what `DiagCache.SPEED` meant |
| `Matvec.PAIRS` | each in-subspace transition once, with its factor | sparse, §3 |
| `Matvec.CSR` | both directions, sorted by target row | sparse, §3 |
| `Matvec.ELL` | the same, rows bucketed by degree | sparse, §3; the fastest kernel measured |

The first three are **dense**: they accept a mesh and are the only ones `run_sqd` and `apply_h` run.
`apply_h` has no `matvec` argument; it picks the kernel from the arrays you pass it, as before.

## 3. The sparse kernels, new since `cache_level`

`PAIRS`, `CSR` and `ELL` store only the transitions that land inside the subspace. A Krylov subspace
contains a small fraction of each group's targets (hit rate 8–12% on the fixtures below), so the dense
kernels spend most of their source-index table on the `-1` "absent" marker.

Warm `sqd` at n=60, `2^17` states, packed input (`poc/sparse-pairs.md` §9):

| fixture | `INDICES` | `PAIRS` | `CSR` | `ELL` |
| --- | --- | --- | --- | --- |
| `type1` | 6.22 s | 1.26 s | 1.62 s | **1.06 s** |
| `type2` | 10.24 s | 2.34 s | 3.42 s | **1.79 s** |

On the same fixture `TABLES` measured 2.22 s (`type1`), so `ELL` is about 2× your current default there.
All kernels return the same energy.

**Constraints:**

- **Single device.** Under a mesh `sqd` raises `ValueError` naming the member. Your sharded runs keep
  `TABLES` or `INDICES`.
- **Built on the host before the solve.** Entry counts depend on the data, so the operator arrays are
  built in `sqd()` rather than inside the jitted solve. That is 0.16–0.36 s of the times above, logged as
  its own phase.
- **Recompiles per size class, not per call.** Shapes are rounded to `m·2^k` (8 ≤ m < 16). `ELL`
  recompiles more often than `CSR`: 4 compiles against 2 over 9 growing prefixes of one subspace. That
  matters for your round-by-round growth.
- **`ELL` costs more peak memory than `CSR`.**
  - The cost is the compile memory of its bucket scans, which is fixed in N.
  - At `2^19` (`type2`) the peak was ×1.5 `CSR`'s: 582 against 381 MiB.
  - A later construction fix (`357ce39`) cut `ELL`'s whole-`sqd` peak at `2^21` from 2076 to 1685 MiB.
    No committed `CSR` figure exists at that size to compare against.
  - Where memory binds, `PAIRS` is the smallest: 336 MiB at `2^19`, against 454 for `INDICES`.
- **Integer limit.** More than `2^31` stored entries raises `ValueError`, suggesting `Matvec.INDICES` or
  a smaller subspace.

## 4. Strings in config files

`Matvec` is a `StrEnum`, so it serializes as its value. To read it back, convert explicitly:

```python
Matvec("tables")      # -> Matvec.TABLES; Matvec("table") raises ValueError
str(Matvec.ELL)       # -> "ell";  json.dumps({"m": Matvec.ELL}) -> '{"m": "ell"}'
```

Convert at your config boundary, not per call. `Matvec.TABLES == "tables"` is `True` (a member *is* a
`str`), yet `sqd(matvec="tables")` raises: the check is `isinstance(matvec, Matvec)`, so that a
misspelling fails loudly rather than landing in some kernel's `else` branch.

If `DiagCache` is itself an `Enum`, the least code is to give it `Matvec` values
(`SPEED = Matvec.TABLES`) and pass `cache.value`. `DiagCache` could also simply become `Matvec`.

## 5. Other changes riding along

- **Imports.** `rqutils.sqd` is now a package that re-exports only its public API. A private helper you
  imported from `rqutils.sqd` (e.g. `_pad_states`) now comes from its owner
  (`rqutils.sqd._states`). The four public names are unaffected.
- **`batch_matvec`.** `run_sqd` always batches, and `sqd` never exposed it, so nothing changes for you.
