"""Multi-device tests for :mod:`rqutils.sqd`. Each subprocesses a ``test/sharded/`` child through
``conftest.run_sharded_child``, except ``TestHostScalar``, the multi-process host-read guard.
"""

import ast
import functools
import inspect
import textwrap

import jax
import numpy as np
import pytest
from conftest import MATVECS, run_sharded_child

from rqutils.sqd._solve import _host_scalar


class TestShardedSqd:
    """``sqd`` must agree sharded and single-device at every ``matvec`` and mesh size.

    **Swept over every kernel, not sampled**, because two sharding defects lived in the
    source-searching cells (now ``"onthefly"``) that nothing covered, and the first masked the second:

    * ``_accumulate_diagonal`` carried its template's rank-2 spec onto a rank-1 accumulator ("Length
      of sharding.spec (2) must be equal to aval's ndim (1)"), failing **every** kernel.
    * ``_spread_seed``'s ``jnp.where`` mixed a replicated predicate with a partitioned ``vec``, because
      ``run_sqd`` reshards ``states_u`` only when the source indices are cached: ``ShardingTypeError`` on the
      uncached kernels. Fixing the first bug halved the failures, not cleared them.

    Runs with ``prefilter=(16, 2)`` on a **padded** subspace (37 states to 64), the one prefilter
    configuration only ``sqd`` reaches: filler masked to zero, partitioned, through ``apply_h``'s
    gather-heavy kernel -- which the filter calls ``cycles * (degree + 1)`` times before the first
    iteration. ``TestChebyshevPrefilter`` covers the dense, unpadded case.

    **Asserts the spec, not only the energy**: the energies agree to 4e-16 whether or not the
    partitioning survives, since a replicated run matches single-device exactly.
    """

    def test_every_matvec_agrees_sharded_and_single_device(self):
        got = run_sharded_child("sqd_grid")
        for devices in ("1", "2", "4"):
            for matvec in MATVECS:
                single, sharded = got["single"][matvec], got["sharded"][devices][matvec]
                assert single == pytest.approx(sharded, abs=1e-12), (
                    f"devices={devices} matvec={matvec}: sharded {sharded} vs single {single}"
                )
            for label in ("part", "repl"):
                vinit_spec, filtered_spec = got["specs"][devices][label]
                assert filtered_spec == vinit_spec, (
                    f"devices={devices} {label}: the prefilter returned {filtered_spec} for a "
                    f"{vinit_spec} input -- it is not sharding-transparent"
                )
        # The partitioned arm must actually be partitioned, or the check above is vacuous.
        for devices in ("2", "4"):
            assert got["specs"][devices]["part"][1] == "P('x',)", got["specs"][devices]


class TestShardedBatchMatvec:
    """``batch_matvec`` on ``sqd``'s own matvec: same answer sharded, data axis kept partitioned.

    Batching stacks ``ground_locg``'s two per-iteration vectors into ``(2, N)``, moving the partitioned
    axis to position 1. ``jnp.stack`` on a ``P('x')`` vector yields ``P(None, 'x')``, so nothing
    reshards, and the operator's gather is paid once per pair: one ``ground_locg`` iteration measures
    3 all-gathers unbatched against 2 batched on 4 devices.

    **The spec assertion is the half values cannot make**: a stack partitioning the batch axis
    (``P('x', None)``) or replicating everything both agree with single-device to exactly 0.0. Drives
    ``ground_locg`` directly with ``run_sqd``'s ``"indices"`` operator, since ``run_sqd`` always batches.
    """

    def test_batched_and_unbatched_agree_sharded(self):
        got = run_sharded_child("batch_matvec")
        runs = got["runs"]
        for batch in ("False", "True"):
            (single, _, _), (sharded, _, _) = runs[batch]
            assert single == pytest.approx(sharded, abs=1e-12), (
                f"batch_matvec={batch}: single-device {single} against sharded {sharded}"
            )
        for arm, name in ((0, "single-device"), (1, "sharded")):
            unbatched, batched = runs["False"][arm], runs["True"][arm]
            # Batching changes only how the operator is called: same theta bit for bit, same path.
            assert batched[:2] == unbatched[:2], (
                f"{name}: batched (theta, iterations) {batched[:2]} against unbatched {unbatched[:2]}"
            )
        gathers = [runs[batch][1][2] for batch in ("False", "True")]
        assert gathers == [3, 2], f"all-gathers per iteration, unbatched then batched: {gathers}"
        assert got["specs"] == ["P('x',)", "P(None, 'x')"], (
            f"stacking must keep the data axis partitioned and replicate the batch axis, got "
            f"{got['specs']}"
        )


class TestShardedApplyHVec:
    """A host ``vec`` must work under a mesh, and an indivisible length must name the fix.

    Two mutants this kills, both of which raised "Resource axis: x of P('x',) is not found in mesh:
    ()": dropping ``_place_vec``'s call, and testing ``isinstance(vec, jax.Array)`` instead of mesh
    identity -- a committed ``jax.Array`` carries an empty mesh exactly as a host array does.

    Only ``zsignatures=`` pairs with ``xsignatures=`` now, the one input set that searches ``states``;
    ``xsources=`` is checked separately, since it has no divisibility requirement.
    """

    def test_host_vec_is_placed_and_indivisible_length_names_the_size(self):
        got = run_sharded_child("apply_h_vec")
        assert got["placed"] == pytest.approx(4.738728797964961e-02, rel=1e-9)
        # Replicated, not partitioned -- a partitioned vec hits `get_diagonal`'s vmap.
        assert got["spec"] == "P(None,)", f"expected a replicated result, got {got['spec']}"
        # Exactly 0.0: the two arms must agree bit-for-bit, not merely to a tolerance.
        assert got["committed_diff"] == 0.0, "a device-committed vec disagreed with a host vec"
        message = got["raised"]["zsignatures"]
        assert message is not None and str(got["size"]) in message, (
            f"zsignatures= did not name the required length: {message!r}"
        )
        # The check must read `states`, not `vec`: reading `vec` let a divisible vec with an
        # indivisible states through to the raw jax error this replaces.
        mismatch = got["raised"]["mismatch"]
        assert mismatch is not None and mismatch.startswith("apply_h:"), mismatch
        # The length check reads shape[-1]: reading shape[0] rejected a valid (2, 24) vec.
        assert got["batched_shape"] == [2, 24], got["batched_shape"]
        assert got["batched_diff"] == 0.0, "a batched row disagreed with the unbatched call"
        # `xsources=` does no search, so no reshard and no divisibility requirement.
        assert got["xsources_len"] == 23, f"xsources length was changed: {got['xsources_len']}"


class TestShardedHproj:
    """``hproj`` must reject a live mesh, and still work once the mesh context exits.

    It returns a host scipy matrix, so a mesh buys it nothing -- and every mesh-enabled call failed
    anyway, at any subspace size: ``columns[valid]`` is a boolean-mask gather on the partitioned array
    ``get_xsource`` returns, which raises ``ShardingTypeError``. Rejected explicitly rather than
    half-supported. ``poc/sharding.py`` is the pattern that must keep working: it builds its dense
    reference with ``hproj`` *outside* its ``with jax.set_mesh(...)`` block.
    """

    def test_hproj_rejects_a_mesh_and_works_outside_one(self):
        got = run_sharded_child("hproj")
        # Without a mesh both sizes work: hproj never cared about mesh divisibility.
        assert got["no_mesh_shapes"] == [23, 24], got["no_mesh_shapes"]
        assert got["scoped_rejects"], "hproj did not reject a scoped mesh"
        # A divisible count is not a loophole -- rejection is unconditional.
        assert got["global_rejects"] == [True, True], got["global_rejects"]
        # Exactly 0.0: leaving the mesh context must restore the single-device result bit-for-bit.
        assert got["after_scope_diff"] == 0.0, "hproj differed after the mesh context exited"


class TestHostScalar:
    """``_host_scalar`` must accept every scalar form ``sqd`` can hand it, sharded or not.

    The defect it fixes is only reachable **multi-process**: ``float(result[0])`` raised "Fetching
    value for `jax.Array` that spans non-addressable (non process local) devices" on a 4-node mesh, on
    the default ``return_eigvec=False`` path. Virtual devices are one process, so every device is
    addressable and *that error cannot be produced here at all*.

    **Mutation-verified as NOT pinning the fix, and kept anyway.** Replacing the whole body with
    ``return value`` -- undoing the fix completely -- leaves all three of these green, because
    single-process ``float()`` succeeds either way. So this class pins the helper's *contract* (it must
    accept a device scalar, a Python float, a bool and a numpy scalar, and must not perturb the value)
    and its *premise* (the scalar really is fully replicated, so reading one shard is exact rather than
    partial). The defect itself is only catchable on a real multi-process run; recorded here so nobody
    reads a green suite as multi-node coverage.

    A reduction over a partitioned vector is the shape that matters: a rank-0 array whose sharding
    still names the whole mesh. ``jax.reshard`` is not the fix (its spec is already ``P()``), so the
    assertion worth making is that the value survives the local-shard read exactly -- a replicated
    scalar holds the same number on every device, so this is exact rather than approximate.
    """

    def test_passes_through_host_scalars_unchanged(self):
        # A plain float, a bool and a numpy scalar must survive untouched: `sqd` reaches this helper
        # on the single-device path too, where `result[0]` may already be host-side.
        assert float(_host_scalar(3.5)) == 3.5
        assert bool(_host_scalar(True)) is True
        assert float(_host_scalar(np.float64(1.25))) == 1.25

    def test_reads_a_device_scalar_exactly(self):
        # Rank-0 outputs of reductions, which is the shape run_sqd returns for eigval and converged.
        vec = jax.numpy.arange(8.0)
        assert float(_host_scalar(jax.numpy.sum(vec))) == 28.0
        assert bool(_host_scalar(jax.numpy.all(vec >= 0.0))) is True

    def test_the_multi_process_branch_does_not_depend_on_this_rank(self):
        """The branch must be rank-uniform, or the ranks disagree and the job hangs.

        Two earlier versions were wrong in different ways, both only visible on real nodes:

        1. ``if not shards: return value`` read an **empty** ``addressable_shards`` as "already
           host-side" and handed the unreadable array back, so the caller's ``float()`` raised the very
           error the helper exists to prevent.
        2. Gating the fast path on ``is_fully_replicated and addressable_shards`` fixed that but
           branched on a **per-rank** property. Measured on 4 nodes, two ranks read the value while two
           raised from the same call -- so that gate would have sent two ranks into a collective and let
           two return early, hanging at the barrier instead of failing. Strictly worse.

        Single-process cannot construct either state, so what is assertable here is the guard's
        *condition*, read off the source: the multi-process decision must come from
        ``jax.process_count()``, which is identical on every rank, and must not consult this rank's
        shards. Not a substitute for a real multi-process run.
        """
        # Strip the docstring and comments: both *quote* the historical wrong forms in order to warn
        # about them, so a naive substring search matches the explanation rather than the code. This
        # test caught itself doing exactly that.
        full = inspect.getsource(_host_scalar)
        tree = ast.parse(textwrap.dedent(full))
        func = tree.body[0]
        assert isinstance(func, ast.FunctionDef)
        if (
            func.body
            and isinstance(func.body[0], ast.Expr)
            and isinstance(func.body[0].value, ast.Constant)
        ):
            del func.body[0]  # the docstring
        source = ast.unparse(func)
        assert "process_count()" in source, (
            "the multi-process branch must be decided by jax.process_count(), which is the same on "
            "every rank; a per-rank condition makes the ranks disagree and hang at the barrier"
        )
        assert "if not shards" not in source, (
            "a bare `if not shards: return value` returns an unreadable array unchanged, which is the "
            "4-node failure this helper exists to prevent"
        )
        # The fast/slow decision must not be gated on a per-rank view of the array. `addressable_shards`
        # may still appear -- it is how the value is finally read -- but not as the branch condition.
        branch_lines = [
            line
            for line in source.splitlines()
            if line.strip().startswith(("if ", "elif "))
            and ("addressable_shards" in line or "is_fully_replicated" in line)
        ]
        assert not branch_lines, (
            f"branching on a per-rank property: {branch_lines}. Ranks that can see the value would "
            "skip the collective the others enter, and the job hangs."
        )
        # The multi-process read must go through process_allgather, not a hand-rolled
        # jit(out_shardings=...). Both hand-rolled forms failed on real nodes: a bare PartitionSpec
        # needs a context mesh ("jit requires a non-empty mesh in context"), and a NamedSharding from
        # value.sharding.mesh replicates only across that array's own mesh -- one device on a 1-device
        # solve, so the result still had no addressable shard on 3 of 4 ranks and `[0]` raised
        # IndexError.
        assert "process_allgather" in source, (
            "the multi-process read must use process_allgather, which handles the mesh-in-context and "
            "array-mesh-scope cases that broke two hand-rolled jit(out_shardings=...) attempts"
        )
        assert "tiled=True" in source, (
            "process_allgather rejects tiled=False for a non-fully-addressable array outright"
        )
        # The return shape depends on addressability -- process_allgather replicates a
        # non-addressable rank-0 array to a scalar but expands a fully addressable one to
        # (process_count,). Measured on 4 nodes: the 1-device row returned (4,) and float() raised
        # "only 0-dimensional arrays can be converted to Python scalars".
        assert "reshape(-1)" in source, (
            "the gathered result must be flattened before indexing: process_allgather returns a "
            "scalar for a non-addressable input and shape (process_count,) for an addressable one, "
            "and branching on which would branch on a per-rank property"
        )

    def test_the_scalar_it_reads_is_fully_replicated(self):
        # The premise the helper rests on. If a future change made `eigval` genuinely partitioned,
        # reading one shard would silently return part of the answer -- so pin the premise, not just
        # the behaviour.
        total = jax.numpy.sum(jax.numpy.arange(8.0))
        assert total.is_fully_replicated
        assert len({float(shard.data) for shard in total.addressable_shards}) == 1


class TestShardedEigvecRoundtrip:
    """``return_eigvec=True`` on a mesh must return a genuine eigenvector of its own basis.

    The only sharded arm through the branch that reshards ``eigvec`` and ``states_u`` back to
    ``P(None)`` before returning; ``poc/sharding.py``'s POC 7c covers it too but costs 59.7 s.

    **It asserts the eigenvector equation, not shapes.** A reshard that dropped or reordered rows keeps
    the shape and dtype, and the two arrays are resharded independently, so ``‖H v - E v‖ / ‖v‖``
    against a dense projection *of the returned basis* is what couples them. The eigenvalue is checked
    against the dense minimum too, since any eigenpair satisfies the residual test.
    """

    def test_returned_eigenvector_satisfies_its_own_projection(self):
        got = run_sharded_child("eigvec_roundtrip")
        assert got["eigvec_len"] == got["basis_rows"] <= 30, got
        assert got["relative_residual"] < 1e-10, got
        assert got["eigval"] == pytest.approx(got["reference"], abs=1e-9), got


class TestShardedDiagonals:
    """The popcount diagonal path on a mesh, with states partitioned rather than replicated.

    ``_z_parity`` is ``sum(bitwise_count(states & z), axis=1) & 1``, so it reduces along the **byte**
    axis while ``P('x', None)`` shards axis 0. That should make the whole path free of collectives --
    the easy half of the rule that only elementwise ops and reductions survive a partitioned axis,
    unlike ``uniquify_states``' ``cumsum``, which reduces *along* the sharded axis and cannot.

    The child checks the **spec and the values together**: a replicated run agrees to exactly 0.0, so
    a silently unsharded builder is invisible to value comparison, and a spec check alone would not
    catch a wrong sign. Both coefficient dtypes, since an odd-Y string makes ``.c`` complex.
    """

    def test_diagonal_builders_shard_and_agree_with_single_device(self):
        got = run_sharded_child("diagonals")
        for dtype in ("real", "complex"):
            for num_devices in (2, 4):
                groups, bad_spec, bad_value = got[f"{dtype} {num_devices}"]
                case = f"{dtype}/{num_devices}"
                assert groups > 1, f"{case}: only {groups} X groups, fixture is degenerate"
                assert bad_spec == 0, f"{case}: {bad_spec} outputs lost their 'x' spec"
                assert bad_value == 0, f"{case}: {bad_value} outputs differ from single-device"


class TestShardedPairs:
    """Term-parallel ``"pairs"``: a contiguous slice of the entries per device, all-gather in, reduce-scatter out.

    ``markdown/pairs-mesh-proposal.md`` section 4.1; ``sqd``'s residual check runs on each sharded solve.
    """

    def test_sharded_pairs_matches_single_device_and_dense(self):
        for label, case in _pairs_mesh().items():
            assert case["single"] == pytest.approx(case["dense"], abs=1e-10), label
            for devices, cell in case["devices"].items():
                where = f"{label}/{devices}"
                assert cell["eigval"] == pytest.approx(case["single"], abs=1e-12), where
                assert cell["product_diff"] < 1e-12, (where, cell["product_diff"])

    def test_entries_are_partitioned_not_replicated(self):
        """A replicated operator agrees in value, so the spec and per-device contents are asserted."""
        for label, case in _pairs_mesh().items():
            for devices, cell in case["devices"].items():
                where = f"{label}/{devices}"
                assert cell["specs"] == ["P('x',)", *["P('x', None, None)"] * 3], (
                    where,
                    cell["specs"],
                )
                assert cell["product_spec"] == "P(None, 'x')", (where, cell["product_spec"])
                assert cell["shards"] == int(devices), where
                counts = cell["entries_per_device"]
                assert max(counts) - min(counts) <= 1, (where, counts)
                assert cell["same_entries"], (
                    f"{where}: the slices are not the flat operator, in order"
                )

    def test_one_all_gather_and_one_reduce_scatter_per_batched_matvec(self):
        expected = {
            "all-gather": 1,
            "reduce-scatter": 1,
            "all-reduce": 0,
            "all-to-all": 0,
            "collective-permute": 0,
        }
        for label, case in _pairs_mesh().items():
            for devices, cell in case["devices"].items():
                assert cell["collectives"] == expected, (f"{label}/{devices}", cell["collectives"])


@functools.cache
def _pairs_mesh():
    """``pairs_mesh``'s result, run once for the three tests above: it is deterministic, ~8 s a run."""
    return run_sharded_child("pairs_mesh")
