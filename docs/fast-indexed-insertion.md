# Indexed fast insertion

`Lattice.addIntentFastIndexed(instance)` preserves the existing `addIntentFast`
search budgets, candidate ordering, equivalence predicate, symbolic edge order,
accepted extent updates, and classification-radius calculation. The word trainer
uses this method and inserts batches in input order. It uses one native PyTorch
thread per score; the lattice owns parallel scoring of independent models.

The indexing optimization adapts this research idea:

- Zhang et al., [A New Rapid Incremental Algorithm for Constructing Concept Lattices](https://doi.org/10.3390/info10020078), Information 10(2), 78 (2019).
  FastAddExtent uses visited/result fields to avoid repeated work and an exact
  concept lookup in its closure search (Algorithm 2, lines 1–4). Here, immutable
  extent keys index canonical nodes, and equivalence results are memoized for
  one insertion. Extent mutations update the indexes, which are rebuilt after
  checkpoint loading. The global learned-intent duplicate check still runs.

Whenever an accepted CVA node gains a new extent member, both fast insertion
methods reuse its append-only Gram-Schmidt basis and unrounded Float32 projector.
The operation order, rank tolerance, common vector, projection, and radius match a
complete fit. If the tolerance invalidates a saved direction, the unchanged basis
prefix is retained and the affected suffix is replayed. The prefix projector is
rebuilt in its original subtraction order to preserve rounding. The radius still
checks the complete extent.

Gram-Schmidt uses private BFloat16 work buffers for each difference, avoiding
native tensor allocation for every multiply, sum, and subtraction. The individual
operations retain their original rounding. Differentiable callers keep the
functional tensor path because PyTorch out operations do not support autograd.

Average models (`noCVA = true`) retain the configured online mean update path.
As with the lattice's shared tensors, sample vectors and model tensors are treated
as read-only. After externally changing sample vectors, call `Node.refit()` before
continuing insertion.

For identity-projector singleton models, a conservative partial-distance bound
uses BFloat16-rounded residuals/products and a Float32 summation-error margin to
reject impossible matches before native scoring. Possible matches retain the
original tensor score and tie breaking. Restored models with unrecognized
projectors use the full tensor check. This optimization does not use approximate
nearest-neighbor search or reduce the duplicate search coverage.

Learned CVA inclusion need not be transitive, so the formal-lattice closure and
pruning proofs in the FCA papers do not directly apply. This method preserves
the repository's bounded, approximate CVA-FCA semantics; it does not change the
engine into an exhaustive formal concept lattice constructor.

`LatticeConfig.fastBasisCacheNodes` bounds a per-lattice LRU of saved bases
(default 64). Eviction releases native basis tensors; zero disables basis reuse
and calls `Node.refit() -> CVA.update(samples)` for every changed CVA extent.
Loading a checkpoint releases the old bases; clones and loaded lattices start
with fresh basis caches. Search distances and equivalence results remain local
to each incoming sample. Singleton bounds retain decoded Float32 centers.
Neither bases nor singleton centers are written to the checkpoint format.

Run `cvafca.LatticeIndexedInsertionTest` for comparisons after each insertion:
node IDs, extents, edges, common vectors, projections and radii must match the
existing fast method. The checks also cover fractional/mutual equivalence,
standalone cache rank-tolerance fallback and suffix replay, bounded cache eviction,
checkpoint/clone continuation, direct extent mutations, bound threshold ties,
and native-pointer lifetime. The reference lattice disables basis reuse.

Run `testing.LatticeIndexedInsertionBenchmark [embedding-file] [sample-count]`
for a reproducible comparison. It warms both engines, alternates their order,
reports medians of three runs, and asserts identical final models and topology.
The reference fast method disables basis reuse; the indexed method enables it.
Reading embeddings and comparing results are excluded from the timing.

A local reproduction extracted a 593-member, 300-dimensional concept from an
existing checkpoint and appended 24 new symbols using its first 24 vectors.
After the initial native-memory cleanup, full refits took 131.5 seconds. Basis
reuse, suffix replay, and work buffers reduced the same workload to 7.5 seconds,
including the initial basis rebuild. Common vectors, projections, radii, and
node counts matched bit for bit after every insertion. This isolated measurement
excludes loading and does not measure searches across the full checkpoint.
Full-corpus speed depends on concept sizes, search costs, cache eviction, and
how much of a basis must be replayed when the rank tolerance changes.
