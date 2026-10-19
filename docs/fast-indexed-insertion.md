# Indexed fast insertion

`Lattice.addIntentFastIndexed(instance)` uses the same intersection-capable
insertion as `addIntentFast`, with exact-extent indexes and conservative singleton
bounds. Both methods preserve their common candidate ordering, extent updates,
symbolic edge order, and classification-radius calculation. Callers should use
one native PyTorch thread per score; the lattice owns parallel scoring of
independent models.

For selected bottom concepts and generator children, insertion computes the
learned intent intersection by fitting a CVA to the union of the operand extents.
New intersections use the full configured fit, including their CVA projector;
they do not start with an average-only model. If the incoming object reuses an
existing concept, the union includes that concept's complete extent.

Parent intersections are inserted recursively, including a new root when no
existing generator contains both operands. Exact extents are reused, including
extents already scheduled to receive the incoming symbol. Learned-intent reuse
can also select nodes whose recorded extents do not yet contain the operands.
Registration prefers symbol overlap at `equalityCoverage` when the learned intent
also matches, then searches other equivalent intents. Both indexed and unindexed
insertion use this approximate lookup; exact snapshots remain a fast path.
Before reusing an intersection, its missing members are added to the selected
node and every ancestor. A candidate is rejected if this expansion would make
an existing parent and child have equal extents.
Cover links are inserted above both operands, moving bypassed edges through the
new intermediate nodes. Existing models remain unchanged until this structural
work is complete, keeping cached model distances valid. Expanded models are
refitted afterward.

Intersection comparisons cache scalar CVA classifications by classifier ID and
tensor identity for one insertion. The cache retains at most 65,536 results across
2,048 classifiers, with at most 4,096 results per classifier; overflow uses the
same scalar classification path. Extent changes do not invalidate these scores
because models stay fixed until refitting. Scores are discarded before the next
insertion and are never shared across lattices.

Approximate registration rejects insufficient symbol overlap early and avoids
checking the same rejected candidate again during the fallback scan. New targets
already contain their complete extent and need no propagation. Reused nodes append
members in batches, publishing one extent-index update per changed node rather
than rebuilding its index for every appended member. Candidate order, intersection
budgets, CVA arithmetic, equivalence thresholds and model refits are preserved.

A frozen before/after comparison on the first 32 local 300-dimensional embeddings
used `CvaConfig(0.5, 0.5)`, `equalityCoverage = 0.8`, a 32-node basis cache and one
native PyTorch thread. Across three warmed runs with alternating execution order,
median indexed insertion time fell from 61.10 seconds to 27.39 seconds (2.23x).
Both versions produced the same 704 node IDs, extents, edges, common vectors,
projections and radii. A separate 32-sample, eight-dimensional clustered fixture
was roughly unchanged (194.9 ms versus 188.7 ms, 11 nodes). Parsing, warmup and
result comparisons were excluded from these timings; these are local fixtures,
not a full-checkpoint or full-corpus measurement.

Intersection discovery uses at most `maxChildrenPerGenerator` starts and parents
per step, `maxRecursionDepth` levels, and `limits.ancestorStarts` incomparable
concepts. These bounds preserve the fast method's local approximation. Once an
existing concept receives a new symbol, all of its existing ancestors receive
it too; truncating that propagation would violate symbolic extent inclusion.
Intersection creation can increase the node count and insertion cost.

Search defaults favor a smaller neighborhood. Regular insertion samples at most
2,048 bottom nodes, retains 128 starts, and visits 256 generator candidates.
Both paths inspect up to 8 children per generator and limit recursion depth to 4.
Fast insertion scales its budgets with
`sqrt(fastBudgetReferenceNodes / nodeCount)`, clamped by
`fastMinimumBudgetScale` and these per-budget floors:

| Fast budget | Maximum (up to 1,024 nodes) | Floor |
| --- | ---: | ---: |
| Bottom samples | 512 | 128 |
| Bottom starts | 64 | 16 |
| Search visits | 192 | 48 |
| Ancestor/intersection starts | 24 | 8 |

At 253,464 nodes, the defaults use the floor values. Floors cannot override a
lowered maximum; previously the floors forced at least 512 starts and 1,024
visits even when their constants were smaller. Narrower searches can select
different concepts and discover fewer intersections. Intersection fitting and
recursive insertion remain enabled, and both insertion paths propagate accepted
extent changes through every existing ancestor to preserve valid edges.

On a local fixture with 8,192 isolated, two-dimensional singleton concepts,
16 insertions took a median 4.638 seconds before tuning and 1.238 seconds after
through regular insertion (3.75x faster). Indexed fast insertion went from
1.877 seconds to 0.715 seconds (2.63x faster). These are medians of three runs
with alternating before/after order, excluding checkpoint loading. Fast insertion
finished with 8,310 nodes instead of 8,512 because of the narrower intersection
search. This fixture measures search-budget effects, not full-corpus performance.

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
An approximate intersection merge can append several samples at once; those
changed average models are fully refitted after structural insertion.
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

Run `cvafca.LatticeFastIntersectionTest` for root construction, recursive parent
intersections, modified generators, intersection reuse, actual CVA projectors,
and cover-edge invariants through both fast entry points.

Run `cvafca.LatticeApproximateExtentTest` for relaxed intersection reuse, ancestor
extent propagation, complete model refits, rejection of collapsing parent edges,
and approximate candidate lookup with and without indexes.

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
Before intersection handling was added, full refits took 131.5 seconds. Basis
reuse, suffix replay, and work buffers reduced the same workload to 7.5 seconds,
including the initial basis rebuild. Common vectors, projections, radii, and
node counts matched bit for bit after every insertion. This isolated measurement
excludes loading and does not measure searches across the full checkpoint or
the additional intersections now computed by fast insertion.
Full-corpus speed depends on concept sizes, search costs, cache eviction, and
how much of a basis must be replayed when the rank tolerance changes.
