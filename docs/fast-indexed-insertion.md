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

Whenever an accepted CVA node gains a new extent member, indexed insertion calls
`Node.refit() -> CVA.update(samples) -> CVA.update_cva(samples) -> CVA.fit_cva(samples)`.
The common vector, projection, and radius are recomputed from the complete extent.
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

`LatticeConfig.fastBasisCacheNodes` is retained for configuration compatibility;
indexed insertion now always performs full CVA refits. The standalone basis-cache
implementation remains available for explicit cached fitting. Singleton bounds
retain decoded Float32 centers, which are not written to the checkpoint format.

Run `cvafca.LatticeIndexedInsertionTest` for comparisons after each insertion:
node IDs, extents, edges, common vectors, projections and radii must match the
existing fast method. The checks also cover fractional/mutual equivalence,
standalone cache rank-tolerance fallback, checkpoint/clone continuation, direct
extent mutations, bound threshold ties, and native-pointer lifetime.

Run `testing.LatticeIndexedInsertionBenchmark [embedding-file] [sample-count]`
for a reproducible comparison. It warms both engines, alternates their order,
reports medians of three runs, and asserts identical final models and topology.
Reading embeddings and comparing results are excluded from the timing.

Earlier local CPU results with CVA enabled, classification threshold 0.3, and
cached CVA refits (these timings do not describe the current full-refit path):

| Input | Existing fast method | Indexed method | Speedup |
| --- | ---: | ---: | ---: |
| First 192 embeddings, 300 dimensions | 1107.7 ms | 292.5 ms | 3.79× |
| Growing 64-member cluster, 64 dimensions | 1840.7 ms | 169.4 ms | 10.87× |

These measure small insertion workloads; full-corpus speed depends on the
number and size of nodes and the frequency of cache eviction or rank fallback.
