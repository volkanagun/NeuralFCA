# FastAddIntent with CVA membership

Use `lattice.fastAddIntent(instance)` for one symbol, or
`lattice.fastAddIntent(instances)` to insert one intent fitted from several
embeddings.

This adapts the closure lookup, recursive intersection insertion and cover
repair of Zou, Zhang and Long's
[FastAddIntent (2015)](https://doi.org/10.1016/j.eswa.2015.01.044).
Because the operands here are stored extents, the traversal is dualized: parents
have larger extents, closure search descends, and intersections become children.

The learned operations are:

* **Membership:** `node.classifies(instance)`, which compares CVA distance with
  that model's classification radius. It does not compare symbol names.
* **Inclusion:** every sample of the smaller node must be classified by the
  larger node's CVA.
* **Intersection:** inspect samples from both operands and retain those
  classified by **both** CVAs. Deduplicate retained samples by symbol, then run
  the configured full `cva.fit` on their embeddings. This rebuilds the common
  vector, projection and radius. A symbol need not already be recorded in both
  extents to belong to the learned intersection. An empty intersection remains
  virtual; it never calls `cva.fit` with empty input.
* **Equivalence:** each classifier must accept at least `equalityCoverage` of
  the **other** node's extent. Both directions must pass. The optional existing
  `requireMutualCommonClassification` setting additionally checks both common
  vectors. Exact symbolic extent matches still require this CVA check.

Closure and equivalence discovery share a graph beam search. Exact extent lookup
still runs first and requires full CVA equivalence. Recursive searches start from
their supplied generator, descending when it contains the requested extent and
otherwise walking upward. Searches without a generator start at the smaller of
the indexed top and bottom frontiers. There is no full-lattice closure derivation,
vocabulary collection, root scan, or global equivalence fallback.

Candidates are ranked by CVA distance to the first, middle and last requested
embeddings, deduplicated by tensor identity. Common vectors from different
projections are not used as search coordinates. These probes only guide traversal:
each retained equivalent node must pass both complete coverage predicates, and
each generator must classify every requested embedding. Closest candidates expand
through children or parents in layers, with deterministic distance/size/ID ordering.

| Setting | Default | Purpose |
| --- | ---: | --- |
| `closureSearchVisits` | 1024 | Maximum distinct candidate models scored per search. |
| `closureSearchFrontier` | 512 | Maximum sampled frontier or neighboring layer; also limits virtual-root intersection starts. |
| `closureSearchBeamWidth` | 16 | Candidates fully checked and expanded per layer. |
| `closureSearchParallel` | true | Parallel probe scoring when its estimated work justifies scheduling. |

Initial layers use at most half the visit budget, leaving room for graph traversal.
Increase the search limits to explore more branches. A bounded search can miss an
equivalent node or a better generator, so node counts and covering relations can
differ from exhaustive discovery. Recursive intersection insertion has no depth
limit, and ancestor propagation remains complete.

At least 32 candidates and an estimated two million projection/vector elements
are required before scoring runs on up to eight workers. Small vectors and average
models usually stay sequential. Searches already running on a fork-join worker
also stay sequential to avoid nested scheduling while holding the lattice monitor.
Workers only read fixed models; the caller publishes all mutable cache entries.

On a local fixture with 256 overlapping parent concepts, 64 shared leaves and
300-dimensional embeddings, median insertion time fell from 2327.64 ms to
84.81 ms (27.45x), excluding fixture and index construction. Serial graph search
took 98.16 ms on the same fixture. These are medians of four measured runs after
warmup with alternating implementation order; node IDs, extents and edges matched
the frozen preceding implementation. This is a synthetic search-heavy fixture,
not a full-corpus training measurement.

Classifications are cached for one insertion, with at most 65,536 entries; cache
overflow performs the same classification again. Cover repair removes candidate
descendants using graph ancestry instead of another round of pairwise CVA
comparisons. Models and extents stay fixed during topology
construction. Reused nodes receive new examples afterward, all ancestor
extents are updated, and changed models are refitted once. Unchanged children
with new parents also seed this propagation. If adjacent extents become equal,
their edge is removed because it no longer represents a strict cover.

This remains a learned adaptation: approximate equivalence and refitted CVAs
do not guarantee the closure and transitivity laws assumed by the Boolean
algorithm. Cycle checks protect the graph. Empty intersections do not create
synthetic union nodes, so disjoint singleton inputs can remain disconnected.

The executable regression is `cvafca.LatticeFastAddIntentTest`. It checks
membership across different labels, empty intersections, full CVA intersection
refitting, mutual coverage thresholds, all 26 nonempty intersections of a cube's
six faces and their covers, recursion beyond four levels, twelve child
intersections, frontier duplicate discovery, overlapping CVA/average-mode
insertions at three coverage thresholds, and checkpoint continuation.
`cvafca.LatticeClosureSearchTest` checks upward/downward traversal, full predicate
verification after probing, serial/parallel model and topology agreement, execution
inside a fork-join worker, and exact matches outside tiny search limits.
