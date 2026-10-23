# FastAddIntent with CVA membership

Use `lattice.fastAddIntent(instance)` for one symbol, or
`lattice.fastAddIntent(instances)` to insert an intent fitted from several
embeddings. The traversal adapts canonical-generator search, recursive
intersections, and cover repair from
[FastAddIntent (2015)](https://doi.org/10.1016/j.eswa.2015.01.044).
Parents represent more general CVAs and children more specific CVAs.

`Instance.symbol` is the word; `Instance.vector` is its embedding. Each node
stores its instances, a common vector, and a CVA projection fitted from those
embeddings. All insertion, generator search, intersection, cover repair, and
persistence helpers are methods of `Lattice`; none is defined inside another
method. Temporary caches belong to a single insertion and are discarded afterward.

The learned operations are:

* **Membership:** `node.classifies(instance)`, using distance < the model's
  classification radius. Full membership updates inspect discovered generators,
  branches retained by the beam, and affected graph neighbors. A classified sample is appended only if its symbol is missing.
  An already recorded sample leaves its stored embedding and CVA unchanged.
* **Equivalence:** each CVA must classify every sample of the other node's
  extent. IDs, matching symbol sets, partial coverage, and common-vector
  similarity do not establish equality.
* **Strict inclusion:** the parent classifies the child's complete extent,
  while the child does not classify the parent's complete extent.
* **Intersection:** inspect samples from both operands and keep those classified
  by both CVAs. Deduplicate retained samples by symbol and rebuild the complete
  common vector, projection, and radius with `cva.fit`. Empty intersections stay
  virtual and never call `cva.fit` with an empty collection.

`node.distance(instance.vector)` returns `r / (1 + r)`, where
`r = ||P * instance.vector - commonVector||₂`. The existing `noCVA` option uses
the mean embedding and identity projection. The model radius starts at
`CvaConfig.classificationThreshold`. If the scaled training residual exceeds
that threshold, the radius becomes the next representable value above that
residual bound. This preserves acceptance of training samples at scale 1 while
keeping the configured threshold strict. A zero radius accepts no samples.
Both single-instance and batch extent updates refit the common vector and
projection; insertion batches its updates between topology passes.

Symbol names identify recorded examples only. Nodes with identical symbol sets
can retain different embeddings and CVAs. An incremental extent index retrieves
same-symbol-set candidates before any node is created, including recursive
intersections. Reuse still requires full mutual CVA classification. Thus a
matching recorded extent outside the beam is checked without a lattice scan.
Extent additions update only the changed node's index entry; loading and cloning
register their nodes as they are restored. There is no global coalescing pass.
Node IDs only address storage, serialized edges, and diagnostics. Object
references key per-pass caches and prevent structural self-links and cycles;
semantic equivalence always evaluates the two classifiers.

Closure and intersection placement use a bounded bottom-up graph beam search.
Searches start at childless nodes and follow parent edges; a recursive hint can
add a local starting concept. A branch stops expanding when its current node
classifies the complete requested extent. This canonical-generator pruning is
approximate for non-transitive CVA predicates. Discovered broader classifiers
are filtered out when a more specific classifier is available. The worst
distance among the first, middle, and last extent embeddings ranks candidates.
A rejected probe disproves complete inclusion immediately. Candidates that pass
the probes still undergo full inclusion and mutual-equivalence verification.
Among rejected candidates, only the closest beam receives full partial-membership
updates and expands upward. Beam width limits expansion, with extent size and
stable storage order breaking ties.
Parent edges are sampled in rounds across the retained branches so a node with
many parents cannot monopolize the next layer. At most `closureSearchFrontier`
distinct parents enter the next layer. Raw edge enumeration is bounded by
`closureSearchFrontier * retainedBeamSize`; duplicate and already visited edges
consume this allowance. This leaves room to pass shared parents without an
unbounded neighbor scan.
Storage order is maintained incrementally, without traversing every node per insertion.
Membership updates apply to visited candidates and propagate through affected
ancestors. Disconnected classifiers outside the beam are not globally evaluated.

| Setting | Default | Purpose |
| --- | ---: | --- |
| `closureSearchVisits` | 256 | Maximum candidate models scored by one closure search. |
| `closureSearchFrontier` | 128 | Maximum bottom seeds or distinct parent candidates per layer. |
| `closureSearchBeamWidth` | 8 | Closest rejected candidates retained for full membership updates and expansion. |
| `closureSearchParallel` | true | Enable parallel probe scoring when worthwhile. |

At least 32 nodes and two million estimated **uncached** projection/vector elements
are required before scoring runs on workers. Searches already on a fork-join worker
remain sequential to avoid nested scheduling while holding the lattice monitor.
Workers only read fixed models; the caller publishes mutable cache entries.
The distance cache retains at most 65,536 scalar scores per topology pass and
serves both membership and probe ranking. Cover changes invalidate closure
results but preserve these distances, avoiding repeated native projections.
Refitting models clears the distance cache before the next pass.
The initial layer uses at most half the visit budget (at least one node), leaving
room to ascend. These budgets bound each generator search; processing the
affected intersections and checking graph reachability still depend on the
affected lattice size.

The previous limits were 1024 visits, 512 frontier candidates, and a beam of 16.
Increase the limits explicitly when broader coverage matters: the bounded
initial frontier samples the first childless nodes in storage order and can miss
equivalent CVAs outside that frontier. The exact-extent index still checks matching
recorded extents regardless of the search budget.

On synthetic 300-dimensional CPU fixtures, median insertion times over five
measured runs (after two warmups, with one native thread) were:

| Fixture | Previous implementation and limits | Tuned implementation and limits |
| --- | ---: | ---: |
| 512 singleton concepts | 596 ms | 127 ms |
| 256 branches, each with a leaf | 475 ms | 227 ms |
| 256 line concepts, 32-instance insertion | 2598 ms | 1300 ms |

All compared runs produced identical extents and graph links on these fixtures.
These measurements cover queries within both frontiers; they do not measure
corpus-wide recall or guarantee the same speedup on other lattices.

`CVAWordTrainer.train` inserts words with `fastAddIntent` and accepts the search
configuration directly:

```scala
val trainer = new CVAWordTrainer(start = 0, limit = 10000, dim = 300)
val lattice = trainer.train(
  "resources/embeddings/vectors.txt",
  minDistance = 0.8,
  latticeConfig = LatticeConfig(
    closureSearchVisits = 256,
    closureSearchFrontier = 64,
    closureSearchBeamWidth = 8
  )
)
```

The trainer resumes and saves its configured checkpoint as before.

Topology and model updates are staged. Each topology pass uses fixed extents and
CVAs. Missing classified samples are then appended, accepted child samples
propagate to ancestors, and only changed CVAs are refitted. Changed nodes are
placed again: invalid incident links are removed, necessary intersection nodes
are inserted, and bypassed cover edges are repaired. Repair stops when no extent
grows or `repairPassLimit` rounds have run (32 by default). The limit bounds
work when learned CVA membership keeps changing after refits; reaching it leaves
the latest refitted graph as an approximation. Extent propagation also stops
after `propagationChecksPerPass` unseen instance-parent checks (100,000 by
default); unprocessed propagation is skipped, so increase this limit when fuller
extents matter more than insertion speed. Independent parents are scored in
parallel when there is enough uncached work; `propagationParallel = false`
keeps this scoring serial. Distance measurements for nodes whose models did not
change are reused across repair rounds. There is no global
equivalence merge. If every visited accepting classifier already records the
input, insertion returns a matching recorded node without changing models,
topology, or persistent IDs.

This is a learned adaptation rather than an exact Boolean concept lattice.
Bounded closure discovery can miss concepts, and CVA predicates need not be
transitive. Graph reachability prevents cycles; validation checks adjacency,
frontier consistency, and strict learned order rather than symbolic uniqueness.

Run `cvafca.LatticeMembershipInsertionTest` for duplicate extents outside tiny
search budgets, incremental index updates, unchanged repeated insertions,
intermediate intersections, topology repair after refitting, and independence
from IDs and symbol equality.
`cvafca.LatticeFastAddIntentTest` covers strict threshold boundaries, calibrated
radii, intersections, full mutual equivalence, recursive insertion, and persistence.
`cvafca.LatticeClosureSearchTest` covers bottom-up generator discovery, branch
diversity and shared-parent traversal within small budgets, bounded membership, full verification after
probing, deterministic parallel execution, and insertion from a fork-join worker.
`cvafca.CVAWordTrainerLatticeTest` trains from a temporary embedding file and checks
search configuration, common vectors, checkpoint saving, and resumed insertion.
