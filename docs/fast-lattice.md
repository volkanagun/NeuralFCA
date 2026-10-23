# FastLattice

`FastLattice` is a separate CVA adaptation of
[FastAddIntent](https://doi.org/10.1016/j.eswa.2015.01.044). It uses the same
`Node`, storage, validation, clone, and checkpoint format as `Lattice`, but has
its own insertion algorithm.

```scala
val lattice = new FastLattice(
  new CVA(300, CvaConfig(classificationThreshold = 0.8)),
  FastLatticeConfig(searchVisits = 128, searchFrontier = 48)
)
lattice.fastAddIntent(instance)
```

Insertion searches upward from childless concepts for canonical generators,
creates intersections with their existing children, and repairs direct cover
links. CVA classification defines membership. An intersection retains samples
from both operands that both CVAs classify, deduplicates them by symbol, and
fits a fresh common vector, projection, and radius from that complete extent.
Empty intersections remain virtual. A node is refitted only when its extent
gains a new symbol; changing graph links does not refit its CVA.

FastLattice identifies an extent by its set of instance symbols. Inserting an
extent already present reuses its stored node even if a later instance with the
same symbol has a different vector. The earlier stored sample and CVA model
remain authoritative. If propagation makes two extents equal, the node with
the lower ID survives; valid parent and child links are redirected to it.
Loaded checkpoints are normalized to the same rule.

`FastLatticeConfig` bounds model visits, frontier and beam width, child cover
candidates, recursive intersections, repair rounds, propagation checks, and
cycle-search visits. Search probes can run in parallel when at least 16 models
and two million estimated projection elements are ready; graph edits and extent
updates stay ordered.

| Parameter | Default | Limit |
| --- | ---: | --- |
| `searchVisits` | 2,048 | Models scored per generator search |
| `searchFrontier` | 128 | Bottom seeds or parents per search layer |
| `searchBeamWidth` | 48 | Rejected branches expanded per layer |
| `maxChildCandidates` | 128 | Existing covers examined per placement |
| `maxIntersections` | 64 | New intersection nodes per insertion |
| `maxIntersectionDepth` | 24 | Recursive intersection depth |
| `repairPasses` | 4 | Model and cover repair rounds |
| `propagationChecks` | 100,000 | Membership checks per extent propagation |
| `reachabilityVisits` | 4,096 | Nodes checked before skipping a possible link |

These bounds can omit concepts or memberships. Learned CVA inclusion is not necessarily transitive,
so the result is an approximate concept lattice. Reachability checks skip a
link when their visit budget is exhausted, which keeps the graph acyclic.
