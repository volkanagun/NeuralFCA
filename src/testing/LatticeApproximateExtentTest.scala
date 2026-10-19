package cvafca

import numeric.BFloatTensors.*
import scala.collection.mutable

/** Run with cvafca.LatticeApproximateExtentTest. */
object LatticeApproximateExtentTest:
  private def sample(symbol: String, x: Float, y: Float = 0f): Instance =
    Instance(symbol, numeric.BFloatTensors(Seq(x, y)))

  private def sameModel(node: Node, cva: CVA): Unit =
    val expected = cva.fit(node.instances.map(_.vector))
    assert(node.commonVector.floatValues.sameElements(expected.commonVector.floatValues))
    assert(node.projection.floatValues.sameElements(expected.projection.floatValues))
    assert(node.classificationRadius == expected.radius)

  def main(args: Array[String]): Unit =
    org.bytedeco.pytorch.global.torch.set_num_threads(1)

    for indexed <- Seq(false, true); noCVA <- Seq(false, true); cacheSize <- Seq(0, 2) do
      val classifier = new CVA(2, CvaConfig(0.05, 1.0, noCVA))
      val left = sample("left", 0f)
      val legacy = Vector(sample("legacy-left", 0f), sample("legacy-right", 4f))
      val extra = sample("extra", 0f, 4f)
      val fixture = new Lattice(classifier, LatticeConfig(equalityCoverage = 1.0,
        requireMutualCommonClassification = true, fastBasisCacheNodes = cacheSize)):
        val leaf = new Node(1000L, List(left), classifier)
        val line = new Node(1001L, legacy, classifier, do_avg = false)
        val parent = new Node(1002L, legacy :+ extra, classifier, do_avg = false)
        nodeMap.put(leaf.id, leaf)
        nodeMap.put(line.id, line)
        nodeMap.put(parent.id, parent)
        parent.children += line.id
        line.parents += parent.id

      val lattice = fixture.cloneLattice()
      val incoming = sample("incoming", 4f)
      val inserted = if indexed then lattice.addIntentFastIndexed(incoming)
        else lattice.addIntentFast(incoming)
      val reused = lattice.node(1001L)
      val parent = lattice.node(1002L)
      assert(lattice.size == 4, "An equivalent existing intersection must prevent a fifth node")
      assert(reused.symbols == Set("legacy-left", "legacy-right", "left", "incoming"))
      assert(reused.children.toSet == Set(1000L, inserted.id))
      assert(parent.symbols == reused.symbols + "extra",
        "Missing intersection members must propagate to every ancestor")
      assert(parent.children.toSet == Set(reused.id))
      lattice.validate()
      sameModel(reused, classifier)
      sameModel(parent, classifier)
      val repeated = if indexed then lattice.addIntentFastIndexed(incoming)
        else lattice.addIntentFast(incoming)
      assert(repeated.id == inserted.id && lattice.size == 4)
      lattice.validate()

      // Growing the first equivalent candidate would erase its distinction
      // from its parent. Reuse that parent instead and retain strict edges.
      val guardedFixture = new Lattice(classifier, lattice.config):
        val leaf = new Node(1000L, List(left), classifier)
        val line = new Node(1001L, legacy, classifier, do_avg = false)
        val parent = new Node(1002L, legacy :+ left, classifier, do_avg = false)
        nodeMap.put(leaf.id, leaf)
        nodeMap.put(line.id, line)
        nodeMap.put(parent.id, parent)
        parent.children += line.id
        line.parents += parent.id
      val guarded = guardedFixture.cloneLattice()
      if indexed then guarded.addIntentFastIndexed(incoming) else guarded.addIntentFast(incoming)
      assert(!guarded.node(1001L).containsSymbol(left.symbol),
        "An approximate merge must not collapse a strict parent edge")
      assert(guarded.node(1002L).containsSymbol(incoming.symbol))
      assert(guarded.node(1000L).parents.toSet == Set(1002L))
      assert(guarded.size == 4)
      guarded.validate()

    // Approximate overlap must select the same preferred candidate even when
    // indexes are enabled and an earlier, disjoint learned intent also matches.
    for indexed <- Seq(false, true) do
      val classifier = new CVA(2, CvaConfig(0.01, 1.0))
      val first = sample("disjoint", 0f)
      val near = (0 until 4).map(index => sample(s"overlap-$index", 0f))
      val fixture = new Lattice(classifier, LatticeConfig(equalityCoverage = 0.8)):
        nodeMap.put(1000L, new Node(1000L, List(first), classifier))
        nodeMap.put(1001L, new Node(1001L, near, classifier))
      val lattice = fixture.cloneLattice()
      if indexed then lattice.addIntentFastIndexed(first)
      val incoming = sample("new-overlap", 0f)
      val target = lattice.targetNode(mutable.HashMap.empty, near.toList :+ incoming)
      val reused = lattice.materialize(mutable.LinkedHashSet.empty,
        mutable.HashMap.empty, mutable.HashMap.empty, mutable.HashMap.empty,
        mutable.HashMap.empty, target, incoming)
      assert(reused.id == 1001L && lattice.size == 2,
        "Indexed lookup must retain approximate overlap preference")
      lattice.validate()

    // High symbolic overlap must not bypass learned equivalence. Exercise the
    // same fallback with indexes enabled and disabled.
    for indexed <- Seq(false, true) do
      val lattice = new Lattice(new CVA(2, CvaConfig(0.01, 0.01, noCVA = true)),
        LatticeConfig(equalityCoverage = 0.75))
      val near = (0 until 4).map(index => sample(s"near-$index", 0f))
      near.foreach { member =>
        if indexed then lattice.addIntentFastIndexed(member) else lattice.addIntentFast(member)
      }
      val existing = lattice.nodes.head
      val members = existing.instances.toList :+ sample("far", 10f)
      val target = lattice.targetNode(mutable.HashMap.empty, members)
      assert(!lattice.equivalent(existing, target))
      val materialized = lattice.materialize(mutable.LinkedHashSet.empty,
        mutable.HashMap.empty, mutable.HashMap.empty, mutable.HashMap.empty,
        mutable.HashMap.empty, target, members.last)
      assert(materialized.id != existing.id,
        "Approximate symbolic overlap alone must not reuse an incompatible intent")
      assert(lattice.size == 2)
      lattice.validate()

    println("Approximate intersections reuse candidates, propagate extents and preserve strict edges")
