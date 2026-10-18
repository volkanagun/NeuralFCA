package testing

import cvafca.{CVA, CvaConfig, Instance, Lattice, LatticeConfig, Node}
import numeric.BFloatTensors.*
import scala.collection.mutable

/** Run with testing.LatticeFastInsertionTest. */
object LatticeFastInsertionTest:
  def main(args: Array[String]): Unit =
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    for noCVA <- Seq(true, false) do
      val lattice = new Lattice(new CVA(2, CvaConfig(0.2, 0.2, noCVA)))
      val first = Instance("first", numeric.BFloatTensors(Seq(1f, 2f)))
      val node = lattice.addIntentFast(first)
      val reused = lattice.addIntentFast(Instance("similar", numeric.BFloatTensors(Seq(1f, 2f))))
      assert(reused.id == node.id && lattice.size == 1)
      assert(reused.symbols == Set("first", "similar"))
      val size = reused.size
      assert(lattice.addIntentFast(first).id == node.id)
      assert(reused.size == size && lattice.size == 1)
      val distinct = lattice.addIntentFast(Instance("far", numeric.BFloatTensors(Seq(100f, 100f))))
      assert(distinct.id != node.id && lattice.size == 2)
      assert(reused.commonVector.floatValues.sameElements(Array(1f, 2f)))
      assert(lattice.nodes.forall(n => !n.parents.contains(n.id) && !n.children.contains(n.id)))
      val near = Instance("near", numeric.BFloatTensors(Seq(1.125f, 2f)))
      assert(lattice.addIntentFast(near).id == node.id)
      assert(reused.symbols == Set("first", "similar", "near"))
      assert(reused.instances.forall(reused.classifies))
      if noCVA then
        assert(math.abs(reused.commonVector.floatValues.head - (3.125f / 3f)) < 0.01)

    // Matching nodes outside the bounded bottom sample must still be reused.
    val global = new Lattice(new CVA(2, CvaConfig(0.01, 0.01, true)),
      LatticeConfig(fastBudgetReferenceNodes = 1, fastMinimumBudgetScale = 0.125))
    val isolated = (0 until 20).map { index =>
      global.addIntentFast(Instance(s"isolated$index", numeric.BFloatTensors(Seq(index.toFloat, 0f))))
    }
    val outside = global.addIntentFast(Instance("outside", numeric.BFloatTensors(Seq(19f, 0f))))
    assert(global.size == 20 && outside.id == isolated.last.id)
    assert(outside.symbols == Set("isolated19", "outside"))

    // Interleaved insertions with the same node IDs and symbols must use fresh
    // scores in both lattices, including when they share a CVA configuration.
    for indexed <- Seq(false, true); noCVA <- Seq(false, true) do
      val cva = new CVA(2, CvaConfig(0.01, 0.01, noCVA))
      val left = new Lattice(cva)
      val right = new Lattice(cva)
      def insert(lattice: Lattice, sample: Instance): Node =
        if indexed then lattice.addIntentFastIndexed(sample) else lattice.addIntentFast(sample)
      for index <- 0 until 128 do
        val value = (index % 2 * 10).toFloat
        insert(left, Instance(s"sample$index", numeric.BFloatTensors(Seq(value, 0f))))
        insert(right, Instance(s"sample$index", numeric.BFloatTensors(Seq(value + 100f, 0f))))
      assert(left.size == 2 && right.size == 2)
      assert(left.nodes.forall(_.size == 64) && right.nodes.forall(_.size == 64))
      assert(left.node(0).commonVector.floatValues.sameElements(Array(0f, 0f)))
      assert(right.node(0).commonVector.floatValues.sameElements(Array(100f, 0f)))
      insert(left, Instance("left-only", numeric.BFloatTensors(Seq(0f, 0f))))
      assert(left.node(0).size == 65 && right.node(0).size == 64)
      assert(!right.nodes.exists(_.containsSymbol("left-only")))

    // Early coverage exits must agree with exhaustive scoring at fractional thresholds.
    for threshold <- Seq(0.25, 0.5, 0.75, 1.0); mutual <- Seq(false, true) do
      val lattice = new Lattice(new CVA(2, CvaConfig(0.2, 0.2, true)),
        LatticeConfig(threshold, mutual))
      val samples = List(0f, 0.1f, 1f, 10f).zipWithIndex.map { (value, index) =>
        Instance(s"sample$index", numeric.BFloatTensors(Seq(value, 0f)))
      }
      val targets = samples.map(sample => List(sample)) ++ List(samples, samples.take(2), samples.drop(2))
      val cache = mutable.HashMap.empty[Set[String], Node]
      val nodes = targets.map(lattice.targetNode(cache, _))
      for a <- nodes; b <- nodes do
        val expected = lattice.coverage(a, b.instances) >= threshold &&
          lattice.coverage(b, a.instances) >= threshold &&
          (!mutual || (a.classifies(b.commonVector) && b.classifies(a.commonVector)))
        assert(lattice.equivalent(a, b) == expected)
    println("Fast insertion and coverage checks passed")
