package cvafca

import numeric.BFloatTensors.*
import org.bytedeco.javacpp.{Pointer, PointerScope}
import scala.util.{Random, Using}
import scala.collection.mutable
import java.nio.file.Files

/** Compare every insertion's result, topology and model with addIntentFast. */
object LatticeIndexedInsertionTest:
  private def sameModel(a: Node, b: Node): Unit =
    def bits(values: Array[Float]) = values.map(java.lang.Float.floatToIntBits)
    assert(bits(a.commonVector.floatValues).sameElements(bits(b.commonVector.floatValues)), "Common vector changed")
    assert(bits(a.projection.floatValues).sameElements(bits(b.projection.floatValues)), "Projection changed")
    assert(java.lang.Double.doubleToLongBits(a.classificationRadius) ==
      java.lang.Double.doubleToLongBits(b.classificationRadius), "Radius changed")

  private def sameLattice(a: Lattice, b: Lattice): Unit =
    assert(a.nodes.map(_.id) == b.nodes.map(_.id))
    a.nodes.zip(b.nodes).foreach { (left, right) =>
      assert(left.symbols == right.symbols)
      assert(left.parents.toVector == right.parents.toVector)
      assert(left.children.toVector == right.children.toVector)
      sameModel(left, right)
    }

  def main(args: Array[String]): Unit =
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    val random = new Random(947L)
    val samples = (0 until 48).map { index =>
      val cluster = (index % 3).toFloat * 3f
      Instance(s"sample$index", numeric.BFloatTensors((0 until 8).map { dimension =>
        cluster + (if index % 7 == 0 then 0f else random.nextFloat() * 0.0625f)
      }))
    }
    for noCVA <- Seq(false, true); coverage <- Seq(0.75, 0.95, 1.0); mutual <- Seq(false, true) do
      val config = LatticeConfig(equalityCoverage = coverage,
        requireMutualCommonClassification = mutual, fastBasisCacheNodes = 2)
      val reference = new Lattice(new CVA(8, CvaConfig(0.2, 0.2, noCVA)), config)
      val indexed = new Lattice(new CVA(8, CvaConfig(0.2, 0.2, noCVA)), config)
      (samples ++ samples.take(4)).foreach { sample =>
        val expected = reference.addIntentFast(sample)
        val actual = indexed.addIntentFastIndexed(sample)
        assert(expected.id == actual.id)
        sameLattice(reference, indexed)
      }

      // Clone/checkpoint continuation must preserve the complete refitted models.
      val cloned = indexed.cloneLattice()
      val checkpoint = Files.createTempFile("indexed-insertion-", ".bin")
      try
        indexed.save(checkpoint.toString)
        val loaded = new Lattice(new CVA(8, CvaConfig(0.2, 0.2, noCVA)), config).load(checkpoint.toString)
        val next = Instance("continuation", samples.head.vector)
        reference.addIntentFast(next)
        indexed.addIntentFastIndexed(next)
        cloned.addIntentFastIndexed(next)
        loaded.addIntentFastIndexed(next)
        sameLattice(reference, indexed)
        sameLattice(reference, cloned)
        sameLattice(reference, loaded)
      finally Files.deleteIfExists(checkpoint)

    // A growing difference scale invalidates an accepted tiny direction.
    val cva = new CVA(2, CvaConfig(0.2, 0.2))
    val basis = new CvaBasisCache
    val rankSamples = Seq(Seq(0f, 0f), Seq(0.000001f, 0f), Seq(0f, 1000f),
      Seq(0f, 1000f), Seq(0f, 1000f)).map(numeric.BFloatTensors(_))
    val countBefore = Pointer.totalCount()
    try
      for size <- 1 to rankSamples.size do
        Using.resource(new PointerScope()) { _ =>
          val expected = cva.fit_cva(rankSamples.take(size))
          val actual = cva.fit_cached(rankSamples.take(size), basis)
          assert(expected.projection.floatValues.sameElements(actual.projection.floatValues))
          assert(expected.commonVector.floatValues.sameElements(actual.commonVector.floatValues))
          assert(expected.radius == actual.radius)
        }
      val stableCount = Pointer.totalCount()
      for _ <- 0 until 30 do
        Using.resource(new PointerScope()) { _ => cva.fit_cached(rankSamples, basis) }
      assert(Pointer.totalCount() <= stableCount, "Basis cache accumulated native temporaries")
    finally basis.close()
    assert(Pointer.totalCount() <= countBefore, "Basis cache failed to release retained tensors")

    // The singleton bound must retain actual matches, including threshold ties.
    for dimension <- Seq(1, 17, 300); noCVA <- Seq(false, true) do
      val classifier = new CVA(dimension, CvaConfig(noCVA = noCVA))
      val centerTensor = numeric.BFloatTensors((0 until dimension).map(_ => random.nextFloat() - 0.5f))
      val model = classifier.fit_avg(Seq(centerTensor))
      val center = model.commonVector.floatValues
      for index <- 0 until 80 do
        val spread = if index % 2 == 0 then 0.001f else 0.5f
        val queryTensor = numeric.BFloatTensors(center.toSeq.map(value => value + random.nextFloat() * spread))
        val query = queryTensor.floatValues
        val score = classifier.distance(model, queryTensor)
        for radius <- Seq(score, java.lang.Math.nextUp(score), 0.2, 0.5, 1.0) do
          if score <= radius then
            assert(SingletonBounds.mayClassify(center, query, radius), "Bound rejected an actual match")

    // Direct extent mutations must rekey the index and remove former singletons.
    val external = new Lattice(new CVA(2, CvaConfig(0.01, 0.01, true)))
    val first = Instance("external", numeric.BFloatTensors(Seq(0f, 0f)))
    val original = external.addIntentFastIndexed(first)
    original.addInstance(Instance("extra", numeric.BFloatTensors(Seq(10f, 0f))))
    original.refit()
    val added = external.addIntentFastIndexed(first)
    assert(added.id != original.id && added.symbols == Set("external"))

    // Converging extents retain nodeMap's first-match order, not rekey order.
    val converged = new Lattice(new CVA(2, CvaConfig(0.01, 0.01, true)))
    val a = Instance("a", numeric.BFloatTensors(Seq(0f, 0f)))
    val b = Instance("b", numeric.BFloatTensors(Seq(10f, 0f)))
    val firstNode = converged.addIntentFastIndexed(a)
    val secondNode = converged.addIntentFastIndexed(b)
    secondNode.addInstances(List(a))
    firstNode.addInstances(List(b))
    val targetCache = mutable.HashMap.empty[Set[String], Node]
    val target = converged.targetNode(targetCache, List(a, b))
    val reused = converged.materialize(mutable.LinkedHashSet.empty,
      mutable.HashMap.empty, mutable.HashMap.empty, mutable.HashMap.empty,
      targetCache, target, a)
    assert(reused.id == firstNode.id, "Exact-extent index changed canonical-node ordering")
    println("Indexed insertion matches full refits, topology, persistence and native lifetime")
