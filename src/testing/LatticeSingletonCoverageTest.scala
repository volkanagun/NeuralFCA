package cvafca

import numeric.BFloatTensors.*
import org.bytedeco.javacpp.Pointer
import scala.collection.mutable

/** Compare batched duplicate checks against exhaustive scalar coverage. */
object LatticeSingletonCoverageTest:
  def main(args: Array[String]): Unit =
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    for noCVA <- Seq(false, true); threshold <- Seq(0.25, 0.5, 0.75, 1.0) do
      val lattice = new Lattice(new CVA(17, CvaConfig(0.2, 0.2, noCVA)),
        LatticeConfig(equalityCoverage = threshold))
      val cache = mutable.HashMap.empty[Set[String], Node]
      val singleton = lattice.targetNode(cache,
        List(Instance("query", numeric.BFloatTensors(Seq.fill(17)(1f)))))
      val samples = (0 until 130).map { index =>
        val offset = Seq(0f, 0.015625f, 0.0625f, 1f)(index % 4)
        Instance(s"sample$index", numeric.BFloatTensors(Seq.fill(17)(1f + offset)))
      }
      for size <- Seq(15, 16, 63, 64, 65, 130) do
        val extent = lattice.targetNode(cache, samples.take(size).toList)
        val expected = lattice.coverage(singleton, extent.instances) >= threshold
        val before = Pointer.totalCount()
        assert(lattice.singletonCoversEnough(singleton, extent) == expected,
          s"Batched coverage changed (noCVA=$noCVA, threshold=$threshold, size=$size)")
        assert(Pointer.totalCount() <= before, "Batched coverage retained native temporaries")
    println("Batched singleton coverage checks passed")
