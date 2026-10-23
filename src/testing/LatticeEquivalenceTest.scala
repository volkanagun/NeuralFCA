package testing

import cvafca.{CVA, CvaConfig, Instance, Lattice}

/** Compare full mutual CVA classification with exhaustive extent scoring. */
object LatticeEquivalenceTest:
  def main(args: Array[String]): Unit =
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    for noCVA <- Seq(false, true) do
      val lattice = new Lattice(new CVA(2, CvaConfig(0.2, 1.0, noCVA)))
      val samples = List(0f, 0.1f, 1f, 10f).zipWithIndex.map { (value, index) =>
        Instance(s"sample$index", numeric.BFloatTensors(Seq(value, 0f)))
      }
      val targets = samples.map(sample => List(sample)) ++ List(samples, samples.take(2), samples.drop(2))
      val nodes = targets.map(members => new Lattice(lattice.cva).fastAddIntent(members))
      for a <- nodes; b <- nodes do
        val expected = a.instancesIterator.forall(b.classifies) && b.instancesIterator.forall(a.classifies)
        assert(lattice.equivalent(a, b) == expected)
    println("Full mutual CVA equivalence checks passed")
