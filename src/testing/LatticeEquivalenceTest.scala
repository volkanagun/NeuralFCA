package testing

import cvafca.{CVA, CvaConfig, Instance, Lattice, LatticeConfig}

/** Run with testing.LatticeEquivalenceTest. */
object LatticeEquivalenceTest:
  def main(args: Array[String]): Unit =
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    // Early coverage exits must agree with exhaustive scoring at fractional thresholds.
    for threshold <- Seq(0.25, 0.5, 0.75, 1.0); mutual <- Seq(false, true) do
      val lattice = new Lattice(new CVA(2, CvaConfig(0.2, 0.2, true)),
        LatticeConfig(threshold, mutual))
      val samples = List(0f, 0.1f, 1f, 10f).zipWithIndex.map { (value, index) =>
        Instance(s"sample$index", numeric.BFloatTensors(Seq(value, 0f)))
      }
      val targets = samples.map(sample => List(sample)) ++ List(samples, samples.take(2), samples.drop(2))
      val nodes = targets.map(members => new Lattice(lattice.cva, lattice.config).fastAddIntent(members))
      for a <- nodes; b <- nodes do
        val expected = lattice.coverage(a, b.instances) >= threshold &&
          lattice.coverage(b, a.instances) >= threshold &&
          (!mutual || (a.classifies(b.commonVector) && b.classifies(a.commonVector)))
        assert(lattice.equivalent(a, b) == expected)
    println("Mutual equivalence and coverage checks passed")
