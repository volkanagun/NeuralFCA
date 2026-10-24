package simple

import java.io.{ByteArrayOutputStream, PrintStream}
import java.nio.file.Files

/** Run with simple.SimpleBeamInsertTest. */
object SimpleBeamInsertTest:
  private def sample(id: String, x: Float): Instance =
    new Instance(id, numeric.BFloatTensors(Seq(x, 0f)))

  def main(args: Array[String]): Unit =
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    val lattice = new Lattice(dim = 2)
    val a = lattice.beamInsert(sample("a", 0f), beamWidth = 2, maxVisits = 4)
    lattice.beamInsert(sample("b", 20f), beamWidth = 2, maxVisits = 4)
    val c = lattice.beamInsert(sample("c", 0.1f), beamWidth = 2, maxVisits = 4)
    assert(a.children.contains(c))
    assert(c.parents.contains(a))
    assert(c.children.contains(lattice.bottom))
    assert(lattice.getInstances.size == 3)
    assert(lattice.getNodes.size == 4)
    assert(lattice.beamInsert(sample("a", 40f), beamWidth = 1, maxVisits = 1) eq a)
    assert(lattice.getInstances.size == 3 && lattice.getNodes.size == 4)
    lattice.beamInsert(sample("alias", 0f), beamWidth = 2, maxVisits = 4)
    assert(lattice.getInstances.size == 4 && lattice.getNodes.size == 5)
    lattice.beamInsert(sample("far", 50f), beamWidth = 1, maxVisits = 1)
    assert(lattice.getInstances.size == 5 && lattice.getNodes.size == 6)

    val defaultLattice = new Lattice(dim = 2)
    defaultLattice.addInstance(sample("first", 0f))
    defaultLattice.addInstance(sample("second", 20f))
    assert(defaultLattice.getInstances.size == 2)
    assert(defaultLattice.getNodes.size == 3)

    val recursive = new Lattice(dim = 2)
    recursive.addInstanceWithFastAddIntent(sample("old-first", 0f))
    recursive.addInstanceWithFastAddIntent(sample("old-second", 20f))
    assert(recursive.getInstances.size == 2)

    val input = Files.createTempFile("simple-beam-", ".txt")
    try
      Files.writeString(input, "one 0 0\ntwo 20 0\nthree 0.1 0\n")
      val output = new ByteArrayOutputStream()
      val trained = Console.withOut(new PrintStream(output)) {
        new Main().train(input.toString, limit = 3, dim = 2,
          beamWidth = 2, maxVisits = 4)
      }
      assert(trained.getInstances.size == 3)
      assert(trained.getNodes.size == 4)
      assert(output.toString("UTF-8").linesIterator.count(_.startsWith("Processed:")) == 3)
    finally Files.deleteIfExists(input)

    println("Simple beam insertion and training passed.")
