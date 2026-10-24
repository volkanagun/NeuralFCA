package simple

import java.io.{ByteArrayOutputStream, PrintStream}
import java.nio.file.Files
import java.nio.file.StandardOpenOption

/** Run with simple.SimpleBeamInsertTest. */
object SimpleBeamInsertTest:
  private def sample(id: String, x: Float): Instance =
    new Instance(id, numeric.BFloatTensors(Seq(x, 0f)))

  def main(args: Array[String]): Unit =
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    val lattice = new Lattice(dim = 2)
    val a = lattice.beamInsert(sample("a", 0f), beamWidth = 2, maxVisits = 4, maxParents = 1)
    lattice.beamInsert(sample("b", 20f), beamWidth = 2, maxVisits = 4, maxParents = 1)
    val c = lattice.beamInsert(sample("c", 0.1f), beamWidth = 2, maxVisits = 4, maxParents = 1)
    val common = c.parents.head
    assert(lattice.getNodes.exists(_ eq common))
    assert(common.id != a.id && common.id != c.id)
    assert(common.children.contains(a) && common.children.contains(c))
    assert(common.extent.map(_.id) == Set("a", "c"))
    assert(a.extent.map(_.id) == Set("a"))
    assert(c.children.contains(lattice.bottom))
    assert(lattice.getInstances.size == 3)
    assert(lattice.getNodes.size == 5)
    assert(lattice.beamInsert(sample("a", 40f), beamWidth = 1, maxVisits = 1) eq a)
    assert(lattice.getInstances.size == 3 && lattice.getNodes.size == 5)
    lattice.beamInsert(sample("alias", 0f), beamWidth = 2, maxVisits = 4, maxParents = 1)
    assert(lattice.getInstances.size == 4 && lattice.getNodes.size == 7)
    lattice.beamInsert(sample("far", 50f), beamWidth = 1, maxVisits = 1)
    assert(lattice.getInstances.size == 5 && lattice.getNodes.size == 8)
    lattice.getNodes.foreach { upper =>
      upper.children.foreach { lower =>
        assert(lattice.getNodes.exists(_ eq lower))
        assert(lower.parents.exists(_ eq upper))
        assert(lower.extent.map(_.id).subsetOf(upper.extent.map(_.id)))
      }
    }

    val multiParent = new Lattice(dim = 2)
    val left = multiParent.beamInsert(sample("left", 0f), beamWidth = 4, maxVisits = 8)
    val right = multiParent.beamInsert(sample("right", 2.2f), beamWidth = 4, maxVisits = 8)
    val middle = sample("middle", 1.4f)
    assert(right.intent.distance(middle.attributes) < left.intent.distance(middle.attributes))
    val inserted = multiParent.beamInsert(middle, beamWidth = 4, maxVisits = 8, maxParents = 2)
    assert(inserted.parents.size == 2)
    assert(inserted.parents.map(_.extent.map(_.id)).toSet ==
      Set(Set("left", "middle"), Set("right", "middle")))
    assert(inserted.parents.forall(multiParent.getNodes.contains))
    assert(left.parents.exists(_.children.contains(inserted)))
    assert(right.parents.exists(_.children.contains(inserted)))

    val capped = new Lattice(dim = 2)
    capped.beamInsert(sample("left", 0f), beamWidth = 4, maxVisits = 8)
    capped.beamInsert(sample("right", 2.2f), beamWidth = 4, maxVisits = 8)
    val cappedObject = capped.beamInsert(sample("middle", 1.4f),
      beamWidth = 4, maxVisits = 8, maxParents = 1)
    assert(cappedObject.parents.size == 1)
    assert(cappedObject.parents.head.extent.map(_.id) == Set("right", "middle"))

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
    val checkpointDirectory = Files.createTempDirectory("simple-lattice-")
    val checkpoint = checkpointDirectory.resolve("checkpoint.bin")
    try
      Files.writeString(input, "one 0 0\ntwo 20 0\nthree 0.1 0\n")
      val output = new ByteArrayOutputStream()
      val trained = Console.withOut(new PrintStream(output)) {
        new Main().train(input.toString, limit = 3, dim = 2,
          beamWidth = 2, maxVisits = 4, checkpointFilename = checkpoint.toString,
          maxParents = 1)
      }
      assert(trained.getInstances.size == 3)
      assert(trained.getNodes.size == 5)
      assert(output.toString("UTF-8").linesIterator.count(_.startsWith("Processed:")) == 1)
      assert(Files.size(checkpoint) > 0)

      val loadOutput = new ByteArrayOutputStream()
      val restored = Console.withOut(new PrintStream(loadOutput)) {
        new Lattice(dim = 2, ratio = trained.ratio, threshold = trained.threshold)
          .load(checkpoint.toString)
      }
      val loadingLog = loadOutput.toString("UTF-8")
      assert(loadingLog.contains("Loading lattice instances: 3/3 (100%)"))
      assert(loadingLog.contains("Loading lattice concepts: 5/5 (100%)"))
      assert(loadingLog.contains("Loading lattice linking concepts: 5/5 (100%)"))
      assert(loadingLog.contains("Loaded lattice checkpoint: 3 instances, 5 concepts"))
      assert(restored.getInstances.map(_.id) == trained.getInstances.map(_.id))
      assert(restored.getNodes.size == trained.getNodes.size)
      trained.getNodes.zip(restored.getNodes).foreach { (before, after) =>
        assert(before.id == after.id)
        assert(before.extent.map(_.id) == after.extent.map(_.id))
        assert(before.children.map(_.id).toSet == after.children.map(_.id).toSet)
        assert(before.parents.map(_.id).toSet == after.parents.map(_.id).toSet)
        assert(numeric.BFloatTensors.toArray(before.intent.commonVector)
          .sameElements(numeric.BFloatTensors.toArray(after.intent.commonVector)))
        assert(numeric.BFloatTensors.toArray(before.intent.projection)
          .sameElements(numeric.BFloatTensors.toArray(after.intent.projection)))
      }
      assert(restored.beamInsert(sample("one", 40f)).id == 1)
      assert(restored.getNodes.size == 5)

      Files.writeString(input, "four 0.2 0\n", StandardOpenOption.APPEND)
      val resumed = Console.withOut(new PrintStream(new ByteArrayOutputStream())) {
        new Main().train(input.toString, limit = 4, dim = 2,
          beamWidth = 2, maxVisits = 4, checkpointFilename = checkpoint.toString,
          maxParents = 1)
      }
      assert(resumed.getInstances.size == 4 && resumed.getNodes.size == 7)
      trained.beamInsert(sample("four", 0.2f), beamWidth = 2, maxVisits = 4,
        maxParents = 1)
      trained.getNodes.zip(resumed.getNodes).foreach { (expected, actual) =>
        assert(expected.extent.map(_.id) == actual.extent.map(_.id))
        assert(expected.children.map(_.id).toSet == actual.children.map(_.id).toSet)
        assert(numeric.BFloatTensors.toArray(expected.intent.commonVector)
          .sameElements(numeric.BFloatTensors.toArray(actual.intent.commonVector)))
      }
      assert(new Lattice(dim = 2, ratio = trained.ratio, threshold = trained.threshold)
        .load(checkpoint.toString).getInstances.size == 4)
    finally
      Files.deleteIfExists(input)
      Files.deleteIfExists(checkpoint)
      Files.deleteIfExists(checkpointDirectory)

    println("Simple beam insertion and training passed.")
