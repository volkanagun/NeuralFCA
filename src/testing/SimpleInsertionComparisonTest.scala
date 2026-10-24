package simple

/** Compare both insertions on the first 500 records in vectors.txt.
  * Run with simple.SimpleInsertionComparisonTest from the project root.
  */
object SimpleInsertionComparisonTest:
  private def extentSignatures(lattice: Lattice): Set[Set[String]] =
    lattice.getNodes.iterator.map(_.extent.iterator.map(_.id).toSet).toSet

  private def edgeCount(lattice: Lattice): Int =
    lattice.getNodes.iterator.map(_.children.size).sum

  private def assertReciprocalLinks(lattice: Lattice): Unit =
    lattice.getNodes.foreach { node =>
      node.children.foreach { child =>
        assert(child.parents.exists(_ eq node))
        assert(child ne node)
      }
      node.parents.foreach { parent =>
        assert(parent.children.exists(_ eq node))
      }
    }

  def main(args: Array[String]): Unit =
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    val filename = args.headOption.getOrElse("resources/embeddings/vectors.txt")
    val count = sys.props.get("simple.comparison.count").map(_.toInt).getOrElse(500)
    require(count > 0, "Comparison count must be positive")
    val input = new cvafca.CVADataset(size = 32, start = 0, limit = count)
      .read(filename, torch.Device.CPU)
      .flatMap(_.iterator)
      .map(embedding => new Instance(embedding.symbol, embedding.vector))
      .toVector
    require(input.size == count, s"Expected $count embeddings, read ${input.size}")
    val dimension = input.head.attributes.numel.toInt
    val fast = new Lattice(dim = dimension)
    val beam = new Lattice(dim = dimension)
    var fastNanos = 0L
    var beamNanos = 0L

    input.zipWithIndex.foreach { (instance, index) =>
      val fastStarted = System.nanoTime()
      val fastConcept = fast.addInstanceWithFastAddIntent(instance)
      fastNanos += System.nanoTime() - fastStarted

      val beamStarted = System.nanoTime()
      val beamConcept = beam.beamInsert(instance)
      beamNanos += System.nanoTime() - beamStarted

      assert(fastConcept.extent.exists(_.id == instance.id))
      assert(beamConcept.extent.exists(_.id == instance.id))
      if (index + 1) % 5 == 0 || index + 1 == input.size then
        println(s"Compared ${index + 1}/$count embeddings")
    }

    assert(fast.getInstances.map(_.id) == input.map(_.id))
    assert(beam.getInstances.map(_.id) == input.map(_.id))
    assertReciprocalLinks(fast)
    assertReciprocalLinks(beam)
    assert(beam.getNodes.size == input.map(_.id).distinct.size + 1)
    val commonExtents = extentSignatures(fast).intersect(extentSignatures(beam)).size
    println(
      f"FastAddIntent: ${fast.getNodes.size} concepts, ${edgeCount(fast)} links, " +
        f"${fastNanos / 1000000.0}%.2f ms"
    )
    println(
      f"Beam insert:   ${beam.getNodes.size} concepts, ${edgeCount(beam)} links, " +
        f"${beamNanos / 1000000.0}%.2f ms"
    )
    println(s"Shared symbolic extents: $commonExtents")
