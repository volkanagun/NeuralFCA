package simple

/** Run with simple.SimpleInsertionComparisonTest. */
object SimpleInsertionComparisonTest:
  private def sample(id: String, x: Float, y: Float): Instance =
    new Instance(id, numeric.BFloatTensors(Seq(x, y)))

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
    val input = Vector(
      sample("origin", 0f, 0f),
      sample("distant", 20f, 0f),
      sample("near-origin", 0.2f, 0f),
      sample("near-distant", 20.2f, 0f)
    )
    val fast = new Lattice(dim = 2)
    val beam = new Lattice(dim = 2)
    var fastNanos = 0L
    var beamNanos = 0L

    input.foreach { instance =>
      val fastStarted = System.nanoTime()
      val fastConcept = fast.addInstanceWithFastAddIntent(instance)
      fastNanos += System.nanoTime() - fastStarted

      val beamStarted = System.nanoTime()
      val beamConcept = beam.beamInsert(instance, beamWidth = 4, maxVisits = 16)
      beamNanos += System.nanoTime() - beamStarted

      assert(fastConcept.extent.exists(_.id == instance.id))
      assert(beamConcept.extent.exists(_.id == instance.id))
      assert(fast.getInstances.map(_.id) == beam.getInstances.map(_.id))
      assertReciprocalLinks(fast)
      assertReciprocalLinks(beam)
    }

    assert(fast.getInstances.map(_.id) == input.map(_.id))
    assert(beam.getInstances.map(_.id) == input.map(_.id))
    assert(beam.getNodes.size == input.size + 1) // One node per object plus bottom.
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
