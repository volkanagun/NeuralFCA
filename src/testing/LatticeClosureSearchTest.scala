package cvafca

import numeric.BFloatTensors.*
import java.util.concurrent.{Callable, ForkJoinPool, TimeUnit}

/** Run with cvafca.LatticeClosureSearchTest. */
object LatticeClosureSearchTest:
  private def sample(symbol: String, values: Float*): Instance =
    Instance(symbol, numeric.BFloatTensors(values))

  private def branching(parallel: Boolean): Lattice =
    val classifier = new CVA(300, CvaConfig(0.01, 1.0))
    val rows = (0 until 64).map { row =>
      Vector(branchSample(s"row-$row-left", 0f, row * 4f),
        branchSample(s"row-$row-right", 4f, row * 4f))
    }.toVector
    val fixture = new Lattice(classifier, LatticeConfig(
      closureSearchParallel = parallel, propagationParallel = parallel)):
      val root = new Node(1000L, rows.flatten, this.cva, do_avg = false)
      nodeMap.put(root.id, root)
      rows.zipWithIndex.foreach { (members, row) =>
        val line = new Node(1100L + row, members, this.cva, do_avg = false)
        nodeMap.put(line.id, line)
        root.children += line.id
        line.parents += root.id
      }
    fixture.cloneLattice()

  private def branchSample(symbol: String, x: Float, y: Float): Instance =
    Instance(symbol, numeric.BFloatTensors(Vector(x, y) ++ Vector.fill(298)(0f)))

  private def assertSame(left: Lattice, right: Lattice): Unit =
    left.validate()
    right.validate()
    assert(left.nodes.map(n => (n.id, n.symbols, n.parents.toVector, n.children.toVector)) ==
      right.nodes.map(n => (n.id, n.symbols, n.parents.toVector, n.children.toVector)))
    left.nodes.zip(right.nodes).foreach { (a, b) =>
      assert(a.commonVector.floatValues.sameElements(b.commonVector.floatValues))
      assert(a.projection.floatValues.sameElements(b.projection.floatValues))
      assert(a.classificationRadius == b.classificationRadius)
    }

  def main(args: Array[String]): Unit =
    org.bytedeco.pytorch.global.torch.set_num_threads(1)

    // Rank the 64 bottom classifiers by actual sample distance.
    val serial = branching(false)
    val parallel = branching(true)
    val worker = branching(true)
    val query = branchSample("inside-row-57", 2f, 228f)
    val inserted = parallel.fastAddIntent(query)
    assert(inserted.parents.toSet == Set(1157L))
    assert(parallel.node(1157L).instances.forall(parallel.node(1157L).classifies))
    assert(parallel.node(1157L).classifies(query))
    serial.fastAddIntent(query)
    assertSame(serial, parallel)

    // The insertion may already run on a fork-join worker holding the monitor.
    val pool = new ForkJoinPool(1)
    try
      pool.submit(new Callable[Node]:
        override def call(): Node = worker.fastAddIntent(query)
      ).get(30, TimeUnit.SECONDS)
    finally pool.shutdown()
    assertSame(serial, worker)

    // A wide parent layer exercises parallel propagation after placement.
    val propagationCva = new CVA(300, CvaConfig(0.01, 1.0))
    val propagationMembers = (0 until 12).map(i =>
      branchSample(s"propagation-child-$i", i * 0.001f, 0f)).toVector
    val center = branchSample("propagation-center", 0f, 0f).vector
    val identity = numeric.BFloatTensors((0 until 300 * 300).map { index =>
      if index / 300 == index % 300 then 1f else 0f
    }).view(300, 300)
    val childModel = CvaModel(center, identity, 0.05)
    val parentModel = CvaModel(center, identity, 0.9)
    def propagationFixture(parallel: Boolean): Lattice =
      new Lattice(propagationCva, LatticeConfig(
        closureSearchVisits = 1, closureSearchFrontier = 1,
        closureSearchBeamWidth = 1, propagationParallel = parallel,
        propagationChecksPerPass = 100000)):
        val child = new Node(7000L, propagationMembers, this.cva,
          restoredModel = Some(childModel))
        nodeMap.put(child.id, child)
        (0 until 4).foreach { index =>
          val extra = branchSample(s"propagation-parent-$index", 4f, 0f)
          val parent = new Node(7100L + index, Vector(extra), this.cva,
            restoredModel = Some(parentModel))
          parent.children += child.id
          child.parents += parent.id
          nodeMap.put(parent.id, parent)
        }
    val serialPropagation = propagationFixture(false)
    val parallelPropagation = propagationFixture(true)
    val propagated = branchSample("propagation-query", 0.003f, 0f)
    serialPropagation.fastAddIntent(propagated)
    parallelPropagation.fastAddIntent(propagated)
    assertSame(serialPropagation, parallelPropagation)
    assert((7100L until 7104L).forall(id => parallelPropagation.node(id).containsSymbol(propagated.symbol)))

    // Walk upward from the bottom frontier to enclosing CVAs.
    val upCva = new CVA(2, CvaConfig(0.01, 1.0))
    val points = (0 until 8).map(i => sample(s"up-$i", i * 4f, 0f)).toVector
    val upFixture = new Lattice(upCva, LatticeConfig(
      closureSearchVisits = 128, closureSearchFrontier = 64)):
      val parents = (0 until 48).map(i =>
        new Node(2000L + i, points :+ sample(s"parent-extra-$i", 32f + i * 4f, 0f),
          this.cva, do_avg = false)).toVector
      val leaves = points.zipWithIndex.map { (point, i) =>
        new Node(3000L + i, Vector(point), this.cva, do_avg = false)
      }
      parents.foreach { parent =>
        leaves.foreach { leaf =>
          parent.children += leaf.id
          leaf.parents += parent.id
        }
      }
      (parents ++ leaves).foreach(node => nodeMap.put(node.id, node))
    val upward = upFixture.cloneLattice()
    val upQuery = sample("upward-query", 1f, 0f)
    val upNode = upward.fastAddIntent(upQuery)
    val expectedClassifyingParents = (2000L until 2048L).toSet
    assert(upNode.parents.toSet == expectedClassifyingParents)
    assert(expectedClassifyingParents.forall(id => upward.node(id).classifies(upQuery)))
    upward.validate()

    // Even with a smaller top frontier, start at the bottom and stop at the
    // most specific generator within the two-model search budget.
    val specificFixture = new Lattice(upCva, LatticeConfig(closureSearchVisits = 2,
      closureSearchFrontier = 1, closureSearchBeamWidth = 1)):
      val leaf = new Node(20L, Vector(sample("specific-origin", 0f, 0f)), this.cva, do_avg = false)
      val middle = new Node(10L, Vector(sample("specific-left", -4f, 0f),
        sample("specific-right", 4f, 0f)), this.cva, do_avg = false)
      val root = new Node(90L, middle.instances :+ sample("specific-above", 0f, 4f),
        this.cva, do_avg = false)
      root.children += middle.id
      middle.parents += root.id
      middle.children += leaf.id
      leaf.parents += middle.id
      Vector(root, middle, leaf).foreach(node => nodeMap.put(node.id, node))
    val specific = specificFixture.cloneLattice()
    val specificQuery = sample("specific-query", 2f, 0f)
    val specificNode = specific.fastAddIntent(specificQuery)
    assert(specificNode.parents.toSet == Set(10L))
    assert(specific.node(10L).containsSymbol(specificQuery.symbol))
    assert(specific.node(90L).containsSymbol(specificQuery.symbol))
    assert(!specific.node(20L).containsSymbol(specificQuery.symbol))
    specific.validate()

    // Give both beam branches access to the limited parent frontier. The first
    // leaf's many parents all reject the query; the second's parent records it.
    val balancedFixture = new Lattice(upCva, LatticeConfig(closureSearchVisits = 4,
      closureSearchFrontier = 2, closureSearchBeamWidth = 2)):
      val leftSample = sample("balanced-left", -1f, 0f)
      val rightSample = sample("balanced-right", 1f, 0f)
      val recorded = sample("balanced-recorded", 0f, 0f)
      val left = new Node(6000L, Vector(leftSample), this.cva)
      val right = new Node(6001L, Vector(rightSample), this.cva)
      val noise = (0 until 20).map { index =>
        val parent = new Node(6100L + index,
          Vector(leftSample, sample(s"balanced-noise-$index", -1f, index + 1f)), this.cva)
        parent.children += left.id
        left.parents += parent.id
        parent
      }
      val containing = new Node(6200L, Vector(rightSample, recorded), this.cva)
      containing.children += right.id
      right.parents += containing.id
      (Vector(left, right) ++ noise :+ containing).foreach(node => nodeMap.put(node.id, node))
    val balanced = balancedFixture.cloneLattice()
    val savedModels = balanced.nodes.map(node => node -> node.model)
    assert(balanced.fastAddIntent(sample("balanced-recorded", 0f, 0f)) eq balanced.node(6200L))
    assert(balanced.size == savedModels.size)
    assert(savedModels.forall((node, model) => node.model eq model))
    balanced.validate()

    // Shared parents consume raw edge visits, but should not exhaust the
    // distinct-candidate frontier before a second enclosing CVA is reached.
    val sharedParentsFixture = new Lattice(upCva, LatticeConfig(closureSearchVisits = 8,
      closureSearchFrontier = 4, closureSearchBeamWidth = 4)):
      val points = Vector(-8f, -4f, 4f, 8f).zipWithIndex.map { (x, index) =>
        sample(s"shared-parent-leaf-$index", x, 0f)
      }
      val recorded = sample("shared-parent-recorded", 0f, 4f)
      val rejecting = new Node(6300L, points, this.cva)
      val containing = new Node(6301L, points :+ recorded, this.cva)
      val leaves = points.zipWithIndex.map { (point, index) =>
        val leaf = new Node(6400L + index, Vector(point), this.cva)
        Vector(rejecting, containing).foreach { parent =>
          parent.children += leaf.id
          leaf.parents += parent.id
        }
        leaf
      }
      (Vector(rejecting, containing) ++ leaves).foreach(node => nodeMap.put(node.id, node))
    val sharedParents = sharedParentsFixture.cloneLattice()
    val parentModels = sharedParents.nodes.map(node => node -> node.model)
    assert(sharedParents.fastAddIntent(sample("shared-parent-recorded", 0f, 4f)) eq sharedParents.node(6301L))
    assert(sharedParents.size == parentModels.size)
    assert(parentModels.forall((node, model) => node.model eq model))
    sharedParents.validate()

    // A recorded bottom member is a no-op. An accepting disconnected classifier
    // outside the beam must not receive a full-lattice membership update.
    val boundedFixture = new Lattice(upCva, LatticeConfig(closureSearchVisits = 1,
      closureSearchFrontier = 1, closureSearchBeamWidth = 1)):
      val recorded = sample("bounded-recorded", 0f, 0f)
      val first = new Node(8L, Vector(recorded), this.cva, do_avg = false)
      val outside = new Node(5L, Vector(sample("bounded-left", -4f, 0f),
        sample("bounded-right", 4f, 0f)), this.cva, do_avg = false)
      nodeMap.put(first.id, first)
      nodeMap.put(outside.id, outside)
    val bounded = boundedFixture.cloneLattice()
    val outsideModel = bounded.node(5L).model
    assert(bounded.fastAddIntent(sample("bounded-recorded", 0f, 0f)) eq bounded.node(8L))
    assert(!bounded.node(5L).containsSymbol("bounded-recorded"))
    assert(bounded.node(5L).model eq outsideModel)
    assert(bounded.size == 2)
    bounded.validate()

    // Probe acceptance alone cannot establish inclusion or mutual equivalence.
    val verified = new Lattice(new CVA(2, CvaConfig(0.01, 1.0)))
    val line = verified.fastAddIntent(Vector(sample("a", 0f, 0f), sample("b", 4f, 0f)))
    val mixed = Vector(sample("first", 0f, 0f), sample("outside-1", 0f, 8f),
      sample("middle", 2f, 0f), sample("outside-2", 4f, 8f), sample("last", 4f, 0f))
    val mixedNode = verified.fastAddIntent(mixed)
    assert((mixedNode ne line) && mixedNode.parents.isEmpty)
    assert(mixedNode.children.contains(line.id))
    verified.validate()

    // Approximate extents are found by CVA classification even when their
    // symbols differ completely.
    val smallCva = new CVA(2, CvaConfig(0.01, 1.0))
    val approximateFixture = new Lattice(smallCva, LatticeConfig(closureSearchVisits = 1200,
      closureSearchFrontier = 600, closureSearchBeamWidth = 16)):
      for i <- 0 until 600 do
        val member = sample(s"exact-$i", (i % 20) * 4f, (i / 20) * 4f)
        val node = new Node(4000L + i, Vector(member), this.cva, do_avg = false)
        nodeMap.put(node.id, node)
    val approximate = approximateFixture.cloneLattice()
    val equivalentVector = approximate.node(4599L).instances.head.vector
    assert(approximate.fastAddIntent(Instance("different-symbol", equivalentVector)) eq approximate.node(4599L))
    assert(approximate.size == 600)
    approximate.validate()

    println("Bottom-up beam generators, bounded membership, full CVA verification and parallel determinism checks passed.")
