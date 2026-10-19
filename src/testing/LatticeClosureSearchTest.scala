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
    val fixture = new Lattice(classifier, LatticeConfig(closureSearchParallel = parallel)):
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

    // Descend from one root and rank its 64 children by actual sample distance.
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

    // Choose the smaller bottom frontier, then walk upward to an enclosing CVA.
    val upCva = new CVA(2, CvaConfig(0.01, 1.0))
    val points = (0 until 8).map(i => sample(s"up-$i", i * 4f, 0f)).toVector
    val upFixture = new Lattice(upCva):
      val parents = (0 until 48).map(i =>
        new Node(2000L + i, points, this.cva, do_avg = false)).toVector
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
    assert(upNode.parents.toSet == Set(2000L))
    assert(upward.node(2000L).classifies(upQuery))
    upward.validate()

    // Probe acceptance alone cannot establish inclusion or mutual equivalence.
    val verified = new Lattice(new CVA(2, CvaConfig(0.01, 1.0)))
    val line = verified.fastAddIntent(Vector(sample("a", 0f, 0f), sample("b", 4f, 0f)))
    val mixed = Vector(sample("first", 0f, 0f), sample("outside-1", 0f, 8f),
      sample("middle", 2f, 0f), sample("outside-2", 4f, 8f), sample("last", 4f, 0f))
    val mixedNode = verified.fastAddIntent(mixed)
    assert(mixedNode.id != line.id && mixedNode.parents.isEmpty)
    assert(mixedNode.children.contains(line.id))
    verified.validate()

    // Exact indexed matches bypass a deliberately tiny approximate search.
    val smallCva = new CVA(2, CvaConfig(0.01, 1.0))
    val exactFixture = new Lattice(smallCva, LatticeConfig(closureSearchVisits = 1,
      closureSearchFrontier = 1, closureSearchBeamWidth = 1)):
      for i <- 0 until 600 do
        val member = sample(s"exact-$i", (i % 20) * 4f, (i / 20) * 4f)
        val node = new Node(4000L + i, Vector(member), this.cva, do_avg = false)
        nodeMap.put(node.id, node)
    val exact = exactFixture.cloneLattice()
    assert(exact.fastAddIntent(exact.node(4599L).instances).id == 4599L)
    assert(exact.size == 600)
    exact.validate()

    // The iterative insertion engine must maintain the same frontier indexes.
    val simple = new LatticeSimple(new CVA(2, CvaConfig(0.05, 1.0)))
    Vector(sample("simple-a", 0f, 0f), sample("simple-b", 4f, 0f),
      sample("simple-c", 0f, 4f)).foreach(simple.addIntent)
    simple.validate()
    println("Closure graph directions, full CVA verification and parallel determinism checks passed.")
