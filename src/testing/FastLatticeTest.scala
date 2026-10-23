package cvafca

import java.nio.file.Files
import numeric.BFloatTensors.*

/** Run with cvafca.FastLatticeTest. */
object FastLatticeTest:
  private def sample(symbol: String, x: Float, y: Float): Instance =
    Instance(symbol, numeric.BFloatTensors(Seq(x, y)))

  def main(args: Array[String]): Unit =
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    val cva = new CVA(2, CvaConfig(0.05, 1.0))
    val lattice = new FastLattice(cva, FastLatticeConfig(
      searchVisits = 128, searchFrontier = 128, searchBeamWidth = 8))
    val shared = sample("shared", 4f, 0f)
    val horizontal = Vector(sample("left", 0f, 0f), shared, sample("right", 8f, 0f))
    val vertical = Vector(sample("down", 4f, -4f), shared, sample("up", 4f, 4f))
    val left = lattice.fastAddIntent(horizontal)
    val right = lattice.fastAddIntent(vertical)
    val common = lattice.nodes.find(_.symbols == Set("shared")).get
    assert(lattice.size == 3)
    assert(common.parents.toSet == Set(left.id, right.id))
    assert(common.commonVector.floatValues.sameElements(shared.vector.floatValues))
    lattice.validate()

    val alias = sample("alias", 4f, 0f)
    assert(lattice.fastAddIntent(alias) eq common)
    assert(left.containsSymbol(alias.symbol) && right.containsSymbol(alias.symbol))
    val oldModel = common.model
    assert(lattice.fastAddIntent(alias) eq common)
    assert(common.model eq oldModel)
    lattice.validate()

    val cloned = lattice.cloneLattice()
    assert(cloned.size == lattice.size)
    cloned.validate()

    val isolated = lattice.fastAddIntent(sample("isolated", 40f, 40f))
    assert(isolated.parents.isEmpty && isolated.children.isEmpty)
    lattice.validate()

    val repeatedExtent = new FastLattice(cva)
    val first = repeatedExtent.fastAddIntent(sample("same-name", 0f, 0f))
    val originalModel = first.model
    (0 until 10).foreach { _ =>
      val second = repeatedExtent.fastAddIntent(sample("same-name", 40f, 40f))
      assert(second eq first)
      assert(repeatedExtent.size == 1)
      assert(first.model eq originalModel)
    }
    repeatedExtent.validate()

    val repeatedPair = new FastLattice(cva)
    val pair = Vector(sample("pair-a", 0f, 0f), sample("pair-b", 8f, 0f))
    val pairNode = repeatedPair.fastAddIntent(pair)
    assert(repeatedPair.fastAddIntent(pair.reverse) eq pairNode)
    assert(repeatedPair.size == 1)
    repeatedPair.validate()

    val converging = new FastLattice(cva):
      val a = sample("a", 0f, 0f)
      val b = sample("b", 0f, 0f)
      nodeMap.put(10L, new Node(10L, Vector(a), this.cva))
      nodeMap.put(11L, new Node(11L, Vector(a, b), this.cva))
    val converged = converging.fastAddIntent(sample("b", 0f, 0f))
    assert(converging.nodes.map(_.symbols).distinct.size == converging.size)
    assert(converging.size == 1)
    assert(converging.nodes.head eq converged)
    converging.validate()

    def duplicateFixture(): FastLattice = new FastLattice(cva):
      nodeMap.put(40L, new Node(40L, Vector(sample("duplicate", 0f, 0f)), this.cva))
      nodeMap.put(41L, new Node(41L, Vector(sample("duplicate", 40f, 40f)), this.cva))
    val prepopulated = duplicateFixture()
    assert(prepopulated.fastAddIntent(sample("duplicate", 0f, 0f)).id == 40L)
    assert(prepopulated.size == 1)
    prepopulated.validate()

    val duplicateCheckpoint = Files.createTempFile("fast-lattice-duplicates-", ".bin")
    try
      duplicateFixture().save(duplicateCheckpoint.toString)
      val loadedUnique = new FastLattice(cva).load(duplicateCheckpoint.toString)
      assert(loadedUnique.size == 1)
      assert(loadedUnique.nodes.head.id == 40L)
      loadedUnique.validate()
    finally Files.deleteIfExists(duplicateCheckpoint)

    val linkedDuplicates = new FastLattice(cva):
      val projection = numeric.BFloatTensors(Seq(1f, 0f, 0f, 1f)).view(2, 2)
      val origin = sample("duplicate", 0f, 0f)
      val childMember = sample("child", 1f, 0f)
      val parent = new Node(1L,
        Vector(sample("parent", 10f, 0f), origin, childMember), this.cva,
        restoredModel = Some(CvaModel(origin.vector, projection, 0.95)))
      val survivor = new Node(2L, Vector(origin, childMember), this.cva,
        restoredModel = Some(CvaModel(origin.vector, projection, 0.7)))
      val duplicate = new Node(3L, Vector(origin, childMember), this.cva,
        restoredModel = Some(CvaModel(origin.vector, projection, 0.7)))
      val child = new Node(4L, Vector(childMember), this.cva,
        restoredModel = Some(CvaModel(childMember.vector, projection, 0.05)))
      Vector(parent, survivor, duplicate, child).foreach(node => nodeMap.put(node.id, node))
      linkFast(parent, duplicate)
      linkFast(duplicate, child)
    linkedDuplicates.fastAddIntent(sample("unrelated", 40f, 0f))
    assert(linkedDuplicates.nodes.map(_.symbols).distinct.size == linkedDuplicates.size)
    assert(linkedDuplicates.node(1L).children.contains(2L))
    assert(linkedDuplicates.node(2L).children.contains(4L))
    linkedDuplicates.validate()

    val bounded = new FastLattice(cva, FastLatticeConfig(maxIntersections = 0))
    bounded.fastAddIntent(horizontal)
    bounded.fastAddIntent(vertical)
    assert(bounded.size == 2)
    bounded.validate()

    val disconnected = new FastLattice(cva):
      val point = new Node(100L, Vector(sample("existing", 4f, 0f)), this.cva)
      val line = new Node(101L,
        Vector(sample("line-left", 0f, 0f), sample("line-right", 8f, 0f)), this.cva)
      nodeMap.put(point.id, point)
      nodeMap.put(line.id, line)
    val reused = disconnected.fastAddIntent(sample("existing", 4f, 0f))
    assert(reused.parents.toSet == Set(101L))
    disconnected.validate()

    val checkpoint = Files.createTempFile("fast-lattice-", ".bin")
    try
      lattice.save(checkpoint.toString)
      val loaded = new FastLattice(cva).load(checkpoint.toString)
      assert(loaded.size == lattice.size)
      loaded.fastAddIntent(sample("after-load", 80f, 80f))
      loaded.validate()
    finally Files.deleteIfExists(checkpoint)

    val wideCva = new CVA(300, CvaConfig(0.01, 1.0))
    val wideProjection = numeric.BFloatTensors((0 until 300 * 300).map { index =>
      if index / 300 == index % 300 then 1f else 0f
    }).view(300, 300)
    def wideSample(symbol: String, x: Float): Instance =
      Instance(symbol, numeric.BFloatTensors(Vector(x) ++ Vector.fill(299)(0f)))
    def wideFixture(parallel: Boolean): FastLattice =
      new FastLattice(wideCva, FastLatticeConfig(
        searchVisits = 64, searchFrontier = 64, parallelSearch = parallel)):
        (0 until 32).foreach { index =>
          val member = wideSample(s"wide-$index", index * 4f)
          nodeMap.put(index.toLong, new Node(index.toLong, Vector(member), this.cva,
            restoredModel = Some(CvaModel(member.vector, wideProjection, 0.01))))
        }
    val wideSerial = wideFixture(false)
    val wideParallel = wideFixture(true)
    val query = wideSample("wide-query", 19 * 4f)
    assert(wideSerial.fastAddIntent(query).id == 19L)
    assert(wideParallel.fastAddIntent(query).id == 19L)
    assert(wideSerial.nodes.map(n => (n.id, n.symbols, n.parents.toSet, n.children.toSet)) ==
      wideParallel.nodes.map(n => (n.id, n.symbols, n.parents.toSet, n.children.toSet)))
    wideSerial.validate()
    wideParallel.validate()
    println("FastLattice extent uniqueness, intersections, refits, and cloning passed.")
