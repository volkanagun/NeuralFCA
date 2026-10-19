package cvafca

import java.nio.file.Files
import numeric.BFloatTensors.*

/** Run with cvafca.LatticeOriginalInsertionTest. */
object LatticeOriginalInsertionTest:
  private def sample(symbol: String, x: Float, y: Float): Instance =
    Instance(symbol, numeric.BFloatTensors(Seq(x, y)))

  private def checkCovers(lattice: Lattice): Unit =
    lattice.validate()
    lattice.nodes.foreach { parent =>
      assert(parent.instances.forall(parent.classifies), s"Unclassified extent in ${parent.id}")
      parent.children.foreach { id =>
        val child = lattice.node(id)
        assert(!lattice.nodes.exists(middle =>
          child.symbols != middle.symbols && middle.symbols != parent.symbols &&
            child.symbols.subsetOf(middle.symbols) && middle.symbols.subsetOf(parent.symbols)),
          s"Bypass edge ${parent.symbols} -> ${child.symbols}")
      }
    }

  def main(args: Array[String]): Unit =
    org.bytedeco.pytorch.global.torch.set_num_threads(1)

    for noCVA <- Seq(false, true) do
      val lattice = new Lattice(new CVA(2, CvaConfig(0.05, 1.0, noCVA)))
      val first = sample("first", 0f, 0f)
      val a = lattice.original(first)
      val same = lattice.original(sample("same-vector", 0f, 0f))
      assert(a.id == same.id && lattice.size == 1)
      assert(a.symbols == Set("first", "same-vector"),
        "Equal learned intents must reuse a node despite different symbols")
      assert(lattice.original(first).id == a.id && a.size == 2)

      val b = lattice.original(sample("far", 4f, 0f))
      assert(b.id != a.id && lattice.size == 3)
      val root = lattice.nodes.find(_.parents.isEmpty).get
      assert(root.symbols == Set("first", "same-vector", "far"))
      assert(root.children.toSet == Set(a.id, b.id))
      if noCVA then
        assert(math.abs(root.commonVector.floatValues.head - 4f / 3f) < 0.01)
      else
        assert(root.projection.floatValues.sameElements(Array(0f, 0f, 0f, 1f)),
          "Intersections must fit CVA projections")
      checkCovers(lattice)

    // Original insertion must maintain indexes established by the fast path.
    val mixed = new Lattice(new CVA(2, CvaConfig(0.05, 1.0)))
    mixed.addIntentFastIndexed(sample("indexed-first", 0f, 0f))
    val originalNode = mixed.original(sample("original-second", 4f, 0f))
    val indexedReuse = mixed.addIntentFastIndexed(sample("indexed-same", 4f, 0f))
    assert(indexedReuse.id == originalNode.id && mixed.size == 3)
    mixed.original(sample("original-third", 0f, 4f))
    checkCovers(mixed)

    val triangle = new Lattice(new CVA(2, CvaConfig(0.05, 1.0)))
    val a = triangle.original(sample("a", 0f, 0f))
    val b = triangle.original(sample("b", 4f, 0f))
    val c = triangle.original(sample("c", 0f, 4f))
    def extent(symbols: String*): Node = triangle.nodes.find(_.symbols == symbols.toSet).get
    assert(triangle.nodes.map(_.symbols).toSet ==
      Set(Set("a"), Set("b"), Set("c"), Set("a", "b"), Set("a", "c"),
        Set("b", "c"), Set("a", "b", "c")))
    val ab = extent("a", "b")
    val top = extent("a", "b", "c")
    assert(top.children.toSet == Set(ab.id, extent("a", "c").id, extent("b", "c").id))
    assert(top.projection.floatValues.forall(_ == 0f))
    checkCovers(triangle)

    // Cloning and checkpoint continuation must retain the same insertion state.
    val checkpoint = Files.createTempFile("original-addintent-", ".bin")
    try
      val cloned = triangle.cloneLattice()
      triangle.save(checkpoint.toString)
      val loaded = new Lattice(new CVA(2, CvaConfig(0.05, 1.0))).load(checkpoint.toString)
      val d = sample("d", 2f, 0f)
      val inserted = triangle.original(d)
      cloned.original(d)
      loaded.original(d)
      assert(extent("a", "b", "d").id == ab.id)
      assert(extent("a", "b", "c", "d").id == top.id)
      assert(extent("c", "d").children.toSet == Set(c.id, inserted.id))
      assert(!triangle.nodes.exists(_.symbols == Set("a", "d")),
        "An intersection equal to an existing learned intent must be reused")
      val count = triangle.size
      assert(triangle.original(d).id == inserted.id && triangle.size == count)
      for copy <- Seq(cloned, loaded) do
        assert(copy.nodes.map(n => (n.id, n.symbols, n.parents.toVector, n.children.toVector)) ==
          triangle.nodes.map(n => (n.id, n.symbols, n.parents.toVector, n.children.toVector)))
        copy.nodes.zip(triangle.nodes).foreach { (left, right) =>
          assert(left.commonVector.floatValues.sameElements(right.commonVector.floatValues))
          assert(left.projection.floatValues.sameElements(right.projection.floatValues))
          assert(left.classificationRadius == right.classificationRadius)
        }
        checkCovers(copy)
      checkCovers(triangle)
    finally Files.deleteIfExists(checkpoint)

    // GetMaximalConcept must inspect bottoms beyond every bounded search budget.
    val manyCva = new CVA(2, CvaConfig(0.0, 1.0))
    val many = new Lattice(manyCva):
      for index <- 0 until 300 do
        val member = sample(s"isolated-$index", (index % 20).toFloat, (index / 20).toFloat)
        val node = new Node(1000L + index, List(member), manyCva)
        nodeMap.put(node.id, node)
    val exhaustive = many.cloneLattice()
    val last = exhaustive.original(sample("outside-budget", 19f, 14f))
    assert(last.id == 1299L && exhaustive.size == 300)
    assert(last.symbols == Set("isolated-299", "outside-budget"))
    checkCovers(exhaustive)

    // Six recursive intersection levels: all nonempty subsets of seven
    // affinely independent vertices.
    val dimension = 6
    val simplex = new Lattice(new CVA(dimension, CvaConfig(0.05, 1.0)))
    val vertices = (0 to dimension).map { index =>
      Instance(s"vertex-$index", numeric.BFloatTensors((0 until dimension).map { axis =>
        if index == axis + 1 then 4f else 0f
      }))
    }
    vertices.foreach(simplex.original)
    val expected = (1 until (1 << vertices.size)).map { mask =>
      vertices.indices.filter(index => (mask & (1 << index)) != 0).map(vertices(_).symbol).toSet
    }.toSet
    assert(simplex.nodes.map(_.symbols).toSet == expected,
      "Original AddIntent must create every intersection without width or depth limits")
    assert(simplex.size == 127)
    checkCovers(simplex)

    // Every one of twelve incomparable parent intersections must be inserted.
    val wideCva = new CVA(2, CvaConfig(0.05, 1.0))
    val wideFixture = new Lattice(wideCva):
      val members = (0 until 12).map(index => sample(s"wide-$index", index - 6f, 0f))
      val root = new Node(2000L, members, wideCva, do_avg = false)
      nodeMap.put(root.id, root)
      members.zipWithIndex.foreach { (member, index) =>
        val leaf = new Node(1000L + index, List(member), wideCva)
        nodeMap.put(leaf.id, leaf)
        root.children += leaf.id
        leaf.parents += root.id
      }
    val wide = wideFixture.cloneLattice()
    val incoming = wide.original(sample("wide-incoming", 0f, 4f))
    assert(incoming.parents.size == 12, "Parent discovery must not stop at a width budget")
    for index <- 0 until 12 do
      assert(wide.nodes.exists(_.symbols == Set(s"wide-$index", "wide-incoming")))
    checkCovers(wide)

    // Every member must be classified, even with permissive equalityCoverage.
    val strictCva = new CVA(2, CvaConfig(0.05, 1.0))
    val strictFixture = new Lattice(strictCva, LatticeConfig(equalityCoverage = 0.5)):
      val near = sample("near", 0f, 0f)
      val far = sample("far", 4f, 0f)
      nodeMap.put(1000L, new Node(1000L, List(near, far), strictCva,
        restoredModel = Some(strictCva.fit(List(near.vector)))))
    val strict = strictFixture.cloneLattice()
    val singleton = strict.original(sample("incoming", 0f, 0f))
    assert(singleton.id != 1000L && singleton.size == 1,
      "Partial extent coverage must not stand in for intent inclusion")
    checkCovers(strict)

    println("Original AddIntent membership, intersections, covers and persistence checks passed")
