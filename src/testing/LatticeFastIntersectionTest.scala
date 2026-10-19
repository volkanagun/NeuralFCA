package cvafca

import numeric.BFloatTensors.*

/** Exercise root, child and reused intersections through both fast entry points. */
object LatticeFastIntersectionTest:
  def main(args: Array[String]): Unit =
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    for indexed <- Seq(false, true); cacheSize <- Seq(0, 2) do
      val lattice = new Lattice(new CVA(2, CvaConfig(0.05, 1.0)),
        LatticeConfig(equalityCoverage = 1.0,
          requireMutualCommonClassification = true, fastBasisCacheNodes = cacheSize))
      def insert(symbol: String, x: Float, y: Float): Node =
        val sample = Instance(symbol, numeric.BFloatTensors(Seq(x, y)))
        if indexed then lattice.addIntentFastIndexed(sample) else lattice.addIntentFast(sample)
      def extent(symbols: String*): Node =
        lattice.nodes.find(_.symbols == symbols.toSet).getOrElse(
          throw new AssertionError(s"Missing intersection ${symbols.toSet}: ${lattice.nodes.map(_.symbols)}"))
      def checkCovers(): Unit =
        lattice.validate()
        lattice.nodes.foreach { parent =>
          parent.children.foreach { id =>
            val child = lattice.node(id)
            assert(!lattice.nodes.exists(middle =>
              child.symbols != middle.symbols && middle.symbols != parent.symbols &&
                child.symbols.subsetOf(middle.symbols) && middle.symbols.subsetOf(parent.symbols)),
              s"Intersection insertion left a bypass edge ${parent.symbols} -> ${child.symbols}")
          }
        }

      val a = insert("a", 0f, 0f)
      val b = insert("b", 4f, 0f)
      val ab = extent("a", "b")
      assert(ab.children.toSet == Set(a.id, b.id), "A new root must connect both operands")
      assert(ab.projection.floatValues.sameElements(Array(0f, 0f, 0f, 1f)),
        "An intersection must fit the CVA projector, rather than average its vectors")
      checkCovers()

      val c = insert("c", 0f, 4f)
      val expected = Set(Set("a"), Set("b"), Set("c"), Set("a", "b"),
        Set("a", "c"), Set("b", "c"), Set("a", "b", "c"))
      assert(lattice.nodes.map(_.symbols).toSet == expected)
      val top = extent("a", "b", "c")
      assert(top.projection.floatValues.sameElements(Array(0f, 0f, 0f, 0f)))
      assert(top.children.toSet == Set(ab.id, extent("a", "c").id, extent("b", "c").id))
      assert(extent("a", "c").children.toSet == Set(a.id, c.id))
      assert(extent("b", "c").children.toSet == Set(b.id, c.id))
      checkCovers()

      val count = lattice.size
      assert(insert("c", 0f, 4f).id == c.id && lattice.size == count,
        "Repeating an insertion must reuse its intersections")

      // The old ab generator accepts d, while c needs a distinct intersection.
      val d = insert("d", 2f, 0f)
      val cd = extent("c", "d")
      assert(cd.children.toSet == Set(c.id, d.id))
      assert(extent("a", "b", "d").id == ab.id, "Equivalent intersections must reuse the generator")
      assert(extent("a", "b", "c", "d").id == top.id)
      assert(!lattice.nodes.exists(_.symbols == Set("a", "d")))
      assert(!lattice.nodes.exists(_.symbols == Set("b", "d")))
      checkCovers()
      assert(lattice.nodes.forall(node => node.instances.forall(node.classifies)))

    for indexed <- Seq(false, true) do
      val lattice = new Lattice(new CVA(2, CvaConfig(0.05, 1.0, noCVA = true)))
      def insert(symbol: String, x: Float): Node =
        val sample = Instance(symbol, numeric.BFloatTensors(Seq(x, 0f)))
        if indexed then lattice.addIntentFastIndexed(sample) else lattice.addIntentFast(sample)
      val a = insert("a", 0f)
      val b = insert("b", 4f)
      val root = lattice.nodes.find(_.symbols == Set("a", "b")).get
      assert(root.commonVector.floatValues.sameElements(Array(2f, 0f)))
      assert(root.children.toSet == Set(a.id, b.id))
      assert(root.instances.forall(root.classifies))
      lattice.validate()

    // Search limits must not leave a growing concept's distant ancestors stale.
    // This chain exceeds both the regular and fast generator-visit budgets.
    val chainCva = new CVA(2, CvaConfig(0.05, 1.0, noCVA = true))
    val sharedVector = numeric.BFloatTensors(Seq(0f, 0f))
    val chainLength = 300
    val fixture = new Lattice(chainCva):
      private val members = (0 until chainLength).map(index => Instance(s"chain-$index", sharedVector))
      private val sharedModel = chainCva.fit_avg(Vector(sharedVector))
      for index <- 0 until chainLength do
        val concept = new Node(1000L + index, members.take(index + 1), chainCva,
          restoredModel = Some(sharedModel))
        nodeMap.put(concept.id, concept)
        if index > 0 then
          val child = nodeMap(concept.id - 1)
          concept.children += child.id
          child.parents += concept.id
    for method <- Seq("regular", "fast", "indexed") do
      val lattice = fixture.cloneLattice()
      val incoming = Instance(s"new-$method", sharedVector)
      val result = method match
        case "regular" => lattice.addIntent(incoming)
        case "fast" => lattice.addIntentFast(incoming)
        case _ => lattice.addIntentFastIndexed(incoming)
      assert(result.id == 1000L)
      assert(lattice.size == chainLength)
      assert(lattice.nodes.forall(_.containsSymbol(incoming.symbol)),
        s"$method insertion truncated required ancestor updates")
      lattice.validate()
    println("Fast intersections preserve roots, child cases, reuse and cover edges")
