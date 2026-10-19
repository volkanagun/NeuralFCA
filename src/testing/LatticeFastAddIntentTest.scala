package cvafca

import java.nio.file.Files
import numeric.BFloatTensors.*

/** Run with cvafca.LatticeFastAddIntentTest. */
object LatticeFastAddIntentTest:
  private def sample(symbol: String, x: Float, y: Float): Instance =
    Instance(symbol, numeric.BFloatTensors(Seq(x, y)))

  private def snapshot(lattice: Lattice) =
    lattice.nodes.map(n => (n.id, n.symbols, n.parents.toVector, n.children.toVector))

  def main(args: Array[String]): Unit =
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    val cva = new CVA(2, CvaConfig(0.05, 1.0))
    val lattice = new Lattice(cva)
    val shared = sample("shared", 4f, 0f)
    val horizontal = Vector(sample("left", 0f, 0f), shared, sample("right", 8f, 0f))
    val vertical = Vector(sample("down", 4f, -4f), shared, sample("up", 4f, 4f))
    val a = lattice.fastAddIntent(horizontal)
    val b = lattice.fastAddIntent(vertical)
    val intersection = lattice.nodes.find(_.symbols == Set("shared")).get
    assert(lattice.size == 3)
    assert(intersection.parents.toSet == Set(a.id, b.id))
    assert(a.children.toSet == Set(intersection.id) && b.children.toSet == Set(intersection.id))
    assert(intersection.commonVector.floatValues.sameElements(shared.vector.floatValues))
    assert(intersection.projection.floatValues.sameElements(Array(1f, 0f, 0f, 1f)),
      "The intersection must be refitted from the common samples, not the union")
    lattice.validate()

    // Different IDs can belong to the same learned intersection.
    val alias = sample("shared-alias", 4f, 0f)
    val aliasLattice = new Lattice(cva)
    aliasLattice.fastAddIntent(horizontal)
    val other = aliasLattice.fastAddIntent(Vector(alias, sample("above", 4f, 8f)))
    val learnedIntersection = aliasLattice.nodes.find(_.symbols == Set("shared", "shared-alias")).get
    assert(learnedIntersection.symbols == Set("shared", "shared-alias"))
    assert(learnedIntersection.instances.forall(a.classifies))
    assert(learnedIntersection.instances.forall(other.classifies))
    val expected = cva.fit(learnedIntersection.instances.map(_.vector))
    assert(learnedIntersection.commonVector.floatValues.sameElements(expected.commonVector.floatValues))
    assert(learnedIntersection.projection.floatValues.sameElements(expected.projection.floatValues))

    // Disjoint intersections stay virtual and must not invoke CVA.fit(empty).
    val isolated = lattice.fastAddIntent(sample("far", 40f, 40f))
    assert(isolated.parents.isEmpty && isolated.children.isEmpty && lattice.size == 4)

    // Mutual coverage reuses equivalent intents and updates all their ancestors.
    val reused = lattice.fastAddIntent(alias)
    assert(reused.id == intersection.id && lattice.size == 4)
    assert(reused.symbols == Set("shared", "shared-alias"))
    assert(a.containsSymbol(alias.symbol) && b.containsSymbol(alias.symbol))
    assert(lattice.fastAddIntent(alias).id == reused.id && lattice.size == 4)
    lattice.validate()

    // One-way classification does not establish equivalence, even at 50%.
    val asymmetric = new Lattice(cva, LatticeConfig(equalityCoverage = 0.5))
    val line = asymmetric.fastAddIntent(horizontal)
    val point = asymmetric.fastAddIntent(sample("point", 2f, 0f))
    assert(line.id != point.id, "Both directional coverage thresholds are required")
    assert(point.parents.contains(line.id))
    asymmetric.validate()

    // Equality is a fraction of the OTHER extent, inclusive at the threshold.
    for threshold <- Seq(0.8, 0.81) do
      val partial = new Lattice(cva, LatticeConfig(equalityCoverage = threshold))
      val left = (0 until 4).map(i => sample(s"left-zero-$i", 0f, 0f)).toVector :+
        sample("left-outlier", 4f, 0f)
      val right = (0 until 4).map(i => sample(s"right-zero-$i", 0f, 0f)).toVector :+
        sample("right-outlier", 0f, 4f)
      val first = partial.fastAddIntent(left)
      val second = partial.fastAddIntent(right)
      assert((first.id == second.id) == (threshold == 0.8))
      assert(partial.size == (if threshold == 0.8 then 1 else 3))
      partial.validate()

    // On exact affine faces, learned membership agrees with Boolean incidence.
    // Compare the complete intersection closure and covers, not just node count.
    val cube = new Lattice(new CVA(3, CvaConfig(0.01, 1.0)))
    val corners = (0 until 8).map { bits =>
      Instance(s"corner-$bits", numeric.BFloatTensors((0 until 3).map(axis =>
        if (bits & (1 << axis)) == 0 then 0f else 4f)))
    }.toVector
    val faces = (for axis <- 0 until 3; side <- 0 to 1 yield
      corners.zipWithIndex.collect { case (member, bits) if ((bits >> axis) & 1) == side => member }
    ).toVector
    faces.foreach(cube.fastAddIntent)
    val expectedExtents = (1 until (1 << faces.size)).map { mask =>
      faces.indices.filter(i => (mask & (1 << i)) != 0)
        .map(i => faces(i).iterator.map(_.symbol).toSet).reduce(_ intersect _)
    }.filter(_.nonEmpty).toSet
    assert(expectedExtents.size == 26)
    assert(cube.nodes.map(_.symbols).toSet == expectedExtents)
    cube.validate()
    cube.nodes.foreach { parent =>
      val expectedChildren = expectedExtents.filter(child =>
        child != parent.symbols && child.subsetOf(parent.symbols) &&
          !expectedExtents.exists(middle => middle != child && middle != parent.symbols &&
            child.subsetOf(middle) && middle.subsetOf(parent.symbols)))
      assert(parent.children.iterator.map(cube.node).map(_.symbols).toSet == expectedChildren,
        s"Incorrect cover links for ${parent.symbols}")
    }

    // Recursive intersection discovery must continue beyond four levels.
    val deepCva = new CVA(7, CvaConfig(0.01, 1.0))
    val origin = Instance("origin", numeric.BFloatTensors(Seq.fill(7)(0f)))
    val axes = (0 until 7).map { axis =>
      Instance(s"axis-$axis", numeric.BFloatTensors((0 until 7).map(i => if i == axis then 4f else 0f)))
    }.toVector
    val deepFixture = new Lattice(deepCva, LatticeConfig(equalityCoverage = 1.0)):
      for level <- 0 to 6 do
        val node = new Node(4000L + level, Vector(origin) ++ axes.take(level), deepCva, do_avg = false)
        nodeMap.put(node.id, node)
        if level > 0 then
          node.children += 4000L + level - 1
          nodeMap(4000L + level - 1).parents += node.id
    val deep = deepFixture.cloneLattice()
    val shifted = axes.head +: (1 until 7).map { axis =>
      Instance(s"shifted-$axis", numeric.BFloatTensors((0 until 7).map(i =>
        if i == 0 || i == axis then 4f else 0f)))
    }.toVector
    deep.fastAddIntent(shifted)
    assert(deep.size == 14, "Every recursive intersection must be materialized")
    (1 to 7).foreach { count =>
      assert(deep.nodes.exists(_.symbols == shifted.take(count).iterator.map(_.symbol).toSet))
    }
    deep.validate()

    // Twelve incomparable child intersections must all be processed.
    val wideSpaceCva = new CVA(3, CvaConfig(0.01, 1.0))
    val crossing = (0 until 12).map { i =>
      Instance(s"crossing-$i", numeric.BFloatTensors(Seq(0f, i.toFloat, 0f)))
    }.toVector
    val farEnds = (0 until 12).map { i =>
      Instance(s"end-$i", numeric.BFloatTensors(Seq(4f, i.toFloat, 0f)))
    }.toVector
    val wideSpaceFixture = new Lattice(wideSpaceCva, LatticeConfig(equalityCoverage = 1.0)):
      val root = new Node(5000L, crossing ++ farEnds, wideSpaceCva, do_avg = false)
      nodeMap.put(root.id, root)
      crossing.indices.foreach { i =>
        val leaf = new Node(5100L + i, Vector(crossing(i), farEnds(i)), wideSpaceCva, do_avg = false)
        nodeMap.put(leaf.id, leaf)
        root.children += leaf.id
        leaf.parents += root.id
      }
    val wideSpace = wideSpaceFixture.cloneLattice()
    wideSpace.fastAddIntent(crossing :+
      Instance("outside-plane", numeric.BFloatTensors(Seq(0f, 0f, 4f))))
    val commonLine = wideSpace.nodes.find(_.symbols == crossing.iterator.map(_.symbol).toSet).get
    assert(commonLine.children.size == 12 && wideSpace.size == 27)
    wideSpace.validate()

    // Frontier ranking finds equivalent nodes beyond the legacy fast budget.
    val wideCva = new CVA(2, CvaConfig(0.0, 1.0))
    val wideFixture = new Lattice(wideCva):
      for index <- 0 until 300 do
        val member = sample(s"isolated-$index", (index % 20).toFloat, (index / 20).toFloat)
        val node = new Node(1000L + index, Vector(member), wideCva, do_avg = false)
        nodeMap.put(node.id, node)
    val wide = wideFixture.cloneLattice()
    val last = wide.fastAddIntent(sample("outside-budget", 19f, 14f))
    assert(last.id == 1299L && wide.size == 300)
    wide.validate()

    // Learned predicates can be non-transitive, especially with partial
    // coverage. Repeated overlaps must still preserve the graph invariants.
    for noCVA <- Seq(false, true); threshold <- Seq(0.5, 0.8, 1.0) do
      val random = new scala.util.Random(741L)
      val overlapping = new Lattice(new CVA(3, CvaConfig(0.1, 1.0, noCVA)),
        LatticeConfig(equalityCoverage = threshold))
      for batch <- 0 until 16 do
        val members = (0 until 3).map { member =>
          Instance(s"overlap-$batch-$member", numeric.BFloatTensors((0 until 3).map(_ =>
            (random.nextInt(3) - 1) * 4f)))
        }
        overlapping.fastAddIntent(members)
        overlapping.validate()

    // Save/load and clone preserve subsequent insertion and rebuilt models.
    val checkpoint = Files.createTempFile("fastaddintent-", ".bin")
    try
      lattice.save(checkpoint.toString)
      val loaded = new Lattice(new CVA(2, cva.config)).load(checkpoint.toString)
      val cloned = lattice.cloneLattice()
      val next = sample("next-shared", 4f, 0f)
      Seq(lattice, loaded, cloned).foreach(_.fastAddIntent(next))
      for copy <- Seq(loaded, cloned) do
        assert(snapshot(copy) == snapshot(lattice))
        copy.nodes.zip(lattice.nodes).foreach { (left, right) =>
          assert(left.commonVector.floatValues.sameElements(right.commonVector.floatValues))
          assert(left.projection.floatValues.sameElements(right.projection.floatValues))
          assert(left.classificationRadius == right.classificationRadius)
        }
        copy.validate()
    finally Files.deleteIfExists(checkpoint)

    val before = snapshot(lattice)
    try
      lattice.fastAddIntent(Vector.empty[Instance])
      assert(false, "Empty input must be rejected")
    catch case _: IllegalArgumentException => ()
    assert(snapshot(lattice) == before)
    println("FastAddIntent CVA intersection, membership, coverage and persistence checks passed.")
