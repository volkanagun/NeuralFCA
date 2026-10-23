package cvafca

import java.nio.file.Files
import numeric.BFloatTensors.*

/** Membership updates and learned identity must not depend on IDs or symbol sets. */
object LatticeMembershipInsertionTest:
  private def sample(symbol: String, x: Float, y: Float = 0f): Instance =
    Instance(symbol, numeric.BFloatTensors(Seq(x, y)))

  private def fitted(node: Node, cva: CVA): Unit =
    val expected = cva.fit(node.instances.map(_.vector))
    assert(node.commonVector.floatValues.sameElements(expected.commonVector.floatValues))
    assert(node.projection.floatValues.sameElements(expected.projection.floatValues))
    assert(node.classificationRadius == expected.radius)

  def main(args: Array[String]): Unit =
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    val cva = new CVA(2, CvaConfig(0.01, 1.0))
    val identity = new Lattice(cva)
    val left = new Node(7L, Vector(sample("same", 0f)), cva, do_avg = false)
    val different = new Node(7L, Vector(sample("same", 4f)), cva, do_avg = false)
    val equivalent = new Node(90L, Vector(sample("alias", 0f)), cva, do_avg = false)
    assert(!identity.equivalent(left, different), "Equal IDs and labels must not establish equality")
    assert(identity.equivalent(left, equivalent), "Different IDs and labels must not prevent equality")

    // Public single-instance updates must also rebuild a CVA from the extent.
    val growingNode = new Node(91L, Vector(sample("initial", 0f, 2f)), cva)
    assert(growingNode.addInstance(sample("additional", 4f, 2f)))
    fitted(growingNode, cva)
    assert(growingNode.classifies(sample("on-fitted-line", 8f, 2f)))

    // Address allocation must not overwrite an externally stored concept.
    val occupied = new Lattice(cva):
      val stored = new Node(0L, Vector(sample("occupied", 0f)), this.cva, do_avg = false)
      nodeMap.put(stored.id, stored)
    val occupiedNode = occupied.node(0L)
    val inserted = occupied.fastAddIntent(sample("far", 4f))
    assert((occupied.node(0L) eq occupiedNode) && (inserted ne occupiedNode))
    occupied.validate()

    // Extents with identical names and different CVAs remain separate.
    val names = new Lattice(cva)
    val first = names.fastAddIntent(sample("word", 0f))
    val second = names.fastAddIntent(sample("word", 4f))
    names.fastAddIntent(sample("unrelated", 0f, 40f))
    assert((first ne second) && names.nodes.count(_.symbols == Set("word")) == 2)
    assert(!names.equivalent(first, second))
    names.validate()

    // Already recorded members of a nonsingleton classifier are complete no-ops.
    val recorded = new Lattice(cva)
    val member = sample("recorded", -4f)
    val line = recorded.fastAddIntent(Vector(member, sample("right", 4f)))
    val model = line.model
    val before = Files.createTempFile("membership-before-", ".bin")
    val after = Files.createTempFile("membership-after-", ".bin")
    try
      recorded.save(before.toString)
      assert(recorded.fastAddIntent(member) eq line)
      recorded.save(after.toString)
      assert(line.model eq model)
      assert(Files.readAllBytes(before).sameElements(Files.readAllBytes(after)),
        "A recorded classified instance must not change models, topology, or the ID allocator")
    finally
      Files.deleteIfExists(before)
      Files.deleteIfExists(after)

    // Local placement and refit passes repair crossing classifiers with a tiny beam.
    val fixture = new Lattice(cva, LatticeConfig(closureSearchVisits = 2,
      closureSearchFrontier = 1, closureSearchBeamWidth = 1)):
      val horizontal = new Node(700L, Vector(sample("h-left", -4f), sample("h-right", 4f)),
        this.cva, do_avg = false)
      val vertical = new Node(3L, Vector(sample("v-down", 0f, -4f), sample("v-up", 0f, 4f)),
        this.cva, do_avg = false)
      nodeMap.put(horizontal.id, horizontal)
      nodeMap.put(vertical.id, vertical)
    val crossing = fixture.cloneLattice()
    val origin = sample("origin", 0f)
    val point = crossing.fastAddIntent(origin)
    assert(crossing.nodes.forall(_.containsSymbol(origin.symbol)))
    assert(point.parents.toSet == Set(700L, 3L))
    crossing.nodes.foreach(fitted(_, cva))
    crossing.validate()
    val unchanged = crossing.nodes.map(node => node -> node.model)
    assert(crossing.fastAddIntent(origin) eq point)
    assert(unchanged.forall((node, saved) => node.model eq saved))

    // Exact extent candidates must be checked even outside every search beam.
    val duplicateFixture = new Lattice(cva, LatticeConfig(closureSearchVisits = 1,
      closureSearchFrontier = 1, closureSearchBeamWidth = 1)):
      (0 until 64).foreach { index =>
        val node = new Node(1000L + index, Vector(sample(s"indexed-$index", index * 4f)),
          this.cva, do_avg = false)
        nodeMap.put(node.id, node)
      }
    val indexed = duplicateFixture.cloneLattice()
    val last = indexed.node(1063L)
    val savedModel = last.model
    assert(indexed.fastAddIntent(sample("indexed-63", 252f)) eq last)
    assert(indexed.size == 64 && (last.model eq savedModel))

    // Public extent changes move only that node's candidate key, and clone/load
    // rebuild the same lookup before any subsequent insertion.
    val extra = sample("indexed-extra", 252f)
    assert(last.addInstances(Vector(extra)))
    val updatedMembers = last.instances.map(member => sample(member.symbol, 252f))
    assert(indexed.fastAddIntent(updatedMembers) eq last)
    assert(indexed.size == 64)
    val clonedIndex = indexed.cloneLattice()
    assert(clonedIndex.fastAddIntent(updatedMembers) eq clonedIndex.node(last.id))
    val indexedCheckpoint = Files.createTempFile("indexed-extent-", ".bin")
    try
      indexed.save(indexedCheckpoint.toString)
      val loadedIndex = new Lattice(cva, indexed.config).load(indexedCheckpoint.toString)
      assert(loadedIndex.fastAddIntent(updatedMembers) eq loadedIndex.node(last.id))
      assert(loadedIndex.size == 64)
    finally Files.deleteIfExists(indexedCheckpoint)
    indexed.validate()

    // A crossing batch inserts the common-vector intersection between both lines.
    val intersections = new Lattice(cva)
    val horizontal = intersections.fastAddIntent(Vector(sample("left", -4f), sample("right", 4f)))
    val common = sample("common", 0f)
    val vertical = intersections.fastAddIntent(Vector(sample("down", 0f, -4f), common,
      sample("up", 0f, 4f)))
    val middle = intersections.nodes.find(_.symbols == Set("common")).get
    assert(middle.parents.toSet == Set(horizontal.id, vertical.id))
    assert(horizontal.containsSymbol(common.symbol))
    intersections.nodes.foreach(fitted(_, cva))
    intersections.validate()

    // Refitting an accepting point into a line discovers an existing child.
    val widening = new Lattice(new CVA(2, CvaConfig(0.6, 1.0)))
    val growing = widening.fastAddIntent(sample("zero", 0f))
    val far = widening.fastAddIntent(sample("four", 4f))
    assert(growing.children.isEmpty && far.parents.isEmpty)
    val widened = widening.fastAddIntent(sample("one", 1f))
    assert(widened eq growing)
    assert(growing.containsSymbol("one") && growing.containsSymbol("four"))
    assert(growing.children.contains(far.id) && far.parents.contains(growing.id))
    fitted(growing, widening.cva)
    widening.validate()

    // Pairwise strict CVA inclusion can be cyclic; validation must detect it.
    val cycle = new Lattice(new CVA(3, CvaConfig(0.01, 1.0))):
      val points = (0 until 3).map { axis =>
        val vector = numeric.BFloatTensors((0 until 3).map(i => if i == axis then 1f else 0f))
        val rejectedAxis = (axis + 2) % 3
        val projection = numeric.BFloatTensors((0 until 9).map(i =>
          if i / 3 == rejectedAxis && i % 3 == rejectedAxis then 1f else 0f)).view(3, 3)
        new Node(100L + axis, Vector(Instance(s"cycle-$axis", vector)), this.cva,
          restoredModel = Some(CvaModel(numeric.BFloatTensors(Seq(0f, 0f, 0f)), projection, 0.01)))
      }
      points.indices.foreach { index =>
        val parent = points(index)
        val child = points((index + 1) % points.size)
        parent.children += child.id
        child.parents += parent.id
      }
      points.foreach(node => nodeMap.put(node.id, node))
    assert(scala.util.Try(cycle.validate()).failed.get.getMessage.contains("Cycle detected"))
    println("CVA membership, no-op insertion, intermediate covers, refits and identity checks passed.")
