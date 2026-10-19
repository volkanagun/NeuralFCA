package cvafca

/** Run with cvafca.LatticeExtentIndexTest. */
object LatticeExtentIndexTest:
  def main(args: Array[String]): Unit =
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    val cva = new CVA(1, CvaConfig(0.01, 1.0, noCVA = true))
    val zero = numeric.BFloatTensors(Seq(0f))
    val model = cva.fit_avg(Vector(zero))

    // Equal cached hashes must still compare complete symbol sets.
    assert(Set("Aa").hashCode() == Set("BB").hashCode())
    val collisionFixture = new Lattice(cva):
      nodeMap.put(100L, new Node(100L, Vector(Instance("Aa", zero)), this.cva,
        restoredModel = Some(model)))
      nodeMap.put(101L, new Node(101L, Vector(Instance("BB", zero)), this.cva,
        restoredModel = Some(model)))
    val collisions = collisionFixture.cloneLattice()
    assert(collisions.fastAddIntent(collisions.node(101L).instances).id == 101L)
    assert(collisions.fastAddIntent(collisions.node(100L).instances).id == 100L)
    collisions.validate()

    // Enough immutable snapshots to exercise parallel hashing. Descending IDs
    // and duplicate extents make completion-order publication observable.
    val count = 4096
    val width = 256
    val members = (0 until count + width).map(i => Instance(s"member-$i", zero)).toVector
    val fixture = new Lattice(cva):
      for order <- 0 until count do
        val start = if order == count - 1 then 0 else order
        val node = new Node(10000L - order, members.slice(start, start + width), this.cva,
          restoredModel = Some(model))
        nodeMap.put(node.id, node)
    val indexed = fixture.cloneLattice()
    assert(indexed.fastAddIntent(members.take(width)).id == 10000L,
      "Duplicate extents must select nodeMap's first node, regardless of ID")
    val lastDistinct = indexed.node(10000L - count + 2)
    assert(indexed.fastAddIntent(lastDistinct.instances).id == lastDistinct.id)
    assert(indexed.size == count)

    // Rekeying must remove the old immutable key and publish the new snapshot.
    lastDistinct.addInstance(Instance("appended", zero))
    lastDistinct.refit()
    assert(indexed.fastAddIntent(lastDistinct.instances).id == lastDistinct.id)
    assert(indexed.size == count)

    // Two extents converging in reverse update order retain insertion priority.
    val duplicate = indexed.node(10000L - count + 1)
    val extra = Instance("common-extra", zero)
    duplicate.addInstance(extra)
    indexed.node(10000L).addInstance(extra)
    assert(indexed.fastAddIntent(duplicate.instances).id == 10000L)
    indexed.validate()

    // A warmed index must track later stores as well as repeated extent edits.
    // All models classify zero, so an incorrect/missing exact key would silently
    // select the earliest equivalent node instead of the requested node.
    class EditableLattice extends Lattice(cva):
      def store(id: Long, symbols: String*): Node =
        val node = new Node(id, symbols.map(Instance(_, zero)), this.cva,
          restoredModel = Some(model))
        nodeMap.put(id, node)
        node
      def replace(id: Long, symbols: String*): Node =
        val node = new Node(id, symbols.map(Instance(_, zero)), this.cva,
          restoredModel = Some(model))
        nodeMap.update(id, node)
        node
      def drop(id: Long): Unit = { nodeMap.remove(id); () }

    val editable = new EditableLattice
    val early = editable.store(1000L, "early")
    val edited = editable.store(1001L, "edited")
    assert(editable.fastAddIntent(early.instances).id == early.id) // Initial build.
    val added = editable.store(1002L, "added")
    for i <- 0 until 64 do edited.addInstance(Instance(s"edit-$i", zero))
    assert(editable.fastAddIntent(edited.instances).id == edited.id)
    assert(editable.fastAddIntent(added.instances).id == added.id)

    // Replacement keeps its position even when the map's size is unchanged.
    val replacement = editable.replace(edited.id, "replacement")
    assert(editable.fastAddIntent(replacement.instances) eq replacement)
    edited.addInstance(Instance("detached-old-node", zero))
    assert(editable.fastAddIntent(replacement.instances) eq replacement)

    // Removing and readding a key gives it a new insertion priority.
    val later = editable.store(1003L, "replacement")
    assert(editable.fastAddIntent(later.instances).id == replacement.id)
    editable.drop(replacement.id)
    val readded = editable.store(replacement.id, "replacement")
    assert(editable.fastAddIntent(readded.instances).id == later.id)
    editable.cloneLattice().validate()
    println("Extent index collisions, incremental updates, replacement and canonical order checks passed.")
