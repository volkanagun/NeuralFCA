package simple
import cvafca.{CVA, CvaModel}
import torch.{BFloat16, Tensor}

import java.io.{BufferedInputStream, BufferedOutputStream, DataInputStream, DataOutputStream, FileInputStream, FileOutputStream}
import java.nio.file.{AtomicMoveNotSupportedException, Files, Paths, StandardCopyOption}
import java.util.stream.IntStream
import scala.collection.mutable
import scala.util.{Random, Using}

class Lattice(val dim:Int = 300,  val ratio:Float=0.7, val threshold:Float = 0.6) {

  // All inserted objects
  private val instances =
    mutable.LinkedHashMap.empty[String, Instance]

  // All formal concepts
  private val nodes =
    mutable.ArrayBuffer.empty[Node]

  // Intent -> Concept index
  private val intentIndex =
    mutable.HashMap.empty[torch.Tensor[torch.BFloat16], Node]

  private val extentIndex =
    mutable.HashMap.empty[Int, Node]

  // Recent concepts supplement the graph frontier when the bottom has many parents.
  private val recentBeamConcepts = mutable.ArrayDeque.empty[Node]
  private val beamConceptsByInstance = mutable.HashMap.empty[String, Node]

  private var nextNodeId = 0
  private val cva = new CVA(dim)
  private val bottomModel = cva.emptyModel()

  // Initial lattice contains only the bottom concept:
  // (empty extent, all attributes)
  private val bottomNode =
    createNode(bottomModel, Set.empty[Instance])


  // ==================================================
  // CREATE NODE
  // ==================================================

  private def createNode(instance: Instance): Node = {
    beamConceptsByInstance.get(instance.id) match {
      case Some(existing) => existing
      case None => {
        val model = cva.fit_cva(mutable.Iterable(instance.attributes))
        Node(0, model, instance)
      }
    }
  }

  private def addNode(extendId:Int, node: Node):Node ={
    nextNodeId += 1
    nodes += node
    //intentIndex(node.intent.commonVector) = node
    extentIndex(extendId) = node
    node
  }



  private def createNode(
                          dummy: Node
                        ): Node = {
    val id = dummy.extentID
    if (!extentIndex.contains(id)) {
      val node = new Node(
        nextNodeId,
        dummy.intent,
        dummy.extent
      )
      addNode(id, node)
    }
    else{
      //intentIndex(intent.commonVector)
      extentIndex(id)
    }
  }
  private def createNode(
                          dummyIntent: CvaModel,
                          dummyExtent:Set[Instance]
                        ): Node = {
    val node =new Node(0,dummyIntent, dummyExtent)
    createNode(node)
  }


  // ==================================================
  // GRAPH OPERATIONS
  // ==================================================

  // upper is more general than lower
  private def setLink(
                       upper: Node,
                       lower: Node
                     ): Unit = {

    upper.children += lower
    lower.parents += upper
  }


  private def removeLink(
                          upper: Node,
                          lower: Node
                        ): Unit = {

    upper.children -= lower
    lower.parents -= upper
  }


  // ==================================================
  // FIND CANONICAL GENERATOR
  // ==================================================

  /**
   * Finds the most general existing concept whose
   * intent contains the requested intent.
   *
   * The search moves upward through parent links.
   *
   * Precondition:
   *   intent subsetOf generator.intent
   */
  private[simple] def findCanonicalGenerator(
                                      intent: Node,
                                      generator: Node
                                    ): Node = {

/*    require(
      intent.subsetOf(generator.intent),
      "The initial generator must contain the intent"
    )*/

    var current = generator
    var searching = true

    while (searching) {

      val next = current.parents.find { parent =>
        intent.subset_of(parent, ratio, threshold)
      }

      next match {

        case Some(parent) =>
          current = parent

        case None =>
          searching = false
      }
    }

    current
  }

  /** Search each parent layer concurrently when it is wide enough to pay for
    * parallel dispatch. `findFirst` preserves the serial search's parent order.
    * Only the current layer is read by workers; graph traversal stays sequential.
    */
  private[simple] def findCanonicalGeneratorParallel(
      intent: Node,
      generator: Node
  ): Node = {
    var current = generator
    var searching = true

    while (searching) {
      val next = if current.parents.size < 8 then {
        current.parents.find(parent => intent.subset_of(parent, ratio, threshold))
      } else {
        val parents = current.parents.toVector
        val first = IntStream.range(0, parents.size).parallel()
          .filter(index => intent.subset_of(parents(index), ratio, threshold))
          .findFirst()
        if first.isPresent then Some(parents(first.getAsInt)) else None
      }

      next match {
        case Some(parent) => current = parent
        case None => searching = false
      }
    }

    current
  }


  // ==================================================
  // SELECT IMMEDIATE PARENTS
  // ==================================================

  /**
   * Keeps the least general concepts among candidates.
   *
   * For two intents A and B:
   *
   * A subset B  => concept(A) is above concept(B)
   *
   * Therefore, among candidate parents, we retain
   * maximal intents under set inclusion.
   */
  private def selectImmediateParents(
                                      candidates: Seq[Node]
                                    ): Set[Node] = {

    val unique = candidates.distinct

    unique.filter { candidate =>

      !unique.exists { other =>

        (other ne candidate) &&
          candidate.subset_of(other, ratio, threshold) &&
          candidate.intent != other.intent
      }

    }.toSet
  }


  // ==================================================
  // UPDATE COVERING RELATIONS
  // ==================================================

  /**
   * Inserts a newly created concept between its
   * canonical generator and its new parents.
   *
   * All parent-child relationships are updated locally.
   */
  private def updateCoveringRelations(
                                       newNode: Node,
                                       generator: Node,
                                       parents: Set[Node]
                                     ): Unit = {

    // Connect the new concept to its immediate parents.
    parents.foreach { parent =>

      // The old edge is no longer an immediate cover
      // if it connected this parent to the generator.
      removeLink(parent, generator)

      setLink(parent, newNode)
    }

    // The new concept lies immediately above generator.
    setLink(newNode, generator)
  }


  // ==================================================
  // FAST ADD INTENT
  // ==================================================

  /**
   * Recursive intent insertion.
   *
   * Returns the concept having exactly the specified
   * intent. Creates it if necessary.
   *
   * This is an optimized AddIntent-style procedure.
   */
  private def fastAddIntent(
                             intent: Node,
                             generator: Node
                           ): Node = {


   /* require(
      intent.subsetOf(attributes),
      "Unknown attributes"
    )*/

    /*// O(1) expected indexed lookup for an existing intent.
    intentIndex.get(intent) match {

      case Some(existing) =>
        return existing

      case None =>
    }*/

    // Step 1: locate canonical generator.
    val canonicalGenerator =
      findCanonicalGeneratorParallel(intent, generator)

    // Step 2: existing exact concept?
    if (canonicalGenerator == intent) {
      return canonicalGenerator
    }

    // Step 3: examine parents of the generator.
    // Snapshot prevents concurrent graph modifications
    // from changing the iteration.
    val generatorParents =
      canonicalGenerator.parents.toVector

    val parentCandidates =
      mutable.ArrayBuffer.empty[Node]

    for (candidate <- generatorParents) {

      val parentNode =

        if (candidate.subset_of(intent, ratio, threshold)) {

          // Existing parent can be reused.
          candidate

        } else {

          // Recursively insert their common intent.
          val intersection =
            candidate.intersect(intent)

          fastAddIntent(intersection, candidate)
        }

      parentCandidates += parentNode
    }

    // Step 4: remove non-immediate parent candidates.
    val newParents =
      selectImmediateParents(parentCandidates.toVector)

    // Another recursive branch might have created this
    // intent, so check the index again.
    intentIndex.get(intent.common) match {

      case Some(existing) =>
        return existing

      case None =>
    }

    // Step 5: create a new concept.
    //
    // Its old extent is initially the extent of its
    // canonical generator.
    val newNode = createNode(
      intent.intent,
      canonicalGenerator.extent
    )

    // Step 6: update only affected lattice edges.
    updateCoveringRelations(
      newNode,
      canonicalGenerator,
      newParents
    )

    newNode
  }


  // ==================================================
  // BOUNDED BEAM INSERT
  // ==================================================

  /** Score a bounded graph frontier, expanding only the closest concepts. */
  private def beamCandidates(
      instance: Instance,
      beamWidth: Int,
      maxVisits: Int
  ): Vector[(Node, Double)] = {
    val visited = mutable.HashSet.empty[Int]
    val scored = mutable.ArrayBuffer.empty[(Node, Double)]
    val seedLimit = math.min(maxVisits.toLong, beamWidth.toLong * 4L).toInt
    val recent = recentBeamConcepts.reverseIterator.take(math.min(beamWidth, seedLimit)).toVector
    var frontier = (recent.iterator ++ bottomNode.parents.iterator
      .take(seedLimit - recent.size)).toVector.distinctBy(_.id)

    while (frontier.nonEmpty && scored.size < maxVisits) {
      val layer = frontier.filter(node => visited.add(node.id))
        .take(maxVisits - scored.size)
      val ranked = layer.map { node =>
        val distance = node.intent.distance(instance.attributes)
        (node, if (distance.isNaN) Double.PositiveInfinity else distance)
      }
      scored ++= ranked
      frontier = ranked.sortBy(_._2).take(beamWidth).iterator
        .flatMap(_._1.parents.iterator)
        .filterNot(node => (node eq bottomNode) || visited.contains(node.id))
        .take(maxVisits - scored.size).toVector.distinctBy(_.id)
    }

    scored.toVector
  }


  /** Keep the closest qualifying candidates, excluding comparable extents so
    * each selected parent contributes a different branch of the lattice.
    */
  private def beamParents(
      candidates: Vector[(Node, Double)],
      maxParents: Int
  ): Vector[Node] = {
    val ranked = candidates.filter(_._2 <= threshold.toDouble)
      .sortBy { case (node, distance) => (distance, node.id) }
    val selected = mutable.ArrayBuffer.empty[(Node, Set[String])]
    val iterator = ranked.iterator

    while (iterator.hasNext && selected.size < maxParents) {
      val candidate = iterator.next()._1
      val extentIds = candidate.extent.iterator.map(_.id).toSet
      if (!selected.exists { case (_, ids) =>
        extentIds.subsetOf(ids) || ids.subsetOf(extentIds)
      }) selected += ((candidate, extentIds))
    }

    selected.iterator.map(_._1).toVector
  }

  /** Approximate insertion: one object and up to maxParents common concepts.
    * A repeated instance ID reuses its existing object concept.
    */
  def beamInsert(
      instance: Instance,
      beamWidth: Int = 64,
      maxVisits: Int = 1024,
      maxParents: Int = 4
  ): Node = this.synchronized {
    require(maxParents > 0, "maxParents must be positive")

   /*
      beamConceptsByInstance.get(instance.id) match {
      case Some(existing) => return existing
      case None =>
    }
    */

    val parents = beamParents(beamCandidates(instance, beamWidth, maxVisits), maxParents)

    val proposed = createNode(instance)
    val objectConcept = createNode(proposed)
    parents.filter(parent=> parent.extentID!=proposed.extentID).foreach { upper =>
      // The common intent has the union of both extents, so it lies above
      // the existing concept and the new object concept.
      val intersection = upper.intersect(proposed)
      val commonConcept = createNode(intersection)
      setLink(commonConcept, upper)
      setLink(commonConcept, objectConcept)
    }
    setLink(objectConcept, bottomNode)

    instances(instance.id) = instance
    beamConceptsByInstance(instance.id) = objectConcept

    recentBeamConcepts.append(objectConcept)
    if (recentBeamConcepts.size > 32) recentBeamConcepts.removeHead()
    objectConcept
  }


  // ==================================================
  // ADD INSTANCE
  // ==================================================

  /**
   * Incrementally inserts an object using the bounded beam path.
   */
  def addInstance(instance: Instance): Node = beamInsert(instance)

  /** Retains the original recursive FastAddIntent insertion path. */
  def addInstanceWithFastAddIntent(instance: Instance): Node = {

/*    require(
      !instances.contains(instance.id),
      s"Duplicate instance: ${instance.id}"
    )

    require(
      instance.attributes.subsetOf(attributes),
      "Instance contains unknown attributes"
    )*/

    // Insert structural concept before changing extents.
    val node = createNode(instance)
    val objectConcept =
      fastAddIntent(node, bottomNode)

    instances(instance.id) = instance

    // Add the object to the extent of its object concept
    // and to all of its ancestors.
    val visited =
      mutable.HashSet.empty[Node]

    val stack =
      mutable.Stack[Node](objectConcept)

    while (stack.nonEmpty) {

      val node = stack.pop()

      if (visited.add(node)) {

        node.extent = node.extent + instance

        node.parents.foreach { parent =>
          stack.push(parent)
        }
      }
    }

    beamConceptsByInstance(instance.id) = objectConcept
    objectConcept
  }


  // ==================================================
  // ACCESSORS
  // ==================================================

  def getNodes: Seq[Node] =
    nodes.toVector

  def nodeSize:Int =
    nodes.size

  def getInstances: Seq[Instance] =
    instances.values.toVector

  def findNode(
                intent: Tensor[BFloat16]
              ): Option[Node] =
    intentIndex.get(intent)

  def top: Node =
    nodes.find(_.parents.isEmpty).get

  def bottom: Node =
    bottomNode

  /** Save the exact graph and fitted CVA models so beam insertion can resume. */
  def save(filename: String): Lattice = this.synchronized {
    val target = Paths.get(filename).toAbsolutePath
    Files.createDirectories(target.getParent)
    val temporary = Files.createTempFile(target.getParent, ".simple-lattice-", ".tmp")
    try {
      Using.resource(new FileOutputStream(temporary.toFile)) { file =>
        Using.resource(new DataOutputStream(new BufferedOutputStream(file))) { out =>
          out.writeInt(Lattice.FileMagic)
          out.writeInt(Lattice.FileVersion)
          out.writeInt(dim)
          out.writeFloat(ratio)
          out.writeFloat(threshold)
          out.writeInt(nextNodeId)

          out.writeInt(instances.size)
          instances.valuesIterator.foreach { instance =>
            out.writeUTF(instance.id)
            writeVector(out, instance.attributes)
          }

          out.writeInt(nodes.size)
          nodes.foreach { node =>
            out.writeInt(node.id)
            writeVector(out, node.intent.commonVector)
            writeProjection(out, node.intent.projection)
            out.writeDouble(node.intent.radius)
            out.writeInt(node.extent.size)
            node.extent.foreach(member => out.writeUTF(member.id))
            out.writeInt(node.children.size)
            node.children.foreach(child => out.writeInt(child.id))
          }

          out.writeInt(recentBeamConcepts.size)
          recentBeamConcepts.foreach(node => out.writeInt(node.id))
          out.writeInt(beamConceptsByInstance.size)
          beamConceptsByInstance.foreach { (id, node) =>
            out.writeUTF(id)
            out.writeInt(node.id)
          }
          out.writeInt(extentIndex.size)
          extentIndex.foreach { (extentId, node) =>
            out.writeInt(extentId)
            out.writeInt(node.id)
          }
          out.flush()
          file.getFD.sync()
        }
      }
      try Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE,
        StandardCopyOption.REPLACE_EXISTING)
      catch {
        case _: AtomicMoveNotSupportedException =>
          Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING)
      }
    } finally Files.deleteIfExists(temporary)
    this
  }

  /** Replace this lattice with a compatible checkpoint. A failed read leaves it unchanged. */
  def load(filename: String): Lattice = this.synchronized {
    val restored = Using.resource(new DataInputStream(new BufferedInputStream(new FileInputStream(filename)))) { in =>
      require(in.readInt() == Lattice.FileMagic, "Not a simple lattice checkpoint")
      val version = in.readInt()
      require(version >= 1 && version <= Lattice.FileVersion,
        "Unsupported simple lattice checkpoint version")
      require(in.readInt() == dim, "Checkpoint dimension does not match lattice dimension")
      in.readFloat() // Saved ratio; the active configuration may differ.
      in.readFloat() // Saved threshold.

      val savedNextId = readCount(in, "next node ID")

      val savedInstances = mutable.LinkedHashMap.empty[String, Instance]
      val instanceCount = readCount(in, "instance count")
      val instanceProgress = loadingProgress("instances", instanceCount)
      instanceProgress(0)
      for (index <- 0 until instanceCount) {
        val id = in.readUTF()
        val instance = new Instance(id, readVector(in))
        require(!savedInstances.contains(id), s"Duplicate instance ID: $id")
        savedInstances(id) = instance
        instanceProgress(index + 1)
      }

      val savedNodes = mutable.ArrayBuffer.empty[Node]
      val byId = mutable.HashMap.empty[Int, Node]
      val edges = mutable.ArrayBuffer.empty[(Int, Vector[Int])]
      val nodeCount = readCount(in, "node count")
      require(nodeCount > 0, "Checkpoint has no bottom node")
      val nodeProgress = loadingProgress("concepts", nodeCount)
      nodeProgress(0)
      for (index <- 0 until nodeCount) {
        val id = readCount(in, "node ID")
        require(!byId.contains(id), s"Duplicate node ID: $id")
        val common = readVector(in)
        val projection = readProjection(in)
        val radius = in.readDouble()
        require(!radius.isNaN && radius >= 0.0, s"Invalid radius for node $id")
        val extentCount = readCount(in, "extent size")
        val memberIds = Vector.fill(extentCount)(in.readUTF())
        require(memberIds.distinct.size == memberIds.size, s"Duplicate extent member in node $id")
        val extent = memberIds.map(memberId => savedInstances.getOrElse(memberId,
          throw new IllegalArgumentException(s"Unknown extent member: $memberId"))).toSet
        val node = if id == bottomNode.id then bottomNode
          else new Node(id, cvafca.CvaModel(common, projection, radius), extent)
        if id == bottomNode.id then {
          // Commit the bottom model only after the complete file has been checked.
          savedNodes += new Node(id, cvafca.CvaModel(common, projection, radius), extent)
        } else savedNodes += node
        byId(id) = node
        val childCount = readCount(in, "child count")
        edges += ((id, Vector.fill(childCount)(readCount(in, "child ID"))))
        nodeProgress(index + 1)
      }
      require(byId.contains(bottomNode.id), "Checkpoint is missing its bottom node")
      require(savedNextId > byId.keysIterator.max, "Invalid next node ID")
      edges.foreach { (id, children) =>
        require(children.distinct.size == children.size, s"Duplicate child link for node $id")
        children.foreach { childId =>
          require(childId != id && byId.contains(childId), s"Invalid child $childId of node $id")
        }
      }

      val recentCount = readCount(in, "recent concept count")
      val recentIds = Vector.fill(recentCount)(readCount(in, "recent concept ID"))
      recentIds.foreach(id => require(byId.contains(id), s"Unknown recent concept $id"))
      val mappingCount = readCount(in, "instance-concept mapping count")
      val mappings = Vector.fill(mappingCount)((in.readUTF(), readCount(in, "object concept ID")))
      require(mappings.map(_._1).distinct.size == mappings.size, "Duplicate object concept mapping")
      mappings.foreach { (id, nodeId) =>
        require(savedInstances.contains(id) && byId.contains(nodeId),
          s"Invalid object concept mapping for $id")
      }
      val savedExtentIndex = if version >= 2 then {
        val indexCount = readCount(in, "extent index count")
        require(indexCount <= nodeCount, "Extent index has more entries than concepts")
        val keys = mutable.HashSet.empty[Int]
        val indexProgress = loadingProgress("extent index", indexCount)
        indexProgress(0)
        Vector.tabulate(indexCount) { index =>
          val extentId = in.readInt()
          val nodeId = readCount(in, "extent index node ID")
          require(keys.add(extentId), s"Duplicate extent index key: $extentId")
          require(byId.contains(nodeId), s"Unknown extent index node: $nodeId")
          indexProgress(index + 1)
          (extentId, nodeId)
        }
      } else {
        // Version 1 did not store this index; derive it from the saved extents.
        savedNodes.iterator.map(node => (node.extentID, node.id)).toVector
      }
      require(in.read() == -1, "Trailing data in simple lattice checkpoint")
      (savedInstances, savedNodes, byId, edges, recentIds, mappings,
        savedExtentIndex, savedNextId)
    }

    val (savedInstances, savedNodes, byId, edges, recentIds, mappings,
      savedExtentIndex, savedNextId) = restored
    val savedBottom = savedNodes.find(_.id == bottomNode.id).get
    bottomNode.intent = savedBottom.intent
    bottomNode.extent = savedBottom.extent
    nodes.clear()
    savedNodes.foreach(node => nodes += (if node.id == bottomNode.id then bottomNode else node))
    nodes.foreach { node =>
      node.parents.clear()
      node.children.clear()
    }
    val linkProgress = loadingProgress("linking concepts", edges.size)
    linkProgress(0)
    edges.iterator.zipWithIndex.foreach { case ((id, children), index) =>
      children.foreach(childId => setLink(byId(id), byId(childId)))
      linkProgress(index + 1)
    }
    instances.clear()
    instances.addAll(savedInstances)
    intentIndex.clear()
    nodes.foreach(node => intentIndex(node.intent.commonVector) = node)
    extentIndex.clear()
    savedExtentIndex.foreach { (extentId, nodeId) => extentIndex(extentId) = byId(nodeId) }
    recentBeamConcepts.clear()
    recentIds.foreach(id => recentBeamConcepts.append(byId(id)))
    beamConceptsByInstance.clear()
    mappings.foreach { (id, nodeId) => beamConceptsByInstance(id) = byId(nodeId) }
    nextNodeId = savedNextId
    println(s"Loaded lattice checkpoint: ${instances.size} instances, ${nodes.size} concepts")
    this
  }

  /** Show completed counts and periodic updates without printing for every node. */
  private def loadingProgress(stage: String, total: Int): Int => Unit = {
    var lastPrintedAt = 0L
    completed => {
      val now = System.nanoTime()
      if (completed == 0 || completed == total || now - lastPrintedAt >= 2000000000L) {
        val percentage = if total == 0 then 100 else completed.toLong * 100 / total
        println(s"Loading lattice $stage: $completed/$total ($percentage%)")
        lastPrintedAt = now
      }
    }
  }

  private def readCount(in: DataInputStream, name: String): Int = {
    val value = in.readInt()
    require(value >= 0, s"Invalid $name: $value")
    value
  }

  private def writeVector(out: DataOutputStream, tensor: Tensor[BFloat16]): Unit = {
    val shape = tensor.shape
    require(shape.nonEmpty && shape.foldLeft(1L)((size, axis) => size * axis) == dim.toLong,
      "Vector shape does not match lattice dimension")
    out.writeInt(shape.length)
    shape.foreach(out.writeInt)
    numeric.BFloatTensors.toArray(tensor).foreach { value =>
      out.writeShort(java.lang.Float.floatToRawIntBits(value) >>> 16)
    }
  }

  private def readVector(in: DataInputStream): Tensor[BFloat16] = {
    val rank = readCount(in, "vector rank")
    require(rank > 0 && rank <= 8, s"Invalid vector rank: $rank")
    val shape = Array.fill(rank)(readCount(in, "vector dimension"))
    require(shape.foldLeft(1L)((size, axis) => size * axis) == dim.toLong,
      "Checkpoint vector shape does not match lattice dimension")
    val values = Array.fill(dim)(java.lang.Float.intBitsToFloat(in.readUnsignedShort() << 16))
    numeric.BFloatTensors(values.toIndexedSeq).view(shape *)
  }

  private def writeProjection(out: DataOutputStream, tensor: Tensor[BFloat16]): Unit = {
    require(tensor.shape.sameElements(Array(dim, dim)), "Invalid CVA projection shape")
    val values = numeric.BFloatTensors.toArray(tensor)
    val identity = values.indices.forall(i => values(i) == (if i / dim == i % dim then 1f else 0f))
    out.writeBoolean(identity)
    if !identity then values.foreach(value => out.writeShort(java.lang.Float.floatToRawIntBits(value) >>> 16))
  }

  private def readProjection(in: DataInputStream): Tensor[BFloat16] = {
    if in.readBoolean() then bottomModel.projection
    else {
      val values = Array.fill(Math.multiplyExact(dim, dim))(
        java.lang.Float.intBitsToFloat(in.readUnsignedShort() << 16))
      numeric.BFloatTensors(values.toIndexedSeq).view(dim, dim)
    }
  }


  // ==================================================
  // PRINT CONCEPTS
  // ==================================================

  def printConcepts(size:Int = 3, limit:Int = 4): Unit = {

    Random.shuffle(nodes.filter(node=> node.size >= size))
      .take(limit)
      .foreach(println)
  }


  // ==================================================
  // PRINT EDGES
  // ==================================================

  def printEdges(): Unit = {

    println("\nCOVERING RELATIONS")

    nodes.sortBy(_.id).foreach { node =>

      node.children.toSeq.sortBy(_.id).foreach { child =>

        println(
          s"C${node.id} -> C${child.id}"
        )
      }
    }
  }


  def printLattice(): Unit = {

    printConcepts()
    printEdges()
  }


  // ==================================================
  // VALIDATE FORMAL CONCEPTS AND LATTICE EDGES
  // ==================================================

  def validate(): Boolean = {

    val allNodes = nodes.toVector

    // Every concept must satisfy A' = B and B' = A.
    val conceptsValid = true
    // Check immediate covering relations.
    val edgesValid = allNodes.forall { upper =>

      upper.children.forall { lower =>

        val ordered =
          upper.subset_of(lower, ratio, threshold) &&
            upper.intent != lower.intent

        val reciprocal =
          lower.parents.contains(upper)

        val intermediateExists =
          allNodes.exists { middle =>

            (middle ne upper) &&
              (middle ne lower) &&
            upper.subset_of(middle, ratio, threshold) &&
              upper.intent != middle.intent &&
            middle.subset_of(lower, ratio, threshold) &&
              middle.intent != lower.intent
          }

        ordered && reciprocal && !intermediateExists
      }
    }

    // Every strict comparable pair must have a path.
    val reachabilityValid = allNodes.forall { upper =>

      val reachable =
        mutable.HashSet.empty[Node]

      val stack =
        mutable.Stack.empty[Node]

      upper.children.foreach(n => stack.push(n))

      while (stack.nonEmpty) {

        val current = stack.pop()

        if (reachable.add(current)) {
          current.children.foreach(n => stack.push(n))
        }
      }

      allNodes.forall { lower =>

        if (
          upper.subset_of(lower, ratio, threshold) &&
            upper.intent != lower.intent
        ) {
          reachable.contains(lower)
        } else {
          true
        }
      }
    }

    val reverseEdgesValid = allNodes.forall { lower =>
      lower.parents.forall(_.children.contains(lower))
    }

    conceptsValid &&
      edgesValid &&
      reachabilityValid &&
      reverseEdgesValid
  }
}

private object Lattice {
  val FileMagic = 0x534C4154 // "SLAT"
  val FileVersion = 2
}
