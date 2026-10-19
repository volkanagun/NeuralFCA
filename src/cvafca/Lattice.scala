package cvafca

import scala.collection.mutable
import java.io.{BufferedInputStream, BufferedOutputStream, DataInputStream, DataOutputStream, EOFException, FileInputStream, FileOutputStream}
import java.nio.ByteBuffer
import java.nio.file.{AtomicMoveNotSupportedException, Files, Paths, StandardCopyOption}
import java.util.concurrent.{ForkJoinPool, ForkJoinWorkerThread}
import java.util.concurrent.atomic.AtomicLong
import org.bytedeco.javacpp.PointerScope

import scala.util.Using
import torch.{BFloat16, Device, Tensor}

import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.collection.parallel.ForkJoinTaskSupport


final case class LatticeConfig(
                                /** Fraction of the other node's instances that must be classified. */
                                equalityCoverage: Double = 0.80,

                                /** If true, also require mutual common-vector classification. */
                                requireMutualCommonClassification: Boolean = false,

                                /** Maximum models scored by one learned closure search. */
                                closureSearchVisits: Int = 1024,

                                /** Maximum candidates sampled from a frontier or neighboring layer. */
                                closureSearchFrontier: Int = 512,

                                /** Closest candidates retained for graph expansion in each layer. */
                                closureSearchBeamWidth: Int = 16,

                                /** Score independent closure-search probes on workers when worthwhile. */
                                closureSearchParallel: Boolean = true
                              )

/**
 * Incremental FCA-like lattice whose intents are CVA common vectors.
 *
 * Semantics
 * ---------
 * 1. node.classifies(x) is the subset/membership test.
 * 2. Intent equivalence requires mutual extent coverage, optionally also common vectors.
 * 3. `fastAddIntent` fits intersections to samples classified by both nodes.
 * 4. Adding an instance to an existing node refits its common vector and projection.
 * 5. `fastAddIntent` uses CVA membership and guards links
 *    against graph cycles, since learned inclusion can be non-transitive.
 *
 * This is therefore an approximate / learned FCA, not a mathematically exact
 * formal concept lattice unless the learned operators happen to satisfy closure,
 * antisymmetry and transitivity.
 */
class Lattice(
               val cva: CVA,
               val config: LatticeConfig = LatticeConfig(),
               val c_device: Device = Device.CPU
             ):

  require(config.equalityCoverage > 0.0 && config.equalityCoverage <= 1.0)
  require(config.closureSearchVisits > 0)
  require(config.closureSearchFrontier > 0)
  require(config.closureSearchBeamWidth > 0)
  private val nextId = new AtomicLong(0L)
  /** Protected so alternative insertion engines can reuse the same lattice state. */
  protected val nodeMap: mutable.Map[Long, Node] = new NodeStore

  /** Preserve insertion order while tracking membership changes in one place,
    * including additions by alternative insertion engines and checkpoint loads.
    */
  private final class NodeStore extends mutable.AbstractMap[Long, Node]:
    private val entries = mutable.LinkedHashMap.empty[Long, Node]
    override def get(id: Long): Option[Node] = entries.get(id)
    override def iterator: Iterator[(Long, Node)] = entries.iterator
    override def valuesIterator: Iterator[Node] = entries.valuesIterator
    override def keysIterator: Iterator[Long] = entries.keysIterator
    override def size: Int = entries.size
    override def knownSize: Int = entries.size
    override def put(id: Long, node: Node): Option[Node] =
      val previous = entries.put(id, node)
      if !previous.exists(_ eq node) then
        trackStoredNode(node)
        if node.parents.isEmpty then topNodeIds += id else topNodeIds -= id
        if node.children.isEmpty then bottomNodeIds += id else bottomNodeIds -= id
      previous
    override def update(id: Long, node: Node): Unit = { put(id, node); () }
    override def addOne(entry: (Long, Node)): this.type =
      put(entry._1, entry._2)
      this
    override def remove(id: Long): Option[Node] =
      val previous = entries.remove(id)
      if previous.nonEmpty then
        removeIndexedNode(id)
        topNodeIds -= id
        bottomNodeIds -= id
      previous
    override def subtractOne(id: Long): this.type =
      remove(id)
      this
    override def clear(): Unit =
      invalidateExtentIndexes()
      entries.clear()
      topNodeIds.clear()
      bottomNodeIds.clear()
  /** Childless-node IDs with constant-time insertion/removal and bounded iteration. */
  private val bottomNodeIds = mutable.LinkedHashSet.empty[Long]
  /** Parentless-node IDs, maintained with the bottom frontier for closure search. */
  private val topNodeIds = mutable.LinkedHashSet.empty[Long]
  // FastAddExtent-style exact lookup. Keys are immutable symbol snapshots;
  // tree keys preserve nodeMap order even when two mutable extents converge.
  private final class ExtentKey(val symbols: Set[String]):
    // Scala set hashes traverse their members. Reuse the hash when a snapshot
    // is looked up, inserted, or removed; equality still checks the full set.
    private val cachedHash = symbols.hashCode()
    override def hashCode(): Int = cachedHash
    override def equals(other: Any): Boolean = other match
      case key: ExtentKey => (symbols eq key.symbols) || symbols == key.symbols
      case _ => false

  private val extentIndex = mutable.HashMap.empty[ExtentKey, mutable.TreeMap[Long, Node]]
  private final class IndexedNode(val order: Long, var node: Node, var key: Option[ExtentKey] = None)
  private val indexedNodes = mutable.LongMap.empty[IndexedNode]
  private val dirtyIndexIds = mutable.LinkedHashSet.empty[Long]
  private var indexesReady = false
  private var nextIndexOrder = 0L

  private def markIndexDirty(id: Long): Unit =
    dirtyIndexIds += id

  private def trackStoredNode(node: Node): Unit =
    indexedNodes.get(node.id) match
      case Some(entry) =>
        entry.node.extentChanged = None
        entry.node = node
      case None =>
        indexedNodes.put(node.id, new IndexedNode(nextIndexOrder, node))
        nextIndexOrder += 1
    node.extentChanged = Some(() => markIndexDirty(node.id))
    markIndexDirty(node.id)

  private def removeExtentKey(key: ExtentKey, order: Long): Unit =
    extentIndex.get(key).foreach { bucket =>
      bucket.remove(order)
      if bucket.isEmpty then extentIndex.remove(key)
    }

  private def removeIndexedNode(id: Long): Unit =
    indexedNodes.remove(id).foreach { entry =>
      entry.node.extentChanged = None
      entry.key.foreach(removeExtentKey(_, entry.order))
    }
    dirtyIndexIds -= id

  private def indexExtent(entry: IndexedNode, key: ExtentKey): Unit =
    entry.key.filterNot(_ eq key).foreach(removeExtentKey(_, entry.order))
    extentIndex.getOrElseUpdate(key, mutable.TreeMap.empty).put(entry.order, entry.node)
    entry.key = Some(key)

  private def invalidateExtentIndexes(): Unit =
    indexedNodes.valuesIterator.foreach(_.node.extentChanged = None)
    extentIndex.clear()
    indexedNodes.clear()
    dirtyIndexIds.clear()
    indexesReady = false
    nextIndexOrder = 0L

  private def ensureExtentIndexes(): Unit =
    if dirtyIndexIds.nonEmpty then
      // Every store/mutation queues an ID once. Do not scan or rehash unchanged
      // nodes, even when additions changed nodeMap.size since the last lookup.
      val snapshot = dirtyIndexIds.iterator.map(indexedNodes).toArray
      if !indexesReady then extentIndex.sizeHint(snapshot.length)
      // Only immutable symbol hashing runs on workers. Publish index changes
      // sequentially; stable order keys retain the nodeMap insertion priority.
      val totalSymbols = snapshot.iterator.map(_.node.size.toLong).sum
      def currentKey(entry: IndexedNode): ExtentKey =
        entry.key.filter(key => key.symbols eq entry.node.symbols)
          .getOrElse(new ExtentKey(entry.node.symbols))
      val keys =
        if snapshot.length >= 4096 && totalSymbols >= 1048576L then
          val parallelSnapshot = snapshot.par
          parallelSnapshot.tasksupport = Lattice.parallelIndexTaskSupport
          parallelSnapshot.map(currentKey).toArray
        else snapshot.map(currentKey)
      snapshot.indices.foreach { index =>
        val entry = snapshot(index)
        indexExtent(entry, keys(index))
        dirtyIndexIds -= entry.node.id
      }
    indexesReady = true

  private def exactExtent(symbols: Set[String]): Option[Node] =
    if indexesReady then ensureExtentIndexes()
    if indexesReady then extentIndex.get(new ExtentKey(symbols)).flatMap(_.headOption.map(_._2))
    else nodeMap.valuesIterator.find(existing =>
      existing.size == symbols.size && symbols.forall(existing.containsSymbol))

  private def rebuildFrontierIndexes(): Unit =
    bottomNodeIds.clear()
    topNodeIds.clear()
    nodeMap.valuesIterator.foreach { node =>
      if node.children.isEmpty then bottomNodeIds += node.id
      if node.parents.isEmpty then topNodeIds += node.id
    }

  protected def unlink(parent: Node, child: Node): Unit =
    parent.children.remove(child.id)
    child.parents.remove(parent.id)
    if parent.children.isEmpty then bottomNodeIds += parent.id
    if child.parents.isEmpty then topNodeIds += child.id

  /**
   * Returns an immutable snapshot of the nodes in deterministic insertion order.
   * The vector is new, but its elements are the live mutable Node objects.
   */
  def nodes: Vector[Node] = nodeMap.values.toVector

  /** Returns the number of nodes currently materialized in the lattice. */
  def size = nodeMap.size

  /** Create an independently mutable structural clone.
    *
    * Node containers, extents, and edges are copied. Instance vectors and model
    * tensors are shared as read-only values; subsequent refits replace a clone's
    * model rather than modifying those tensors in place.
    */
  def cloneLattice(): Lattice = this.synchronized {
    val cloned = new Lattice(cva, config, c_device)
    nodeMap.foreach { case (id, source) =>
      val copied = new Node(
        id,
        source.instances,
        cva,
        restoredModel = Some(source.model)
      )
      copied.parents.addAll(source.parents)
      copied.children.addAll(source.children)
      cloned.nodeMap.put(id, copied)
    }
    cloned.rebuildFrontierIndexes()
    cloned.nextId.set(nextId.get())
    cloned
  }

  /**
   * Returns the live node having `id`.
   * Throws NoSuchElementException when the ID is not present in `nodeMap`.
   */
  def node(id: Long): Node = nodeMap(id)

  /** Updates both directions of an edge and the indexed frontiers. */
  private def linkFast(parent: Node, child: Node): Unit =
    if parent.id != child.id then
      parent.children += child.id
      child.parents += parent.id
      bottomNodeIds -= parent.id
      topNodeIds -= child.id

  /**
   * Collects `start` and all its ancestors by following parent IDs.
   * The deque provides depth-first-like traversal and the hash set prevents
   * repeated visits/cycles. Search budgets must not truncate extent propagation:
   * every parent of a growing concept must receive the incoming instance.
   */
  def ancestorClosure(start: Node): Set[Long] =
    val pending = mutable.ArrayDeque(start)
    val closure = mutable.HashSet.empty[Long]
    while pending.nonEmpty do
      val concept = pending.removeLast()
      if closure.add(concept.id) then
        concept.parents.iterator.flatMap(nodeMap.get).foreach(pending.append)
    closure.toSet

  /** FastAddIntent with CVA operations in place of Boolean intent operations.
    *
    * Based on Zou, Zhang and Long (2015), doi:10.1016/j.eswa.2015.01.044.
    * The order is dualized because this graph stores larger extents as parents:
    * closure search descends from a virtual universal extent, and recursive
    * intersections become children. Empty intersections remain virtual.
    *
    * Membership is Node.classifies (distance <= the model's classification
    * radius). Inclusion requires classification of every extent member;
    * equivalence requires equalityCoverage in BOTH directions. Symbol sets are
    * only lookup/cache keys, never membership or equivalence tests.
    *
    * Closure/equivalence discovery uses a bounded graph beam search configured
    * by closureSearchVisits, closureSearchFrontier and closureSearchBeamWidth.
    * Candidate scores may run in parallel; accepted candidates still pass the
    * complete CVA predicates. Recursive intersections have no depth limit.
    * This remains a learned approximation of the paper's Boolean lattice.
    */
  def fastAddIntent(instance: Instance): Node = fastAddIntent(List(instance))

  /** Insert a CVA extent, rather than treating its members as separate intents. */
  def fastAddIntent(instances: Iterable[Instance]): Node = this.synchronized {
    require(instances.nonEmpty, "FastAddIntent requires a nonempty extent")
    ensureExtentIndexes()

    // All predicates see fixed models AND extents until topology is complete.
    // Identity keys avoid confusing different embeddings with the same label.
    val membership = mutable.LongMap.empty[java.util.IdentityHashMap[Tensor[BFloat16], java.lang.Boolean]]
    var membershipEntries = 0
    val inclusion = mutable.HashMap.empty[(Long, Long), Boolean]
    val equality = mutable.HashMap.empty[(Long, Long), Boolean]
    final case class ClosureSearch(generator: Option[Node], equivalent: Option[Node])
    val closures = mutable.LongMap.empty[ClosureSearch]
    val targets = mutable.HashMap.empty[Set[String], Node]
    val additions = mutable.LinkedHashMap.empty[Node, mutable.LinkedHashMap[String, Instance]]
    val created = mutable.ArrayBuffer.empty[Node]
    val linkedChildren = mutable.LinkedHashSet.empty[Node]

    def retainAcceptance(classifier: Node, member: Instance, accepted: Boolean): Unit =
      if membershipEntries < 65536 then
        val retained = membership.getOrElseUpdate(classifier.id,
          new java.util.IdentityHashMap[Tensor[BFloat16], java.lang.Boolean]())
        if retained.get(member.vector) == null then
          retained.put(member.vector, java.lang.Boolean.valueOf(accepted))
          membershipEntries += 1

    def accepts(classifier: Node, member: Instance): Boolean =
      val cache = membership.get(classifier.id).orNull
      val cached = if cache == null then null else cache.get(member.vector)
      if cached != null then cached.booleanValue()
      else
        val accepted = classifier.classifies(member)
        // Overflow recomputes the same predicate without retaining the result.
        retainAcceptance(classifier, member, accepted)
        accepted

    def includes(classifier: Node, extent: Node): Boolean =
      classifier.id == extent.id || inclusion.getOrElseUpdate((classifier.id, extent.id),
        extent.instancesIterator.forall(accepts(classifier, _)))

    def enough(classifier: Node, extent: Node): Boolean =
      var accepted = 0
      var remaining = extent.size
      val members = extent.instancesIterator
      while members.hasNext do
        if accepts(classifier, members.next()) then accepted += 1
        remaining -= 1
        if accepted.toDouble / extent.size >= config.equalityCoverage then return true
        if (accepted + remaining).toDouble / extent.size < config.equalityCoverage then return false
      false

    def same(left: Node, right: Node): Boolean =
      val key = (math.min(left.id, right.id), math.max(left.id, right.id))
      left.id == right.id || equality.getOrElseUpdate(key,
        enough(left, right) && enough(right, left) &&
          (!config.requireMutualCommonClassification ||
            (left.classifies(right.commonVector) && right.classifies(left.commonVector))))

    def target(members: Vector[Instance]): Node =
      targets.getOrElseUpdate(members.iterator.map(_.symbol).toSet,
        createNode(members))

    def remember(existing: Node, requested: Node): Node =
      val pending = additions.getOrElseUpdate(existing, mutable.LinkedHashMap.empty)
      requested.instancesIterator.foreach(member => pending.getOrElseUpdate(member.symbol, member))
      existing

    def childrenOf(generator: Option[Node]): Vector[Node] = generator match
      case Some(node) => node.children.iterator.map(nodeMap).toVector
      case None => topNodeIds.iterator.take(config.closureSearchFrontier).map(nodeMap).toVector

    def searchClosure(requested: Node, start: Option[Node]): ClosureSearch =
      closures.getOrElseUpdate(requested.id, {
        val downward = start match
          case Some(node) => includes(node, requested)
          case None => topNodeIds.size <= bottomNodeIds.size
        // Reserve room to move beyond the initial frontier, including upward
        // walks from many bottom nodes to a smaller enclosing ancestor.
        val layerLimit = math.min(config.closureSearchFrontier,
          math.max(1, config.closureSearchVisits / 2))
        var frontier = start.fold {
          val ids = if downward then topNodeIds else bottomNodeIds
          ids.iterator.take(layerLimit).flatMap(nodeMap.get).toVector
        }(Vector(_))

        // Probe actual extent embeddings, since common vectors from different
        // projections are not comparable search coordinates. These samples only
        // rank candidates; they never establish inclusion or equivalence.
        val members = requested.instances
        val probeVectors = new java.util.IdentityHashMap[Tensor[BFloat16], java.lang.Boolean]()
        val probes = Vector(members.head, members(members.size / 2), members.last).filter { member =>
          probeVectors.put(member.vector, java.lang.Boolean.TRUE) == null
        }
        val scheduled = mutable.HashSet.empty[Long]
        val scores = mutable.LongMap.empty[Double]
        var generator = Option.empty[Node]
        var equivalent = Option.empty[Node]

        def measure(node: Node): (Node, Double, Array[Double]) =
          val distances = probes.map(member => node.distance(member.vector)).toArray
          val score = distances.iterator.map(d => if d.isNaN then Double.PositiveInfinity else d).sum / probes.size
          (node, score, distances)

        while frontier.nonEmpty && scheduled.size < config.closureSearchVisits && equivalent.isEmpty do
          val remaining = config.closureSearchVisits - scheduled.size
          val candidates = frontier.iterator.filter(node => scheduled.add(node.id)).take(remaining).toArray
          val scoreWork = candidates.length.toDouble * probes.size * cva.dimension *
            (if cva.config.noCVA then 1 else cva.dimension)
          val measured =
            // Small vectors/average models cost less than worker scheduling.
            if !config.closureSearchParallel || candidates.length < 32 || scoreWork < 2000000.0 ||
              Thread.currentThread().isInstanceOf[ForkJoinWorkerThread] then candidates.map(measure)
            else
              val parallelCandidates = candidates.par
              parallelCandidates.tasksupport = Lattice.parallelClosureTaskSupport
              parallelCandidates.map(measure).toArray

          // Workers only read fixed models. Publish scores and classifications
          // here, keeping every mutable insertion cache on the caller's thread.
          measured.foreach { (node, score, distances) =>
            scores.put(node.id, score)
            probes.indices.foreach(i =>
              retainAcceptance(node, probes(i), distances(i) <= node.classificationRadius))
          }
          val beam = measured.sortBy { (node, score, _) => (score, node.size, node.id) }
            .take(config.closureSearchBeamWidth).map(_._1)
          val iterator = beam.iterator
          while iterator.hasNext && equivalent.isEmpty do
            val candidate = iterator.next()
            if same(candidate, requested) then equivalent = Some(candidate)
            else if includes(candidate, requested) then
              val rank = (candidate.size, scores(candidate.id), candidate.id)
              if generator.forall(old =>
                summon[Ordering[(Int, Double, Long)]].lt(rank, (old.size, scores(old.id), old.id))) then
                generator = Some(candidate)

          val queued = mutable.HashSet.empty[Long]
          frontier = beam.iterator.flatMap { node =>
            val neighbors = if downward then node.children else node.parents
            neighbors.iterator.flatMap(nodeMap.get)
          }.filter(node => !scheduled.contains(node.id) && queued.add(node.id))
            .take(math.min(layerLimit, config.closureSearchVisits - scheduled.size)).toVector
        ClosureSearch(generator, equivalent)
      })

    def findEquivalent(requested: Node, start: Option[Node]): Option[Node] =
      // An exact symbol key remains a fast path, subject to full CVA equality.
      exactExtent(requested.symbols).filter(same(_, requested))
        .orElse(searchClosure(requested, start).equivalent)

    def getClosureConcept(requested: Node, start: Option[Node]): Option[Node] =
      searchClosure(requested, start).generator

    def connect(parent: Node, child: Node): Boolean =
      // Numerical inclusion is not transitive. Do not introduce a graph cycle
      // or a redundant edge when the child is already reachable.
      if parent.id == child.id || ancestorClosure(parent).contains(child.id) then false
      else
        if !ancestorClosure(child).contains(parent.id) then
          linkFast(parent, child)
          linkedChildren += child
        true

    def insert(requested: Node, start: Option[Node]): Node =
      findEquivalent(requested, start) match
        case Some(existing) => remember(existing, requested)
        case None =>
          val generator = getClosureConcept(requested, start)
          val generatorChildren = childrenOf(generator)

          // Publish the requested model before recursion so a repeated learned
          // intersection can reuse it even if CVA inclusion is non-transitive.
          nodeMap.put(requested.id, requested)
          bottomNodeIds += requested.id
          created += requested

          val candidates = mutable.LinkedHashMap.empty[Long, Node]
          generatorChildren.foreach { oldChild =>
            val candidate =
              if includes(requested, oldChild) then Some(oldChild)
              else
                val members = intersectionInstances(oldChild, requested, accepts)
                if members.isEmpty then None
                else Some(insert(target(members), Some(oldChild)))

            candidate.filter(_.id != requested.id).foreach { child =>
              candidates.getOrElseUpdate(child.id, child)
            }
          }

          // Cover repair uses the order already established by recursive calls,
          // avoiding a quadratic round of pairwise CVA extent comparisons.
          val candidateIds = candidates.keySet.toSet
          val newChildren = candidates.valuesIterator.filter { child =>
            !ancestorClosure(child).exists(id => id != child.id && candidateIds.contains(id))
          }.toVector
          newChildren.foreach { child =>
            if connect(requested, child) then generator.foreach(unlink(_, child))
          }
          generator.foreach(connect(_, requested))
          requested

    val objectNode = insert(target(instances.toVector), None)

    // Reused approximate intents retain the new examples. Propagate complete
    // child extents and refit each changed node once, after predicate caches are
    // no longer used. Unchanged intersections keep their fitted common vector.
    val changed = mutable.LinkedHashSet.empty[Node]
    additions.foreach { (concept, members) =>
      if concept.addInstancesWithoutRefit(members.valuesIterator) then changed += concept
    }
    // An unchanged old child can acquire a new parent through CVA inclusion,
    // even when that parent has not recorded its symbol names yet. Those child
    // extents must also seed propagation, not just new or extended concepts.
    val seeds = mutable.LinkedHashSet.from(created.iterator ++ changed.iterator ++ linkedChildren.iterator)
    val pending = mutable.ArrayDeque.from(seeds)
    val queued = mutable.HashSet.from(pending.iterator.map(_.id))
    while pending.nonEmpty do
      val child = pending.removeHead()
      queued -= child.id
      child.parents.iterator.map(nodeMap).foreach { parent =>
        if parent.addInstancesWithoutRefit(child.instancesIterator) then
          changed += parent
          if queued.add(parent.id) then pending.append(parent)
      }
    changed.foreach(_.refit())

    // Learned inclusion/equivalence can make adjacent nodes acquire identical
    // extents. Propagation guarantees containment, so equal sizes mean there is
    // no strict cover left between these endpoints. Retain both learned intents
    // but remove that edge; membership/equivalence still use their CVA models.
    (seeds.iterator ++ changed.iterator).foreach { child =>
      child.parents.iterator.map(nodeMap).filter(_.size == child.size).toVector
        .foreach(unlink(_, child))
    }
    objectNode
  }

  private def intersectionInstances(
      left: Node,
      right: Node,
      accepts: (Node, Instance) => Boolean
  ): Vector[Instance] =
    val retained = mutable.LinkedHashMap.empty[String, Instance]
    (left.instancesIterator ++ right.instancesIterator).foreach { member =>
      if !retained.contains(member.symbol) && accepts(left, member) && accepts(right, member) then
        retained.put(member.symbol, member)
    }
    retained.valuesIterator.toVector

  /**
   * Tests learned mutual equivalence of two nodes.
   *
   * Each node's classifier must cover at least `equalityCoverage` of the other's
   * extent. When configured, each classifier must additionally accept the other's
   * common vector. The two directional coverage scans make the result symmetric.
   */
  def equivalent(a: Node, b: Node): Boolean =
    coversEnough(a, b) && coversEnough(b, a) &&
      (!config.requireMutualCommonClassification ||
        (a.classifies(b.commonVector) && b.classifies(a.commonVector)))

  /** Stop once the coverage threshold is either reached or impossible.
    * This preserves coverage semantics without scoring every extent member.
    */
  private def coversEnough(classifier: Node, extent: Node): Boolean =
    val total = extent.size
    var accepted = 0
    var remaining = total
    val instances = extent.instancesIterator
    while instances.hasNext do
      if classifier.classifies(instances.next()) then accepted += 1
      remaining -= 1
      if accepted.toDouble / total >= config.equalityCoverage then return true
      if (accepted + remaining).toDouble / total < config.equalityCoverage then return false
    total == 0

  /**
   * Returns the fraction of `xs` classified by `classifier` in one pass.
   * Empty input has coverage 1.0 (vacuous truth). Runtime is linear in the number
   * of supplied instances and each iteration performs one CVA classification.
   */
  def coverage(classifier: Node, xs: Iterable[Instance]): Double =
    val iterator = xs.iterator
    var accepted = 0
    var total = 0
    while iterator.hasNext do
      if classifier.classifies(iterator.next()) then accepted += 1
      total += 1
    if total == 0 then 1.0 else accepted.toDouble / total.toDouble

  /**
   * Tests the authoritative graph-order relation on current symbolic extents.
   * A parent is more general exactly when its symbols strictly contain the child's.
   */
  def isMoreGeneral(parent: Node, child: Node): Boolean =
    child.symbols.subsetOf(parent.symbols) && parent.symbols != child.symbols

  /**
   * Validates basic graph invariants for every stored edge.
   *
   * Every referenced node must exist, parent/child adjacency must be symmetric,
   * and every edge must follow strict symbolic extent order. `require` throws on
   * the first violation. This does not prove that all possible cover edges exist
   * or that an intermediate concept is absent from every edge.
   */
  def validate(): Unit =
    val expectedBottomIds = nodeMap.valuesIterator.filter(_.children.isEmpty).map(_.id).toSet
    require(bottomNodeIds.toSet == expectedBottomIds, "Bottom-node index is inconsistent")
    val expectedTopIds = nodeMap.valuesIterator.filter(_.parents.isEmpty).map(_.id).toSet
    require(topNodeIds.toSet == expectedTopIds, "Top-node index is inconsistent")
    nodeMap.values.foreach { n =>
      n.parents.foreach { pId =>
        require(nodeMap.contains(pId), s"Missing parent $pId of node ${n.id}")
        val p = nodeMap(pId)
        require(p.children.contains(n.id), s"Asymmetric edge $pId -> ${n.id}")
        require(isMoreGeneral(p, n), s"Invalid extent order $pId -> ${n.id}")
      }
      n.children.foreach { cId =>
        require(nodeMap.contains(cId), s"Missing child $cId of node ${n.id}")
        val c = nodeMap(cId)
        require(c.parents.contains(n.id), s"Asymmetric edge ${n.id} -> $cId")
        require(isMoreGeneral(n, c), s"Invalid extent order ${n.id} -> $cId")
      }
    }

  /**
   * Builds a deterministic multiline description of the current lattice.
   * Nodes are ordered by descending extent size and then ID; each line includes
   * the node representation and its parent/child ID sets. No state is changed.
   */
  def describe(): String =
    nodes.sortBy(n => (-n.size, n.id)).map { n =>
      val ps = n.parents.mkString("[", ",", "]")
      val cs = n.children.mkString("[", ",", "]")
      s"${n.toString}, parents=$ps, children=$cs"
    }.mkString("\n")

  /** Replaces this lattice with the lattice stored in `filename`, reporting loading progress. */
  def load(filename: String, maxCount:Int = 10000000): Lattice =
    val startedAt = System.nanoTime()
    // Models use projections read-only and replace them on refit. Reuse the
    // overwhelmingly common identity instead of retaining dimension² floats per node.
    var identityProjection = Option.empty[Tensor[BFloat16]]
    var sharedIdentityCount = 0
    lazy val identityBytes =
      val bytes = new Array[Byte](Math.multiplyExact(Math.multiplyExact(cva.dimension, cva.dimension), 4))
      val buffer = ByteBuffer.wrap(bytes)
      for index <- 0 until cva.dimension do
        buffer.putFloat((index * cva.dimension + index) * 4, 1.0f)
      bytes
    lazy val projectionBuffer = new Array[Byte](identityBytes.length)

    def tensorFromBytes(bytes: Array[Byte], shape: Array[Int]): Tensor[BFloat16] =
      // The returned native handle belongs to the caller's scope. Copy assignment
      // keeps its storage alive while this inner scope releases options and views.
      val retained = new org.bytedeco.pytorch.Tensor()
      try
        Using.resource(new PointerScope()) { _ =>
          val values = new Array[Float](bytes.length / 4)
          ByteBuffer.wrap(bytes).asFloatBuffer().get(values)
          val tensor = numeric.BFloatTensors(values.toSeq, c_device).view(shape *)
          retained.put(tensor.native)
        }
        Tensor.fromNative[BFloat16](retained)
      catch
        case error: Throwable =>
          retained.close()
          throw error

    def readCount(input: DataInputStream, name: String): Int =
      val value = input.readInt()
      if value < 0 then
        throw new IllegalArgumentException(s"Invalid $name in lattice file: $value")
      value

    def readTensor(input: DataInputStream, expectedValues: Long, isProjection: Boolean = false): Tensor[BFloat16] =
      val rank = readCount(input, "tensor rank")
      val shape = Array.fill(rank)(readCount(input, "tensor dimension"))
      val valueCount = readCount(input, "tensor value count")
      require(valueCount.toLong == expectedValues,
        s"Expected $expectedValues tensor values, found $valueCount")
      val shapedValues = shape.foldLeft(1L)((size, dimension) => Math.multiplyExact(size, dimension.toLong))
      require(shapedValues == valueCount.toLong, "Tensor shape does not match its value count")
      if isProjection then
        require(shape.sameElements(Array(cva.dimension, cva.dimension)), "Invalid projection shape")
      val bytes = if isProjection then projectionBuffer else new Array[Byte](Math.multiplyExact(valueCount, 4))
      input.readFully(bytes)
      if isProjection && java.util.Arrays.equals(bytes, identityBytes) then
        identityProjection match
          case Some(projection) =>
            sharedIdentityCount += 1
            projection
          case None =>
            val projection = tensorFromBytes(bytes, shape)
            identityProjection = Some(projection)
            projection
      else tensorFromBytes(bytes, shape)

    Using.resource(
      new DataInputStream(new BufferedInputStream(new FileInputStream(filename)))
    ) { input =>
      val magic = input.readInt()
      if magic != Lattice.FileMagic then
        throw new IllegalArgumentException("Not a CVA-FCA lattice file")
      val version = input.readInt()
      if version != 1 && version != Lattice.FileVersion then
        throw new IllegalArgumentException(s"Unsupported lattice file version: $version")

      val storedDimension = readCount(input, "CVA dimension")
      if storedDimension != cva.dimension then
        throw new IllegalArgumentException(
          s"Lattice dimension $storedDimension does not match CVA dimension ${cva.dimension}"
        )

      val storedNextId = input.readLong()
      val nodeCount = readCount(input, "node count")
      val loadedNodes = mutable.LinkedHashMap.empty[Long, Node]
      var lastProgressAt = System.nanoTime()

      def reportProgress(stage: String, completed: Int, force: Boolean): Unit =
        val now = System.nanoTime()
        if force || now - lastProgressAt >= 1000000000L then
          val percentage = if nodeCount == 0 then 100.0 else completed.toDouble * 100.0 / nodeCount
          val elapsed = (now - startedAt) / 1e9
          println(f"Lattice load ($stage): $completed%,d / $nodeCount%,d | $percentage%6.2f%% | $elapsed%.1f s")
          lastProgressAt = now

      reportProgress("nodes", 0, force = true)
      var nodeIndex = 0
      val minCount = math.min(nodeCount, maxCount)
      while nodeIndex < minCount do
        try
            val id = input.readLong()
            if loadedNodes.contains(id) then
              throw new IllegalArgumentException(s"Duplicate node ID in lattice file: $id")
            val restoredModel = if version >= 2 then
              val commonVector = readTensor(input, cva.dimension.toLong)
              val projection = readTensor(input, cva.dimension.toLong * cva.dimension, isProjection = true)
              val radius = input.readDouble()
              require(!radius.isNaN && radius >= 0.0, s"Invalid classification radius for node $id")
              Some(CvaModel(commonVector, projection, radius))
            else None
            val instanceCount = readCount(input, "instance count")
            if instanceCount == 0 then
              throw new IllegalArgumentException(s"Node $id has no instances")
            // Node consumes this iterator directly into its extent, without a second collection.
            val instances = Iterator.fill(instanceCount) {
              val symbol = input.readUTF()
              val vector = readTensor(input, cva.dimension.toLong)
              Instance(symbol, vector)
            }
            val loaded = new Node(id, instances, cva, restoredModel = restoredModel)
            // Finish the whole record before publishing it. If EOF occurs here,
            // no partially read node is retained.
            val parents = Array.fill(readCount(input, "parent count"))(input.readLong())
            val children = Array.fill(readCount(input, "child count"))(input.readLong())
            loaded.parents.addAll(parents)
            loaded.children.addAll(children)
            loadedNodes.put(id, loaded)
            nodeIndex += 1
            reportProgress("nodes", nodeIndex, force = nodeIndex == nodeCount)
        catch
          case error: EOFException =>
            throw new IllegalArgumentException(
              s"Truncated lattice file: read ${loadedNodes.size} of $nodeCount declared nodes. " +
                "The checkpoint cannot be resumed safely.",
              error
            )

      reportProgress("edges", 0, force = true)
      var restoredNodes = 0
      loadedNodes.valuesIterator.foreach { loaded =>
        loaded.parents.foreach { parentId =>
          if !loadedNodes.contains(parentId) then
            throw new IllegalArgumentException(s"Missing parent $parentId of node ${loaded.id}")
        }
        loaded.children.foreach { childId =>
          if !loadedNodes.contains(childId) then
            throw new IllegalArgumentException(s"Missing child $childId of node ${loaded.id}")
        }
        restoredNodes += 1
        reportProgress("edges", restoredNodes, force = restoredNodes == nodeCount)
      }

      val minimumNextId = loadedNodes.keysIterator.maxOption.map(_ + 1L).getOrElse(0L)
      if storedNextId < minimumNextId then
        throw new IllegalArgumentException(s"Invalid next node ID in lattice file: $storedNextId")

      nodeMap.clear()
      nodeMap.addAll(loadedNodes)
      rebuildFrontierIndexes()
      nextId.set(storedNextId)
      //println("Lattice load: validating...")
      //validate()
      val elapsed = (System.nanoTime() - startedAt) / 1e9
      println(
        f"Lattice load complete: ${loadedNodes.size}%,d / $nodeCount%,d nodes | " +
          f"$sharedIdentityCount%,d identity projections reused | $elapsed%.1f s"
      )
    }
    this

  /** Writes this lattice to `filename`. */
  def save(filename: String): Lattice = this.synchronized {
    saveUnsafe(filename)
  }

  private def saveUnsafe(filename: String): Lattice =
    def writeTensor(output: DataOutputStream, tensor: Tensor[BFloat16]): Unit =
      val shape = tensor.shape
      output.writeInt(shape.length)
      shape.foreach(output.writeInt)
      val values = numeric.BFloatTensors.toArray(tensor)
      output.writeInt(values.length)
      values.foreach(output.writeFloat)

    // Freeze the sequence and count together so the header always describes
    // exactly the records written by this save operation.
    val latticeNodes = nodeMap.iterator.toArray
    if latticeNodes.length != nodeMap.size then
      throw new IllegalStateException(
        s"Lattice map is internally inconsistent: size=${nodeMap.size}, iterable=${latticeNodes.length}. " +
          "This usually indicates concurrent mutation from an older training run."
      )
    val storedNextId = nextId.get()
    val target = Paths.get(filename).toAbsolutePath
    val temporary = Files.createTempFile(target.getParent, ".lattice-save-", ".tmp")

    try
      Using.resource(new FileOutputStream(temporary.toFile)) { fileOutput =>
        Using.resource(new DataOutputStream(new BufferedOutputStream(fileOutput))) { output =>
          output.writeInt(Lattice.FileMagic)
          output.writeInt(Lattice.FileVersion)
          output.writeInt(cva.dimension)
          output.writeLong(storedNextId)
          output.writeInt(latticeNodes.length)

          latticeNodes.foreach { (id, latticeNode) =>
            output.writeLong(id)
            writeTensor(output, latticeNode.commonVector)
            writeTensor(output, latticeNode.projection)
            output.writeDouble(latticeNode.classificationRadius)
            output.writeInt(latticeNode.size)
            latticeNode.instances.foreach { instance =>
              output.writeUTF(instance.symbol)
              writeTensor(output, instance.vector)
            }
            output.writeInt(latticeNode.parents.size)
            latticeNode.parents.foreach(output.writeLong)
            output.writeInt(latticeNode.children.size)
            latticeNode.children.foreach(output.writeLong)
          }

          // Ensure all buffered data reaches the filesystem before publication.
          output.flush()
          fileOutput.getFD.sync()
        }
      }
      try
        Files.move(
          temporary,
          target,
          StandardCopyOption.ATOMIC_MOVE,
          StandardCopyOption.REPLACE_EXISTING
        )
      catch
        case _: AtomicMoveNotSupportedException =>
          Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING)
    finally
      Files.deleteIfExists(temporary)
    this

  /**
   * Allocates a unique ID and fits a new Node from `instances`.
   * The node is returned as a temporary object and is not inserted into `nodeMap`.
   * Node's symbol-keyed extent performs the required duplicate-symbol handling, so
   * this method deliberately avoids copying all instances for separate deduplication.
   */
  private def createNode(instances: Iterable[Instance]): Node =
    new Node(nextId.getAndIncrement(), instances, cva, do_avg = false)

private object Lattice:
  val FileMagic = 0x43564146 // "CVAF"
  val FileVersion = 2
  lazy val parallelIndexTaskSupport = new ForkJoinTaskSupport(
    new ForkJoinPool(math.min(24, math.max(1, Runtime.getRuntime.availableProcessors())))
  )
  lazy val parallelClosureTaskSupport = new ForkJoinTaskSupport(
    new ForkJoinPool(math.min(8, math.max(1, Runtime.getRuntime.availableProcessors())))
  )
