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

import scala.collection.parallel.CollectionConverters.VectorIsParallelizable
import scala.collection.parallel.ForkJoinTaskSupport


final case class LatticeConfig(
                                /** Maximum models scored by one learned closure search. */
                                closureSearchVisits: Int = 256,

                                /** Maximum bottom seeds or distinct parent candidates scored per layer. */
                                closureSearchFrontier: Int = 1024,

                                /** Closest candidates retained for graph expansion in each layer. */
                                closureSearchBeamWidth: Int = 24,

                                /** Score independent closure-search probes on workers when worthwhile. */
                                closureSearchParallel: Boolean = true,

                                /** Maximum topology repair rounds after the initial insertion pass. */
                                repairPassLimit: Int = 48,

                                /** Maximum unseen instance-parent classifications per repair pass. */
                                propagationChecksPerPass: Int = 24,

                                /** Score independent propagation parents on workers when worthwhile. */
                                propagationParallel: Boolean = true
                              )

/**
 * Incremental FCA-like lattice whose intents are CVA common vectors.
 *
 * Semantics
 * ---------
 * 1. node.classifies(x) is the subset/membership test.
 * 2. Intent equivalence requires classification of every extent sample in both directions.
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

  require(config.closureSearchVisits > 0, "closureSearchVisits must be positive")
  require(config.closureSearchFrontier > 0, "closureSearchFrontier must be positive")
  require(config.closureSearchBeamWidth > 0, "closureSearchBeamWidth must be positive")
  require(config.repairPassLimit > 0, "repairPassLimit must be positive")
  require(config.propagationChecksPerPass > 0, "propagationChecksPerPass must be positive")
  private val nextId = new AtomicLong(0L)

  /** Node IDs are immutable storage addresses, never semantic identities. */
  /** Protected so alternative insertion engines can reuse the same lattice state. */
  protected val nodeMap: mutable.Map[Long, Node] = new NodeStore

  /** Preserve insertion order while maintaining extent candidates and graph frontiers, including
   * additions by alternative insertion engines and checkpoint loads.
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
        previous.foreach(untrackNode)
        // Alternative insertion engines may supply an occupied storage address.
        nextId.updateAndGet(current => math.max(current, id + 1L))
        storageOrder.put(node, nextStorageOrder)
        nextStorageOrder += 1
        indexExtent(node)
        node.extentChanged = Some(() => indexExtent(node))
        if node.parents.isEmpty then topNodeIds += id else topNodeIds -= id
        if node.children.isEmpty then bottomNodeIds += id else bottomNodeIds -= id
      previous

    override def update(id: Long, node: Node): Unit = {
      put(id, node); ()
    }

    override def addOne(entry: (Long, Node)): this.type =
      put(entry._1, entry._2)
      this

    override def remove(id: Long): Option[Node] =
      val previous = entries.remove(id)
      previous.foreach(untrackNode)
      if previous.nonEmpty then
        topNodeIds -= id
        bottomNodeIds -= id
      previous

    override def subtractOne(id: Long): this.type =
      remove(id)
      this

    override def clear(): Unit =
      entries.valuesIterator.foreach(node => node.extentChanged = None)
      entries.clear()
      extentCandidates.clear()
      indexedExtents.clear()
      storageOrder.clear()
      nextStorageOrder = 0L
      topNodeIds.clear()
      bottomNodeIds.clear()

  /** Childless-node IDs with constant-time insertion/removal and bounded iteration. */
  private val bottomNodeIds = mutable.LinkedHashSet.empty[Long]
  /** Parentless-node IDs, maintained with the bottom frontier for closure search. */
  private val topNodeIds = mutable.LinkedHashSet.empty[Long]
  /** Symbol sets only retrieve candidates: mutual CVA classification decides reuse. */
  private val extentCandidates = mutable.HashMap.empty[Set[String], mutable.LinkedHashSet[Node]]
  private val indexedExtents = mutable.HashMap.empty[Node, Set[String]]
  private val storageOrder = mutable.HashMap.empty[Node, Long]
  private var nextStorageOrder = 0L

  /** Storage indexes shared with alternative insertion engines. */
  protected def indexedExtentCandidates(symbols: Set[String]): Iterator[Node] =
    extentCandidates.get(symbols).iterator.flatMap(_.iterator)

  protected def bottomCandidates(limit: Int): Vector[Node] =
    bottomNodeIds.iterator.flatMap(nodeMap.get).take(limit).toVector

  protected def topCandidates(limit: Int): Vector[Node] =
    topNodeIds.iterator.flatMap(nodeMap.get).take(limit).toVector

  private def removeExtentIndex(node: Node): Unit =
    indexedExtents.remove(node).foreach { symbols =>
      extentCandidates.get(symbols).foreach { candidates =>
        candidates -= node
        if candidates.isEmpty then extentCandidates -= symbols
      }
    }

  private def indexExtent(node: Node): Unit =
    removeExtentIndex(node)
    val symbols = node.symbols
    extentCandidates.getOrElseUpdate(symbols, mutable.LinkedHashSet.empty) += node
    indexedExtents.put(node, symbols)

  private def untrackNode(node: Node): Unit =
    node.extentChanged = None
    removeExtentIndex(node)
    storageOrder -= node

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
  protected def linkFast(parent: Node, child: Node): Unit =
    if parent ne child then
      parent.children += child.id
      child.parents += parent.id
      bottomNodeIds -= parent.id
      topNodeIds -= child.id

  /**
   * Collects `start` and all its ancestors by following parent IDs.
   * The deque provides depth-first-like traversal and the hash set prevents
   * repeated visits/cycles. Search budgets must not truncate this traversal.
   */
  private def ancestorClosure(start: Node): Set[Node] =
    val pending = mutable.ArrayDeque(start)
    val closure = mutable.HashSet.empty[Node]
    while pending.nonEmpty do
      val concept = pending.removeLast()
      if closure.add(concept) then
        concept.parents.iterator.flatMap(nodeMap.get).foreach(pending.append)
    closure.toSet

  private final case class ClosureSearch(generators: Vector[Node], equivalents: Vector[Node])
  private final case class PropagationWork(parent: Node, members: Vector[Instance])

  /** Insertion-local bookkeeping. All operations live on Lattice; no state
    * survives an insertion or leaks between a lattice and its clone.
    */
  private final class InsertionState:
    // References key caches/traversal state, never learned equality.
    val distances = mutable.HashMap.empty[Node,
      java.util.IdentityHashMap[Tensor[BFloat16], java.lang.Double]]
    var distanceEntries = 0
    val inclusion = mutable.HashMap.empty[(Node, Node), Boolean]
    val equality = mutable.HashMap.empty[(Node, Node), Boolean]
    val closures = mutable.HashMap.empty[(Node, Option[Node], Option[Node]), ClosureSearch]
    val vectorIds = new java.util.IdentityHashMap[Tensor[BFloat16], java.lang.Integer]()
    var nextVectorId = 0
    var temporaryId = -1L
    val targets = mutable.HashMap.empty[Set[(String, Int)], Node]
    val results = mutable.HashMap.empty[Node, Node]
    val active = mutable.HashSet.empty[Node]
    val additions = mutable.LinkedHashMap.empty[Node, mutable.LinkedHashMap[String, Instance]]
    val phaseCreated = mutable.ArrayBuffer.empty[Node]
    val linkedChildren = mutable.LinkedHashSet.empty[Node]


  /** FastAddIntent with CVA membership and full mutual classification for equality.
    * IDs address storage and edges only. Predicates see fixed models/extents in
    * each topology pass; changed CVAs are refitted before their next placement.
    */
  def fastAddIntent(instance: Instance): Node = fastAddIntent(List(instance))

  def fastAddIntent(instances: Iterable[Instance]): Node = this.synchronized {
    val incoming = instances.iterator.toVector.distinctBy(_.symbol)
    require(incoming.nonEmpty, "FastAddIntent requires a nonempty extent")

    val state = new InsertionState
    val requested = target(state, incoming)
    val duplicate = duplicateExtent(state, requested)
    duplicate.foreach(remember(state, _, incoming))
    val initial = searchClosure(state, requested, None, None)
    val classifiers = initial.equivalents ++ initial.generators
    if state.additions.isEmpty then
      (duplicate.iterator ++ classifiers.iterator)
        .filter(node => incoming.forall(member => node.containsSymbol(member.symbol)))
        .minByOption(_.size) match
          case Some(node) => return node
          case None => ()
    val represented = duplicate.orElse(initial.equivalents.headOption)

    var objectNode = Option.empty[Node]
    var repair = Vector.empty[Node]
    var firstPass = true
    var repairPasses = 0
    while firstPass || (repair.nonEmpty && repairPasses < config.repairPassLimit) do
      if !firstPass then prepareRepairPass(state, repair, incoming)
      state.phaseCreated.clear()
      state.linkedChildren.clear()
      val placed = placeInsertionPass(state, firstPass, represented, requested,
        classifiers, repair, incoming)
      if firstPass then objectNode = placed
      repair = growExtents(state)
      if !firstPass then repairPasses += 1
      firstPass = false
    objectNode.get
  }

  private def prepareRepairPass(
      state: InsertionState,
      repair: Vector[Node],
      incoming: Vector[Instance]
  ): Unit =
    resetPredicates(state, repair)
    repair.foreach(remember(state, _, incoming))
    val examined = mutable.HashSet.empty[(Long, Long)]
    repair.foreach { node =>
      node.parents.iterator.flatMap(nodeMap.get).toVector.foreach { parent =>
        if examined.add((parent.id, node.id)) && !strict(state, parent, node) then
          disconnect(state, parent, node)
      }
      node.children.iterator.flatMap(nodeMap.get).toVector.foreach { child =>
        if examined.add((node.id, child.id)) && !strict(state, node, child) then
          disconnect(state, node, child)
      }
    }

  private def placeInsertionPass(
      state: InsertionState,
      firstPass: Boolean,
      represented: Option[Node],
      requested: Node,
      classifiers: Vector[Node],
      repair: Vector[Node],
      incoming: Vector[Instance]
  ): Option[Node] =
    val placed =
      if firstPass then
        val node = represented match
          case Some(existing) =>
            state.results.put(requested, existing)
            position(state, existing, requested, None)
            existing
          case None => insert(state, requested, None)
        classifiers.foreach(connect(state, _, node))
        Some(node)
      else
        repair.foreach(node => position(state, node, node, None))
        None
    state.phaseCreated.foreach(remember(state, _, incoming))
    placed

  private def growExtents(state: InsertionState): Vector[Node] =
    val extentGrew = mutable.LinkedHashSet.empty[Node]
    state.additions.foreach { (node, members) =>
      val fresh = members.valuesIterator.filterNot(member => node.containsSymbol(member.symbol)).toVector
      if node.addInstancesWithoutRefit(fresh) then extentGrew += node
    }
    state.additions.clear()

    val seeds = (state.phaseCreated.iterator ++ extentGrew.iterator ++ state.linkedChildren.iterator)
      .toVector.distinct
    propagateExtentDeltas(state, seeds, extentGrew)
    extentGrew.foreach(_.refit())
    extentGrew.toVector

  private def propagateExtentDeltas(
      state: InsertionState,
      seeds: Vector[Node],
      extentGrew: mutable.LinkedHashSet[Node]
  ): Unit =
    val pending = mutable.ArrayDeque.empty[Node]
    val queued = mutable.HashSet.empty[Node]
    val pendingInstances = mutable.HashMap.empty[Node, mutable.LinkedHashMap[String, Instance]]
    def enqueue(node: Node, members: IterableOnce[Instance]): Unit =
      val batch = pendingInstances.getOrElseUpdate(node, mutable.LinkedHashMap.empty)
      members.iterator.foreach(member => batch.getOrElseUpdate(member.symbol, member))
      if batch.nonEmpty && queued.add(node) then pending.append(node)

    seeds.foreach(node => enqueue(node, node.instancesIterator))
    var checks = 0
    while pending.nonEmpty && checks < config.propagationChecksPerPass do
      val child = pending.removeHead()
      queued -= child
      val delta = pendingInstances.remove(child).iterator.flatMap(_.valuesIterator).toVector
      val work = mutable.ArrayBuffer.empty[PropagationWork]
      val parents = child.parents.iterator.flatMap(nodeMap.get)
      while parents.hasNext && checks < config.propagationChecksPerPass do
        val parent = parents.next()
        val unseen = mutable.ArrayBuffer.empty[Instance]
        val members = delta.iterator
        while members.hasNext && checks < config.propagationChecksPerPass do
          val member = members.next()
          if !parent.containsSymbol(member.symbol) then
            checks += 1
            unseen += member
        if unseen.nonEmpty then work += PropagationWork(parent, unseen.toVector)

      def addAccepted(parent: Node, accepted: IterableOnce[Instance]): Unit =
        val fresh = accepted.iterator.toVector
        if parent.addInstancesWithoutRefit(fresh) then
          extentGrew += parent
          enqueue(parent, fresh)

      val planned = work.toVector
      val parallelEligible = config.propagationParallel && planned.size >= 4 &&
        !Thread.currentThread().isInstanceOf[ForkJoinWorkerThread]
      val useParallel = parallelEligible && {
        val uncached = planned.iterator.map(item => missingProbeCount(state, item.parent, item.members)).sum
        val cost = uncached.toDouble * cva.dimension *
          (if cva.config.noCVA then 1 else cva.dimension)
        cost >= 2000000.0
      }
      if !useParallel then
        planned.foreach { item =>
          addAccepted(item.parent, item.members.filter(accepts(state, item.parent, _)))
        }
      else
        // Workers only read fixed models and cached distances. Publish cache
        // entries and mutate extents in parent order after all workers finish.
        val parallelWork = planned.par
        parallelWork.tasksupport = Lattice.parallelClosureTaskSupport
        val measured = parallelWork.map(item => (item, measurePropagationWork(state, item))).toVector
        measured.foreach { (item, distances) =>
          val accepted = mutable.ArrayBuffer.empty[Instance]
          item.members.indices.foreach { index =>
            val member = item.members(index)
            val distance = distances(index)
            retainDistance(state, item.parent, member, distance)
            if distance < item.parent.classificationRadius then accepted += member
          }
          addAccepted(item.parent, accepted)
        }

  private def measurePropagationWork(state: InsertionState, work: PropagationWork): Vector[Double] =
    val cache = state.distances.get(work.parent).orNull
    work.members.map { member =>
      val cached = if cache == null then null else cache.get(member.vector)
      if cached == null then work.parent.distance(member.vector) else cached.doubleValue()
    }

  private def resetPredicates(state: InsertionState, refitted: IterableOnce[Node]): Unit =
    // Refits replace these models; cached distances for every other node remain valid.
    refitted.iterator.foreach { node =>
      state.distances.remove(node).foreach(cache => state.distanceEntries -= cache.size())
    }
    state.inclusion.clear()
    state.equality.clear()
    state.closures.clear()

  private def retainDistance(state: InsertionState, classifier: Node, member: Instance, distance: Double): Unit =
    if state.distanceEntries < 65536 then
      val retained = state.distances.getOrElseUpdate(classifier,
        new java.util.IdentityHashMap[Tensor[BFloat16], java.lang.Double]())
      if retained.get(member.vector) == null then
        retained.put(member.vector, java.lang.Double.valueOf(distance))
        state.distanceEntries += 1

  private def accepts(state: InsertionState, classifier: Node, member: Instance): Boolean =
    val cache = state.distances.get(classifier).orNull
    val cached = if cache == null then null else cache.get(member.vector)
    if cached != null then cached.doubleValue() < classifier.classificationRadius
    else
      val distance = classifier.distance(member.vector)
      retainDistance(state, classifier, member, distance)
      distance < classifier.classificationRadius

  private def includes(state: InsertionState, classifier: Node, extent: Node): Boolean =
    state.inclusion.getOrElseUpdate((classifier, extent),
      extent.instancesIterator.forall(accepts(state, classifier, _)))

  private def same(state: InsertionState, left: Node, right: Node): Boolean =
    state.equality.getOrElseUpdate((left, right), includes(state, left, right) && includes(state, right, left))

  private def strict(state: InsertionState, parent: Node, child: Node): Boolean =
    includes(state, parent, child) && !same(state, parent, child)

  private def vectorId(state: InsertionState, vector: Tensor[BFloat16]): Int =
    val known = state.vectorIds.get(vector)
    if known != null then known.intValue()
    else
      val assigned = state.nextVectorId
      state.nextVectorId += 1
      state.vectorIds.put(vector, java.lang.Integer.valueOf(assigned))
      assigned

  private def target(state: InsertionState, members: Vector[Instance]): Node =
    val key = members.iterator.map(member => (member.symbol, vectorId(state, member.vector))).toSet
    state.targets.getOrElseUpdate(key, {
      // A no-op insertion never consumes persistent node IDs.
      val node = new Node(state.temporaryId, members, cva, do_avg = false)
      state.temporaryId -= 1
      node
    })

  private def remember(state: InsertionState, existing: Node, members: IterableOnce[Instance]): Unit =
    members.iterator.foreach { member =>
      if !existing.containsSymbol(member.symbol) && accepts(state, existing, member) then
        state.additions.getOrElseUpdate(existing, mutable.LinkedHashMap.empty)
          .getOrElseUpdate(member.symbol, member)
    }

  private def duplicateExtent(state: InsertionState, requested: Node): Option[Node] =
    extentCandidates.get(requested.symbols).flatMap(_.iterator.find(same(state, _, requested)))

  private def childrenOf(generators: Vector[Node]): Vector[Node] =
    if generators.nonEmpty then
      generators.iterator.flatMap(_.children.iterator.flatMap(nodeMap.get)).toVector.distinct
    else topNodeIds.iterator.take(config.closureSearchFrontier).flatMap(nodeMap.get).toVector

  private def searchClosure(state: InsertionState, requested: Node, start: Option[Node], excluded: Option[Node]): ClosureSearch =
    state.closures.getOrElseUpdate((requested, start, excluded), {
      val layerLimit = math.min(config.closureSearchFrontier,
        math.max(1, config.closureSearchVisits / 2))
      // Always walk upward from specific concepts. A recursive hint adds a
      // local starting point; it never switches the traversal to top-down.
      var frontier = (start.iterator ++ bottomNodeIds.iterator.flatMap(nodeMap.get))
        .take(layerLimit).toVector.distinct
      val members = requested.instances
      val probeVectors = new java.util.IdentityHashMap[Tensor[BFloat16], java.lang.Boolean]()
      val probes = Vector(members.head, members(members.size / 2), members.last).filter { member =>
        probeVectors.put(member.vector, java.lang.Boolean.TRUE) == null
      }
      val scheduled = mutable.HashSet.empty[Node]
      val scores = mutable.HashMap.empty[Node, Double]
      val classifying = mutable.LinkedHashSet.empty[Node]
      val equivalents = mutable.ArrayBuffer.empty[Node]
      while frontier.nonEmpty && scheduled.size < config.closureSearchVisits do
        val remaining = config.closureSearchVisits - scheduled.size
        val selected = frontier.iterator.filterNot(scheduled).take(remaining).toVector
        scheduled ++= selected
        val missingScores = selected.iterator.map(missingProbeCount(state, _, probes)).sum
        val work = missingScores.toDouble * cva.dimension *
          (if cva.config.noCVA then 1 else cva.dimension)
        val measured =
          if !config.closureSearchParallel || selected.size < 32 || work < 2000000.0 ||
            Thread.currentThread().isInstanceOf[ForkJoinWorkerThread] then selected.map(measureClosureCandidate(state, _, probes))
          else
            val parallelNodes = selected.par
            parallelNodes.tasksupport = Lattice.parallelClosureTaskSupport
            parallelNodes.map(measureClosureCandidate(state, _, probes)).toVector
        measured.foreach { (node, score, distances) =>
          scores.put(node, score)
          probes.indices.foreach(i => retainDistance(state, node, probes(i), distances(i)))
          // A single rejected probe disproves complete inclusion. Do not scan
          // the extent to rediscover the same rejection at a later sample.
          if distances.exists(d => !(d < node.classificationRadius)) then
            state.inclusion.put((node, requested), false)
        }
        val ordered = measured.sortBy { (node, score, _) => (score, node.size, storageOrder(node)) }
        ordered.foreach { (candidate, _, _) =>
          if !excluded.contains(candidate) then
            if same(state, candidate, requested) then equivalents += candidate
            else if includes(state, candidate, requested) then classifying += candidate
        }
        // Stop ascending a classifying branch, as in canonical-generator search.
        // For learned, non-transitive predicates this is an approximation.
        val beam = ordered.iterator.filter { (node, _, _) =>
          excluded.contains(node) || !includes(state, node, requested)
        }.take(config.closureSearchBeamWidth).map(_._1).toVector
        val retained = beam.toSet
        // Verify complete extents for every possible generator, but spend the
        // partial-membership work only on the branches retained by the beam.
        ordered.foreach { (candidate, _, _) =>
          if retained.contains(candidate) || includes(state, candidate, requested) then
            remember(state, candidate, members.iterator)
        }
        frontier = nextClosureFrontier(beam, scheduled,
          math.min(layerLimit, config.closureSearchVisits - scheduled.size))
      val candidates = classifying.toVector
      val generators = candidates.filterNot { broader =>
        candidates.exists(narrower => (narrower ne broader) && strict(state, broader, narrower))
      }.sortBy(node => (node.size, scores(node), storageOrder(node)))
      ClosureSearch(generators, equivalents.toVector)
    })

  /** Give every retained branch a turn. Allow one frontier of raw edges per
    * branch so shared parents do not exhaust the distinct-candidate budget.
    * Duplicate/visited edges count against this finite enumeration limit.
    */
  private def nextClosureFrontier(
      beam: Vector[Node],
      visited: mutable.Set[Node],
      limit: Int
  ): Vector[Node] =
    val branches = mutable.ArrayDeque.from(beam.map(_.parents.iterator))
    val selected = mutable.LinkedHashSet.empty[Node]
    val edgeLimit = math.min(Int.MaxValue.toLong,
      config.closureSearchFrontier.toLong * beam.size).toInt
    var examined = 0
    while branches.nonEmpty && selected.size < limit && examined < edgeLimit do
      val parents = branches.removeHead()
      if parents.hasNext then
        val parentId = parents.next()
        examined += 1
        nodeMap.get(parentId).filterNot(visited).foreach(selected.add)
        if parents.hasNext then branches.append(parents)
    selected.toVector

  private def disconnect(state: InsertionState, parent: Node, child: Node): Unit =
    if parent.children.contains(child.id) then
      unlink(parent, child)
      state.closures.clear()

  private def connect(state: InsertionState, parent: Node, child: Node): Boolean =
    val pending = mutable.ArrayDeque((parent, child))
    val processed = mutable.HashSet.empty[(Node, Node)]
    while pending.nonEmpty do
      val (upper, lower) = pending.removeLast()
      // Reference checks protect graph structure only; strict uses both CVAs.
      if (upper ne lower) && processed.add((upper, lower)) && strict(state, upper, lower) &&
        !ancestorClosure(upper).contains(lower) && !ancestorClosure(lower).contains(upper) then
        val children = upper.children.iterator.flatMap(nodeMap.get).toVector
        children.find(middle => (middle ne lower) && strict(state, middle, lower)) match
          case Some(middle) => pending.append((middle, lower))
          case None =>
            val parents = lower.parents.iterator.flatMap(nodeMap.get).toVector
            linkFast(upper, lower)
            state.linkedChildren += lower
            state.closures.clear()
            parents.filter(old => (old ne upper) && strict(state, old, upper)).foreach { old =>
              disconnect(state, old, lower)
              pending.append((old, upper))
            }
            children.filter(old => (old ne lower) && strict(state, lower, old)).foreach { old =>
              disconnect(state, upper, old)
              pending.append((lower, old))
            }
    ancestorClosure(child).contains(parent)

  private def position(state: InsertionState, concept: Node, requested: Node, start: Option[Node]): Unit =
    if state.active.add(concept) then
      try
        val placement = searchClosure(state, requested, start.filterNot(_ eq concept), Some(concept))
        placement.equivalents.foreach(node => remember(state, node, requested.instancesIterator))
        val generators = placement.generators.filterNot(_ eq concept)
        val oldChildren = childrenOf(generators).filterNot(_ eq concept)
        val candidates = mutable.LinkedHashSet.empty[Node]
        oldChildren.foreach { oldChild =>
          val candidate =
            if includes(state, requested, oldChild) then Some(oldChild)
            else
              val common = intersectionInstances(oldChild, requested, accepts(state, _, _))
              if common.isEmpty then None else Some(insert(state, target(state, common), Some(oldChild)))
          candidate.filterNot(_ eq concept).foreach(candidates.add)
        }
        val newChildren = candidates.filterNot(child =>
          ancestorClosure(child).exists(other => (other ne child) && candidates.contains(other)))
        newChildren.foreach { child =>
          if connect(state, concept, child) then
            generators.filter(_.children.contains(child.id)).foreach(disconnect(state, _, child))
        }
        generators.foreach(connect(state, _, concept))
      finally state.active -= concept

  private def insert(state: InsertionState, requested: Node, start: Option[Node]): Node =
    state.results.get(requested) match
      // An active target is a lifecycle cache entry, never learned equality.
      case Some(existing) if state.active.contains(existing) || same(state, existing, requested) => existing
      case _ =>
        // Check the index even when the beam cannot visit the recorded node.
        // Matching symbols only select candidates; same evaluates both CVAs.
        val matched = duplicateExtent(state, requested)
          .orElse(searchClosure(state, requested, start, None).equivalents.headOption)
          .orElse(state.phaseCreated.find(same(state, _, requested)))
        val concept = matched.getOrElse {
          var address = nextId.getAndIncrement()
          while nodeMap.contains(address) do address = nextId.getAndIncrement()
          val stored = new Node(address, requested.instances, cva,
            restoredModel = Some(requested.model))
          nodeMap.put(stored.id, stored)
          state.phaseCreated += stored
          state.closures.clear()
          stored
        }
        state.results.put(requested, concept)
        remember(state, concept, requested.instancesIterator)
        position(state, concept, requested, start)
        concept

  /** Models stay fixed throughout a topology pass. Reuse measured distances
    * after cover changes invalidate the closure cache. Workers only read this
    * cache; searchClosure publishes their new measurements after they finish.
    */
  private def measureClosureCandidate(state: InsertionState, node: Node, probes: Vector[Instance]): (Node, Double, Vector[Double]) =
    val cache = state.distances.get(node).orNull
    val distances = probes.map { member =>
      val cached = if cache == null then null else cache.get(member.vector)
      if cached != null then cached.doubleValue() else node.distance(member.vector)
    }
    val score = distances.iterator.map(d => if d.isNaN then Double.PositiveInfinity else d).max
    (node, score, distances)

  private def missingProbeCount(state: InsertionState, node: Node, probes: Vector[Instance]): Int =
    val cache = state.distances.get(node).orNull
    if cache == null then probes.size
    else probes.count(member => cache.get(member.vector) == null)

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

  /** Nodes are equivalent only when each CVA classifies the other's full extent. */
  def equivalent(a: Node, b: Node): Boolean =
    a.instancesIterator.forall(b.classifies) && b.instancesIterator.forall(a.classifies)

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
   * A strict learned parent classifies the child's full extent, while the child
   * does not classify the parent's full extent. Symbols never establish order.
   */
  def isMoreGeneral(parent: Node, child: Node): Boolean =
    child.instancesIterator.forall(parent.classifies) &&
      !parent.instancesIterator.forall(child.classifies)

  /**
   * Validates basic graph invariants for every stored edge.
   *
   * Every referenced node must exist, parent/child adjacency must be symmetric,
   * and every edge must follow strict CVA extent order. `require` throws on
   * the first violation. This does not prove that all possible cover edges exist
   * or that an intermediate concept is absent from every edge.
   */
  def validate(): Unit =
    val expectedBottomIds = nodeMap.valuesIterator.filter(_.children.isEmpty).map(_.id).toSet
    require(bottomNodeIds.toSet == expectedBottomIds,
      s"Bottom-node index is inconsistent: indexed=${bottomNodeIds.toSet}, expected=$expectedBottomIds")
    val expectedTopIds = nodeMap.valuesIterator.filter(_.parents.isEmpty).map(_.id).toSet
    require(topNodeIds.toSet == expectedTopIds,
      s"Top-node index is inconsistent: indexed=${topNodeIds.toSet}, expected=$expectedTopIds")
    nodeMap.values.foreach { n =>
      n.parents.foreach { pId =>
        require(nodeMap.contains(pId), s"Missing parent $pId of node ${n.id}")
        val p = nodeMap(pId)
        require(p.children.contains(n.id), s"Asymmetric edge $pId -> ${n.id}")
        require(isMoreGeneral(p, n), s"Invalid CVA order $pId -> ${n.id}")
      }
      n.children.foreach { cId =>
        require(nodeMap.contains(cId), s"Missing child $cId of node ${n.id}")
        val c = nodeMap(cId)
        require(c.parents.contains(n.id), s"Asymmetric edge ${n.id} -> $cId")
        require(isMoreGeneral(n, c), s"Invalid CVA order ${n.id} -> $cId")
      }
    }

    // Learned strict inclusion need not be transitive, so it cannot by itself
    // prove acyclicity as strict symbolic containment did.
    val remainingParents = mutable.HashMap.from(nodeMap.valuesIterator.map(n => n -> n.parents.size))
    val pending = mutable.ArrayDeque.from(remainingParents.iterator.collect {
      case (node, 0) => node
    })
    var visited = 0
    while pending.nonEmpty do
      val parent = pending.removeHead()
      visited += 1
      parent.children.iterator.map(nodeMap).foreach { child =>
        val remaining = remainingParents(child) - 1
        remainingParents.update(child, remaining)
        if remaining == 0 then pending.append(child)
      }
    require(visited == nodeMap.size, "Cycle detected in lattice graph")

  /**
   * Builds a deterministic multiline description of the current lattice.
   * Nodes are ordered by descending extent size, retaining storage order for ties;
   * each line includes the node representation and its parent/child ID sets.
   * No state is changed.
   */
  def describe(): String =
    nodes.sortBy(n => -n.size).map { n =>
      val ps = n.parents.mkString("[", ",", "]")
      val cs = n.children.mkString("[", ",", "]")
      s"${n.toString}, parents=$ps, children=$cs"
    }.mkString("\n")

  private final class LoadState:
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

    var nodeCount = 0
    var lastProgressAt = startedAt

  private def tensorFromBytes(bytes: Array[Byte], shape: Array[Int]): Tensor[BFloat16] =
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

  private def readCount(input: DataInputStream, name: String): Int =
    val value = input.readInt()
    if value < 0 then
      throw new IllegalArgumentException(s"Invalid $name in lattice file: $value")
    value

  private def readTensor(state: LoadState, input: DataInputStream, expectedValues: Long, isProjection: Boolean = false): Tensor[BFloat16] =
    val rank = readCount(input, "tensor rank")
    val shape = Array.fill(rank)(readCount(input, "tensor dimension"))
    val valueCount = readCount(input, "tensor value count")
    require(valueCount.toLong == expectedValues,
      s"Expected $expectedValues tensor values, found $valueCount")
    val shapedValues = shape.foldLeft(1L)((size, dimension) => Math.multiplyExact(size, dimension.toLong))
    require(shapedValues == valueCount.toLong, "Tensor shape does not match its value count")
    if isProjection then
      require(shape.sameElements(Array(cva.dimension, cva.dimension)), "Invalid projection shape")
    val bytes = if isProjection then state.projectionBuffer else new Array[Byte](Math.multiplyExact(valueCount, 4))
    input.readFully(bytes)
    if isProjection && java.util.Arrays.equals(bytes, state.identityBytes) then
      state.identityProjection match
        case Some(projection) =>
          state.sharedIdentityCount += 1
          projection
        case None =>
          val projection = tensorFromBytes(bytes, shape)
          state.identityProjection = Some(projection)
          projection
    else tensorFromBytes(bytes, shape)

  private def reportLoadProgress(state: LoadState, stage: String, completed: Int, force: Boolean): Unit =
    val now = System.nanoTime()
    if force || now - state.lastProgressAt >= 1000000000L then
      val percentage = if state.nodeCount == 0 then 100.0 else completed.toDouble * 100.0 / state.nodeCount
      val elapsed = (now - state.startedAt) / 1e9
      println(f"Lattice load ($stage): $completed%,d / ${state.nodeCount}%,d | $percentage%6.2f%% | $elapsed%.1f s")
      state.lastProgressAt = now

  /** Replaces this lattice with the lattice stored in `filename`, reporting loading progress. */
  def load(filename: String, maxCount: Int = 10000000): Lattice = this.synchronized {
    val state = new LoadState
    Using.resource(
      new DataInputStream(new BufferedInputStream(new FileInputStream(filename)))
    ) { input =>
      val magic = input.readInt()
      if magic != Lattice.FileMagic then
        throw new IllegalArgumentException("Not a CVA-FCA lattice file")
      val version = input.readInt()
      if version < 1 || version > Lattice.FileVersion then
        throw new IllegalArgumentException(s"Unsupported lattice file version: $version")

      val storedDimension = readCount(input, "CVA dimension")
      if storedDimension != cva.dimension then
        throw new IllegalArgumentException(
          s"Lattice dimension $storedDimension does not match CVA dimension ${cva.dimension}"
        )

      val storedNextId = input.readLong()
      state.nodeCount = readCount(input, "node count")
      val loadedNodes = mutable.LinkedHashMap.empty[Long, Node]
      reportLoadProgress(state, "nodes", 0, force = true)
      var nodeIndex = 0
      val minCount = math.min(state.nodeCount, maxCount)
      while nodeIndex < minCount do
        try
          val id = input.readLong()
          if loadedNodes.contains(id) then
            throw new IllegalArgumentException(s"Duplicate node ID in lattice file: $id")
          val restoredTensors = if version >= 2 then
            val commonVector = readTensor(state, input, cva.dimension.toLong)
            val projection = readTensor(state, input, cva.dimension.toLong * cva.dimension, isProjection = true)
            if version == 2 then
              val radius = input.readDouble()
              require(!radius.isNaN && radius >= 0.0, s"Invalid classification radius for node $id")
            Some((commonVector, projection))
          else None
          val instanceCount = readCount(input, "instance count")
          if instanceCount == 0 then
            throw new IllegalArgumentException(s"Node $id has no instances")
          val instances = Vector.fill(instanceCount) {
            val symbol = input.readUTF()
            val vector = readTensor(state, input, cva.dimension.toLong)
            Instance(symbol, vector)
          }
          val restoredModel = restoredTensors.map { (commonVector, projection) =>
            cva.restore(commonVector, projection, instances.map(_.vector))
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
          reportLoadProgress(state, "nodes", nodeIndex, force = nodeIndex == state.nodeCount)
        catch
          case error: EOFException =>
            throw new IllegalArgumentException(
              s"Truncated lattice file: read ${loadedNodes.size} of ${state.nodeCount} declared nodes. " +
                "The checkpoint cannot be resumed safely.",
              error
            )

      reportLoadProgress(state, "edges", 0, force = true)
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
        reportLoadProgress(state, "edges", restoredNodes, force = restoredNodes == state.nodeCount)
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
      val elapsed = (System.nanoTime() - state.startedAt) / 1e9
      println(
        f"Lattice load complete: ${loadedNodes.size}%,d / ${state.nodeCount}%,d nodes | " +
          f"${state.sharedIdentityCount}%,d identity projections reused | $elapsed%.1f s"
      )
    }
    this
  }

  /** Writes this lattice to `filename`. */
  def save(filename: String): Lattice = this.synchronized {
    saveUnsafe(filename)
  }

  private def writeTensor(output: DataOutputStream, tensor: Tensor[BFloat16]): Unit =
    val shape = tensor.shape
    output.writeInt(shape.length)
    shape.foreach(output.writeInt)
    val values = numeric.BFloatTensors.toArray(tensor)
    output.writeInt(values.length)
    values.foreach(output.writeFloat)

  private def saveUnsafe(filename: String): Lattice =
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

private object Lattice:
  val FileMagic = 0x43564146 // "CVAF"
  val FileVersion = 3
  lazy val parallelClosureTaskSupport = new ForkJoinTaskSupport(
    new ForkJoinPool(math.min(24, math.max(1, Runtime.getRuntime.availableProcessors())))
  )
