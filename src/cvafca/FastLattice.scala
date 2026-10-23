package cvafca

import scala.collection.mutable
import scala.collection.parallel.CollectionConverters.VectorIsParallelizable
import scala.collection.parallel.ForkJoinTaskSupport
import java.util.concurrent.{ForkJoinPool, ForkJoinWorkerThread}
import torch.{BFloat16, Device, Tensor}

/** Budgets for the learned FastAddIntent approximation. */
final case class FastLatticeConfig(
    searchVisits: Int = 2048,
    searchFrontier: Int = 128,
    searchBeamWidth: Int = 48,
    maxChildCandidates: Int = 128,
    maxIntersections: Int = 64,
    maxIntersectionDepth: Int = 24,
    repairPasses: Int = 4,
    propagationChecks: Int = 100000,
    reachabilityVisits: Int = 4096,
    parallelSearch: Boolean = true
)

/** FastAddIntent with CVA membership and an extent-fitted intersection.
  *
  * The insertion engine is independent of Lattice.fastAddIntent. It uses the
  * inherited node store, extent index, persistence, and graph validation.
  */
class FastLattice(
    override val cva: CVA,
    val fastConfig: FastLatticeConfig = FastLatticeConfig(),
    override val c_device: Device = Device.CPU
) extends Lattice(cva, LatticeConfig(), c_device):

  require(fastConfig.searchVisits > 0)
  require(fastConfig.searchFrontier > 0)
  require(fastConfig.searchBeamWidth > 0)
  require(fastConfig.maxChildCandidates > 0)
  require(fastConfig.maxIntersections >= 0)
  require(fastConfig.maxIntersectionDepth >= 0)
  require(fastConfig.repairPasses >= 0)
  require(fastConfig.propagationChecks > 0)
  require(fastConfig.reachabilityVisits > 0)

  private final case class Search(generators: Vector[Node], equivalents: Vector[Node])

  private final class Insert:
    val distances = mutable.HashMap.empty[Node,
      java.util.IdentityHashMap[Tensor[BFloat16], java.lang.Double]]
    val inclusion = mutable.HashMap.empty[(Node, Node), Boolean]
    val targets = mutable.HashMap.empty[Set[(String, Int)], Node]
    val vectorIds = new java.util.IdentityHashMap[Tensor[BFloat16], java.lang.Integer]()
    val results = mutable.HashMap.empty[Node, Node]
    val aliases = mutable.HashMap.empty[Node, Node]
    val active = mutable.HashSet.empty[Node]
    val additions = mutable.LinkedHashMap.empty[Node, mutable.LinkedHashMap[String, Instance]]
    val phaseCreated = mutable.ArrayBuffer.empty[Node]
    val linkedChildren = mutable.LinkedHashSet.empty[Node]
    var nextVectorId = 0
    var temporaryId = -1L
    var newIntersections = 0

    def resetPredicates(refitted: IterableOnce[Node]): Unit =
      refitted.iterator.foreach(distances.remove)
      inclusion.clear()

    def resolve(node: Node): Node =
      var current = node
      while aliases.contains(current) do current = aliases(current)
      current

  private var nextStorageId = 0L
  private var knownNodeCount = 0
  private var normalizedNodeCount = 0

  override def fastAddIntent(instance: Instance): Node = fastAddIntent(List(instance))

  override def fastAddIntent(instances: Iterable[Instance]): Node = this.synchronized {
    val incoming = instances.iterator.toVector.distinctBy(_.symbol)
    require(incoming.nonEmpty, "FastAddIntent requires a nonempty extent")
    val state = new Insert
    // A subclass or checkpoint can populate the shared store before the first
    // FastLattice insertion. Normalize that input once; ordinary insertions
    // maintain uniqueness incrementally using the extent index.
    if normalizedNodeCount != nodeMap.size then
      coalesceExtents(state, nodeMap.valuesIterator.toVector, Set.empty[Node])
    val requested = target(state, incoming)
    val result = addIntent(state, requested, depth = 0)
    var repair = growExtents(state)
    var pass = 0
    while repair.nonEmpty && pass < fastConfig.repairPasses do
      state.resetPredicates(repair)
      state.phaseCreated.clear()
      state.linkedChildren.clear()
      pruneInvalidLinks(state, repair)
      repair.foreach(node => remember(state, node, incoming))
      repair.foreach(node => place(state, node, node, depth = 0))
      repair = growExtents(state)
      pass += 1
    // The final refit may invalidate an incident edge even at the repair cap.
    if repair.nonEmpty then
      state.resetPredicates(repair)
      pruneInvalidLinks(state, repair)
    normalizedNodeCount = nodeMap.size
    state.resolve(result)
  }

  private def allocateId(): Long =
    if knownNodeCount != nodeMap.size then
      nextStorageId = math.max(nextStorageId,
        nodeMap.keysIterator.maxOption.map(_ + 1L).getOrElse(0L))
      knownNodeCount = nodeMap.size
    while nodeMap.contains(nextStorageId) do nextStorageId += 1
    val id = nextStorageId
    nextStorageId += 1
    knownNodeCount += 1
    id

  private def target(state: Insert, members: Vector[Instance]): Node =
    val key = members.iterator.map { member =>
      val known = state.vectorIds.get(member.vector)
      val id = if known != null then known.intValue() else
        val fresh = state.nextVectorId
        state.nextVectorId += 1
        state.vectorIds.put(member.vector, java.lang.Integer.valueOf(fresh))
        fresh
      (member.symbol, id)
    }.toSet
    state.targets.getOrElseUpdate(key, {
      val temporary = new Node(state.temporaryId, members, cva)
      state.temporaryId -= 1
      temporary
    })

  private def distance(state: Insert, node: Node, member: Instance): Double =
    val scores = state.distances.getOrElseUpdate(node,
      new java.util.IdentityHashMap[Tensor[BFloat16], java.lang.Double]())
    val cached = scores.get(member.vector)
    if cached != null then cached.doubleValue()
    else
      val measured = node.distance(member.vector)
      scores.put(member.vector, java.lang.Double.valueOf(measured))
      measured

  private def measureProbes(state: Insert, node: Node, probes: Vector[Instance]): Vector[Double] =
    val cache = state.distances.get(node).orNull
    probes.map { member =>
      val cached = if cache == null then null else cache.get(member.vector)
      if cached == null then node.distance(member.vector) else cached.doubleValue()
    }

  private def accepts(state: Insert, node: Node, member: Instance): Boolean =
    distance(state, node, member) < node.classificationRadius

  private def includes(state: Insert, parent: Node, child: Node): Boolean =
    state.inclusion.getOrElseUpdate((parent, child),
      child.instancesIterator.forall(accepts(state, parent, _)))

  private def same(state: Insert, left: Node, right: Node): Boolean =
    includes(state, left, right) && includes(state, right, left)

  private def strict(state: Insert, parent: Node, child: Node): Boolean =
    (parent ne child) && includes(state, parent, child) && !includes(state, child, parent)

  private def remember(state: Insert, node: Node, members: IterableOnce[Instance]): Unit =
    members.iterator.foreach { member =>
      if !node.containsSymbol(member.symbol) && accepts(state, node, member) then
        state.additions.getOrElseUpdate(node, mutable.LinkedHashMap.empty)
          .getOrElseUpdate(member.symbol, member)
    }

  /** Search upward from specific concepts, stopping at classifying branches.
    * The frontier and beam bound the canonical-generator approximation.
    */
  private def canonicalGenerators(state: Insert, requested: Node, excluded: Option[Node]): Search =
    val seedLimit = math.min(Int.MaxValue.toLong,
      fastConfig.searchFrontier.toLong + (if excluded.nonEmpty then 1L else 0L)).toInt
    var frontier = bottomCandidates(seedLimit)
      .filterNot(excluded.contains).take(fastConfig.searchFrontier)
    val visited = mutable.HashSet.empty[Node]
    val classifiers = mutable.LinkedHashSet.empty[Node]
    val equivalents = mutable.ArrayBuffer.empty[Node]
    val probes =
      val members = requested.instances
      Vector(members.head, members(members.size / 2), members.last).distinctBy(_.vector)

    while frontier.nonEmpty && visited.size < fastConfig.searchVisits do
      val selected = frontier.filterNot(visited).take(fastConfig.searchVisits - visited.size)
      visited ++= selected
      val missingProbes = selected.iterator.map { node =>
        val cache = state.distances.get(node).orNull
        if cache == null then probes.size
        else probes.count(member => cache.get(member.vector) == null)
      }.sum
      val estimatedWork = missingProbes.toDouble * cva.dimension *
        (if cva.config.noCVA then 1 else cva.dimension)
      val measured =
        if fastConfig.parallelSearch && selected.size >= 16 && estimatedWork >= 2000000.0 &&
          !Thread.currentThread().isInstanceOf[ForkJoinWorkerThread] then
          val parallel = selected.par
          parallel.tasksupport = FastLattice.searchTaskSupport
          parallel.map(node => (node, measureProbes(state, node, probes))).toVector
        else selected.map(node => (node, measureProbes(state, node, probes)))
      measured.foreach { (node, scores) =>
        probes.indices.foreach { i =>
          state.distances.getOrElseUpdate(node,
            new java.util.IdentityHashMap[Tensor[BFloat16], java.lang.Double]())
            .put(probes(i).vector, java.lang.Double.valueOf(scores(i)))
        }
      }
      val ordered = measured.sortBy { (node, scores) =>
        (scores.iterator.map(d => if d.isNaN then Double.PositiveInfinity else d).max, node.size, node.id)
      }
      ordered.foreach { (node, scores) =>
        if scores.forall(_ < node.classificationRadius) && includes(state, node, requested) then
          if same(state, node, requested) then equivalents += node
          else classifiers += node
          remember(state, node, requested.instancesIterator)
      }
      val beam = ordered.iterator.collect { case (node, scores)
          if scores.exists(d => !(d < node.classificationRadius)) ||
            !includes(state, node, requested) => node
      }.take(fastConfig.searchBeamWidth).toVector
      beam.foreach(node => remember(state, node, requested.instancesIterator))
      frontier = beam.iterator.flatMap(_.parents.iterator.flatMap(nodeMap.get))
        .filterNot(node => visited.contains(node) || excluded.contains(node))
        .take(fastConfig.searchFrontier).toVector.distinct

    val candidates = classifiers.toVector
    val generators = candidates.filterNot { broader =>
      candidates.exists(narrower => (narrower ne broader) && strict(state, broader, narrower))
    }
    Search(generators, equivalents.toVector)

  /** Symbolic extents are unique in FastLattice, even when their CVAs differ. */
  private def indexedExtentNode(requested: Node): Option[Node] =
    indexedExtentCandidates(requested.symbols).minByOption(_.id)

  /** AddIntent: reuse a concept or create it, then fix its immediate covers. */
  private def addIntent(state: Insert, requested: Node, depth: Int): Node =
    state.results.get(requested) match
      case Some(existing) => state.resolve(existing)
      case None =>
        val indexed = indexedExtentNode(requested)
        val search = canonicalGenerators(state, requested, indexed)
        val equivalent = indexed.orElse(search.equivalents.headOption)
        val concept = equivalent.getOrElse {
          val created = new Node(allocateId(), requested.instances, cva,
            restoredModel = Some(requested.model))
          nodeMap.put(created.id, created)
          state.phaseCreated += created
          if depth > 0 then state.newIntersections += 1
          created
        }
        state.results.put(requested, concept)
        remember(state, concept, requested.instancesIterator)
        val placement = if equivalent.nonEmpty then
          canonicalGenerators(state, requested, Some(concept))
        else search
        place(state, concept, requested, depth, Some(placement))
        concept

  private def intersection(state: Insert, left: Node, right: Node): Option[Node] =
    val common = mutable.LinkedHashMap.empty[String, Instance]
    (left.instancesIterator ++ right.instancesIterator).foreach { member =>
      if !common.contains(member.symbol) && accepts(state, left, member) &&
        accepts(state, right, member) then common.put(member.symbol, member)
    }
    if common.isEmpty then None else Some(target(state, common.valuesIterator.toVector))

  private def place(
      state: Insert,
      concept: Node,
      requested: Node,
      depth: Int,
      knownSearch: Option[Search] = None
  ): Unit =
    if state.active.add(concept) then
      try
        val found = knownSearch.getOrElse(canonicalGenerators(state, requested, Some(concept)))
        found.equivalents.foreach(node => remember(state, node, requested.instancesIterator))
        val parents = found.generators.filterNot(_ eq concept)
        val oldChildren = if parents.nonEmpty then
          parents.iterator.flatMap(_.children.iterator.flatMap(nodeMap.get))
            .take(fastConfig.maxChildCandidates).toVector.distinct
        else topCandidates(math.min(fastConfig.searchFrontier, fastConfig.maxChildCandidates))
        val newChildren = mutable.LinkedHashSet.empty[Node]
        oldChildren.filterNot(_ eq concept).foreach { child =>
          if strict(state, concept, child) then newChildren += child
          else if depth < fastConfig.maxIntersectionDepth &&
              state.newIntersections < fastConfig.maxIntersections then
            intersection(state, child, requested).foreach { common =>
              if !same(state, common, requested) then
                val inserted = addIntent(state, common, depth + 1)
                if inserted ne concept then newChildren += inserted
            }
        }
        parents.foreach(parent => link(state, parent, concept))
        newChildren.foreach(child => link(state, concept, child))
        parents.foreach { parent =>
          newChildren.foreach { child =>
            if parent.children.contains(child.id) &&
              concept.parents.contains(parent.id) && concept.children.contains(child.id) then
              unlink(parent, child)
          }
        }
      finally state.active -= concept

  /** Unknown reachability is treated as reachable, so a budget cannot add a cycle. */
  private def reachable(start: Node, goal: Node): Boolean =
    val pending = mutable.ArrayDeque(start)
    val seen = mutable.HashSet.empty[Node]
    var truncated = false
    while pending.nonEmpty && seen.size < fastConfig.reachabilityVisits && !truncated do
      val current = pending.removeLast()
      if current eq goal then return true
      if seen.add(current) then
        val children = current.children.iterator.flatMap(nodeMap.get)
        while children.hasNext && !truncated do
          if seen.size + pending.size >= fastConfig.reachabilityVisits then truncated = true
          else pending.append(children.next())
    pending.nonEmpty || truncated

  private def link(state: Insert, parent: Node, child: Node): Unit =
    if strict(state, parent, child) && !reachable(child, parent) &&
      !reachable(parent, child) then
      linkFast(parent, child)
      state.linkedChildren += child

  private def pruneInvalidLinks(state: Insert, changed: Vector[Node]): Unit =
    val inspected = mutable.HashSet.empty[(Long, Long)]
    changed.foreach { node =>
      node.parents.iterator.flatMap(nodeMap.get).toVector.foreach { parent =>
        if inspected.add((parent.id, node.id)) && !strict(state, parent, node) then
          unlink(parent, node)
      }
      node.children.iterator.flatMap(nodeMap.get).toVector.foreach { child =>
        if inspected.add((node.id, child.id)) && !strict(state, node, child) then
          unlink(node, child)
      }
    }

  /** Add the object to classified ancestors, then refit only grown extents. */
  private def growExtents(state: Insert): Vector[Node] =
    val grown = mutable.LinkedHashSet.empty[Node]
    state.additions.foreach { (node, members) =>
      if node.addInstancesWithoutRefit(members.valuesIterator) then grown += node
    }
    state.additions.clear()
    val pending = mutable.ArrayDeque.empty[Node]
    val queued = mutable.HashSet.empty[Node]
    val pendingMembers = mutable.HashMap.empty[Node, mutable.LinkedHashMap[String, Instance]]
    def enqueue(node: Node, members: IterableOnce[Instance]): Unit =
      val batch = pendingMembers.getOrElseUpdate(node, mutable.LinkedHashMap.empty)
      members.iterator.foreach(member => batch.getOrElseUpdate(member.symbol, member))
      if batch.nonEmpty && queued.add(node) then pending.append(node)
    (state.phaseCreated.iterator ++ grown.iterator ++ state.linkedChildren.iterator)
      .toVector.distinct.foreach(node => enqueue(node, node.instancesIterator))
    var checks = 0
    while pending.nonEmpty && checks < fastConfig.propagationChecks do
      val child = pending.removeHead()
      queued -= child
      val delta = pendingMembers.remove(child).iterator.flatMap(_.valuesIterator).toVector
      val parents = child.parents.iterator.flatMap(nodeMap.get)
      while parents.hasNext && checks < fastConfig.propagationChecks do
        val parent = parents.next()
        val accepted = mutable.ArrayBuffer.empty[Instance]
        val members = delta.iterator
        while members.hasNext && checks < fastConfig.propagationChecks do
          val member = members.next()
          if !parent.containsSymbol(member.symbol) then
            checks += 1
            if accepts(state, parent, member) then accepted += member
        if parent.addInstancesWithoutRefit(accepted) then
          grown += parent
          enqueue(parent, accepted)
    // Two different concepts can acquire the same symbolic extent in this
    // propagation wave. Keep one stored concept for each resulting extent.
    coalesceExtents(state, grown.iterator ++ state.phaseCreated.iterator, grown)

  /** Keep the lowest node ID for an extent and redirect valid incident links. */
  private def coalesceExtents(
      state: Insert,
      affected: IterableOnce[Node],
      grown: collection.Set[Node]
  ): Vector[Node] =
    val examined = mutable.HashSet.empty[Set[String]]
    val redirects = mutable.ArrayBuffer.empty[(Node, Vector[Node], Vector[Node])]
    affected.iterator.foreach { node =>
      if nodeMap.contains(node.id) && examined.add(node.symbols) then
        val candidates = indexedExtentCandidates(node.symbols)
        if candidates.hasNext then
          val first = candidates.next()
          if candidates.hasNext then
            val matches = first +: candidates.toVector
            val survivor = matches.minBy(_.id)
            matches.filterNot(_ eq survivor).foreach { duplicate =>
              val parents = duplicate.parents.iterator.flatMap(nodeMap.get).toVector
              val children = duplicate.children.iterator.flatMap(nodeMap.get).toVector
              parents.foreach(unlink(_, duplicate))
              children.foreach(unlink(duplicate, _))
              nodeMap.remove(duplicate.id)
              state.aliases.put(duplicate, survivor)
              redirects += ((survivor, parents, children))
            }
    }
    // Discarded nodes need no CVA update. A link-only change also needs none.
    val liveGrown = grown.iterator.map(state.resolve).filter(node => nodeMap.contains(node.id))
      .toVector.distinct
    liveGrown.filter(grown.contains).foreach(_.refit())
    if redirects.nonEmpty then
      state.resetPredicates(liveGrown)
      redirects.foreach { (survivor, parents, children) =>
        parents.iterator.map(state.resolve).filter(node => nodeMap.contains(node.id))
          .foreach(parent => link(state, parent, survivor))
        children.iterator.map(state.resolve).filter(node => nodeMap.contains(node.id))
          .foreach(child => link(state, survivor, child))
      }
    liveGrown

  override def cloneLattice(): FastLattice = this.synchronized {
    val cloned = new FastLattice(cva, fastConfig, c_device)
    nodeMap.foreach { (id, source) =>
      val copied = new Node(id, source.instances, cva, restoredModel = Some(source.model))
      copied.parents.addAll(source.parents)
      copied.children.addAll(source.children)
      cloned.nodeMap.put(id, copied)
    }
    cloned
  }

  override def validate(): Unit =
    super.validate()
    val extents = mutable.HashMap.empty[Set[String], Long]
    nodeMap.valuesIterator.foreach { node =>
      extents.put(node.symbols, node.id).foreach { otherId =>
        throw new IllegalArgumentException(
          s"Duplicate symbolic extent on nodes $otherId and ${node.id}")
      }
    }

  override def load(filename: String, maxCount: Int = 10000000): FastLattice = this.synchronized {
    super.load(filename, maxCount)
    knownNodeCount = 0
    coalesceExtents(new Insert, nodeMap.valuesIterator.toVector, Set.empty[Node])
    normalizedNodeCount = nodeMap.size
    this
  }

private object FastLattice:
  lazy val searchTaskSupport = new ForkJoinTaskSupport(
    new ForkJoinPool(math.min(16, math.max(1, Runtime.getRuntime.availableProcessors())))
  )
