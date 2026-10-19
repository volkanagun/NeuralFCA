package cvafca

import scala.collection.mutable
import java.io.{BufferedInputStream, BufferedOutputStream, DataInputStream, DataOutputStream, EOFException, FileInputStream, FileOutputStream}
import java.nio.ByteBuffer
import java.nio.file.{AtomicMoveNotSupportedException, Files, Paths, StandardCopyOption}
import java.util.concurrent.{ForkJoinPool, ForkJoinWorkerThread}
import java.util.concurrent.atomic.AtomicLong
import org.bytedeco.javacpp.PointerScope

import scala.util.Random
import scala.util.Using
import torch.{BFloat16, Device, Tensor}

import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.collection.parallel.ForkJoinTaskSupport


final case class LatticeConfig(
                                /** Fraction of the other node's instances that must be classified. */
                                equalityCoverage: Double = 0.80,

                                /** If true, also require mutual common-vector classification. */
                                requireMutualCommonClassification: Boolean = false,

                                /** Fast-search budgets stay at their maximum through this node count. */
                                fastBudgetReferenceNodes: Int = 1024,

                                /** Lowest fraction of each fast-search budget used on large lattices. */
                                fastMinimumBudgetScale: Double = 0.125,

                                /** Maximum concepts retaining append-only CVA bases; zero forces full refits. */
                                fastBasisCacheNodes: Int = 32
                              )

/**
 * Incremental FCA-like lattice whose intents are CVA common vectors.
 *
 * Semantics
 * ---------
 * 1. node.classifies(x) is the subset/membership test.
 * 2. Bounded insertion compares intents using mutual extent coverage, optionally
 *    also their common vectors. `original` requires full mutual classification.
 * 3. Intersection(a,b) is a new CVA model fitted to extent(a) U extent(b).
 * 4. Adding an instance to an existing node refits its common vector and projection.
 * 5. Bounded insertion methods use symbolic extent inclusion to order edges.
 *    `original` instead uses CVA membership throughout AddIntent and guards links
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
  require(config.fastBudgetReferenceNodes > 0)
  require(config.fastMinimumBudgetScale > 0.0 && config.fastMinimumBudgetScale <= 1.0)
  require(config.fastBasisCacheNodes >= 0)
  // Bound candidate discovery; required extent propagation remains complete.
  private val maxRecursionDepth = 4
  private val maxSearchVisits = 256
  private val maxBottomSamples = 2048
  private val maxBottomStarts = 128
  private val maxChildrenPerGenerator = 8
  private val fastBottomSamples = 512
  private val fastBottomStarts = 64
  private val fastSearchVisits = 192
  private val fastAncestorStarts = 24

  private final case class FastSearchLimits(
                                              bottomSamples: Int,
                                              bottomStarts: Int,
                                              searchVisits: Int,
                                              ancestorStarts: Int
                                            )

  private def fastSearchLimits(nodeCount: Int): FastSearchLimits =
    val sizeScale = math.sqrt(
      config.fastBudgetReferenceNodes.toDouble / math.max(1, nodeCount).toDouble
    )
    val scale = math.max(config.fastMinimumBudgetScale, math.min(1.0, sizeScale))
    def scaled(maximum: Int, minimum: Int): Int =
      // A floor must never override a deliberately lowered maximum.
      math.min(maximum, math.max(minimum, math.round(maximum * scale).toInt))
    FastSearchLimits(
      bottomSamples = scaled(fastBottomSamples, 128),
      bottomStarts = scaled(fastBottomStarts, 16),
      searchVisits = scaled(fastSearchVisits, 48),
      ancestorStarts = scaled(fastAncestorStarts, 8)
    )

  private val nextId = new AtomicLong(0L)
  /** Protected so alternative insertion engines can reuse the same lattice state. */
  protected val nodeMap = mutable.LinkedHashMap.empty[Long, Node]
  /** Childless-node IDs with constant-time insertion/removal and bounded iteration. */
  private val bottomNodeIds = mutable.LinkedHashSet.empty[Long]
  /** Per-lattice LRU; sample-dependent search scores are never retained here. */
  private val fastBasisNodes = mutable.LinkedHashMap.empty[Long, Node]

  private def refitFast(concept: Node): Unit =
    if config.fastBasisCacheNodes == 0 then concept.refit()
    else
      fastBasisNodes.remove(concept.id)
      while fastBasisNodes.size >= config.fastBasisCacheNodes do
        val (id, evicted) = fastBasisNodes.head
        fastBasisNodes.remove(id)
        evicted.clearBasisCache()
      fastBasisNodes.put(concept.id, concept)
      try concept.refitCached()
      catch
        case error: Throwable =>
          fastBasisNodes.remove(concept.id)
          concept.clearBasisCache()
          throw error

  // FastAddExtent-style exact lookup. Keys are immutable symbol snapshots;
  // tree keys preserve nodeMap order even when two mutable extents converge.
  private val extentIndex = mutable.HashMap.empty[Set[String], mutable.TreeMap[Long, Node]]
  private val singletonIndex = mutable.TreeMap.empty[Long, Node]
  private val singletonCenters = mutable.LongMap.empty[Array[Float]]
  private val indexedNodes = mutable.LongMap.empty[(Set[String], Long, Node)]
  private var indexesReady = false
  private var nextIndexOrder = 0L

  private def indexExtent(node: Node): Unit =
    indexedNodes.get(node.id).foreach { case (oldSymbols, order, _) =>
      extentIndex.get(oldSymbols).foreach { bucket =>
        bucket.remove(order)
        if bucket.isEmpty then extentIndex.remove(oldSymbols)
      }
      singletonIndex.remove(order)
      singletonCenters.remove(node.id)
      val symbols = node.symbols
      extentIndex.getOrElseUpdate(symbols, mutable.TreeMap.empty).put(order, node)
      if node.size == 1 then
        singletonIndex.put(order, node)
        if cva.config.noCVA || node.identityProjectionKnown then
          singletonCenters.put(node.id, numeric.BFloatTensors.toArray(node.commonVector))
      indexedNodes.put(node.id, (symbols, order, node))
    }

  private def indexNode(node: Node): Unit =
    val order = nextIndexOrder
    nextIndexOrder += 1
    indexedNodes.put(node.id, (Set.empty, order, node))
    indexExtent(node)
    node.extentChanged = Some(() => indexExtent(node))

  private def invalidateExtentIndexes(): Unit =
    indexedNodes.valuesIterator.foreach { case (_, _, node) => node.extentChanged = None }
    extentIndex.clear()
    singletonIndex.clear()
    singletonCenters.clear()
    indexedNodes.clear()
    indexesReady = false
    nextIndexOrder = 0L

  private def ensureExtentIndexes(): Unit =
    if !indexesReady || indexedNodes.size != nodeMap.size then
      invalidateExtentIndexes()
      nodeMap.valuesIterator.foreach(indexNode)
      indexesReady = true

  /** Prefer approximate extent matches satisfying the caller's intent predicate.
    * Look beyond the bounded generator search when checking existing nodes.
    * Models remain unchanged until the insertion's final extent updates.
    */
  private def registerNode(node: Node): Node =
    registerNode(node, existing => equivalent(existing, node))

  private def exactExtent(symbols: Set[String]): Option[Node] =
    if indexesReady && indexedNodes.size != nodeMap.size then ensureExtentIndexes()
    if indexesReady then extentIndex.get(symbols).flatMap(_.headOption.map(_._2))
    else nodeMap.valuesIterator.find(existing =>
      existing.size == symbols.size && symbols.forall(existing.containsSymbol))

  /** Exact snapshots are a fast path; approximate matches must also have the
    * requested learned intent. Scan the same candidates with or without indexes.
    */
  private def approximateExtent(symbols: Set[String], sameIntent: Node => Boolean): Option[Node] =
    exactExtent(symbols).orElse {
      nodeMap.valuesIterator.find(existing =>
        overlapsEnough(symbols, existing) &&
          sameIntent(existing))
    }

  /** Stop once enough shared symbols are found or the remaining ones cannot
    * reach the threshold. Reject undersized candidates before scanning symbols.
    */
  private def overlapsEnough(symbols: Set[String], existing: Node): Boolean =
    val total = symbols.size
    if total == 0 || existing.size.toDouble / total < config.equalityCoverage then return false
    var shared = 0
    var remaining = total
    val iterator = symbols.iterator
    while iterator.hasNext do
      if existing.containsSymbol(iterator.next()) then shared += 1
      remaining -= 1
      if shared.toDouble / total >= config.equalityCoverage then return true
      if (shared + remaining).toDouble / total < config.equalityCoverage then return false
    false

  private def registerNode(node: Node, sameIntent: Node => Boolean): Node =
    val symbols = node.symbols
    // No model, extent or edge changes between these two scans. Do not repeat
    // expensive CVA comparisons already rejected by the preferred overlap scan.
    val tested = mutable.HashSet.empty[Long]
    approximateExtent(symbols, existing =>
      tested += existing.id
      sameIntent(existing))
      .orElse(nodeMap.valuesIterator.find(existing =>
        !tested.contains(existing.id) && sameIntent(existing)))
      .getOrElse {
        nodeMap.put(node.id, node)
        if indexesReady then indexNode(node)
        if node.children.isEmpty then bottomNodeIds += node.id
        else bottomNodeIds -= node.id
        node
      }

  private def rebuildBottomNodeIndex(): Unit =
    bottomNodeIds.clear()
    nodeMap.valuesIterator.filter(_.children.isEmpty).foreach(node => bottomNodeIds += node.id)

  private def unlink(parent: Node, child: Node): Unit =
    parent.children.remove(child.id)
    child.parents.remove(parent.id)
    if parent.children.isEmpty then bottomNodeIds += parent.id

  /**
   * Returns an immutable snapshot of the nodes in deterministic insertion order.
   * The vector is new, but its elements are the live mutable Node objects.
   */
  def nodes: Vector[Node] = nodeMap.values.toVector

  /** Returns the number of nodes currently materialized in the lattice. */
  def size = nodeMap.size

  /** Filters a stable node snapshot in parallel while preserving snapshot order. */
  private def parallelNodeSearch(
                                  snapshot: Array[Node],
                                  limit: Int
                                )(predicate: Node => Boolean): Array[Node] =
    if snapshot.length <= 1 || Thread.currentThread().isInstanceOf[ForkJoinWorkerThread] then
      snapshot.filter(predicate).take(limit)
    else
      val parallelSnapshot = snapshot.par
      parallelSnapshot.tasksupport = Lattice.parallelSearchTaskSupport
      parallelSnapshot.filter(predicate).take(limit).toArray

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
    cloned.rebuildBottomNodeIndex()
    cloned.nextId.set(nextId.get())
    cloned
  }

  /**
   * Returns the live node having `id`.
   * Throws NoSuchElementException when the ID is not present in `nodeMap`.
   */
  def node(id: Long): Node = nodeMap(id)

  /**
   * Returns the per-insertion target node for `instances`.
   *
   * The cache key is the set of instance symbols, so distinct construction paths
   * for the same symbolic extent share one fitted CVA node. A cache miss allocates
   * a fresh node and fits its model, but does not add it to the lattice; that is
   * done later by `materialize` or the no-generator branch.
   */
  def targetNode(targetCache: mutable.HashMap[Set[String], Node], instances: List[Instance]): Node =
    val key = instances.iterator.map(_.symbol).toSet
    targetCache.getOrElseUpdate(key, createNode(instances))

  /**
   * Returns `concept`'s CVA distance to the incoming instance, computing it once
   * per insertion. NaN is normalized to positive infinity so invalid numerical
   * output is treated as the least desirable distance during ordering/search.
   */
  def incomingDistance(incomingDistanceCache: mutable.HashMap[Long, Double], concept: Node, instance: Instance): Double = {
    incomingDistanceCache.getOrElseUpdate(
      concept.id,
      {
        val distance = concept.distance(instance.vector)
        if distance.isNaN then Double.PositiveInfinity else distance
      }
    )

  }

  /**
   * Tests whether an existing concept is locally compatible with the incoming item.
   *
   * A concept is coherent when it is within the classification threshold of the
   * item and the singleton target is within `trainingRadiusScale` of every member
   * of the concept. The result is cached by concept ID for this insertion because
   * neither the incoming item nor existing models change during structural search.
   */
  def coherentWithIncoming(coherenceCache: mutable.HashMap[Long, Boolean], incomingDistanceCache: mutable.HashMap[Long, Double], singletonTarget: Node, concept: Node, instance: Instance): Boolean = {
    val distance = incomingDistance(incomingDistanceCache, concept, instance)
    coherenceCache.getOrElseUpdate(
      concept.id,
      distance <= cva.config.classificationThreshold &&
        concept.instances.forall { member => {
          val proximity = singletonTarget.distance(member.vector)
          proximity <= cva.config.trainingRadiusScale
        }}
    )
  }

  /**
   * Implements directional learned containment and caches the answer.
   *
   * It returns true when `left`'s classifier accepts at least `equalityCoverage`
   * of `right`'s instances. Despite the method name, this is not symbolic Set
   * inclusion and is not necessarily symmetric or transitive.
   */
  def intentSubset(subsetCache: mutable.HashMap[(Long, Long), Boolean], left: Node, right: Node): Boolean =
    subsetCache.getOrElseUpdate(
      (left.id, right.id),
      coverage(left, right.instances) >= config.equalityCoverage
    )

  /**
   * Performs a bounded, bottom-up best-first search for a generator of `target`.
   *
   * Search starts at `preferred` when supplied; otherwise it starts from at most
   * `maxBottomStarts` bottom-frontier nodes. Nodes closest to the incoming vector
   * are visited first. An incompatible node contributes its parents to the queue;
   * a compatible node is retained and its less-specific ancestors are pruned.
   * At most `maxSearchVisits` distinct nodes are inspected. Of the compatible
   * frontier, the smallest extent (then lowest ID) is returned.
   *
   * This method only reads lattice topology and populates the supplied caches.
   */
  def maximalConcept(bottomFrontier: mutable.LinkedHashSet[Node],
                     coherenceCache: mutable.HashMap[Long, Boolean],
                     subsetCache: mutable.HashMap[(Long, Long), Boolean],
                     incomingDistanceCache: mutable.HashMap[Long, Double],
                     target: Node, instance: Instance, preferred: Option[Node]): Option[Node] =
    val closestFirst = Ordering.by[Node, (Double, Long)] { concept =>
      (-incomingDistance(incomingDistanceCache, concept, instance), -concept.id)
    }
    val pending = mutable.PriorityQueue.empty[Node](using closestFirst)
    preferred match
      case Some(generator) => pending.enqueue(generator)
      case None =>
        bottomFrontier.iterator.take(maxBottomStarts).foreach(pending.enqueue(_))

    val visited = mutable.HashSet.empty[Long]
    val compatibleFrontier = mutable.ArrayBuffer.empty[Node]

    while pending.nonEmpty && visited.size < maxSearchVisits do
      val concept = pending.dequeue()
      if visited.add(concept.id) then
        if intentSubset(subsetCache, concept, target) && coherentWithIncoming(coherenceCache, incomingDistanceCache, target, concept, instance) then
          // This is the first compatible concept reached on this bottom-up
          // path. Its ancestors are less specific and cannot be maximal.
          compatibleFrontier += concept
        else
          concept.parents.iterator.flatMap(nodeMap.get).foreach(pending.enqueue(_))

    compatibleFrontier.minByOption(concept => (concept.size, concept.id))

  /**
   * Converts a temporary target into its canonical node for this insertion.
   *
   * A local generator search first tries to find an existing node mutually
   * equivalent to `target`. If one exists it is reused; otherwise `target` is
   * inserted into `nodeMap`. The target cache is then redirected to the selected
   * node so later requests for the same symbolic extent reuse it. This method
   * materializes a node but does not create lattice edges.
   */
  def materialize(bottomFrontier: mutable.LinkedHashSet[Node],
                  coherenceCache: mutable.HashMap[Long, Boolean],
                  subsetCache: mutable.HashMap[(Long, Long), Boolean],
                  incomingDistanceCache: mutable.HashMap[Long, Double],
                  targetCache: mutable.HashMap[Set[String], Node],
                  target: Node, instance: Instance): Node =

    val materialized = maximalConcept(bottomFrontier, coherenceCache, subsetCache, incomingDistanceCache, target, instance, None)
      .filter(existing => equivalent(existing, target))
      .getOrElse(registerNode(target))
    targetCache(target.symbols) = materialized
    materialized

  /**
   * Inserts `target` below/around a nearby generator and repairs local cover edges.
   *
   * Algorithm:
   *   1. Find a compatible generator, optionally beginning at `preferredGenerator`.
   *   2. If none exists, insert the target as a bottom node; if the generator is
   *      equivalent, reuse it.
   *   3. Guard the (target symbols, generator ID) pair against recursive cycles.
   *   4. Unless the recursion limit was reached, inspect the closest bounded set
   *      of generator children. Reuse children that contain the target, reject
   *      incoherent children, and recursively insert union candidates for the rest.
   *   5. Remove semantically redundant prospective parents.
   *   6. Materialize the target, remove locally bypassed edges, and link the new
   *      concept in the direction required by strict symbolic extent inclusion.
   *   7. Refresh bottom-frontier membership and return the canonical target node.
   *
   * Existing CVA models are not refitted here; `addIntent` performs that step only
   * after structural insertion so cached predicates remain stable.
   */
  def addIntentRecursively(bottomFrontier: mutable.LinkedHashSet[Node],
                           activeCalls: mutable.HashSet[(Set[String], Long)],
                           coherenceCache: mutable.HashMap[Long, Boolean],
                           subsetCache: mutable.HashMap[(Long, Long), Boolean],
                           incomingDistanceCache: mutable.HashMap[Long, Double],
                           targetCache: mutable.HashMap[Set[String], Node],
                           acceptedIds: Set[Long],
                           target: Node,
                           instance: Instance,
                           preferredGenerator: Option[Node],
                           depth: Int
                          ): Node =
    maximalConcept(bottomFrontier,
      coherenceCache,
      subsetCache, incomingDistanceCache,
      target, instance, preferredGenerator) match
      case None =>
        val materialized = registerNode(target)
        targetCache(target.symbols) = materialized
        if materialized.children.isEmpty then bottomFrontier += materialized
        materialized
      case Some(generator) =>
        if equivalent(generator, target) then generator
        else
          val callKey = (target.symbols, generator.id)
          if !activeCalls.add(callKey) then
            materialize(bottomFrontier, coherenceCache, subsetCache, incomingDistanceCache, targetCache, target, instance)
          else
            val generatorChildren =
              if depth >= maxRecursionDepth then List.empty
              else
                generator.children.iterator
                  .flatMap(nodeMap.get)
                  .toList
                  .sortBy(concept => (incomingDistance(incomingDistanceCache, concept, instance), concept.id))
                  .take(maxChildrenPerGenerator)

            var newParents = List.empty[Node]

            generatorChildren.foreach { originalCandidate =>
              val candidate =
                if intentSubset(subsetCache, originalCandidate, target) then Some(originalCandidate)
                else if !coherentWithIncoming(coherenceCache, incomingDistanceCache, target, originalCandidate, instance) then None
                else
                  val candidateSymbols = originalCandidate.symbols
                  val missingInstances = target.instances.iterator
                    .filterNot(member => candidateSymbols.contains(member.symbol))
                    .toList
                  if missingInstances.isEmpty then
                    Some(originalCandidate)
                  else
                    val intersectionInstances = originalCandidate.instances.toList ::: missingInstances
                    Some(
                      addIntentRecursively(
                        bottomFrontier,
                        activeCalls,
                        coherenceCache,
                        subsetCache,
                        incomingDistanceCache,
                        targetCache,
                        acceptedIds,
                        targetNode(targetCache, intersectionInstances),
                        instance,
                        Some(originalCandidate),
                        depth + 1
                      )
                    )

              candidate.foreach { acceptedCandidate =>
                var addCandidate = acceptedCandidate.id != generator.id
                val snapshot = newParents
                val iterator = snapshot.iterator
                while iterator.hasNext && addCandidate do
                  val parent = iterator.next()
                  if intentSubset(subsetCache, acceptedCandidate, parent) then
                    addCandidate = false
                  else if intentSubset(subsetCache, parent, acceptedCandidate) then
                    newParents = newParents.filterNot(_.id == parent.id)

                if addCandidate then newParents = acceptedCandidate :: newParents
              }
            }

            val newConcept = materialize(bottomFrontier, coherenceCache, subsetCache, incomingDistanceCache, targetCache, target, instance)

            newParents.foreach { parent =>
              if locallyMoreGeneral(acceptedIds, generator, newConcept, instance) &&
                locallyMoreGeneral(acceptedIds, newConcept, parent, instance) then
                unlink(generator, parent)
              if locallyMoreGeneral(acceptedIds, parent, newConcept, instance) then link(bottomFrontier, acceptedIds, parent, newConcept, instance)
              else link(bottomFrontier, acceptedIds, newConcept, parent, instance)
            }

            preferredGenerator.foreach { originalCandidate =>
              if locallyMoreGeneral(acceptedIds, newConcept, originalCandidate, instance) then
                if locallyMoreGeneral(acceptedIds, generator, newConcept, instance) then
                  unlink(generator, originalCandidate)
                link(bottomFrontier, acceptedIds, newConcept, originalCandidate, instance)
            }

            val hasMiddle = newParents.exists { parent =>
              locallyMoreGeneral(acceptedIds, generator, parent, instance) &&
                locallyMoreGeneral(acceptedIds, parent, newConcept, instance)
            }
            if !hasMiddle then
              if locallyMoreGeneral(acceptedIds, generator, newConcept, instance) then link(bottomFrontier, acceptedIds, generator, newConcept, instance)
              else link(bottomFrontier, acceptedIds, newConcept, generator, instance)

            if newConcept.children.isEmpty then bottomFrontier += newConcept
            else bottomFrontier.remove(newConcept)
            activeCalls.remove(callKey)
            newConcept


  /**
   * Returns the symbolic extent used while planning this insertion's edges.
   * Accepted ancestors are viewed as if they already contained the incoming
   * symbol; all other nodes expose their current symbols. No node is mutated.
   */
  def effectiveSymbols(acceptedIds: Set[Long], concept: Node, instance: Instance): Set[String] =
    if acceptedIds.contains(concept.id) then concept.symbols + instance.symbol
    else concept.symbols

  /**
   * Tests the strict symbolic order used for a prospective local edge.
   * `parent` is more general exactly when its effective symbol set is a strict
   * superset of `child`'s effective symbol set.
   */
  def locallyMoreGeneral(acceptedIds: Set[Long], parent: Node, child: Node, instance: Instance): Boolean =
    val parentSymbols = effectiveSymbols(acceptedIds, parent, instance)
    val childSymbols = effectiveSymbols(acceptedIds, child, instance)
    childSymbols.subsetOf(parentSymbols) && parentSymbols != childSymbols

  /**
   * Adds a bidirectional parent -> child edge when strict local symbolic order holds.
   * Duplicate links are harmless because adjacency uses sets. A linked parent is
   * removed from the bottom frontier, while a still-childless child is retained
   * there. Invalid or self links are ignored.
   */
  def link(bottomFrontier: mutable.LinkedHashSet[Node], acceptedIds: Set[Long], parent: Node, child: Node, instance: Instance): Unit =
    if parent.id != child.id && locallyMoreGeneral(acceptedIds, parent, child, instance) then
      parent.children += child.id
      child.parents += parent.id
      bottomNodeIds -= parent.id
      bottomFrontier.remove(parent)
      if child.children.isEmpty then bottomFrontier += child

  /** Links a generator to a new singleton in the fast path.
    * The generator is in acceptedIds, so its effective extent already contains
    * the incoming symbol and is known to be strictly above the new singleton.
    */
  private def linkFast(parent: Node, child: Node): Unit =
    if parent.id != child.id then
      parent.children += child.id
      child.parents += parent.id
      bottomNodeIds -= parent.id

  /** Inserts a cover relation, moving bypassed links through intermediate nodes. */
  private def linkFastCover(acceptedIds: Set[Long], parent: Node, child: Node, instance: Instance): Unit =
    val pending = mutable.ArrayDeque((parent, child))
    val processed = mutable.HashSet.empty[(Long, Long)]
    while pending.nonEmpty do
      val (upper, lower) = pending.removeLast()
      if processed.add((upper.id, lower.id)) && locallyMoreGeneral(acceptedIds, upper, lower, instance) then
        val children = upper.children.iterator.flatMap(nodeMap.get).toVector
        children.find(middle => locallyMoreGeneral(acceptedIds, middle, lower, instance)) match
          case Some(middle) =>
            unlink(upper, lower)
            pending.append((middle, lower))
          case None =>
            val parents = lower.parents.iterator.flatMap(nodeMap.get).toVector
            linkFast(upper, lower)
            parents.filter(locallyMoreGeneral(acceptedIds, _, upper, instance)).foreach { ancestor =>
              unlink(ancestor, lower)
              pending.append((ancestor, upper))
            }
            children.filter(locallyMoreGeneral(acceptedIds, lower, _, instance)).foreach { descendant =>
              unlink(upper, descendant)
              pending.append((lower, descendant))
            }


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

  /**
   * Bounded approximation of incremental CVA-AddIntent insertion.
   *
   * The incoming instance is treated as a singleton intent. Existing nodes that
   * classify it are modified. For sampled non-classifying neighbors, intersection
   * candidates are constructed from oldExtent U {instance}; equivalent candidates
   * are reused and affected edges are updated locally.
   *
   * Candidate intersections are inserted from a small bottom-up neighborhood.
   * Search width, search visits and recursion depth are deliberately capped.
   *
   * Algorithm:
   *   1. Create the first singleton immediately when the lattice is empty.
   *   2. Fit/cache a temporary singleton target and create per-insertion predicate
   *      caches plus a recursion guard.
   *   3. Sample bottom nodes, sort by incoming distance, and retain the closest
   *      bounded frontier.
   *   4. Find a generator and collect its complete ancestor closure as accepted.
   *   5. If no generator exists, build a root/incoming union concept to provide a
   *      usable learned root, then retry generator search.
   *   6. Insert the singleton through `addIntentRecursively` and update local edges.
   *   7. Only after topology is stable, add the instance to accepted ancestors and
   *      refit their models. Delaying this keeps all ID-keyed caches valid.
   *
   * Returns the existing or newly materialized node representing the singleton
   * target. Bounded candidate discovery favors local insertion over exhaustive
   * global AddIntent behavior. Ancestor updates and refits still depend on the
   * existing graph and extent sizes.
   */
  def addIntent(instance: Instance): Node = this.synchronized {
    addIntentUnsafe(instance)
  }

  /** Original AddIntent traversal with learned intent operations.
    *
    * Follows van der Merwe, Obiedkov and Kourie (2004), section 3:
    * https://www.researchgate.net/publication/225177947
    * Intent inclusion means that the left CVA classifies every member of the
    * right extent; equality is mutual inclusion. Intersections fit a CVA to the
    * two extents' union. No symbolic inclusion, coverage tolerance, coherence
    * filter, or search/depth budget participates in these decisions.
    *
    * None represents the paper's empty-extent bottom, whose parents are all
    * childless materialized nodes. It is virtual because Node requires samples.
    * Models remain fixed during traversal; extent propagation and full refits
    * follow insertion. CVA membership need not obey formal closure laws, so the
    * resulting learned graph is still an approximation of a formal lattice.
    */
  def original(instance: Instance): Node = this.synchronized {
    def store(concept: Node): Node =
      nodeMap.put(concept.id, concept)
      if indexesReady then indexNode(concept)
      bottomNodeIds += concept.id
      concept

    if nodeMap.isEmpty then
      return store(createNode(List(instance), doAverage = false))

    val inclusion = mutable.HashMap.empty[(Long, Long), Boolean]
    val created = mutable.ArrayBuffer.empty[Node]

    def included(left: Node, right: Node): Boolean =
      inclusion.getOrElseUpdate((left.id, right.id),
        right.instancesIterator.forall(left.classifies))

    def parentsOf(concept: Option[Node]): Vector[Node] = concept match
      case Some(node) => node.parents.iterator.map(nodeMap).toVector
      case None => nodeMap.valuesIterator.filter(_.children.isEmpty).toVector

    def getMaximalConcept(intent: Node, start: Option[Node]): Option[Node] =
      var generator = start
      var parent = parentsOf(generator).find(included(intent, _))
      while parent.nonEmpty do
        generator = parent
        parent = parentsOf(generator).find(included(intent, _))
      generator

    def connect(parent: Node, child: Node): Unit =
      // Learned inclusion is not necessarily transitive. Graph reachability,
      // rather than symbol sets, prevents a numerical inconsistency making a cycle.
      if parent.id != child.id && !ancestorClosure(parent).contains(child.id) then
        linkFast(parent, child)

    def insert(intent: Node, start: Option[Node]): Node =
      val generator = getMaximalConcept(intent, start)
      generator match
        case Some(existing) if included(existing, intent) && included(intent, existing) => existing
        case _ =>
          // Snapshot before recursion, which can replace these parent links.
          val generatorParents = parentsOf(generator)
          var newParents = Vector.empty[Node]
          generatorParents.foreach { oldParent =>
            val candidate =
              if included(oldParent, intent) then oldParent
              else
                val intersection = createNode(oldParent.instances ++ intent.instances, doAverage = false)
                insert(intersection, Some(oldParent))

            var addCandidate = true
            val iterator = newParents.iterator
            while iterator.hasNext && addCandidate do
              val parent = iterator.next()
              if included(candidate, parent) then addCandidate = false
              else if included(parent, candidate) then
                newParents = newParents.filterNot(_.id == parent.id)
            if addCandidate then newParents :+= candidate
          }

          // Copy the generator's extent while retaining the requested intent.
          // Target samples also stand in for the incoming object's intent and
          // keep virtual-bottom insertions nonempty. Refit only after traversal.
          val extent = generator.fold(intent.instances)(_.instances ++ intent.instances)
          val concept = store(new Node(nextId.getAndIncrement(), extent, cva,
            restoredModel = Some(intent.model)))
          created += concept
          newParents.foreach { parent =>
            generator.foreach(unlink(parent, _))
            connect(parent, concept)
          }
          generator.foreach(connect(concept, _))
          concept

    val objectNode = insert(createNode(List(instance), doAverage = false), None)

    // Add the object to its complete ancestor closure. Also propagate copied
    // generator extents: approximate CVA equality can reuse a parent whose
    // recorded extent does not yet contain every descendant sample.
    val changed = mutable.LinkedHashSet.from(created)
    if objectNode.addInstance(instance) then changed += objectNode
    val pending = mutable.ArrayDeque.from(created.iterator ++ Iterator(objectNode))
    val queued = mutable.HashSet.from(pending.iterator.map(_.id))
    while pending.nonEmpty do
      val child = pending.removeHead()
      queued -= child.id
      val members = child.instances
      child.parents.iterator.map(nodeMap).foreach { parent =>
        var extended = false
        members.foreach { member =>
          if parent.addInstance(member) then extended = true
        }
        if extended then
          changed += parent
          if queued.add(parent.id) then pending.append(parent)
      }
    changed.foreach(_.refit())
    objectNode
  }

  /** Faster bounded insertion for large training runs.
    *
    * Keeps closest-first generator search and computes extent-union CVA
    * intersections in a bounded bottom/child neighborhood. Parent intersections
    * are inserted recursively and bypassed cover links are repaired before
    * updating the accepted ancestor extents. Full coherence scans are omitted.
    * Average models use online means. CVA models reuse append-only bases with
    * an unrounded Float32 projector and the same results as a complete refit.
    * A bounded per-lattice LRU releases bases for concepts no longer in use.
    */
  def addIntentFast(instance: Instance): Node = this.synchronized {
    addIntentFastUnsafe(instance)
  }

  /** Same search, equivalence, extent updates and edges as addIntentFast.
    * Adds exact-extent indexes and per-insertion equivalence memoization.
    * Changed CVA extents use the same bounded basis reuse as addIntentFast.
    * Identity-projector singleton bounds reject only impossible matches; all
    * possible matches still use the original tensor scores and tie breaking.
    */
  def addIntentFastIndexed(instance: Instance): Node = this.synchronized {
    ensureExtentIndexes()
    addIntentFastUnsafe(instance, indexedSearch = true)
  }

  private def addIntentFastUnsafe(instance: Instance, indexedSearch: Boolean = false): Node =
    if nodeMap.isEmpty then
      val first = createNode(List(instance))
      return registerNode(first)

    // Scores depend on this incoming sample and the current models. Keep them
    // local so neither later insertions nor other lattices can reuse them.
    val incomingDistanceCache = mutable.HashMap.empty[Long, Double]
    val singletonTarget = createNode(List(instance))
    val limits = fastSearchLimits(nodeMap.size)

    // Models stay fixed until structural insertion finishes. Extents overlap
    // heavily, so retain scalar CVA results by tensor identity across candidates.
    // Bound both the number of classifiers and scores, and never share them
    // across insertions or lattices. Identity keys also distinguish different
    // vectors supplied with the same symbolic label.
    val classificationScores = mutable.LongMap.empty[
      java.util.IdentityHashMap[Tensor[BFloat16], java.lang.Boolean]]
    var cachedClassifications = 0
    def scoresFor(classifier: Node): java.util.IdentityHashMap[Tensor[BFloat16], java.lang.Boolean] =
      classificationScores.getOrElse(classifier.id, {
        if classificationScores.size >= 2048 || cachedClassifications >= 65536 then null
        else
          val scores = new java.util.IdentityHashMap[Tensor[BFloat16], java.lang.Boolean]()
          classificationScores.put(classifier.id, scores)
          scores
      })

    def cachedClassifies(
        classifier: Node,
        scores: java.util.IdentityHashMap[Tensor[BFloat16], java.lang.Boolean],
        vector: Tensor[BFloat16]
    ): Boolean =
      val known = if scores == null then null else scores.get(vector)
      if known != null then known.booleanValue()
      else
        val result =
          if vector eq instance.vector then
            incomingDistanceCache.get(classifier.id)
              .map(distance => distance.isFinite && distance <= classifier.classificationRadius)
              .getOrElse(classifier.classifies(vector))
          else classifier.classifies(vector)
        if scores != null && cachedClassifications < 65536 && scores.size() < 4096 then
          scores.put(vector, java.lang.Boolean.valueOf(result))
          cachedClassifications += 1
        result

    def cachedCoverage(classifier: Node, extent: Node): Boolean =
      val scores = scoresFor(classifier)
      val total = extent.size
      var accepted = 0
      var remaining = total
      val members = extent.instancesIterator
      while members.hasNext do
        if cachedClassifies(classifier, scores, members.next().vector) then accepted += 1
        remaining -= 1
        if accepted.toDouble / total >= config.equalityCoverage then return true
        if (accepted + remaining).toDouble / total < config.equalityCoverage then return false
      total == 0

    def equivalentIntersection(existing: Node, target: Node): Boolean =
      cachedCoverage(existing, target) && cachedCoverage(target, existing) &&
        (!config.requireMutualCommonClassification ||
          (cachedClassifies(existing, scoresFor(existing), target.commonVector) &&
            cachedClassifies(target, scoresFor(target), existing.commonVector)))

    /** Measure independent model distances on workers, then update the mutable
      * per-insertion cache on this thread before queue ordering reads it.
      */
    def cacheDistancesParallel(concepts: Array[Node]): Unit =
      val missing = concepts.filterNot(concept => incomingDistanceCache.contains(concept.id))
      val measured =
        // A parallel collection invoked by another fork-join worker can park
        // while holding the lattice monitor and strand its nested task. In that
        // context sequential evaluation is bounded and guarantees progress.
        if missing.length < 16 || Thread.currentThread().isInstanceOf[ForkJoinWorkerThread] then
          missing.map { concept =>
            val distance = concept.distance(instance.vector)
            (concept.id, if distance.isNaN then Double.PositiveInfinity else distance)
          }
        else
          val parallelMissing = missing.par
          parallelMissing.tasksupport = Lattice.parallelSearchTaskSupport
          parallelMissing.map { concept =>
            val distance = concept.distance(instance.vector)
            (concept.id, if distance.isNaN then Double.PositiveInfinity else distance)
          }.toArray
      measured.foreach { case (id, distance) => incomingDistanceCache.put(id, distance) }

    val closestFirst = Ordering.by[Node, (Double, Long)] { concept =>
      (-incomingDistance(incomingDistanceCache, concept, instance), -concept.id)
    }
    val pending = mutable.PriorityQueue.empty[Node](using closestFirst)
    val scheduled = mutable.HashSet.empty[Long]
    val bottomCandidates = bottomNodeIds.iterator
      .flatMap(nodeMap.get)
      .take(limits.bottomSamples)
      .toArray
    cacheDistancesParallel(bottomCandidates)
    bottomCandidates
      .sortBy(concept => (incomingDistance(incomingDistanceCache, concept, instance), concept.id))
      .take(limits.bottomStarts)
      .foreach { concept =>
        if scheduled.add(concept.id) then pending.enqueue(concept)
      }

    val visited = mutable.HashSet.empty[Long]
    val compatible = mutable.ArrayBuffer.empty[Node]
    while pending.nonEmpty && visited.size < limits.searchVisits do
      val concept = pending.dequeue()
      if visited.add(concept.id) then
        val distance = incomingDistance(incomingDistanceCache, concept, instance)
        val acceptsSingleton = distance <= concept.classificationRadius
        val isNear = distance <= cva.config.classificationThreshold
        if acceptsSingleton && isNear then compatible += concept
        else
          val remainingSlots = limits.searchVisits - scheduled.size
          if remainingSlots > 0 then
            val parents = concept.parents.iterator
              .flatMap(nodeMap.get)
              .filter(parent => !visited.contains(parent.id) && scheduled.add(parent.id))
              .take(remainingSlots)
              .toArray
            cacheDistancesParallel(parents)
            parents.foreach(pending.enqueue(_))

    def fastRank(concept: Node): (Double, Int, Long) =
      (incomingDistance(incomingDistanceCache, concept, instance), concept.size, concept.id)

    // The target has one member: its forward coverage is already cached by
    // generator search. Reuse that score during every duplicate check.
    val equivalenceCache = mutable.LongMap.empty[Boolean]
    lazy val incomingValues = numeric.BFloatTensors.toArray(instance.vector)
    def singletonMayMatch(concept: Node): Boolean =
      val possible = !indexedSearch || singletonCenters.get(concept.id).forall { center =>
        SingletonBounds.mayClassify(center, incomingValues, concept.classificationRadius)
      }
      if !possible then equivalenceCache.put(concept.id, false)
      possible

    def equivalentToSingleton(concept: Node): Boolean =
      equivalenceCache.getOrElseUpdate(concept.id,
        incomingDistance(incomingDistanceCache, concept, instance) <= concept.classificationRadius &&
          singletonCoversEnough(singletonTarget, concept) &&
          (!config.requireMutualCommonClassification ||
            (concept.classifies(singletonTarget.commonVector) &&
              singletonTarget.classifies(concept.commonVector))))

    def fastAncestorClosure(starts: Iterable[Node]): Set[Long] =
      val startNodes = starts.toArray
      cacheDistancesParallel(startNodes)
      val ancestorOrdering = Ordering.by[Node, (Double, Long)] { concept =>
        (-incomingDistance(incomingDistanceCache, concept, instance), -concept.id)
      }
      val ancestors = mutable.PriorityQueue.empty[Node](using ancestorOrdering)
      val scheduledAncestors = mutable.HashSet.empty[Long]
      startNodes
        .sortBy(fastRank)
        .take(limits.ancestorStarts)
        .foreach { start =>
          if scheduledAncestors.add(start.id) then ancestors.enqueue(start)
        }

      val closure = mutable.LinkedHashSet.empty[Long]
      // Once a concept grows, every existing ancestor must receive the symbol;
      // truncating this closure would invalidate parent/child extent inclusion.
      while ancestors.nonEmpty do
        val concept = ancestors.dequeue()
        if closure.add(concept.id) then
          val parents = concept.parents.iterator
            .flatMap(nodeMap.get)
            .filter(parent => scheduledAncestors.add(parent.id))
            .toArray
          cacheDistancesParallel(parents)
          parents.foreach(ancestors.enqueue(_))
      closure.toSet

    val searchedNodes = visited.iterator.flatMap(nodeMap.get).toArray
    val existingSingletons =
      if indexesReady then singletonIndex.valuesIterator.toArray
      else nodeMap.valuesIterator.filter(_.size == 1).toArray
    val existingObject = existingSingletons
      .find(_.containsSymbol(instance.symbol))
      .orElse {
        val candidates = existingSingletons.filter(singletonMayMatch)
        cacheDistancesParallel(candidates)
        candidates
          .sortBy(fastRank)
          .find(equivalentToSingleton)
      }
      .orElse {
        searchedNodes
          .sortBy(fastRank)
          .find(equivalentToSingleton)
      }

    val objectNode = existingObject.getOrElse(registerNode(singletonTarget, equivalentToSingleton))
    val rankedCompatible = compatible.sortBy(fastRank)
    val generator = rankedCompatible.headOption
    val extentSeeds = (Seq(objectNode) ++ rankedCompatible)
      .foldLeft(Vector.empty[Node]) { (selected, candidate) =>
        if selected.exists(_.id == candidate.id) then selected else selected :+ candidate
      }
      .take(limits.ancestorStarts)
    val acceptedIds = fastAncestorClosure(extentSeeds) + objectNode.id

    val extendedIds = mutable.LinkedHashSet.empty[Long]

    /** Extending a candidate must not collapse an existing strict parent edge.
      * Every ancestor will receive the same missing samples, so each edge must
      * retain at least one distinguishing parent member after that propagation.
      */
    def canExtend(existing: Node, symbols: Set[String]): Boolean =
      ancestorClosure(existing).forall { id =>
        val child = nodeMap(id)
        val grown = effectiveSymbols(acceptedIds, child, instance) ++ symbols
        child.parents.forall(parentId =>
          !effectiveSymbols(acceptedIds, nodeMap(parentId), instance).subsetOf(grown))
      }

    def extendIntersection(existing: Node, members: Vector[Instance]): Unit =
      ancestorClosure(existing).toVector.sorted.foreach { id =>
        val concept = nodeMap(id)
        if concept.addInstancesWithoutRefit(members) then extendedIds += id
      }

    def objectSymbols = effectiveSymbols(acceptedIds, objectNode, instance)
    val childCandidates = generator.toArray.flatMap(_.children.iterator
      .flatMap(nodeMap.get).take(maxChildrenPerGenerator))
    val intersectionCandidates = (bottomCandidates.sortBy(fastRank).take(maxChildrenPerGenerator) ++
      childCandidates).distinctBy(_.id)
    cacheDistancesParallel(intersectionCandidates)
    val intersectionStarts = intersectionCandidates.sortBy(fastRank).take(maxChildrenPerGenerator)

    // Snapshot the old parent paths before cover insertion moves any edges.
    // The existing depth/width and seed budgets also bound intersection work.
    val intersectionParents = mutable.LongMap.empty[Array[Node]]
    val toSnapshot = mutable.ArrayDeque.from(intersectionStarts.map(node => (node, 0)))
    while toSnapshot.nonEmpty && intersectionParents.size < limits.ancestorStarts do
      val (concept, depth) = toSnapshot.removeHead()
      val symbols = effectiveSymbols(acceptedIds, concept, instance)
      if !intersectionParents.contains(concept.id) &&
        !objectSymbols.subsetOf(symbols) && !symbols.subsetOf(objectSymbols) then
        val parents =
          if depth >= maxRecursionDepth then Array.empty[Node]
          else concept.parents.iterator.flatMap(nodeMap.get).take(maxChildrenPerGenerator).toArray
        intersectionParents.put(concept.id, parents)
        parents.foreach(parent => toSnapshot.append((parent, depth + 1)))

    val intersectionTargets = mutable.HashMap.empty[Set[String], Node]
    val intersectionResults = mutable.LongMap.empty[Node]
    val pendingExtents = mutable.HashMap.empty[Set[String], Node]
    acceptedIds.iterator.flatMap(nodeMap.get).toVector.sortBy(_.id).foreach { node =>
      pendingExtents.getOrElseUpdate(effectiveSymbols(acceptedIds, node, instance), node)
    }
    def insertIntersection(concept: Node): Option[Node] =
      val symbols = effectiveSymbols(acceptedIds, concept, instance)
      if objectSymbols.subsetOf(symbols) then Some(concept)
      else if symbols.subsetOf(objectSymbols) then Some(objectNode)
      else intersectionResults.get(concept.id).orElse {
        intersectionParents.get(concept.id).map { parents =>
          val unionSymbols = symbols ++ objectSymbols
          val intersection = intersectionTargets.getOrElseUpdate(unionSymbols, {
            exactExtent(unionSymbols)
              .orElse(pendingExtents.get(unionSymbols))
              .getOrElse {
                val members = mutable.LinkedHashMap.empty[String, Instance]
                (concept.instancesIterator ++ objectNode.instancesIterator ++ Iterator.single(instance))
                  .foreach(member => members.getOrElseUpdate(member.symbol, member))
                val target = createNode(members.values, doAverage = false)
                // Learned equality supplies candidates even with incomplete
                // symbolic extents. Expand a reused node before placing edges;
                // keep its model fixed until all structural work is finished.
                val intersection = registerNode(target, existing =>
                  equivalentIntersection(existing, target) && canExtend(existing, unionSymbols))
                // A newly registered target already owns its complete extent
                // and has no parents. Only reused nodes need propagation.
                if intersection.id != target.id then extendIntersection(intersection, target.instances)
                intersection
              }
          })
          intersectionResults.put(concept.id, intersection)
          parents.foreach { parent =>
            insertIntersection(parent).foreach { upper =>
              linkFastCover(acceptedIds, upper, intersection, instance)
            }
          }
          linkFastCover(acceptedIds, intersection, concept, instance)
          linkFastCover(acceptedIds, intersection, objectNode, instance)
          intersection
        }
      }

    generator.foreach { parent =>
      linkFastCover(acceptedIds, parent, objectNode, instance)
    }
    intersectionStarts.foreach { concept =>
      insertIntersection(concept).foreach { intersection =>
        linkFastCover(acceptedIds, intersection, concept, instance)
        linkFastCover(acceptedIds, intersection, objectNode, instance)
      }
    }

    acceptedIds.foreach { conceptId =>
      nodeMap.get(conceptId).foreach { concept =>
        if extendedIds.contains(conceptId) then concept.addInstance(instance)
        else if cva.config.noCVA then concept.addInstanceFast(instance)
        else if concept.addInstance(instance) then
          refitFast(concept)
      }
    }
    extendedIds.foreach { id =>
      val concept = nodeMap(id)
      if cva.config.noCVA then concept.refit()
      else refitFast(concept)
    }
    objectNode

  /** The lattice graph and its linked maps form one mutation transaction. */
  private def addIntentUnsafe(instance: Instance): Node =
    // Bounded AddIntent approximation. These limits keep insertion cost tied
    // to a small local neighborhood instead of total lattice size.

    if nodeMap.isEmpty then
      val first = createNode(List(instance))
      return registerNode(first)

    val targetCache = mutable.HashMap.empty[Set[String], Node]
    var acceptedIds = Set.empty[Long]

    val singletonTarget = targetNode(targetCache, List(instance))
    val subsetCache = mutable.HashMap.empty[(Long, Long), Boolean]
    val incomingDistanceCache = mutable.HashMap.empty[Long, Double]
    val coherenceCache = mutable.HashMap.empty[Long, Boolean]
    val activeCalls = mutable.HashSet.empty[(Set[String], Long)]

    // Sample a fixed-size bottom neighborhood, then retain only its closest
    // representatives. This prevents insertion order from deciding the search.
    val bottomFrontier = mutable.LinkedHashSet.from(
      bottomNodeIds.iterator
        .flatMap(nodeMap.get)
        .take(maxBottomSamples)
        .toArray
        .sortBy(concept => (incomingDistance(incomingDistanceCache, concept, instance), concept.id))
        .take(maxBottomStarts))


    val existingGenerator = maximalConcept(bottomFrontier, coherenceCache, subsetCache, incomingDistanceCache, singletonTarget, instance, None)
    val accepted = existingGenerator.map(ancestorClosure).getOrElse(Set.empty)
    acceptedIds = accepted

    if existingGenerator.isEmpty then {
      val nodeSnapshot = nodeMap.valuesIterator.toArray
      val rootCandidates = parallelNodeSearch(nodeSnapshot, maxBottomSamples)(_.parents.isEmpty).par
      rootCandidates.tasksupport = Lattice.parallelSearchTaskSupport

      // The mutable insertion caches must not be updated by worker threads.
      // Snapshot their current values, calculate independent root scores in
      // parallel, then publish the results and select the minimum sequentially.
      val knownDistances = incomingDistanceCache.toMap
      val knownCoherence = coherenceCache.toMap
      val scoredRoots = rootCandidates.map { root =>
        val distance = knownDistances.getOrElse(
          root.id,
          {
            val measured = root.distance(instance.vector)
            if measured.isNaN then Double.PositiveInfinity else measured
          }
        )
        val coherent = knownCoherence.getOrElse(
          root.id,
          distance <= cva.config.classificationThreshold &&
            root.instances.forall { member =>
              singletonTarget.distance(member.vector) <= cva.config.trainingRadiusScale
            }
        )
        (root, distance, coherent)
      }.toArray

      scoredRoots.foreach { case (root, distance, coherent) =>
        incomingDistanceCache.put(root.id, distance)
        coherenceCache.put(root.id, coherent)
      }
      scoredRoots.iterator
        .filter(_._3)
        .minByOption { case (root, distance, _) => (distance, root.id) }
        .map(_._1)
        .foreach { root => {
          val rootIntersection =
            if root.symbols.contains(instance.symbol) then root.instances.toList
            else root.instances.toList :+ instance
          val intersection = materialize(bottomFrontier,
            coherenceCache,
            subsetCache,
            incomingDistanceCache,
            targetCache,
            targetNode(targetCache, rootIntersection),
            instance)
          if intersection.id != root.id then
            link(bottomFrontier, acceptedIds, intersection, root, instance)
        }
        }
    }

    val insertionGenerator =
      existingGenerator.orElse(maximalConcept(bottomFrontier, coherenceCache, subsetCache, incomingDistanceCache, singletonTarget, instance, None))
    val objectNode = addIntentRecursively(bottomFrontier, activeCalls, coherenceCache, subsetCache, incomingDistanceCache, targetCache, acceptedIds, singletonTarget, instance, insertionGenerator, depth = 0)

    // Models stayed unchanged during structural insertion; refit the accepted
    // ancestor closure now that all local edges have been placed.
    (accepted ++ ancestorClosure(objectNode)).foreach { conceptId =>
      nodeMap.get(conceptId).foreach { concept =>
        if concept.addInstance(instance) then concept.refit()
      }
    }
    objectNode


  /**
   * Prints a random diagnostic sample of sufficiently large nodes.
   * Eligible nodes are shuffled with the supplied Random, truncated to `nodeCount`,
   * sorted by descending extent size and ID, then printed with parent/child IDs.
   * Supplying a seeded Random makes the output reproducible.
   */
  def printRandomNodeSymbols(
                              nodeCount: Int = 3,
                              symbolsPerNode: Int = 20,
                              random: Random = Random
                            ): Unit = {

    val selected = random
      .shuffle(nodes.iterator.filter(_.instances.length >= symbolsPerNode).toIndexedSeq)
      .take(nodeCount)
    
    println(selected.sortBy(n => (-n.size, n.id)).map { n =>
      val ps = n.parents.mkString("[", ",", "]")
      val cs = n.children.mkString("[", ",", "]")
      s"${n.toString}, parents=$ps, children=$cs"
    }.mkString("\n"))

  }


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

  /** Score singleton coverage in bounded tensor batches. Each row retains the
    * same BFloat16 residual and reduction as the scalar distance calculation.
    * Only decoded sums escape the native pointer scope.
    */
  private[cvafca] def singletonCoversEnough(singleton: Node, extent: Node): Boolean =
    require(singleton.size == 1)
    if extent.size < 16 then return coversEnough(singleton, extent)
    val total = extent.size
    val instances = extent.instancesIterator
    var accepted = 0
    var remaining = total
    while instances.hasNext do
      val batch = instances.take(64).map(_.vector).toVector
      val squaredDistances = Using.resource(new PointerScope()) { _ =>
        val vectors = torch.stack(batch.map(_.flatten))
        val projected =
          if cva.config.noCVA then vectors
          else singleton.projection.matmul(vectors.transpose(0, 1)).transpose(0, 1)
        val residual = projected - singleton.commonVector.flatten
        val squared = (residual * residual).native.sum(
          Array(1L), false,
          new org.bytedeco.pytorch.ScalarTypeOptional(
            org.bytedeco.pytorch.global.torch.ScalarType.BFloat16))
        numeric.BFloatTensors.toArray(Tensor.fromNative[BFloat16](squared))
      }
      squaredDistances.foreach { squared =>
        val euclidean = math.sqrt(squared.toDouble)
        val distance = if euclidean.isPosInfinity then 1.0 else euclidean / (1.0 + euclidean)
        if distance <= singleton.classificationRadius then accepted += 1
      }
      remaining -= batch.size
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
   * Unlike `locallyMoreGeneral`, this method does not preview pending insertion.
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

      def reportProgress(stage: String, completed: Int, force: Boolean = false): Unit =
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
            loaded.identityProjectionKnown = restoredModel.isEmpty || restoredModel.exists { model =>
              identityProjection.exists(_ eq model.projection)
            }
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

      invalidateExtentIndexes()
      nodeMap.valuesIterator.foreach(_.clearBasisCache())
      fastBasisNodes.clear()
      nodeMap.clear()
      nodeMap.addAll(loadedNodes)
      rebuildBottomNodeIndex()
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
  private def createNode(instances: Iterable[Instance], doAverage: Boolean = true): Node =
    new Node(nextId.getAndIncrement(), instances, cva, do_avg = doAverage)

private object Lattice:
  val FileMagic = 0x43564146 // "CVAF"
  val FileVersion = 2
  val parallelSearchTaskSupport = new ForkJoinTaskSupport(
    new ForkJoinPool(math.max(1, Runtime.getRuntime.availableProcessors()))
  )
