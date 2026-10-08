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
                                equalityCoverage: Double = 1.0,

                                /** If true, also require mutual common-vector classification. */
                                requireMutualCommonClassification: Boolean = false,

                                /** Fast-search budgets stay at their maximum through this node count. */
                                fastBudgetReferenceNodes: Int = 4096,

                                /** Lowest fraction of each fast-search budget used on large lattices. */
                                fastMinimumBudgetScale: Double = 0.125
                              )

/**
 * Incremental FCA-like lattice whose intents are CVA common vectors.
 *
 * Semantics
 * ---------
 * 1. node.classifies(x) is the subset/membership test.
 * 2. Two nodes are equivalent when their CVA models mutually classify the
 *    other's extent (up to equalityCoverage), optionally also their common vectors.
 * 3. Intersection(a,b) is a new CVA model fitted to extent(a) U extent(b).
 * 4. Adding an instance to an existing node refits its common vector and projection.
 * 5. Symbolic extent inclusion is used ONLY to impose a true partial order for
 *    lattice edges. Learned classifier inclusion can be non-transitive, so using
 *    it alone for topology can create cycles.
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
  private val maxRecursionDepth = 7
  private val maxSearchVisits = 400
  private val maxBottomSamples = 500000
  private val maxBottomStarts = 256
  private val maxChildrenPerGenerator = 32
  private val fastBottomSamples = 64
  private val fastBottomStarts = 16
  private val fastSearchVisits = 256
  private val fastAncestorRefits = 16

  private final case class FastSearchLimits(
                                              bottomSamples: Int,
                                              bottomStarts: Int,
                                              searchVisits: Int,
                                              ancestorRefits: Int
                                            )

  private def fastSearchLimits(nodeCount: Int): FastSearchLimits =
    val sizeScale = math.sqrt(
      config.fastBudgetReferenceNodes.toDouble / math.max(1, nodeCount).toDouble
    )
    val scale = math.max(config.fastMinimumBudgetScale, math.min(1.0, sizeScale))
    def scaled(maximum: Int, minimum: Int): Int =
      math.max(minimum, math.round(maximum * scale).toInt)
    FastSearchLimits(
      bottomSamples = scaled(fastBottomSamples, 8),
      bottomStarts = scaled(fastBottomStarts, 2),
      searchVisits = scaled(fastSearchVisits, 32),
      ancestorRefits = scaled(fastAncestorRefits, 1)
    )

  private val nextId = new AtomicLong(0L)
  /** Protected so alternative insertion engines can reuse the same lattice state. */
  protected val nodeMap = mutable.LinkedHashMap.empty[Long, Node]
  /** Childless-node IDs with constant-time insertion/removal and bounded iteration. */
  private val bottomNodeIds = mutable.LinkedHashSet.empty[Long]

  private def registerNode(node: Node): Unit =
    nodeMap.put(node.id, node)
    if node.children.isEmpty then bottomNodeIds += node.id
    else bottomNodeIds -= node.id

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
      .getOrElse {
        registerNode(target)
        target
      }
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
        registerNode(target)
        bottomFrontier += target
        target
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


  /**
   * Collects `start` and a bounded set of its ancestors by following parent IDs.
   * The deque provides depth-first-like traversal, the hash set prevents repeated
   * visits/cycles, and `maxSearchVisits` bounds the closure. Returned IDs identify
   * the existing concepts that will receive and refit for the incoming instance.
   */
  def ancestorClosure(start: Node): Set[Long] =
    val pending = mutable.ArrayDeque(start)
    val closure = mutable.HashSet.empty[Long]
    while pending.nonEmpty && closure.size < maxSearchVisits do
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
   *   4. Find a generator and collect its bounded ancestor closure as accepted.
   *   5. If no generator exists, build a root/incoming union concept to provide a
   *      usable learned root, then retry generator search.
   *   6. Insert the singleton through `addIntentRecursively` and update local edges.
   *   7. Only after topology is stable, add the instance to accepted ancestors and
   *      refit their models. Delaying this keeps all ID-keyed caches valid.
   *
   * Returns the existing or newly materialized node representing the singleton
   * target. Because every search bound is fixed, this favors predictable local
   * insertion cost over exhaustive global AddIntent behavior.
   */
  def addIntent(instance: Instance): Node = this.synchronized {
    addIntentUnsafe(instance)
  }

  /** Faster bounded insertion for large training runs.
    *
    * This keeps closest-first generator search and ancestor updates, but omits
    * full-extent coherence scans and recursive intersection construction. It
    * samples a small bottom frontier, links the singleton directly below the
    * most specific compatible generator, and refits a small ancestor closure.
    */
  def addIntentFast(instance: Instance): Node = this.synchronized {
    addIntentFastUnsafe(instance)
  }

  private def addIntentFastUnsafe(instance: Instance): Node =
    if nodeMap.isEmpty then
      val first = createNode(List(instance))
      registerNode(first)
      return first

    val incomingDistanceCache = mutable.HashMap.empty[Long, Double]
    val singletonTarget = createNode(List(instance))
    val limits = fastSearchLimits(nodeMap.size)

    /** Measure independent model distances on workers, then update the mutable
      * per-insertion cache on this thread before queue ordering reads it.
      */
    def cacheDistancesParallel(concepts: Array[Node]): Unit =
      val missing = concepts.filterNot(concept => incomingDistanceCache.contains(concept.id))
      val measured =
        // A parallel collection invoked by another fork-join worker can park
        // while holding the lattice monitor and strand its nested task. In that
        // context sequential evaluation is bounded and guarantees progress.
        if missing.length <= 1 || Thread.currentThread().isInstanceOf[ForkJoinWorkerThread] then
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

    val generator = compatible.minByOption(concept => (concept.size, concept.id))
    val acceptedIds = generator.map { start =>
      val closure = mutable.LinkedHashSet.empty[Long]
      val ancestors = mutable.ArrayDeque(start)
      val scheduledAncestors = mutable.HashSet(start.id)
      while ancestors.nonEmpty && closure.size < limits.ancestorRefits do
        val concept = ancestors.removeLast()
        if closure.add(concept.id) then
          val remainingSlots = limits.ancestorRefits - scheduledAncestors.size
          if remainingSlots > 0 then
            concept.parents.iterator
              .flatMap(nodeMap.get)
              .filter(parent => scheduledAncestors.add(parent.id))
              .take(remainingSlots)
              .foreach(ancestors.append)
      closure.toSet
    }.getOrElse(Set.empty[Long])

    val objectNode = generator
      .filter(concept => concept.size == 1 && concept.containsSymbol(instance.symbol))
      .getOrElse {
        registerNode(singletonTarget)
        singletonTarget
      }
    generator.foreach { parent =>
      if parent.id != objectNode.id then linkFast(parent, objectNode)
    }

    acceptedIds.foreach { conceptId =>
      nodeMap.get(conceptId).foreach { concept =>
        concept.addInstanceFast(instance)
      }
    }
    objectNode

  /** The lattice graph and its linked maps form one mutation transaction. */
  private def addIntentUnsafe(instance: Instance): Node =
    // Bounded AddIntent approximation. These limits keep insertion cost tied
    // to a small local neighborhood instead of total lattice size.

    if nodeMap.isEmpty then
      val first = createNode(List(instance))
      registerNode(first)
      return first

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
    accepted.foreach { conceptId =>
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
                              symbolsPerNode: Int = 2,
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
    val coverageAB = coverage(a, b.instances)
    val coverageBA = coverage(b, a.instances)

    val extentCondition =
      coverageAB >= config.equalityCoverage &&
        coverageBA >= config.equalityCoverage

    val commonCondition =
      if !config.requireMutualCommonClassification then true
      else a.classifies(b.commonVector) && b.classifies(a.commonVector)

    extentCondition && commonCondition

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
  private def createNode(instances: Iterable[Instance]): Node =
    new Node(nextId.getAndIncrement(), instances, cva)

private object Lattice:
  val FileMagic = 0x43564146 // "CVAF"
  val FileVersion = 2
  val parallelSearchTaskSupport = new ForkJoinTaskSupport(
    new ForkJoinPool(math.max(1, Runtime.getRuntime.availableProcessors()))
  )
