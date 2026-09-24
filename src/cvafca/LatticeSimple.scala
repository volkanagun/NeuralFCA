package cvafca

import scala.collection.mutable

/**
 * A loop-based version of [[Lattice]].
 *
 * It preserves Lattice's insertion rules but replaces recursive calls with an
 * explicit stack of frames. `maxDepth = 1` matches the original bounded
 * recursion. Larger values explore child-union candidates for more levels
 * without consuming the JVM call stack.
 */
final class LatticeSimple(
    cva: CVA,
    config: LatticeConfig = LatticeConfig(),
    val maxDepth: Int = 1,
    val maxChildrenPerFrame: Int = 4
) extends Lattice(cva, config):

  require(maxDepth >= 0, "maxDepth cannot be negative")
  require(maxChildrenPerFrame > 0, "maxChildrenPerFrame must be positive")

  private final class Frame(
      val target: Node,
      val preferredGenerator: Option[Node],
      val depth: Int
  ):
    var initialized = false
    var generator: Node = null
    var callKey: Option[(Set[String], Long)] = None
    var children = List.empty[Node]
    var nextChild = 0
    var waitingForChild = false
    var newParents = List.empty[Node]

  override def addIntentRecursively(
      bottomFrontier: mutable.LinkedHashSet[Node],
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
    val frames = mutable.ArrayDeque.empty[Frame]
    frames.append(new Frame(target, preferredGenerator, depth))
    var completed: Option[Node] = None

    def finish(result: Node): Unit =
      frames.removeLast()
      completed = Some(result)

    def acceptParent(frame: Frame, candidate: Node): Unit =
      var addCandidate = candidate.id != frame.generator.id
      val iterator = frame.newParents.iterator
      while iterator.hasNext && addCandidate do
        val parent = iterator.next()
        if intentSubset(subsetCache, candidate, parent) then
          addCandidate = false
        else if intentSubset(subsetCache, parent, candidate) then
          frame.newParents = frame.newParents.filterNot(_.id == parent.id)
      if addCandidate then frame.newParents = candidate :: frame.newParents

    while frames.nonEmpty do
      val frame = frames.last

      // A nested frame has finished; its returned node is this frame's candidate.
      if frame.waitingForChild && completed.nonEmpty then
        acceptParent(frame, completed.get)
        completed = None
        frame.waitingForChild = false

      if !frame.initialized then
        maximalConcept(
          bottomFrontier,
          coherenceCache,
          subsetCache,
          incomingDistanceCache,
          frame.target,
          instance,
          frame.preferredGenerator
        ) match
          case None =>
            nodeMap.put(frame.target.id, frame.target)
            bottomFrontier += frame.target
            finish(frame.target)

          case Some(generator) if equivalent(generator, frame.target) =>
            finish(generator)

          case Some(generator) =>
            val key = (frame.target.symbols, generator.id)
            if !activeCalls.add(key) then
              finish(materialize(
                bottomFrontier, coherenceCache, subsetCache,
                incomingDistanceCache, targetCache, frame.target, instance
              ))
            else
              frame.initialized = true
              frame.generator = generator
              frame.callKey = Some(key)
              frame.children =
                if frame.depth >= maxDepth then List.empty
                else
                  generator.children.iterator
                    .flatMap(nodeMap.get)
                    .toList
                    .sortBy(node => (incomingDistance(incomingDistanceCache, node, instance), node.id))
                    .take(maxChildrenPerFrame)

      else if !frame.waitingForChild && frame.nextChild < frame.children.size then
        val original = frame.children(frame.nextChild)
        frame.nextChild += 1

        if intentSubset(subsetCache, original, frame.target) then
          acceptParent(frame, original)
        else if coherentWithIncoming(
          coherenceCache, incomingDistanceCache, frame.target, original, instance
        ) then
          val missing = frame.target.instances.iterator
            .filterNot(member => original.symbols.contains(member.symbol))
            .toList
          if missing.isEmpty then
            acceptParent(frame, original)
          else
            val unionTarget = targetNode(targetCache, original.instances.toList ::: missing)
            frame.waitingForChild = true
            completed = None
            frames.append(new Frame(unionTarget, Some(original), frame.depth + 1))

      else if !frame.waitingForChild then
        val newConcept = materialize(
          bottomFrontier, coherenceCache, subsetCache,
          incomingDistanceCache, targetCache, frame.target, instance
        )

        frame.newParents.foreach { parent =>
          if locallyMoreGeneral(acceptedIds, frame.generator, newConcept, instance) &&
              locallyMoreGeneral(acceptedIds, newConcept, parent, instance) then
            frame.generator.children.remove(parent.id)
            parent.parents.remove(frame.generator.id)

          if locallyMoreGeneral(acceptedIds, parent, newConcept, instance) then
            link(bottomFrontier, acceptedIds, parent, newConcept, instance)
          else
            link(bottomFrontier, acceptedIds, newConcept, parent, instance)
        }

        frame.preferredGenerator.foreach { original =>
          if locallyMoreGeneral(acceptedIds, newConcept, original, instance) then
            if locallyMoreGeneral(acceptedIds, frame.generator, newConcept, instance) then
              frame.generator.children.remove(original.id)
              original.parents.remove(frame.generator.id)
            link(bottomFrontier, acceptedIds, newConcept, original, instance)
        }

        val hasMiddle = frame.newParents.exists { parent =>
          locallyMoreGeneral(acceptedIds, frame.generator, parent, instance) &&
            locallyMoreGeneral(acceptedIds, parent, newConcept, instance)
        }
        if !hasMiddle then
          if locallyMoreGeneral(acceptedIds, frame.generator, newConcept, instance) then
            link(bottomFrontier, acceptedIds, frame.generator, newConcept, instance)
          else
            link(bottomFrontier, acceptedIds, newConcept, frame.generator, instance)

        if newConcept.children.isEmpty then bottomFrontier += newConcept
        else bottomFrontier.remove(newConcept)
        frame.callKey.foreach(activeCalls.remove)
        finish(newConcept)

    completed.getOrElse(
      throw new IllegalStateException("Loop insertion finished without a result")
    )
