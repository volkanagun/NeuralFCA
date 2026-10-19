package concept

import scala.collection.mutable

class CVALattice(val device:torch.Device, val attributeCount: Int, val attributeDimension: Int) {
  var id = 1
  val bottom: CVANode = CVANode.empty(0, attributeCount, attributeDimension,device)
  var concepts = Array[CVANode](bottom)
  val subset_threshold = 0.95f

  private type Intent = Array[torch.Tensor[torch.BFloat16]]
  private type IntentKey = Vector[Int]

  private val intentIndex = mutable.HashMap.empty[IntentKey, CVANode]

  private def intentKey(intent: Intent): IntentKey = {
    val key = Vector.newBuilder[Int]
    intent.foreach { commonVector =>
      val values = numeric.BFloatTensors.toArray(commonVector)
      key += values.length
      values.foreach { value =>
        key += java.lang.Float.floatToIntBits(if value == 0.0f then 0.0f else value)
      }
    }
    key.result()
  }

  private def indexIntent(concept: CVANode): Unit =
    intentIndex.getOrElseUpdate(intentKey(concept.cv), concept)

  private def indexedIntent(intent: Intent): Option[CVANode] =
    intentIndex.get(intentKey(intent)).filter(concept => equality(concept.cv, intent))

  indexIntent(bottom)

  private def subset(left: Intent, right: Intent): Boolean =
    left.length == right.length &&
      left.zip(right).forall { case (leftCommon, rightCommon) =>
        CVAUtil.commonVectorSubset(leftCommon, rightCommon, subset_threshold)
      }

  private def equality(left: Intent, right: Intent): Boolean =
    subset(left, right) && subset(right, left)

  private def intersection(left: Intent, right: Intent): Intent = {
    require(left.length == right.length, "Intent sizes must match")
    left.zip(right).map { case (leftCommon, rightCommon) =>
      leftCommon.flatten.minimum(rightCommon.flatten)
    }
  }

  private def maximal(intent: Intent, generator: CVANode): CVANode = {
    val start =
      if subset(generator.cv, intent) then generator else bottom
    val pending = mutable.ArrayDeque((start, 0))
    val visited = mutable.HashSet.empty[CVANode]
    var best = start
    var bestDepth = 0

    while pending.nonEmpty do
      val (current, depth) = pending.removeLast()
      if visited.add(current) then
        if equality(current.cv, intent) then return current

        val compatibleParents = current.parents.filter(parent => subset(parent.cv, intent))
        if compatibleParents.isEmpty then
          val strictlyMoreSpecific =
            subset(best.cv, current.cv) && !subset(current.cv, best.cv)
          if strictlyMoreSpecific || depth > bestDepth then
            best = current
            bestDepth = depth
        else
          compatibleParents.foreach(parent => pending.append((parent, depth + 1)))

    best
  }

  private def appendParent(node: CVANode, parent: CVANode): Unit =
    if (node ne parent) && !node.parents.contains(parent) then node.parents :+= parent

  /** Direct implementation of ADD-INTENT(Y, Generator, L). */
  private def addIntent(
      intent: Intent,
      generator: CVANode,
      expandedCandidates: Set[CVANode] = Set.empty
  ): CVANode = {
    val existingIntent = indexedIntent(intent)
    if existingIntent.nonEmpty then return existingIntent.get

    val maximalGenerator = maximal(intent, generator)
    if equality(maximalGenerator.cv, intent) then
      return maximalGenerator

    val generatorParents =
      maximalGenerator.parents.filterNot(expandedCandidates.contains)
    var newParents = Vector.empty[CVANode]

    generatorParents.foreach { originalCandidate =>
      val candidate =
        if subset(originalCandidate.cv, intent) then originalCandidate
        else
          addIntent(
            intersection(originalCandidate.cv, intent),
            originalCandidate,
            expandedCandidates + originalCandidate
          )

      // Fuzzy common-vector equality can make the recursive result equal to
      // the generator. Step 23 already supplies that edge, so exclude a
      // self-parent here.
      if candidate ne maximalGenerator then
        var addCandidate = true
        val parentsSnapshot = newParents
        val iterator = parentsSnapshot.iterator
        while iterator.hasNext && addCandidate do
          val parent = iterator.next()
          if subset(candidate.cv, parent.cv) then
            addCandidate = false
          else if subset(parent.cv, candidate.cv) then
            newParents = newParents.filterNot(_ == parent)

        if addCandidate then newParents :+= candidate
    }

    // Recursive processing of sibling candidates may already have produced
    // this exact intersection. Reuse it instead of allocating a duplicate.
    val recursivelyCreatedIntent = indexedIntent(intent)
    if recursivelyCreatedIntent.nonEmpty then return recursivelyCreatedIntent.get

    val newConcept = CVANode(
      id,
      intent.map(_.clone()),
      maximalGenerator.projections.map(_.clone())
    ).instances(maximalGenerator.instances.clone())
      .items(maximalGenerator.items.clone())
    add(newConcept)

    newParents.foreach { parent =>
      maximalGenerator.remove(parent)
      appendParent(newConcept, parent)
    }
    appendParent(maximalGenerator, newConcept)
    newConcept
  }

  private def add(
      symbol: String,
      instance: Array[torch.Tensor[torch.BFloat16]]
  ): Unit = {
    val pending = mutable.ArrayDeque(bottom)
    val visited = mutable.HashSet.empty[CVANode]
    while pending.nonEmpty do
      val concept = pending.removeLast()
      if visited.add(concept) && concept.membership(instance) then
        concept.add(symbol, instance)
        concept.parents.foreach(pending.append)
  }

  def add(concept: CVANode): this.type = {

    if !concepts.contains(concept) then {
      concepts :+= concept
      indexIntent(concept)
      id = id + 1
    }
    this
  }

  def intent(symbol: String, instance: Array[torch.Tensor[torch.BFloat16]]): CVALattice = {
    intent(symbol, instance, bottom)
    this
  }

  def intent(symbol: String, instance: Array[torch.Tensor[torch.BFloat16]], generator: CVANode): CVANode = {
    require(concepts.contains(generator), "Generator concept is not in this lattice")
    require(
      instance.length == attributeCount,
      s"Expected $attributeCount attributes, received ${instance.length}"
    )
    val objectConcept = addIntent(instance.map(_.clone()), generator)
    add(symbol, instance)
    objectConcept
  }
}
