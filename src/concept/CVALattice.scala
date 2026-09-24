package concept

import java.io.{BufferedInputStream, BufferedOutputStream, DataInputStream, DataOutputStream, FileInputStream, FileOutputStream}
import java.util.IdentityHashMap
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.collection.immutable.ArraySeq
import scala.collection.mutable
import scala.util.Using
import scala.util.Random

class CVALattice(val device:torch.Device, val attributeCount: Int, val attributeDimension: Int) {
  var id = 1
  var bottom: CVANode = CVANode.empty(0, attributeCount, attributeDimension,device)
  var concepts = Array[CVANode](bottom)
  var subset_threshold = 0.95f

  def load(filename: String): CVALattice = {
    def readCount(input: DataInputStream, name: String): Int = {
      val count = input.readInt()
      if count < 0 then
        throw new IllegalArgumentException(s"Invalid $name in lattice file: $count")
      count
    }

    def readTensor(input: DataInputStream): torch.Tensor[torch.BFloat16] = {
      val rank = readCount(input, "tensor rank")
      val shape = Array.fill(rank)(readCount(input, "tensor dimension"))
      val valueCount = readCount(input, "tensor value count")
      val expectedValueCount = shape.foldLeft(1L)(_ * _.toLong)

      if expectedValueCount != valueCount.toLong then
        throw new IllegalArgumentException(
          s"Tensor shape ${shape.mkString("[", ", ", "]")} requires " +
            s"$expectedValueCount values, but the lattice file contains $valueCount"
        )

      val values = Array.fill(valueCount)(input.readFloat())
      numeric.BFloatTensors(values.toSeq).view(shape *)
    }

    def readTensorArray(input: DataInputStream): Array[torch.Tensor[torch.BFloat16]] = {
      Array.fill(readCount(input, "tensor array length"))(readTensor(input))
    }

    Using.resource(
      new DataInputStream(new BufferedInputStream(new FileInputStream(filename)))
    ) { input =>
      val loadedAttributeCount = readCount(input, "attribute count")
      val loadedAttributeDimension = readCount(input, "attribute dimension")
      val loadedId = input.readInt()
      val loadedSubsetThreshold = input.readFloat()
      val nodeCount = readCount(input, "node count")
      if nodeCount == 0 then
        throw new IllegalArgumentException("A CVAL lattice must contain a bottom node")

      val loaded = new CVALattice(torch.Device.CPU, loadedAttributeCount, loadedAttributeDimension)
      val nodes = new Array[CVANode](nodeCount)
      val parentIndices = new Array[Array[Int]](nodeCount)

      for index <- nodes.indices do
        val nodeId = input.readInt()
        val cv = readTensorArray(input)
        val projections = readTensorArray(input)
        val instances = Array.fill(readCount(input, "instance count")) {
          readTensorArray(input)
        }
        nodes(index) = CVANode(nodeId, cv, projections).instances(instances)
        parentIndices(index) = Array.fill(readCount(input, "parent count"))(input.readInt())

      for index <- nodes.indices do
        nodes(index).parents = parentIndices(index).map { parentIndex =>
          if parentIndex < 0 || parentIndex >= nodeCount then
            throw new IllegalArgumentException(
              s"Invalid parent index $parentIndex for node ${nodes(index).id}"
            )
          nodes(parentIndex)
        }

      loaded.concepts = nodes
      loaded.bottom = nodes(0)
      loaded.id = loadedId
      loaded.subset_threshold = loadedSubsetThreshold
      loaded.rebuildIntentIndex()
      loaded
    }
  }
  def save(filename: String): CVALattice = {
    val nodes = concepts.clone()
    val nodeIndices = new IdentityHashMap[CVANode, Integer]()
    nodes.zipWithIndex.foreach { case (node, index) =>
      nodeIndices.put(node, Integer.valueOf(index))
    }

    def writeTensor(output: DataOutputStream, tensor: torch.Tensor[torch.BFloat16]): Unit = {
      val shape = tensor.shape
      output.writeInt(shape.length)
      shape.foreach(output.writeInt)

      val values = numeric.BFloatTensors.toArray(tensor)
      output.writeInt(values.length)
      values.foreach(output.writeFloat)
    }

    def writeTensorArray(
                          output: DataOutputStream,
                          tensors: Array[torch.Tensor[torch.BFloat16]]
                        ): Unit = {
      output.writeInt(tensors.length)
      tensors.foreach(writeTensor(output, _))
    }

    Using.resource(
      new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filename)))
    ) { output =>
      output.writeInt(attributeCount)
      output.writeInt(attributeDimension)
      output.writeInt(id)
      output.writeFloat(subset_threshold)
      output.writeInt(nodes.length)

      nodes.foreach { node =>
        output.writeInt(node.id)
        writeTensorArray(output, node.cv)
        writeTensorArray(output, node.projections)

        output.writeInt(node.instances.length)
        node.instances.foreach(writeTensorArray(output, _))

        output.writeInt(node.parents.length)
        node.parents.foreach { parent =>
          val parentIndex = nodeIndices.get(parent)
          if parentIndex == null then
            throw new IllegalStateException(
              s"Node ${node.id} has a parent that is not present in the lattice"
            )
          output.writeInt(parentIndex.intValue())
        }
      }
    }
    this
  }
  def compute():CVALattice = {
    concepts.par.map(node=> node.compute())
    rebuildIntentIndex()
    this
  }

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

  private def rebuildIntentIndex(): Unit = {
    intentIndex.clear()
    concepts.foreach(indexIntent)
  }

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

  def printRandomNodeSymbols(
      nodeCount: Int = 5,
      symbolsPerNode: Int = 2,
      random: Random = Random
  ): Unit = {

    val selected = random
      .shuffle(concepts.iterator.filter(_.items.length >= symbolsPerNode).toIndexedSeq)
      .take(nodeCount)

    selected.foreach { concept =>
      val symbols = concept.randomSymbols(symbolsPerNode, random)
      println(s"Node ${concept.id}: ${symbols.mkString(", ")}")
    }

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
