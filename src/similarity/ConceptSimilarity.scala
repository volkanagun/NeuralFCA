package similarity

import numeric.BFloatTensors.*
import cvafca.Node
import org.bytedeco.javacpp.PointerScope

import scala.collection.mutable
import scala.util.Using

class ConceptSimilarity( val dictionary:Set[String],
                         val nodes: Vector[Node],
                         val embeddingMap: Map[String, torch.Tensor[torch.BFloat16]],
                         val conceptCandidate: Int) {
  require(conceptCandidate > 0, "conceptCandidate must be positive")

  lazy val indexMap: Map[String, Array[Node]] = {
    val index = mutable.HashMap.empty[String, mutable.ArrayBuffer[Node]]
    nodes.iterator.foreach { node =>
      node.instances.iterator.foreach { instance =>
        if dictionary.contains(instance.symbol) then
          index.getOrElseUpdate(instance.symbol, mutable.ArrayBuffer.empty) += node
      }
    }
    index.iterator.map { case (token, concepts) =>
      token -> concepts.toArray.sortBy(_.children.size)
    }.toMap
  }

  lazy val candidates = indexMap.map((token, nodes) => {
    // One embedding/common-vector concatenation per selected concept (2 * dimension).
    val embedding = embeddingMap(token).flatten
    token -> nodes.map(node =>
      torch.cat(Seq(embedding, node.commonVector.flatten.to(embedding.device))))
  })

  def concept_exists(item: (String,String,String,String)): Boolean =
    indexMap.contains(item._1) && indexMap.contains(item._2) && indexMap.contains(item._3) &&indexMap.contains(item._4)

  def euclidean(difference: torch.Tensor[torch.BFloat16]): Double = {

    val square = torch.sum(difference * difference)
    val distance = torch.sqrt(square).scalar.doubleValue()
    1.0 / (distance + 1.0)
  }

  /** Return the closest topK node extents, with input order breaking distance ties. */
  def projectAdd(target:String, x: torch.Tensor[torch.BFloat16], topK: Int): Array[Set[String]] = {
    if topK <= 0 then return Array.empty[Set[String]]
    // PriorityQueue is a max-heap: the farthest retained node is replaced first.
    val ordering = Ordering.by[(Node, Double, Int), (Double, Int)](entry => (entry._2, entry._3))
    val best = mutable.PriorityQueue.empty[(Node, Double, Int)](using ordering)
    nodes.iterator.zipWithIndex.foreach { case (node, index) =>
      val distance = Using.resource(new PointerScope()) { _ => node.distance(x) }
      val entry = (node, distance, index)
      if node.size == 1 && node.symbols.contains(target) then
      {
        //no single target accepted
      }
      else if best.size < topK then best.enqueue(entry)
      else if ordering.lt(entry, best.head) then
        best.dequeue()
        best.enqueue(entry)
    }
    best.dequeueAll.reverseIterator.map(_._1.symbols).toArray
  }

  def cosineSimilarity(x: torch.Tensor[torch.BFloat16], y: torch.Tensor[torch.BFloat16]): Double =
    Using.resource(new PointerScope()) { _ =>
      val normX = torch.sqrt((x * x).sum).scalar
      val normY = torch.sqrt((y * y).sum).scalar
      (x * y).sum.scalar / (normX * normY + 1e-8)
    }

  /** Keep at most topK scores. The queue's head is the worst retained result. */
  private def topCandidates(scores: Iterator[(String, Double)], topK: Int,
                            alphabeticalTies: Boolean): Array[String] = {
    if topK <= 0 then return Array.empty[String]
    val ordering = if alphabeticalTies then
      Ordering.by[(String, Double, Int), (Double, String)](entry => (-entry._2, entry._1))
    else
      // The original cosine sort reversed both scores and input order for ties.
      Ordering.by[(String, Double, Int), (Double, Int)](entry => (entry._2, entry._3)).reverse
    val best = mutable.PriorityQueue.empty[(String, Double, Int)](using ordering)
    scores.zipWithIndex.foreach { case ((token, score), index) =>
      val entry = (token, score, index)
      if best.size < topK then best.enqueue(entry)
      else if ordering.lt(entry, best.head) then
        best.dequeue()
        best.enqueue(entry)
    }
    best.dequeueAll.reverseIterator.map(_._1).toArray
  }
  
  def cosAdd(source: String, destination: String, item: String, topK: Int): Array[String] = {
    if topK <= 0 then return Array.empty[String]
    Using.resource(new PointerScope()) { _ =>
      val difference = embeddingMap(destination) - embeddingMap(source) + embeddingMap(item)
      val scores = embeddingMap.iterator
        .filter { case (token, _) => token != source && token != destination && token != item }
        .map { case (token, vector) => token -> cosineSimilarity(vector, difference) }
      topCandidates(scores, topK, alphabeticalTies = false)
    }
  }
  
  def conceptAdd(source: String, destination: String, item: String, topK: Int): Array[String] = {
    if topK <= 0 then return Array.empty[String]
    // Initialize the persistent candidate tensors outside any temporary pointer scope.
    val filterCandidates = candidates.iterator
      .filter { case (token, _) => token != source && token != destination && token != item }.toArray
    if filterCandidates.isEmpty then return Array.empty[String]
    val additiveSource = candidates(source)
    val additiveDestination = candidates(destination)
    val additiveItem = candidates(item)

    // Stream the Cartesian product: retain one difference and one maximum per word.
    val bestScores = Array.fill(filterCandidates.length)(Double.NegativeInfinity)
    val scoreOrdering = scala.Predef.implicitly[Ordering[Double]]
    additiveSource.foreach { sourceEmbedding =>
      additiveDestination.foreach { destinationEmbedding =>
        additiveItem.foreach { itemEmbedding =>
          Using.resource(new PointerScope()) { _ =>
            val difference = destinationEmbedding - sourceEmbedding + itemEmbedding
            var index = 0
            while index < filterCandidates.length do
              filterCandidates(index)._2.foreach { vector =>
                bestScores(index) = scoreOrdering.max(bestScores(index), cosineSimilarity(difference, vector))
              }
              index += 1
          }
        }
      }
    }
    val scores = filterCandidates.iterator.zipWithIndex.map { case ((token, _), index) =>
      token -> bestScores(index)
    }
    topCandidates(scores, topK, alphabeticalTies = true)
  }
}
