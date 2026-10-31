package similarity


import org.bytedeco.javacpp.{FloatPointer, PointerScope}
import simple.{Instance, Node}

import scala.collection.mutable
import scala.util.Using

class Concept3CosAdd(val nodes: Vector[Node], parallelBatches: Boolean = true) {
  private val batchSize = 4096

  // Rows in the Torch index are [concept common vector, instance embedding].
  private final case class IndexedVector(
      symbol: String,
      batch: Int,
      row: Int,
      inverseNorm: Double
  )
  private final case class DecodedEmbedding(values: Array[Float], squaredNorm: Double)
  private final case class CandidateBatch(
      vectors: Vector[IndexedVector],
      matrix: torch.Tensor[torch.Float32]
  )
  private final case class CosineIndex(
      bySymbol: Map[String, Vector[IndexedVector]],
      candidates: Vector[IndexedVector],
      batches: Vector[CandidateBatch]
  )

  private lazy val cosineIndex: CosineIndex = {
    require(nodes.nonEmpty, "Concept3CosAdd needs at least one concept")
    val commonDim = Math.toIntExact(nodes.head.commonVector.numel)
    val embeddingDim = nodes.iterator.flatMap(_.extent.iterator)
      .map(instance => Math.toIntExact(instance.attributes.numel))
      .nextOption().getOrElse(throw new IllegalArgumentException("Concepts have no instances"))
    val decodedEmbeddings = new java.util.IdentityHashMap[Instance, DecodedEmbedding]()
    val bySymbol = mutable.HashMap.empty[String, mutable.ArrayBuffer[IndexedVector]]
    val candidates = mutable.ArrayBuffer.empty[IndexedVector]
    val batches = mutable.ArrayBuffer.empty[CandidateBatch]
    val pending = mutable.ArrayBuffer.empty[(String, Array[Float], Double)]
    val concatDim = commonDim + embeddingDim
    def flushBatch(): Unit =
      if pending.nonEmpty then {
        val values = new Array[Float](pending.size * concatDim)
        pending.iterator.zipWithIndex.foreach { case ((_, concat, _), row) =>
          System.arraycopy(concat, 0, values, row * concatDim, concatDim)
        }
        val retained = new org.bytedeco.pytorch.Tensor()
        val matrix = try {
          Using.resource(new PointerScope()) { _ =>
            retained.put(torch.Tensor[Float](values.toSeq).view(pending.size, concatDim).native)
          }
          torch.Tensor.fromNative[torch.Float32](retained)
        } catch {
          case error: Throwable =>
            retained.close()
            throw error
        }
        val batchIndex = batches.size
        val vectors = pending.iterator.zipWithIndex.map { case ((symbol, _, inverseNorm), row) =>
          val entry = IndexedVector(symbol, batchIndex, row, inverseNorm)
          bySymbol.getOrElseUpdate(symbol, mutable.ArrayBuffer.empty) += entry
          candidates += entry
          entry
        }.toVector
        batches += CandidateBatch(vectors, matrix)
        pending.clear()
      }
    val startedAt = System.nanoTime()
    var lastProgressAt = startedAt
    println(s"[Concept3CosAdd] Indexing ${nodes.size} concepts...")

    nodes.iterator.zipWithIndex.foreach { (node, nodeIndex) =>
      if node.extent.nonEmpty then {
        val common = numeric.BFloatTensors.toArray(node.commonVector)
        require(common.length == commonDim, s"Invalid common-vector size in node ${node.id}")
        val commonSquaredNorm = common.iterator.map(value => value.toDouble * value).sum
        node.extent.toVector.sortBy(_.id).foreach { instance =>
          var decoded = decodedEmbeddings.get(instance)
          if decoded == null then {
            val values = numeric.BFloatTensors.toArray(instance.attributes)
            require(values.length == embeddingDim,
              s"Invalid embedding size for ${instance.id} in node ${node.id}")
            decoded = DecodedEmbedding(values, values.iterator.map(value => value.toDouble * value).sum)
            decodedEmbeddings.put(instance, decoded)
          }
          val squaredNorm = commonSquaredNorm + decoded.squaredNorm
          if squaredNorm.isFinite && squaredNorm > 0.0 then {
            pending += ((instance.id, common.concat(decoded.values), 1.0 / math.sqrt(squaredNorm)))
            if pending.size == batchSize then flushBatch()
          }
        }
      }
      val now = System.nanoTime()
      if now - lastProgressAt >= 2000000000L then {
        println(s"[Concept3CosAdd] Indexed ${nodeIndex + 1}/${nodes.size} concepts, " +
          s"${candidates.size + pending.size} vectors")
        lastProgressAt = now
      }
    }
    flushBatch()
    println(s"[Concept3CosAdd] Indexed ${candidates.size} vectors for ${bySymbol.size} words " +
      f"in ${(System.nanoTime() - startedAt) / 1e9}%.1fs")
    CosineIndex(bySymbol.iterator.map((symbol, vectors) => symbol -> vectors.toVector).toMap,
      candidates.toVector, batches.toVector)
  }

  def indexedVectorCount(symbol: String): Int =
    cosineIndex.bySymbol.get(symbol).fold(0)(_.size)

  def indexedVectorCount: Int = cosineIndex.candidates.size

  def indexedSymbolCount: Int = cosineIndex.bySymbol.size

  /** Exhaustive cosine search over every destination - source + target variant. */
  def projectAdd(
      source: String,
      destination: String,
      target: String,
      topK: Int,
      candidateSymbols: Set[String] = Set.empty,
      onProgress: (Long, Long) => Unit = (_, _) => ()
  ): Vector[String] = {
    if topK <= 0 then return Vector.empty
    val index = cosineIndex
    val sources = index.bySymbol.getOrElse(source, Vector.empty)
    val destinations = index.bySymbol.getOrElse(destination, Vector.empty)
    val targets = index.bySymbol.getOrElse(target, Vector.empty)
    if sources.isEmpty || destinations.isEmpty || targets.isEmpty then return Vector.empty
    val excluded = Set(source, destination, target)
    val candidateBatches = index.batches.flatMap { batch =>
      val rows = batch.vectors.indices.filter { row =>
        val symbol = batch.vectors(row).symbol
        !excluded.contains(symbol) &&
          (candidateSymbols.isEmpty || candidateSymbols.contains(symbol))
      }.toVector
      if rows.isEmpty then None else Some((batch, rows))
    }
    val candidateCount = candidateBatches.iterator.map(_._2.size).sum
    if candidateCount == 0 then return Vector.empty

    val bestBySymbol = mutable.HashMap.empty[String, Double]
    def rowTensor(vector: IndexedVector): torch.Tensor[torch.Float32] =
      torch.Tensor.fromNative[torch.Float32](
        index.batches(vector.batch).matrix.native.select(0L, vector.row.toLong))
    val totalVariants = sources.size.toLong * destinations.size.toLong * targets.size.toLong
    if totalVariants > 1000 then
      println(s"[Concept3CosAdd] $destination - $source + $target: " +
        s"$totalVariants query variants against $candidateCount indexed vectors")
    var completedVariants = 0L
    sources.foreach { sourceVector =>
      destinations.foreach { destinationVector =>
        targets.foreach { targetVector =>
          Using.resource(new PointerScope()) { _ =>
            val query = rowTensor(destinationVector) - rowTensor(sourceVector) + rowTensor(targetVector)
            val queryNorm = query.native.norm().item_double()
            if queryNorm.isFinite && queryNorm > 0.0 then {
              val inverseQueryNorm = 1.0 / queryNorm
              val scoredBatches = BatchScoring.map(candidateBatches, parallelBatches) { (batch, rows) =>
                Using.resource(new PointerScope()) { _ =>
                  val dots = new Array[Float](batch.vectors.size)
                  val scores = batch.matrix.matmul(query)
                  new FloatPointer(scores.native.data_ptr()).get(dots)
                  val batchBest = mutable.HashMap.empty[String, Double]
                  rows.foreach { row =>
                    val candidate = batch.vectors(row)
                    val score = dots(row).toDouble * inverseQueryNorm * candidate.inverseNorm
                    if score.isFinite && score > batchBest.getOrElse(candidate.symbol, Double.NegativeInfinity) then
                      batchBest(candidate.symbol) = score
                  }
                  // A global topK word must be in the topK of at least one batch
                  // and query variant where it reaches its best score.
                  val local = mutable.PriorityQueue.empty[(String, Double)](using BatchScoring.ordering)
                  batchBest.foreach { (symbol, score) =>
                    BatchScoring.offer(local, (symbol, score), topK)
                  }
                  local.dequeueAll.reverseIterator.toVector
                }
              }
              scoredBatches.foreach { winners =>
                winners.foreach { (symbol, score) =>
                  if score > bestBySymbol.getOrElse(symbol, Double.NegativeInfinity) then
                    bestBySymbol(symbol) = score
                }
              }
            }
          }
          completedVariants += 1
          onProgress(completedVariants, totalVariants)
        }
      }
    }
    val best = mutable.PriorityQueue.empty[(String, Double)](using BatchScoring.ordering)
    bestBySymbol.foreach { (symbol, score) =>
      BatchScoring.offer(best, (symbol, score), topK)
    }
    best.dequeueAll.reverseIterator.map(_._1).toVector
  }

  /** Rank individual words by the closest concept that contains them. */
  def rankSymbols(
      query: torch.Tensor[torch.BFloat16],
      candidateSymbols: Set[String],
      excludedSymbols: Set[String],
      topK: Int,
      onProgress: (Int, Int) => Unit = (_, _) => ()
  ): Vector[String] = {
    if topK <= 0 then return Vector.empty
    val scores = mutable.HashMap.empty[String, (Double, Int)]
    nodes.iterator.zipWithIndex.foreach { (node, nodeIndex) =>
      val distance = Using.resource(new PointerScope()) { _ => node.distance(query) }
      if distance.isFinite then
        node.symbols.iterator.filter(symbol =>
          candidateSymbols.contains(symbol) && !excludedSymbols.contains(symbol)).foreach { symbol =>
          scores.get(symbol) match {
            case Some((previous, previousIndex))
                if previous < distance || (previous == distance && previousIndex <= nodeIndex) =>
            case _ => scores(symbol) = (distance, nodeIndex)
          }
        }
      if (nodeIndex + 1) % 4096 == 0 || nodeIndex + 1 == nodes.size then
        onProgress(nodeIndex + 1, nodes.size)
    }
    val ordering = Ordering.by[(String, Double, Int), (Double, Int, String)] {
      case (symbol, distance, nodeIndex) => (distance, nodeIndex, symbol)
    }
    val best = mutable.PriorityQueue.empty[(String, Double, Int)](using ordering)
    scores.foreach { (symbol, score) =>
      val entry = (symbol, score._1, score._2)
      if best.size < topK then best.enqueue(entry)
      else if ordering.lt(entry, best.head) then {
        best.dequeue()
        best.enqueue(entry)
      }
    }
    best.dequeueAll.reverseIterator.map(_._1).toVector
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

}
