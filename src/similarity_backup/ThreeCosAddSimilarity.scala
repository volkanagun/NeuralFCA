package similarity

import org.bytedeco.javacpp.{FloatPointer, PointerScope}

import scala.collection.mutable
import scala.util.Using

/** Conventional 3CosAdd: rank words by cosine(b - a + c, candidate). */
class ThreeCosAddSimilarity(
    embeddings: Map[String, torch.Tensor[torch.BFloat16]],
    candidateSymbols: Set[String],
    parallelBatches: Boolean = true
) {
  private val dimension = embeddings.headOption.map(entry => Math.toIntExact(entry._2.numel))
    .getOrElse(throw new IllegalArgumentException("The embedding map is empty"))

  private val batchSize = 4096

  // Keep normalized candidates in Torch matrices so each query can score a
  // whole batch with one matrix-vector product.
  private val candidateBatches = candidateSymbols.toVector.sorted.grouped(batchSize).flatMap { symbols =>
    val candidates = symbols.flatMap { symbol =>
      embeddings.get(symbol).flatMap { tensor =>
        val values = numeric.BFloatTensors.toArray(tensor)
        require(values.length == dimension, s"Embedding dimension differs for $symbol")
        val squaredNorm = values.iterator.map(value => value.toDouble * value).sum
        if !squaredNorm.isFinite || squaredNorm <= 0.0 then None
        else {
          val norm = math.sqrt(squaredNorm)
          Some((symbol, values.map(value => (value / norm).toFloat)))
        }
      }
    }
    if candidates.isEmpty then None
    else {
      val values = new Array[Float](candidates.size * dimension)
      candidates.zipWithIndex.foreach { case ((_, vector), row) =>
        System.arraycopy(vector, 0, values, row * dimension, dimension)
      }
      // Retain the matrix while releasing the temporary flat tensor and view.
      val retained = new org.bytedeco.pytorch.Tensor()
      try {
        Using.resource(new PointerScope()) { _ =>
          retained.put(torch.Tensor[Float](values.toSeq).view(candidates.size, dimension).native)
        }
        Some((candidates.map(_._1), torch.Tensor.fromNative[torch.Float32](retained)))
      } catch {
        case error: Throwable =>
          retained.close()
          throw error
      }
    }
  }.toVector
  val vocabulary: Set[String] = candidateBatches.iterator.flatMap(_._1).toSet

  def queryVector(source: String, destination: String, item: String): Array[Float] = {
    def values(symbol: String): Array[Float] =
      val tensor = embeddings.getOrElse(symbol,
        throw new IllegalArgumentException(s"Unknown analogy word: $symbol"))
      val result = numeric.BFloatTensors.toArray(tensor)
      require(result.length == dimension, s"Embedding dimension differs for $symbol")
      result

    val a = values(source)
    val b = values(destination)
    val c = values(item)
    Array.tabulate(dimension)(index => b(index) - a(index) + c(index))
  }

  /** Return the highest-cosine words, excluding all analogy query words. */
  def projectAdd(source: String, destination: String, item: String, topK: Int): Vector[String] =
    rank(queryVector(source, destination, item), Set(source, destination, item), topK)

  def rank(query: Array[Float], excluded: Set[String], topK: Int): Vector[String] = {
    if topK <= 0 then return Vector.empty
    require(query.length == dimension, "Analogy query dimension does not match embeddings")
    val queryNorm = math.sqrt(query.iterator.map(value => value.toDouble * value).sum)
    if !queryNorm.isFinite || queryNorm == 0.0 then return Vector.empty
    val best = mutable.PriorityQueue.empty[(String, Double)](using BatchScoring.ordering)
    Using.resource(new PointerScope()) { _ =>
      val queryTensor = torch.Tensor[Float](query.toSeq)
      val scoredBatches = BatchScoring.map(candidateBatches, parallelBatches) { (symbols, matrix) =>
        Using.resource(new PointerScope()) { _ =>
          val scores = matrix.matmul(queryTensor)
          val dots = new Array[Float](symbols.size)
          new FloatPointer(scores.native.data_ptr()).get(dots)
          val local = mutable.PriorityQueue.empty[(String, Double)](using BatchScoring.ordering)
          symbols.indices.foreach { index =>
            val symbol = symbols(index)
            if !excluded.contains(symbol) then {
              val score = dots(index).toDouble / queryNorm
              if score.isFinite then
                BatchScoring.offer(local, (symbol, score), topK)
            }
          }
          local.dequeueAll.reverseIterator.toVector
        }
      }
      scoredBatches.foreach(_.foreach(entry => BatchScoring.offer(best, entry, topK)))
    }
    best.dequeueAll.reverseIterator.map(_._1).toVector
  }
}
