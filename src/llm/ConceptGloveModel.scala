package llm

import torch.{Device, Float32, Tensor, nn}

import java.util.concurrent.{Callable, Executors}
import scala.collection.mutable
import scala.io.Source
import scala.util.Using

final case class CooccurrenceSample(target: Int, context: Int, count: Double)
final case class ConceptCooccurrenceSample(target: Int, targetConcept: Int,
                                           context: Int, contextConcept: Int, count: Double)

/** GloVe on word pairs conditioned on their selected FCA concepts. */
final class ConceptGloveModel(
    val sentenceFilename: String,
    val targetWords: Set[String],
    latticeFilename: String,
    embeddingFilename: String,
    val device: Device = Device.CUDA,
    dimension: Int = 300,
    candidateExtents: Int = 30,
    val contextRadius: Int = 3,
    val maxPositiveConcepts: Int = 8,
    val contextConditioned: Boolean = true
) extends nn.modules.Module with BaseModel(latticeFilename, embeddingFilename, dimension, candidateExtents) {
  require(contextRadius > 0 && maxPositiveConcepts > 0)
  val candidateDimension: Int = 2 * dim

  private val initialWords: Tensor[Float32] = {
    val byRow = embeddingMap.iterator.map { (token, vector) =>
      index(token) -> vector.flatten.to(dtype = torch.float32)
    }.toMap
    torch.stack((0 until vocabLength).map { row =>
      byRow.getOrElse(row, torch.randn(Seq(dim)))
    })
  }

  val wordEmb: Tensor[Float32] = registerParameter({
    val matrix = initialWords.clone().to(device)
    matrix.requiresGrad = true
    matrix
  }, n = "wordEmb")
  val contextEmb: Tensor[Float32] = registerParameter({
    val matrix = initialWords.clone().to(device)
    matrix.requiresGrad = true
    matrix
  }, n = "contextEmb")
  val conceptEmb: Tensor[Float32] = registerParameter({
    val matrix = torch.stack((0 to conceptSize).map(row =>
      commonMap(row).flatten.to(dtype = torch.float32))).to(device)
    matrix.requiresGrad = true
    matrix
  }, n = "conceptEmb")
  val wordBias: Tensor[Float32] = registerParameter({
    val bias = torch.zeros(Seq(vocabLength), device = device)
    bias.requiresGrad = true
    bias
  }, n = "wordBias")
  val contextBias: Tensor[Float32] = registerParameter({
    val bias = torch.zeros(Seq(vocabLength), device = device)
    bias.requiresGrad = true
    bias
  }, n = "contextBias")

  private val conceptRowsById = candidateNodes.iterator.zipWithIndex.map { (node, row) =>
    node.id -> row
  }.toMap

  private val conceptRows: Vector[Vector[Int]] = (0 until vocabLength).map { word =>
    val direct = conceptMap.getOrElse(word, Seq.empty).distinct.filter(_ < conceptSize)
    val parents = direct.iterator.flatMap(row => candidateNodes(row).parents.iterator.flatMap(node =>
      conceptRowsById.get(node.id))).toVector
    parents.distinct
      .sortBy(row => (candidateNodes(row).size, row)).take(maxPositiveConcepts).toVector
  }.toVector

  def conceptCandidates(word: Int): Vector[Int] = conceptRows(word)

  private val joinSelector = new FcaJoinSelector(candidateNodes, lattice.getNodes,
    lattice.getInstances.size)
  private val pairChoiceCache = new java.util.concurrent.ConcurrentHashMap[Long, (Int, Int)]()

  private def options(word: Int): Vector[Int] =
    if conceptRows(word).nonEmpty then conceptRows(word) else Vector(conceptSize)

  private def contextSupport(concept: Int, neighbors: Seq[Int]): Double = {
    if neighbors.isEmpty then 0.0 else {
      neighbors.iterator.map { word =>
        options(word).iterator.map(joinSelector.informationContent(concept, _)).max
      }.sum / neighbors.size
    }
  }

  /** The join scores a pair; the returned rows remain distinct trainable vectors. */
  def selectConceptPair(target: Int, context: Int,
                        targetNeighbors: Seq[Int] = Seq.empty,
                        contextNeighbors: Seq[Int] = Seq.empty): (Int, Int) = {
    val hasWindow = contextConditioned &&
      (targetNeighbors.nonEmpty || contextNeighbors.nonEmpty)
    val key = (target.toLong << 32) | (context.toLong & 0xffffffffL)
    if !hasWindow then {
      val cached = pairChoiceCache.get(key)
      if cached != null then return cached
    }
    val targetOptions = options(target)
    val contextOptions = options(context)
    val targetSupport = if hasWindow then targetOptions.iterator.map(a =>
      a -> contextSupport(a, targetNeighbors)).toMap else Map.empty[Int, Double]
    val contextSupportByRow = if hasWindow then contextOptions.iterator.map(b =>
      b -> contextSupport(b, contextNeighbors)).toMap else Map.empty[Int, Double]
    var best = (conceptSize, conceptSize)
    var bestScore = Double.NegativeInfinity
    for a <- targetOptions; b <- contextOptions do {
      val score = joinSelector.informationContent(a, b) +
        targetSupport.getOrElse(a, 0.0) + contextSupportByRow.getOrElse(b, 0.0)
      if score > bestScore then {
        best = (a, b)
        bestScore = score
      }
    }
    if !hasWindow && pairChoiceCache.size < 100000 then
      pairChoiceCache.putIfAbsent(key, best)
    best
  }

  private case class CountableSentence(tokens: Vector[Int], from: Int, until: Int)

  private val targetWordIds = targetWords.flatMap(index.get)

  private def includesTarget(tokens: Vector[Int]): Boolean =
    targetWords.isEmpty || tokens.exists(targetWordIds.contains)

  /** Count independent sentences in parallel; each center retains its full sentence context. */
  def cooccurrences(skipSamples: Int = 0, maxSamples: Int = Int.MaxValue,
                    workers: Int = 1): Vector[CooccurrenceSample] = {
    require(skipSamples >= 0 && maxSamples > 0 && workers > 0)
    val counts = mutable.LongMap.empty[Double]
    val executor = if workers > 1 then Some(Executors.newFixedThreadPool(workers)) else None
    val chunk = mutable.ArrayBuffer.empty[CountableSentence]
    def count(part: Seq[CountableSentence]): mutable.LongMap[Double] = {
      val local = mutable.LongMap.empty[Double]
      part.foreach { sentence =>
        val tokens = sentence.tokens
        var center = sentence.from
        while center < sentence.until do {
          val first = math.max(0, center - contextRadius)
          val last = math.min(tokens.size - 1, center + contextRadius)
          var neighbor = first
          while neighbor <= last do {
            if neighbor != center then {
              val key = (tokens(center).toLong << 32) | (tokens(neighbor).toLong & 0xffffffffL)
              local.update(key, local.getOrElse(key, 0.0) + 1.0 / math.abs(center - neighbor))
            }
            neighbor += 1
          }
          center += 1
        }
      }
      local
    }
    def merge(local: mutable.LongMap[Double]): Unit =
      local.iterator.foreach { (key, value) =>
        counts.update(key, counts.getOrElse(key, 0.0) + value)
      }
    def flush(): Unit = {
      if chunk.nonEmpty then {
        val sentences = chunk.toVector
        chunk.clear()
        executor match {
          case None => merge(count(sentences))
          case Some(_) if sentences.size < 64 => merge(count(sentences))
          case Some(pool) =>
            val shardSize = math.max(1, (sentences.size + workers - 1) / workers)
            val tasks = sentences.grouped(shardSize).map { shard =>
              pool.submit(new Callable[mutable.LongMap[Double]] {
                override def call(): mutable.LongMap[Double] = count(shard)
              })
            }.toVector
            tasks.foreach(task => merge(task.get()))
        }
      }
    }
    var seen = 0L
    val stop = skipSamples.toLong + maxSamples.toLong
    try {
      Using.resource(Source.fromFile(sentenceFilename, "UTF-8")) { source =>
        val lines = source.getLines()
        while lines.hasNext && seen < stop do {
          val tokens = tokenize(lines.next()).toVector
          if includesTarget(tokens) then {
            val from = math.max(0L, skipSamples.toLong - seen).min(tokens.size.toLong).toInt
            val until = math.max(0L, stop - seen).min(tokens.size.toLong).toInt
            if from < until then chunk += CountableSentence(tokens, from, until)
            seen += tokens.size
            if chunk.size >= 1024 then flush()
          }
        }
        flush()
      }
    } finally executor.foreach(_.shutdown())
    counts.iterator.map { (key, count) =>
      CooccurrenceSample((key >>> 32).toInt, key.toInt, count)
    }.toVector.sortBy(sample => (sample.target, sample.context))
  }

  private case class SelectedKey(target: Int, targetConcept: Int,
                                 context: Int, contextConcept: Int)

  /** Select per occurrence, then aggregate evidence by word and concept rows. */
  def selectedCooccurrences(skipSamples: Int = 0, maxSamples: Int = Int.MaxValue,
                            workers: Int = 1): Vector[ConceptCooccurrenceSample] = {
    require(skipSamples >= 0 && maxSamples > 0 && workers > 0)
    val counts = mutable.HashMap.empty[SelectedKey, Double]
    val executor = if workers > 1 then Some(Executors.newFixedThreadPool(workers)) else None
    val chunk = mutable.ArrayBuffer.empty[CountableSentence]
    def count(part: Seq[CountableSentence]): mutable.HashMap[SelectedKey, Double] = {
      val local = mutable.HashMap.empty[SelectedKey, Double]
      part.foreach { sentence =>
        val tokens = sentence.tokens
        def neighbors(position: Int, other: Int): Vector[Int] = {
          (math.max(0, position - contextRadius) to
            math.min(tokens.size - 1, position + contextRadius)).iterator
            .filter(index => index != position && index != other)
            .map(tokens).toVector
        }
        var center = sentence.from
        while center < sentence.until do {
          val first = math.max(0, center - contextRadius)
          val last = math.min(tokens.size - 1, center + contextRadius)
          var neighbor = first
          while neighbor <= last do {
            if neighbor != center then {
              val targetWindow = if contextConditioned then neighbors(center, neighbor) else Vector.empty
              val contextWindow = if contextConditioned then neighbors(neighbor, center) else Vector.empty
              val (a, b) = selectConceptPair(tokens(center), tokens(neighbor),
                targetWindow, contextWindow)
              val key = SelectedKey(tokens(center), a, tokens(neighbor), b)
              local.update(key, local.getOrElse(key, 0.0) + 1.0 / math.abs(center - neighbor))
            }
            neighbor += 1
          }
          center += 1
        }
      }
      local
    }
    def merge(local: mutable.HashMap[SelectedKey, Double]): Unit =
      local.iterator.foreach { (key, value) =>
        counts.update(key, counts.getOrElse(key, 0.0) + value)
      }
    def flush(): Unit = {
      if chunk.nonEmpty then {
        val sentences = chunk.toVector
        chunk.clear()
        executor match {
          case None => merge(count(sentences))
          case Some(_) if sentences.size < 64 => merge(count(sentences))
          case Some(pool) =>
            val shardSize = math.max(1, (sentences.size + workers - 1) / workers)
            val tasks = sentences.grouped(shardSize).map { shard =>
              pool.submit(new Callable[mutable.HashMap[SelectedKey, Double]] {
                override def call(): mutable.HashMap[SelectedKey, Double] = count(shard)
              })
            }.toVector
            tasks.foreach(task => merge(task.get()))
        }
      }
    }
    var seen = 0L
    val stop = skipSamples.toLong + maxSamples.toLong
    try {
      Using.resource(Source.fromFile(sentenceFilename, "UTF-8")) { source =>
        val lines = source.getLines()
        while lines.hasNext && seen < stop do {
          val tokens = tokenize(lines.next()).toVector
          if includesTarget(tokens) then {
            val from = math.max(0L, skipSamples.toLong - seen).min(tokens.size.toLong).toInt
            val until = math.max(0L, stop - seen).min(tokens.size.toLong).toInt
            if from < until then chunk += CountableSentence(tokens, from, until)
            seen += tokens.size
            if chunk.size >= 1024 then flush()
          }
        }
        flush()
      }
    } finally executor.foreach(_.shutdown())
    counts.iterator.map { (key, value) =>
      ConceptCooccurrenceSample(key.target, key.targetConcept, key.context,
        key.contextConcept, value)
    }.toVector.sortBy(sample =>
      (sample.target, sample.targetConcept, sample.context, sample.contextConcept))
  }

  /** Build [word; selected concept], zeroing the missing-concept block. */
  private[llm] def candidateVectors(samples: Seq[ConceptCooccurrenceSample]):
      (Tensor[Float32], Tensor[Float32]) = {
    require(samples.nonEmpty)
    def ids(values: Seq[Int]): Tensor[torch.Int64] =
      torch.Tensor(values).to(dtype = torch.int64, device = device)
    def conceptVectors(rows: Seq[Int]): Tensor[Float32] = {
      val selected = torch.indexSelect(conceptEmb, 0, ids(rows))
      val present = torch.Tensor(rows.map(row => if row == conceptSize then 0f else 1f),
        device = device).unsqueeze(1)
      selected * present
    }
    val target = torch.cat(Seq(
      torch.indexSelect(wordEmb, 0, ids(samples.map(_.target))),
      conceptVectors(samples.map(_.targetConcept))), dim = 1)
    val context = torch.cat(Seq(
      torch.indexSelect(contextEmb, 0, ids(samples.map(_.context))),
      conceptVectors(samples.map(_.contextConcept))), dim = 1)
    (target, context)
  }

  /** Weighted GloVe on counts aggregated by (word, concept, context, concept). */
  def loss(samples: Seq[ConceptCooccurrenceSample], xMax: Double = 100.0,
           alpha: Double = 0.75): Tensor[Float32] = {
    require(samples.nonEmpty && xMax > 0 && alpha > 0)
    val (targetVectors, contextVectors) = candidateVectors(samples)
    val targets = torch.Tensor(samples.map(_.target)).to(dtype = torch.int64, device = device)
    val contexts = torch.Tensor(samples.map(_.context)).to(dtype = torch.int64, device = device)
    val prediction = (targetVectors * contextVectors).sum(dim = 1) +
      torch.indexSelect(wordBias, 0, targets) + torch.indexSelect(contextBias, 0, contexts)
    val expected = torch.Tensor(samples.map(sample => math.log(sample.count).toFloat), device = device)
    val gloveWeights = torch.Tensor(samples.map(sample =>
      math.pow(math.min(sample.count / xMax, 1.0), alpha).toFloat), device = device)
    val error = prediction - expected
    ((error * error) * gloveWeights).mean
  }
}
