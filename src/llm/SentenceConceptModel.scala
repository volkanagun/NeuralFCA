package llm

import torch.{Device, Float32, Tensor, nn}
import torch.nn.functional

/** A masked, bidirectional sentence window with trainable word and FCA concept tables. */
final class SentenceConceptModel(
    latticeFilename: String,
    embeddingFilename: String,
    val device: Device = Device.CUDA,
    dimension: Int = 300,
    candidateExtents: Int = 30,
    val radius: Int = 3,
    val maxPositiveConcepts: Int = 4
) extends nn.modules.Module with BaseModel(latticeFilename, embeddingFilename, dimension, candidateExtents) {
  require(radius > 0 && maxPositiveConcepts > 0)
  val width: Int = radius * 2 + 1

  private def projection(in: Int, out: Int): Tensor[Float32] =
    torch.randn(Seq(in, out), device = device) * (1.0 / math.sqrt(in.toDouble)).toFloat

  val wordEmb: Tensor[Float32] = registerParameter({
    val pretrained = embeddingMap.iterator.map { (token, vector) =>
      index(token) -> vector.flatten.to(dtype = torch.float32)
    }.toMap
    val matrix = torch.stack((0 until vocabLength).map(row =>
      pretrained.getOrElse(row, torch.randn(Seq(dim)) * 0.02f))).to(device)
    matrix.requiresGrad = true
    matrix
  }, n = "wordEmb")
  val conceptEmb: Tensor[Float32] = registerParameter({
    val matrix = torch.stack((0 to conceptSize).map(row =>
      commonMap(row).flatten.to(dtype = torch.float32))).to(device)
    matrix.requiresGrad = true
    matrix
  }, n = "conceptEmb")
  val maskEmb = registerParameter(torch.randn(Seq(dim), device = device) * 0.02f, n = "maskEmb")
  val positionEmb = registerParameter(torch.randn(Seq(width, dim), device = device) * 0.02f, n = "positionEmb")
  val Wq = registerParameter(projection(dim, dim), n = "Wq")
  val Wk = registerParameter(projection(dim, dim), n = "Wk")
  val Wv = registerParameter(projection(dim, dim), n = "Wv")
  val Wo = registerParameter(projection(dim, dim), n = "Wo")
  val Wff1 = registerParameter(projection(dim, dim * 2), n = "Wff1")
  val Wff2 = registerParameter(projection(dim * 2, dim), n = "Wff2")

  private val rowsById = candidateNodes.iterator.zipWithIndex.map((node, row) => node.id -> row).toMap
  val conceptRows: Vector[Vector[Int]] = (0 until vocabLength).map { word =>
    val direct = conceptMap.getOrElse(word, Seq.empty).distinct.filter(_ < conceptSize)
    val parents = direct.iterator.flatMap(row =>
      candidateNodes(row).parents.iterator.flatMap(node => rowsById.get(node.id))).toVector
    (direct ++ parents).distinct.sortBy(row => (candidateNodes(row).size, row))
      .take(maxPositiveConcepts).toVector
  }.toVector
  private val selector = new FcaJoinSelector(candidateNodes, lattice.getNodes, lattice.getInstances.size)

  /** Select a target concept from visible neighboring words only. */
  def selectConcept(word: Int, neighbors: Seq[Int]): Int = {
    val options = conceptRows(word)
    if options.isEmpty then conceptSize
    else if neighbors.isEmpty then options.head
    else options.maxBy { candidate =>
      (neighbors.iterator.map { neighbor =>
        val other = conceptRows(neighbor)
        if other.isEmpty then 0.0 else other.iterator.map(selector.informationContent(candidate, _)).max
      }.sum, -candidate)
    }
  }

  private def ids(values: Seq[Int]): Tensor[torch.Int64] =
    torch.Tensor(values).to(dtype = torch.int64, device = device)

  private def normalize(x: Tensor[Float32]): Tensor[Float32] =
    x / torch.sqrt((x * x).sum(dim = x.shape.size - 1, keepdim = true) + 1e-8f)

  private def rmsNorm(x: Tensor[Float32]): Tensor[Float32] =
    x / torch.sqrt((x * x).mean(dim = 2, keepdim = true) + 1e-5f)

  /** Center word and center concept are hidden, including from the concept head. */
  def encode(samples: Seq[SentenceConceptSample]): Tensor[Float32] = {
    require(samples.nonEmpty && samples.forall(_.words.size == width))
    val batch = samples.size
    val words = samples.flatMap(_.words)
    val concepts = samples.flatMap(_.concepts)
    val visible = samples.flatMap(_.visible)
    val wordTensor = torch.indexSelect(wordEmb, 0, ids(words)).reshape(batch, width, dim)
    val conceptTensor = torch.indexSelect(conceptEmb, 0, ids(concepts)).reshape(batch, width, dim)
    val visibleTensor = torch.Tensor(visible.map(value => if value then 1f else 0f), device = device)
      .reshape(batch, width, 1)
    val conceptMask = torch.Tensor(concepts.map(row => if row == conceptSize then 0f else 1f), device = device)
      .reshape(batch, width, 1)
    val centerMask = torch.Tensor((0 until batch).flatMap(_ =>
      (0 until width).map(position => if position == radius then 1f else 0f)), device = device)
      .reshape(batch, width, 1)
    val input = (normalize(wordTensor) + normalize(conceptTensor) * conceptMask) * visibleTensor +
      maskEmb.reshape(1, 1, dim) * centerMask + positionEmb.unsqueeze(0)
    val keyBias = torch.Tensor(samples.flatMap(_.visible.zipWithIndex.map { (isVisible, position) =>
      if isVisible || position == radius then 0f else -1e9f
    }), device = device).reshape(batch, 1, width)
    val scores = ((input matmul Wq) matmul (input matmul Wk).transpose(1, 2)) /
      math.sqrt(dim.toDouble).toFloat
    val weights = torch.softmax[Float32, Float32](scores + keyBias, dim = 2, dtype = torch.float32)
    val attended = (weights matmul (input matmul Wv)) matmul Wo
    val hidden = rmsNorm(input + attended)
    val output = rmsNorm(hidden + (functional.relu(hidden matmul Wff1) matmul Wff2))
    torch.select(output, 1, radius)
  }

  /** Sampled softmax predicts masked words and the context-selected FCA concepts. */
  def loss(samples: Seq[SentenceConceptSample], wordNegatives: Seq[Int],
           conceptNegatives: Seq[Int], conceptWeight: Float): Tensor[Float32] = {
    require(samples.nonEmpty && conceptWeight >= 0f)
    val hidden = normalize(encode(samples))
    val wordRows = (samples.map(_.target) ++ wordNegatives).distinct
    val wordLabels = ids(samples.map(sample => wordRows.indexOf(sample.target)))
    val wordScores = (hidden matmul normalize(torch.indexSelect(wordEmb, 0, ids(wordRows))).t) * 10f
    val wordLoss = functional.crossEntropy(wordScores, wordLabels)
    val positiveConcepts = samples.zipWithIndex.filter(_._1.concept != conceptSize)
    if positiveConcepts.isEmpty || conceptWeight == 0f then wordLoss
    else {
      val conceptRows = (positiveConcepts.map(_._1.concept) ++
        positiveConcepts.flatMap { (sample, _) => this.conceptRows(sample.target) } ++
        conceptNegatives).distinct
      val positions = ids(positiveConcepts.map(_._2))
      val conceptLabels = ids(positiveConcepts.map { (sample, _) =>
        conceptRows.indexOf(sample.concept)
      })
      val conceptScores = (torch.indexSelect(hidden, 0, positions) matmul
        normalize(torch.indexSelect(conceptEmb, 0, ids(conceptRows))).t) * 10f
      wordLoss + functional.crossEntropy(conceptScores, conceptLabels) * conceptWeight
    }
  }
}

/** A fixed sentence window with its center word and concept hidden from the encoder. */
final case class SentenceConceptSample(words: Vector[Int], concepts: Vector[Int],
                                       visible: Vector[Boolean], target: Int, concept: Int)
