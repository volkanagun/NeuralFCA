package llm

import torch.{Device, Float32, Tensor, nn}

import scala.io.Source

class AttentionModel(sentenceFilename: String,
                     latticeFilename: String,
                     embeddingFilename: String, device: Device = Device.CUDA) extends nn.modules.Module with BaseModel(latticeFilename, embeddingFilename) {

  val dw = 300
  val dc = 300
  val window = 7 // Number of input tokens used to predict the following token.
  val attentionWidth = 512
  val attentionHeads = 8
  val feedForwardWidth = 2048
  private val headWidth = attentionWidth / attentionHeads
  private val logitScale = 20.0f

  private def projection(inputWidth: Int, outputWidth: Int): Tensor[Float32] =
    torch.randn(Seq(inputWidth, outputWidth)) * (1.0 / math.sqrt(inputWidth.toDouble)).toFloat

  private def normalizeRows(matrix: Tensor[Float32]): Tensor[Float32] =
    matrix / torch.sqrt((matrix * matrix).sum(dim = 1, keepdim = true) + 1e-8f)

  val wordEmb: torch.Tensor[Float32] = registerParameter({
    val pretrainedRows = embeddingMap.iterator.map { case (token, vector) =>
      index(token) -> vector.flatten.to(dtype = torch.float32)
    }.toMap
    val rows = (0 until vocabLength).map { row =>
      pretrainedRows.getOrElse(row, torch.randn(Seq(dim)))
    }
    val matrix = normalizeRows(torch.stack(rows))
    matrix.requiresGrad = true
    matrix
  }, n = "wordEmb")

  // One row per selected FCA concept, followed by the fallback concept.
  val conceptEmb: torch.Tensor[Float32] = registerParameter({
    val rows = (0 to conceptSize).map { row =>
      commonMap(row).flatten.to(dtype = torch.float32)
    }
    val matrix = normalizeRows(torch.stack(rows))
    matrix.requiresGrad = true
    matrix
  }, n = "conceptEmb")

  // Attend to each word's FCA concepts before attending across word positions.
  val WconceptQuery = registerParameter(projection(dim, dim), n = "WconceptQuery")
  val WconceptKey = registerParameter(projection(dim, dim), n = "WconceptKey")
  val Win = registerParameter(projection(dim, attentionWidth), n = "Win")
  val positionEmb = registerParameter(torch.randn(Seq(window, attentionWidth)) * 0.02f, n = "positionEmb")

  // Causal multi-head self-attention over the five observed words.
  val Wq = registerParameter(projection(attentionWidth, attentionWidth), n = "Wq")
  val Wk = registerParameter(projection(attentionWidth, attentionWidth), n = "Wk")
  val Wv = registerParameter(projection(attentionWidth, attentionWidth), n = "Wv")
  val Wc = registerParameter(projection(attentionWidth, attentionWidth), n = "Wc")

  val Wff1 = registerParameter(projection(attentionWidth, feedForwardWidth), n = "Wff1")
  val Wff2 = registerParameter(projection(feedForwardWidth, attentionWidth), n = "Wff2")
  val Wo = registerParameter(projection(attentionWidth, dim), n = "Wo")

  private def rmsNorm(x: Tensor[Float32]): Tensor[Float32] =
    x / torch.sqrt((x * x).mean(dim = 2, keepdim = true) + 1e-5f)

  def wordVector(word: Int): Tensor[Float32] = {
    torch.select(wordEmb, 0, word)
  }

  def conceptVector(concept: Int): Tensor[Float32] = {
    torch.select(conceptEmb, 0, concept)
  }


  def dataset(startLines : Int = 0, maxLines : Int = 100000): Seq[Sentence] = {
    Source.fromFile(sentenceFilename).getLines().flatMap(sentence => {
      val tokens = tokenize(sentence)
      val conceptSeq = concepts(tokens)
      val sampleLength = window + 1

      if (tokens.size >= sampleLength){
        tokens.zip(conceptSeq).sliding(sampleLength, 1)
          .filter(_.size == sampleLength)
          .map(windowSeq => Sentence(windowSeq.map(_._1), windowSeq.map(_._2)))
      }
      else{
        Seq[Sentence]()
      }

    }).zipWithIndex.filter((sentence, index)=> index > startLines).take(maxLines).map(_._1).toSeq
  }

  def target(input:Sentence):Int = {
    input.tokens.last
  }

  def apply(input: Sentence): Tensor[Float32] =
    torch.squeeze(applyBatch(Seq(input)), 0)

  def applyBatch(inputs: Seq[Sentence]): Tensor[Float32] = {
    require(inputs.nonEmpty, "A training batch must contain at least one sentence")
    val predictions = predictEmbedding(inputs)
    val normalizedPredictions = predictions /
      torch.sqrt((predictions * predictions).sum(dim = 1, keepdim = true) + 1e-8f)
    (normalizedPredictions matmul normalizeRows(wordEmb).t) * logitScale
  }

  private def predictEmbedding(inputs: Seq[Sentence]): Tensor[Float32] = {
    inputs.foreach { input =>
      require(input.tokens.size == window + 1, s"Attention requires $window input tokens and one target word")
      require(input.concepts.size == input.tokens.size, "Tokens and concepts must align")
      require(input.concepts.take(window).forall(_.nonEmpty), "Each input token needs at least one concept")
    }
    val batchSize = inputs.size
    val maxConcepts = inputs.flatMap(_.concepts.take(window)).map(_.size).max
    val wordIds = inputs.flatMap(_.tokens.take(window))
    val conceptRows = inputs.flatMap(_.concepts.take(window))

    val wordIndices = torch.Tensor(wordIds).to(dtype = torch.int64, device = device)
    val conceptIndices = torch.Tensor(conceptRows.flatMap(_.padTo(maxConcepts, 0)))
      .to(dtype = torch.int64, device = device)
    val conceptBias = torch.Tensor(conceptRows.flatMap(row =>
      Seq.fill(row.size)(0.0f) ++ Seq.fill(maxConcepts - row.size)(Float.NegativeInfinity)
    ), device = device).reshape(batchSize, window, maxConcepts)

    val words = torch.indexSelect(wordEmb, 0, wordIndices).reshape(batchSize, window, dim)
    val concepts = torch.indexSelect(conceptEmb, 0, conceptIndices).reshape(batchSize, window, maxConcepts, dim)
    val conceptQueries = (words matmul WconceptQuery).unsqueeze(2)
    val conceptKeys = concepts matmul WconceptKey
    val conceptScores = torch.squeeze(conceptQueries matmul conceptKeys.transpose(2, 3), 2) /
      math.sqrt(dim.toDouble).toFloat
    val conceptWeights = torch.softmax[Float32, Float32](conceptScores + conceptBias, dim = 2, dtype = torch.float32)
    val conceptContext = torch.squeeze(conceptWeights.unsqueeze(2) matmul concepts, 2)

    val tokens = ((words + conceptContext) matmul Win) + positionEmb.unsqueeze(0)
    def splitHeads(x: Tensor[Float32]): Tensor[Float32] =
      x.reshape(batchSize, window, attentionHeads, headWidth).transpose(1, 2)

    val Q = splitHeads(tokens matmul Wq)
    val K = splitHeads(tokens matmul Wk)
    val V = splitHeads(tokens matmul Wv)
    val causalBias = torch.Tensor((0 until window).flatMap(query =>
      (0 until window).map(key => if (key <= query) 0.0f else Float.NegativeInfinity)
    ), device = device).reshape(1, 1, window, window)
    val scores = (Q matmul K.transpose(2, 3)) / math.sqrt(headWidth.toDouble).toFloat
    val weights = torch.softmax[Float32, Float32](scores + causalBias, dim = 3, dtype = torch.float32)
    val attended = (weights matmul V).transpose(1, 2).reshape(batchSize, window, attentionWidth)

    val hidden = rmsNorm(tokens + (attended matmul Wc))
    val feedForward = torch.nn.functional.relu(hidden matmul Wff1) matmul Wff2
    val finalState = rmsNorm(hidden + feedForward)
    torch.select(finalState, 1, window - 1) matmul Wo
  }

}
