package llm

import torch.{BFloat16, Device, Float32, Tensor, nn}

import scala.io.Source

class AttentionModel(sentenceFilename: String,
                     latticeFilename: String,
                     embeddingFilename: String, device: Device = Device.CUDA) extends nn.modules.Module with BaseModel(latticeFilename, embeddingFilename) {

  val dw = 300
  val dc = 300
  val window = 5
  private val projectionScale = (1.0 / math.sqrt(dim.toDouble)).toFloat

  val wordEmb: torch.Tensor[Float32] = registerParameter({
    val pretrainedRows = embeddingMap.iterator.map { case (token, vector) =>
      index(token) -> vector.flatten.to(dtype = torch.float32)
    }.toMap
    val rows = (0 until vocabLength).map { row =>
      pretrainedRows.getOrElse(row, torch.randn(Seq(dim)))
    }
    val matrix = torch.stack(rows)
    matrix.requiresGrad = true
    matrix
  }, n = "wordEmb")

  // One row per selected FCA concept, followed by the fallback concept.
  val conceptEmb: torch.Tensor[Float32] = registerParameter({
    val rows = (0 to conceptSize).map { row =>
      commonMap(row).flatten.to(dtype = torch.float32)
    }
    val matrix = torch.stack(rows)
    matrix.requiresGrad = true
    matrix
  }, n = "conceptEmb")

  // Q, K, V projections
  val Wq = registerParameter(torch.randn(Seq(dim, dim)) * projectionScale, n = "Wq")
  val Wk = registerParameter(torch.randn(Seq(dim, dim)) * projectionScale, n = "Wk")
  val Wv = registerParameter(torch.randn(Seq(dim, dim)) * projectionScale, n = "Wv")

  // Candidate representation and activation
  val Wc = registerParameter(torch.randn(Seq(dim, dim)) * projectionScale, n = "Wc")
  val Wg = registerParameter(torch.randn(Seq(dim, 1)) * projectionScale, n = "Wg")

  // Predict next-word embedding
  val Wo = registerParameter(torch.randn(Seq(dim, dim)) * projectionScale, n = "Wo")

  def oneHot(id: Int, size: Int): Tensor[Float32] = {
    torch.Tensor(
      (0 until size).map { i =>
        if (i == id) 1.0f else 0.0f
      }, device = device
    )
  }

  def wordVector(word: Int): Tensor[Float32] = {
    oneHot(word, vocabLength) matmul wordEmb
  }

  def conceptVector(concept: Int): Tensor[Float32] = {
    oneHot(concept, conceptSize + 1) matmul conceptEmb
  }


  def dataset(maxLines : Int = 100000): Seq[Sentence] = {
    Source.fromFile(sentenceFilename).getLines().flatMap(sentence => {
      val tokens = tokenize(sentence)
      val concept_seq = concepts(tokens)

      if (tokens.size >= window){
        tokens.zipWithIndex.sliding(window, 1).map { windowSeq =>{
          windowSeq.map((token, indice) => (token, concept_seq(indice)))
        }}.map(windowSeq => Sentence(windowSeq.map(_._1), windowSeq.map(_._2)))
      }
      else{
        Seq[Sentence]()
      }

    }).take(maxLines).toSeq
  }

  def target(input:Sentence):Int = {
    input.tokens.last
  }

  def apply(input: Sentence): Tensor[Float32] = {

    // Flatten the concept candidates of the prefix.
    // Each candidate keeps its associated word.
    require(input.tokens.size >= 2, "Attention requires a prefix and a target word")
    val prefix = input.tokens.dropRight(1).zip(input.concepts.dropRight(1))
    val candidates = prefix.flatMap { case (word, concepts) =>
      concepts.map(concept => (word, concept))
    }
    val lastIndex = input.tokens.length - 2

    // Concepts belonging to the last observed word
    // act as Queries.
    val queryConcepts = input.concepts(lastIndex)

    // Q: concept embeddings
    val Q = torch.stack(
      queryConcepts.map { concept =>
        conceptVector(concept) matmul Wq
      }
    )

    // K: all candidate concept embeddings
    val K = torch.stack(
      candidates.map { case (_, concept) =>
        conceptVector(concept) matmul Wk
      }
    )

    // V: corresponding word embeddings
    val V = torch.stack(
      candidates.map { case (word, _) =>
        wordVector(word) matmul Wv
      }
    )

    // -----------------------------------
    // Concept Q/K attention
    // -----------------------------------

    val scores =
      (Q matmul K.t) / math.sqrt(dim.toDouble).toFloat

    val attention = torch.softmax[Float32, Float32](scores, dim = 1, dtype = torch.float32)

    // Values contain word information.
    val H = attention matmul V

    // -----------------------------------
    // Update contextual concept vectors
    // -----------------------------------

    val Z = torch.stack(
      queryConcepts.map(conceptVector)
    )

    val updatedConcepts =
      torch.tanh((H + Z) matmul Wc)

    // -----------------------------------
    // Learn concept activation weights
    // -----------------------------------

    val activationScores =
      torch.squeeze(updatedConcepts matmul Wg, 1)

    val beta =
      torch.softmax[Float32, Float32](activationScores, dim = 0, dtype = torch.float32)

    // Weighted contextual concept embedding
    val context =
      torch.squeeze(beta.unsqueeze(0) matmul updatedConcepts, 0)

    // -----------------------------------
    // Predict next-word embedding
    // -----------------------------------

    val predictedEmbedding = context matmul Wo

    // Compare with every vocabulary embedding
    val logits =
      (predictedEmbedding matmul wordEmb.t) / math.sqrt(dim.toDouble).toFloat

    logits
  }

}
