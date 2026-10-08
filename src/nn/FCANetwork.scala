package nn

import torch.{Float32, Tensor, nn}

class FCANetwork( lambda1:Float, lambda2:Float,
                  inputDim: Int,
                  hiddenDim: Int,
                  outputDim: Int) extends torch.nn.modules.TensorModule[Float32]{

  require(inputDim > 0 && hiddenDim > 0 && outputDim > 0)
  require(inputDim == outputDim, "Residual CVA prediction requires equal input and output dimensions")

  val nnSet = register(DeepSetCVA(inputDim, hiddenDim))

  private val summaryDim = inputDim * 2 + hiddenDim * 3 + 1

  val nnTransformation = register(
    nn.Sequential[Float32](
      nn.Linear[Float32](summaryDim, hiddenDim),
      nn.ReLU[Float32](),
      nn.Linear[Float32](hiddenDim, hiddenDim),
      nn.ReLU[Float32](),
      nn.Linear[Float32](hiddenDim, outputDim)
    )
  )

  def cosineSimilarity(
                        a: Tensor[Float32],
                        b: Tensor[Float32],
                        eps: Float = 1e-6f
                      ): Tensor[Float32] = {

    val dot = (a * b).sum(dim = -1)

    // Add epsilon to each squared norm. Clamping only their product produces
    // gradients near 1/eps when either prediction is close to zero.
    val normA = torch.sqrt((a * a).sum(dim = -1) + eps)
    val normB = torch.sqrt((b * b).sum(dim = -1) + eps)

    dot / (normA * normB)
  }

  def geometry(x_a:Tensor[Float32], x_b:Tensor[Float32],
             orig_a:Tensor[Float32], orig_b:Tensor[Float32]):Tensor[Float32] =
  {
      torch.square(cosineSimilarity(x_a, x_b)-cosineSimilarity(orig_a, orig_b)).mean
  }

  def direct(x_cva:Tensor[Float32], original:Tensor[Float32]):Tensor[Float32]={
      val beta = 0.1f
      val absoluteError = (x_cva - original).abs
      val smoothL1 = torch.where(
        absoluteError < beta,
        torch.square(absoluteError) * (0.5f / beta),
        absoluteError - beta * 0.5f
      ).mean
      val originalSquaredNorm = torch.square(original).sum(dim = -1)
      val directional = torch.where(
        originalSquaredNorm > 1e-6f,
        cosineSimilarity(x_cva, original) * -1f + 1f,
        0f
      ).mean
      val predictedNorm = torch.sqrt(torch.square(x_cva).sum(dim = -1) + 1e-6f)
      val originalNorm = torch.sqrt(originalSquaredNorm + 1e-6f)
      val normLoss = torch.square(predictedNorm - originalNorm).mean
      smoothL1 + directional * 0.05f + normLoss * 0.01f
  }

  private def predict(x: Tensor[Float32]): Tensor[Float32] = {
    val mean = x.mean(dim = 1)
    val centered = x - mean.unsqueeze(1)
    val variance = torch.square(centered).mean(dim = 1)
    val standardDeviation = torch.sqrt(variance + 1e-6f)
    val encodedStatistics = nnSet(x)
    val setSize = x.shape(1).toLong
    val sizeFeature = torch.ones(
      Seq(x.shape.head, 1),
      dtype = torch.float32,
      device = x.device
    ) * math.log1p(setSize.toDouble).toFloat
    val summary = torch.cat(
      Seq(mean, standardDeviation, encodedStatistics, sizeFeature),
      dim = -1
    )
    val rawCorrection = nnTransformation(summary)
    val correctionScale = torch.sqrt(
      torch.square(mean).mean(dim = -1) + variance.mean(dim = -1) + 1e-6f
    ).unsqueeze(-1) * 4f
    val singletonGate = ((setSize - 1).toDouble / setSize.toDouble).toFloat
    mean + torch.tanh(rawCorrection) * correctionScale * singletonGate
  }


  def apply(x: Tensor[Float32], y:Tensor[Float32],
            x_cva:Tensor[Float32], y_cva:Tensor[Float32]): Tensor[Float32] = {
    // x: [batchSize, setSize, inputDim]
    val nncva1 = predict(x)
    val nncva2 = predict(y)

    val loss1 = (direct(nncva1, x_cva) + direct(nncva2, y_cva)) / 2f
    val loss2 = geometry(nncva1,nncva2, x_cva, y_cva)


    loss1 * (lambda1) +(loss2 *(lambda2))

  }

  def apply(x: Tensor[Float32]): Tensor[Float32] = {
    // x: [batchSize, setSize, inputDim]
    predict(x)
  }
}
