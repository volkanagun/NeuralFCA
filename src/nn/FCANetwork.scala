package nn

import torch.{Float32, Tensor, nn}

class FCANetwork( lambda1:Float, lambda2:Float,
                  inputDim: Int,
                  hiddenDim: Int,
                  outputDim: Int) extends torch.nn.modules.TensorModule[Float32]{

  require(inputDim > 0 && hiddenDim > 0 && outputDim > 0)

  val inputNormalization = register(nn.LayerNorm[Float32](Seq(inputDim)))
  val nnSet = register(DeepSetCVA(inputDim, hiddenDim, hiddenDim))

  val nnTransformation = register(
    nn.Sequential[Float32](
      nn.Linear[Float32](hiddenDim, hiddenDim),
      nn.GeLU[Float32](),
      nn.Linear[Float32](hiddenDim, hiddenDim),
      nn.GeLU[Float32](),
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
      // Mean squared error is independent of common-vector dimensionality and
      // has a stable gradient at an exact match.
      torch.square(x_cva - original).mean
  }


  def apply(x: Tensor[Float32], y:Tensor[Float32],
            x_cva:Tensor[Float32], y_cva:Tensor[Float32]): Tensor[Float32] = {
    // x: [batchSize, setSize, inputDim]
    val s1 = nnSet.apply(inputNormalization(x))
    val s2 = nnSet.apply(inputNormalization(y))
    val nncva1 = nnTransformation(s1)
    val nncva2 = nnTransformation(s2)

    val loss1 = (direct(nncva1, x_cva) + direct(nncva2, y_cva)) / 2f
    val loss2 = geometry(nncva1,nncva2, x_cva, y_cva)


    loss1 * (lambda1) +(loss2 *(lambda2))

  }

  def apply(x: Tensor[Float32]): Tensor[Float32] = {
    // x: [batchSize, setSize, inputDim]
    val s1 = nnSet.apply(inputNormalization(x))
    val nncva1 = nnTransformation(s1)
    nncva1
  }
}
