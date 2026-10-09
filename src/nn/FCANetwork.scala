package nn

import torch.{Float32, Tensor, nn}

class FCANetwork( lambda1:Float, lambda2:Float,
                  inputDim: Int,
                  hiddenDim: Int,
                  outputDim: Int,
                  outputLimit: Float = 4.0f,
                  huberDelta: Float = 1.0f) extends torch.nn.modules.TensorModule[Float32]{

  val nnSet = register(DeepSet(inputDim, hiddenDim, hiddenDim))

  val nnTransformation = register(
    nn.Sequential[Float32](
      nn.Linear[Float32](hiddenDim, hiddenDim/2),
      nn.GeLU[Float32](),
      nn.Linear[Float32](hiddenDim/2, hiddenDim/4),
      nn.GeLU[Float32](),
      nn.Linear[Float32](hiddenDim/4, outputDim),
      nn.Tanh[Float32]()
    )
  )

  def cosineSimilarity(
                        a: Tensor[Float32],
                        b: Tensor[Float32],
                        dim: Int = 1,
                        eps: Double = 1e-8
                      ): Tensor[Float32] = {

    val dt = (a * b)
    val dot = dt.sum(dim = dim)

    val normA = torch.sqrt ((a * a).sum(dim = dim))
    val normB = torch.sqrt ((b * b).sum(dim = dim))

    dot / torch.clamp ((normA * normB), Some(eps), None)
  }

  def euclidNorm(a:Tensor[Float32]):Tensor[Float32]={
   torch.sqrt(torch.sum(torch.square(a)))
  }

  def robustLoss(predicted: Tensor[Float32], target: Tensor[Float32]): Tensor[Float32] = {
    val absDiff = torch.abs(predicted - target)
    val linear = torch.clamp(absDiff - huberDelta, Some(0.0), None)
    val quadratic = absDiff - linear
    ((torch.square(quadratic) * 0.5f) + (linear * huberDelta)).mean
  }

  def geometry(x_a:Tensor[Float32], x_b:Tensor[Float32],
             orig_a:Tensor[Float32], orig_b:Tensor[Float32]):Tensor[Float32] =
  {
      torch.square(cosineSimilarity(x_a, x_b)-cosineSimilarity(orig_a, orig_b)).mean
  }

  def direct(x_cva:Tensor[Float32], original:Tensor[Float32]):Tensor[Float32]={
      robustLoss(x_cva, original)
  }

  private def bounded(output: Tensor[Float32]): Tensor[Float32] =
    output * outputLimit


  def apply(x: Tensor[Float32], y:Tensor[Float32],
            x_cva:Tensor[Float32], y_cva:Tensor[Float32]): Tensor[Float32] = {
    // x: [batchSize, setSize, inputDim]
    val s1 = nnSet.apply(x)
    val s2 = nnSet.apply(y)
    val nncva1 = bounded(nnTransformation(s1))
    val nncva2 = bounded(nnTransformation(s2))

    val loss1 = direct(nncva1, x_cva)
    val loss2 = geometry(nncva1,nncva2, x_cva, y_cva)


    loss1 * (lambda1) +(loss2 *(lambda2))

  }

  def apply(x: Tensor[Float32]): Tensor[Float32] = {
    // x: [batchSize, setSize, inputDim]
    val s1 = nnSet.apply(x)
    val nncva1 = bounded(nnTransformation(s1))
    nncva1

  }
}
