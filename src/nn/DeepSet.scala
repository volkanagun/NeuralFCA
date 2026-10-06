package nn

import torch.*
import torch.nn.*
import torch.nn.functional as F

class DeepSet(
               inputDim: Int,
               hiddenDim: Int,
               outputDim: Int
             ) extends torch.nn.modules.TensorModule[Float32] {

  val phi = register(
    nn.Sequential[Float32](
      nn.Linear[Float32](inputDim, hiddenDim),
      nn.ReLU[Float32](),
      nn.Linear[Float32](hiddenDim, hiddenDim),
      nn.ReLU[Float32]()
    )
  )

  val rho = register(
    nn.Sequential[Float32](
      nn.Linear[Float32](hiddenDim, hiddenDim),
      nn.ReLU[Float32](),
      nn.Linear[Float32](hiddenDim, outputDim)
    )
  )


  def apply(x: Tensor[Float32]): Tensor[Float32] = {
    // x: [batchSize, setSize, inputDim]
    val encoded = phi(x)
    // encoded:
    // [batchSize, setSize, hiddenDim]
    val pooled = encoded.mean(dim = 1)
    // pooled:
    // [batchSize, hiddenDim]
    rho(pooled)
  }
}