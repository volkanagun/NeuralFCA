package nn

import torch.*
import torch.nn.*

class DeepSetCVA(
               inputDim: Int,
               hiddenDim: Int
             ) extends torch.nn.modules.TensorModule[Float32] {

  require(inputDim > 0 && hiddenDim > 0)

  /** Encode raw values, centered differences, and diagonal second moments. */
  val phi = register(
    nn.Sequential[Float32](
      nn.Linear[Float32](inputDim * 3, hiddenDim),
      nn.ReLU[Float32](),
      nn.Linear[Float32](hiddenDim, hiddenDim),
      nn.ReLU[Float32]()
    )
  )

  def apply(x: Tensor[Float32]): Tensor[Float32] = {
    val mean = x.mean(dim = 1)
    val centered = x - mean.unsqueeze(1)
    val features = torch.cat(Seq(x, centered, torch.square(centered)), dim = -1)
    val encoded = phi(features)
    val encodedMean = encoded.mean(dim = 1)
    val encodedMax = encoded.max(1).values
    val encodedCentered = encoded - encodedMean.unsqueeze(1)
    val encodedStd = torch.sqrt(torch.square(encodedCentered).mean(dim = 1) + 1e-6f)
    torch.cat(Seq(encodedMean, encodedMax, encodedStd), dim = -1)
  }
}
