package testing

import nn.FCANetwork
import org.bytedeco.javacpp.PointerScope
import torch.{Float32, Tensor}

import scala.util.Using

object FCANetworkDeviceTest:
  private val BatchSize = 1024
  private val InputDim = 300
  private val SetSize = 2

  private def randomMatrix(device: torch.Device): Tensor[Float32] =
    torch.randn(Seq(BatchSize, InputDim), dtype = torch.float32, device = device)

  private def randomSetBatch(device: torch.Device): Tensor[Float32] =
    torch.randn(Seq(BatchSize, SetSize, InputDim), dtype = torch.float32, device = device)

  private def assertFiniteScalar(tensor: Tensor[Float32], label: String): Unit =
    assert(tensor.numel == 1, s"$label must be scalar, got shape ${tensor.shape.mkString("[", ", ", "]")}")
    val value = tensor.item.asInstanceOf[Number].doubleValue()
    assert(value.isFinite, s"$label must be finite, got $value")

  def featureDimReductionWorks(device: torch.Device): Unit =
    Using.resource(new PointerScope()) { _ =>
      val a = randomMatrix(device)
      val b = randomMatrix(device)
      val sum = (a * b).sum(dim = 1)
      assert(
        sum.shape.sameElements(Array(BatchSize)),
        s"feature reduction should have shape [$BatchSize], got ${sum.shape.mkString("[", ", ", "]")}"
      )
    }

  def cosineSimilarityUsesFeatureDimension(device: torch.Device): Unit =
    Using.resource(new PointerScope()) { _ =>
      val network = new FCANetwork(1.0f, 0.1f, InputDim, InputDim, InputDim)
      network.to(device)

      val cosine = network.cosineSimilarity(randomMatrix(device), randomMatrix(device))
      assert(
        cosine.shape.sameElements(Array(BatchSize)),
        s"cosine shape should be [$BatchSize], got ${cosine.shape.mkString("[", ", ", "]")}"
      )
      assert(cosine.device.device == device.device, s"cosine is on ${cosine.device}, expected $device")
    }

  def lossIsScalarAndBackpropagates(device: torch.Device): Unit =
    Using.resource(new PointerScope()) { _ =>
      val network = new FCANetwork(1.0f, 0.1f, InputDim, InputDim, InputDim)
      network.to(device)
      network.train()

      val loss = network(
        randomSetBatch(device),
        randomSetBatch(device),
        randomMatrix(device),
        randomMatrix(device)
      )
      assertFiniteScalar(loss, "FCANetwork loss")
      loss.backward()

      val gradientCount = network.parameters.flatMap(_.grad).size
      assert(gradientCount > 0, "backward should produce gradients")
    }

  def check(device: torch.Device): Unit =
    featureDimReductionWorks(device)
    cosineSimilarityUsesFeatureDimension(device)
    lossIsScalarAndBackpropagates(device)
    println(s"FCANetwork device checks passed on $device")

  def main(args: Array[String]): Unit =
    check(torch.Device.CPU)
    if torch.cuda.isAvailable then check(torch.Device.CUDA)
    else println("CUDA is not available; skipped CUDA FCANetwork device checks.")
