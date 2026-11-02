package llm

import torch.Device
import torch.nn.functional
import torch.optim.Adam

class BaseTrainer(val epocs:Int, val lrate:Float) {
  val sentenceFilename = "resources/sentences/sentences-tr.txt"
  val latticeFilename = "resources/binary/simple-lattice.bin"
  val embeddingFilename = "resources/embeddings/vectors.txt"
  val device = Device.CUDA
  val attentionModel = new AttentionModel(sentenceFilename, latticeFilename, embeddingFilename)
    .to(device)
  val optimizer = new Adam(
    attentionModel.parameters,
    lr = lrate.toDouble
  )

  def train(): Unit = {
    println("Loading training samples...")
    val dataset = attentionModel.dataset()
    val totalSamples = dataset.size
    if (totalSamples == 0) {
      println("No training samples found.")
      return
    }

    println(s"Training on $totalSamples samples for $epocs epochs.")
    val progressEvery = math.min(5, totalSamples / 10)
    for (epoch <- 1 to epocs) {

      var totalLoss = 0.0
      println(s"Epoch $epoch/$epocs started.")

      for ((sentence, sampleIndex) <- dataset.iterator.zipWithIndex) {

        optimizer.zeroGrad()

        val logits = attentionModel.apply(sentence)

        val targetTensor =
          torch.Tensor(Seq(attentionModel.target(sentence)))
            .to(dtype = torch.int64, device=device)

        val loss =
          functional.crossEntropy(
            logits.unsqueeze(0),
            targetTensor
          )

        loss.backward()
        torch.nn.utils.clipGradNorm_(attentionModel.parameters, 1.0)
        optimizer.step()

        totalLoss += loss.item.toDouble
        val processed = sampleIndex + 1
        if (processed % progressEvery == 0 || processed == totalSamples) {
          println(f"Epoch $epoch%d/$epocs%d: $processed%d/$totalSamples%d samples, avg loss ${totalLoss / processed}%.4f")
        }
      }
    }
  }
}

object BaseTrainer {
  def main(args: Array[String]): Unit = {
    new BaseTrainer(10, 0.001f).train()
  }
}
