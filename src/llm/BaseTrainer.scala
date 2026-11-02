package llm

import torch.Device
import torch.nn.functional
import torch.optim.Adam

import java.nio.file.{AtomicMoveNotSupportedException, Files, Path, Paths, StandardCopyOption}
import scala.util.Using

class BaseTrainer(val epocs:Int, val lrate:Float,
                  val modelFilename: String = "resources/models/attention-model.bin") {
  val sentenceFilename = "resources/sentences/sentences-tr.txt"
  val latticeFilename = "resources/binary/simple-lattice.bin"
  val embeddingFilename = "resources/embeddings/vectors.txt"
  val device = Device.CUDA
  val attentionModel = new AttentionModel(sentenceFilename, latticeFilename, embeddingFilename, device)

  def saveModel(filename: String = modelFilename): Path = {
    val target = Paths.get(filename).toAbsolutePath
    val directory = target.getParent
    Files.createDirectories(directory)
    val temporary = Files.createTempFile(directory, target.getFileName.toString + ".", ".tmp")
    try {
      Using.resource(new org.bytedeco.pytorch.OutputArchive()) { archive =>
        attentionModel.save(archive)
        archive.save_to(temporary.toString)
      }
      try Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
      catch {
        case _: AtomicMoveNotSupportedException =>
          Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING)
      }
      target
    } finally Files.deleteIfExists(temporary)
  }

  def loadModel(filename: String = modelFilename): Path = {
    val source = Paths.get(filename).toAbsolutePath
    require(Files.isRegularFile(source), s"Neural model checkpoint not found: $source")
    Using.resource(new org.bytedeco.pytorch.InputArchive()) { archive =>
      Using.resource(new org.bytedeco.pytorch.Device("cpu")) { cpuDevice =>
        Using.resource(new org.bytedeco.pytorch.DeviceOptional(cpuDevice)) { mapLocation =>
          archive.load_from(source.toString, mapLocation)
        }
      }
      attentionModel.load(archive)
    }
    source
  }

  def train(): Unit = {
    if (Files.exists(Paths.get(modelFilename))) {
      println(s"Loaded neural model from ${loadModel()}")
    }
    attentionModel.to(device)
    val optimizer = new Adam(attentionModel.parameters, lr = lrate.toDouble)

    println("Loading training samples...")
    val dataset = attentionModel.dataset()
    val totalSamples = dataset.size
    if (totalSamples == 0) {
      println("No training samples found.")
      return
    }

    println(s"Training on $totalSamples samples for $epocs epochs.")
    val progressEvery = math.max(1, math.min(5, totalSamples / 10))
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
      println(s"Saved neural model to ${saveModel()}")
    }
  }
}

object BaseTrainer {
  def main(args: Array[String]): Unit = {
    require(args.length <= 2, "Usage: BaseTrainer [train|load] [checkpoint.bin]")
    val mode = args.headOption.getOrElse("train")
    val filename = args.lift(1).getOrElse("resources/models/attention-model.bin")
    val trainer = new BaseTrainer(10, 0.001f, filename)
    mode match {
      case "train" => trainer.train()
      case "load" =>
        println(s"Loaded neural model from ${trainer.loadModel()}")
        trainer.attentionModel.to(trainer.device)
      case _ => throw new IllegalArgumentException("Usage: BaseTrainer [train|load] [checkpoint.bin]")
    }
  }
}
