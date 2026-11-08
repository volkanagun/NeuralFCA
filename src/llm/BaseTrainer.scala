package llm

import torch.Device
import torch.nn.functional
import torch.optim.AdamW
import org.bytedeco.javacpp.PointerScope

import java.nio.file.{AtomicMoveNotSupportedException, Files, Path, Paths, StandardCopyOption}
import scala.util.{Random, Using}

class BaseTrainer(val epocs:Int, val lrate:Float,
                  val modelFilename: String = "resources/models/attention-model-causal-stable.bin",
                  val batchSize: Int = 128, val startLines:Int = 100000, val maxLines:Int = 100000) {

  require(epocs > 0, "Epoch count must be positive")
  require(lrate > 0.0f, "Learning rate must be positive")
  //require(lrate <= 0.0005f, "Learning rate above 0.0005 caused divergence in this model")
  require(batchSize > 0, "Batch size must be positive")
  require(maxLines > 0, "Sample limit must be positive")

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
    val optimizer = new AdamW(attentionModel.parameters, lr = lrate.toDouble, weightDecay = 0.01)

    println("Loading training samples...")
    val dataset = attentionModel.dataset(startLines, maxLines)
    val totalSamples = dataset.size
    if (totalSamples == 0) {
      println("No training samples found.")
      return
    }

    val totalBatches = (totalSamples - 1) / batchSize + 1
    val totalSteps = epocs.toLong * totalBatches
    val warmupSteps = math.min(500L, math.max(1L, totalSteps / 20))

    def learningRateAt(step: Long): Double = {
      if (step <= warmupSteps) lrate.toDouble * step / warmupSteps
      else {
        val progress = (step - warmupSteps).toDouble / (totalSteps - warmupSteps)
        lrate.toDouble * (0.1 + 0.45 * (1.0 + math.cos(math.Pi * progress)))
      }
    }

    println(s"Training on $totalSamples samples in $totalBatches batches for $epocs epochs.")
    val progressEvery = math.max(1, totalBatches / 20)
    for (epoch <- 1 to epocs) {

      var totalLoss = 0.0
      var recentLoss = 0.0
      var recentSamples = 0
      println(s"Epoch $epoch/$epocs started.")
      val shuffled = Random.shuffle(dataset)
      for ((batch, batchIndex) <- shuffled.iterator.grouped(batchSize).zipWithIndex) {

        val step = (epoch - 1).toLong * totalBatches + batchIndex + 1
        val currentRate = learningRateAt(step)
        torch.OptimizerLearningRate.set(optimizer, currentRate)
        optimizer.zeroGrad()
        val (batchLoss, gradNorm) = Using.resource(new PointerScope()) { _ =>
          val logits = attentionModel.applyBatch(batch.toSeq)
          val targetTensor =
            torch.Tensor(batch.map(attentionModel.target).toSeq)
              .to(dtype = torch.int64, device=device)
          val loss = functional.crossEntropy(logits, targetTensor)
          val lossValue = loss.item.toDouble
          if (!lossValue.isFinite) {
            throw new IllegalStateException(s"Non-finite training loss at epoch $epoch, batch ${batchIndex + 1}")
          }
          loss.backward()
          val gradNorm = torch.nn.utils.clipGradNorm_(attentionModel.parameters, 1.0)
          if (!gradNorm.isFinite) {
            throw new IllegalStateException(s"Non-finite gradient norm at epoch $epoch, batch ${batchIndex + 1}")
          }
          (lossValue, gradNorm)
        }
        optimizer.step()

        totalLoss += batchLoss * batch.size
        recentLoss += batchLoss * batch.size
        recentSamples += batch.size
        val processed = batchIndex * batchSize + batch.size
        if ((batchIndex + 1) % progressEvery == 0 || processed == totalSamples) {
          println(f"Epoch $epoch%d/$epocs%d: $processed%d/$totalSamples%d samples, " +
            f"recent loss ${recentLoss / recentSamples}%.4f, epoch loss ${totalLoss / processed}%.4f, " +
            f"grad norm $gradNorm%.3f, lr $currentRate%.6g")
          recentLoss = 0.0
          recentSamples = 0
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
    val max = 7000000
    val start = 5000000
    val filename = args.lift(1).getOrElse("resources/models/attention-model-causal-stable.bin")
    val trainer = new BaseTrainer(6, 0.01f, filename, batchSize = 512, maxLines = max)
    mode match {
      case "train" => trainer.train()
      case "load" =>
        println(s"Loaded neural model from ${trainer.loadModel()}")
        trainer.attentionModel.to(trainer.device)
      case _ => throw new IllegalArgumentException("Usage: BaseTrainer [train|load] [checkpoint.bin]")
    }
  }
}
