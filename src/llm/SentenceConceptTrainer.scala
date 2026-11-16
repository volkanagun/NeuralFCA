package llm

import org.bytedeco.javacpp.PointerScope
import torch.Device
import torch.optim.AdamW

import java.nio.file.{AtomicMoveNotSupportedException, Files, Path, Paths, StandardCopyOption}
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.{Random, Using}

/** Streams sentences and trains static word and FCA concept tables through masked prediction. */
final class SentenceConceptTrainer(
                                    val sentenceFilename: String = "resources/sentences/sentences-tr.txt",
                                    val latticeFilename: String = "resources/binary/simple-lattice.bin",
                                    val embeddingFilename: String = "resources/embeddings/vectors.txt",
                                    val checkpointFilename: String = "resources/models/lattice-sentence-concept.bin",
                                    val outputLatticeFilename: String = "resources/binary/simple-sentence-concept-lattice.bin",
                                    val outputEmbeddingFilename: String = "resources/embeddings/vectors-sentence-concept.txt",
                                    val device: Device = Device.CUDA,
                                    val epochs: Int = 5,
                                    val learningRate: Double = 0.0003,
                                    val batchSize: Int = 4096,
                                    val skipSamples: Int = 0,
                                    val maxSamples: Int = 5000000,
                                    val dimension: Int = 300,
                                    val candidateExtents: Int = 30,
                                    val contextRadius: Int = 3,
                                    val maxPositiveConcepts: Int = 4,
                                    val wordNegatives: Int = 256,
                                    val conceptNegatives: Int = 128,
                                    val conceptWeight: Float = 0.2f
                                  ) {
  require(epochs > 0 && learningRate > 0 && batchSize > 0)
  require(skipSamples >= 0 && maxSamples > 0 && wordNegatives > 0 && conceptNegatives > 0)
  require(conceptWeight >= 0f)

  val model = new SentenceConceptModel(latticeFilename, embeddingFilename, device,
    dimension, candidateExtents, contextRadius, maxPositiveConcepts)

  private def configPath(checkpoint: Path): Path = Paths.get(checkpoint.toString + ".sentence-config")

  private def config: String =
    s"masked-sentence-v1\t$dimension\t$candidateExtents\t$contextRadius\t$maxPositiveConcepts"

  private def verifyConfig(checkpoint: Path): Unit = {
    val path = configPath(checkpoint)
    require(Files.isRegularFile(path) && Files.readString(path).trim == config,
      s"Sentence model configuration differs from checkpoint: $path")
  }

  def saveCheckpoint(): Path = {
    val target = Paths.get(checkpointFilename).toAbsolutePath
    Files.createDirectories(target.getParent)
    val temporary = Files.createTempFile(target.getParent, target.getFileName.toString + ".", ".tmp")
    try {
      Using.resource(new org.bytedeco.pytorch.OutputArchive()) { archive =>
        model.save(archive)
        archive.save_to(temporary.toString)
      }
      try Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
      catch {
        case _: AtomicMoveNotSupportedException =>
          Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING)
      }
      EmbeddingRowMapping.save(target, model)
      Files.writeString(configPath(target), config + "\n")
      target
    } finally Files.deleteIfExists(temporary)
  }

  def loadCheckpoint(): Path = {
    val source = Paths.get(checkpointFilename).toAbsolutePath
    require(Files.isRegularFile(source), s"Model checkpoint not found: $source")
    require(Files.isRegularFile(Paths.get(source.toString + ".rows.tsv")),
      s"Model checkpoint row mapping not found: $source.rows.tsv")
    verifyConfig(source)
    EmbeddingRowMapping.verifyIfPresent(source, model)
    Using.resource(new org.bytedeco.pytorch.InputArchive()) { archive =>
      Using.resource(new org.bytedeco.pytorch.Device("cpu")) { cpu =>
        Using.resource(new org.bytedeco.pytorch.DeviceOptional(cpu)) { location =>
          archive.load_from(source.toString, location)
        }
      }
      model.load(archive)
    }
    model.to(device)
    source
  }

  private def sample(tokens: Vector[Int], center: Int): SentenceConceptSample = {
    val start = center - contextRadius
    val positions = (0 until model.width).map(start + _).toVector
    val visible = positions.map(position => position >= 0 && position < tokens.size && position != center)
    val words = positions.zip(visible).map { (position, isVisible) =>
      if isVisible then tokens(position) else model.index.size
    }
    val concepts = positions.zip(visible).map { (position, isVisible) =>
      if !isVisible then model.conceptSize
      else model.conceptRows(tokens(position)).headOption.getOrElse(model.conceptSize)
    }
    val targetConcept = model.selectConcept(tokens(center), words.zip(visible)
      .collect { case (word, true) => word })
    SentenceConceptSample(words, concepts, visible, tokens(center), targetConcept)
  }

  /** Includes every sentence in the chosen token range; no evaluation-word filter. */
  private def foreachBatch(consume: Seq[SentenceConceptSample] => Unit): Unit = {
    val stop = skipSamples.toLong + maxSamples.toLong
    var seen = 0L
    val batch = ArrayBuffer.empty[SentenceConceptSample]
    Using.resource(Source.fromFile(sentenceFilename, "UTF-8")) { source =>
      val lines = source.getLines()
      while lines.hasNext && seen < stop do {
        val tokens = model.tokenize(lines.next()).toVector
        val from = math.max(0L, skipSamples.toLong - seen).min(tokens.size.toLong).toInt
        val until = math.max(0L, stop - seen).min(tokens.size.toLong).toInt
        seen += tokens.size
        for position <- from until until if tokens(position) != model.index.size do {
          batch += sample(tokens, position)
          if batch.size >= batchSize then {
            consume(batch.toVector)
            batch.clear()
          }
        }
      }
      if batch.nonEmpty then consume(batch.toVector)
    }
  }

  def train(): Path = {
    if Files.exists(Paths.get(checkpointFilename)) then
      println(s"Loaded sentence concept model from ${loadCheckpoint()}")
    model.to(device)
    val optimizer = new AdamW(model.parameters, lr = learningRate, weightDecay = 0.01)
    val random = new Random(42)
    for epoch <- 1 to epochs do {
      var totalLoss = 0.0
      var batches = 0
      var processed = 0L
      foreachBatch { batch =>
        val wordNoise = Vector.fill(wordNegatives)(random.nextInt(model.index.size))
        val conceptNoise = if model.conceptSize == 0 then Vector.empty
        else Vector.fill(conceptNegatives)(random.nextInt(model.conceptSize))
        optimizer.zeroGrad()
        val value = Using.resource(new PointerScope()) { _ =>
          val loss = model.loss(batch.toSeq, wordNoise, conceptNoise, conceptWeight)
          val result = loss.item.toDouble
          require(result.isFinite, s"Non-finite sentence loss in epoch $epoch, batch ${batches + 1}")
          loss.backward()
          val gradNorm = torch.nn.utils.clipGradNorm_(model.parameters, 1.0)
          require(gradNorm.isFinite,
            s"Non-finite sentence gradient in epoch $epoch, batch ${batches + 1}")
          result
        }
        optimizer.step()
        batches += 1
        processed += batch.size
        totalLoss += value
        if batches % 100 == 0 then
          println(f"Epoch $epoch/$epochs: $processed%d masked words, loss=${totalLoss / batches}%.4f")
      }
      require(batches > 0, "No known words found in the selected sentence range")
      println(f"Epoch $epoch/$epochs: $processed%d masked words, mean loss=${totalLoss / batches}%.4f")
      println(s"Saved sentence concept model to ${saveCheckpoint()}")
    }
    Paths.get(checkpointFilename).toAbsolutePath
  }

  def exportEmbeddings(): Path = {
    val checkpoint = Paths.get(checkpointFilename).toAbsolutePath
    require(Files.isRegularFile(checkpoint), s"Train or load a checkpoint before export: $checkpoint")
    require(Files.isRegularFile(Paths.get(checkpoint.toString + ".rows.tsv")),
      s"Model checkpoint row mapping not found: $checkpoint.rows.tsv")
    verifyConfig(checkpoint)
    new AttentionLatticeUpdater(checkpoint.toString, latticeFilename,
      outputLatticeFilename, dimension, candidateExtents, embeddingFilename,
      outputEmbeddingFilename).update()
  }
}


object SentenceConceptTrainer {
  def main(args: Array[String]): Unit = {

    val trainer = new SentenceConceptTrainer()
    trainer.train()
    trainer.exportEmbeddings()

  }
}