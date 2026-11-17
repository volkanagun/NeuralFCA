package llm

import org.bytedeco.javacpp.PointerScope
import org.bytedeco.pytorch.{Adagrad, AdagradOptions, TensorVector}
import torch.Device

import java.nio.file.{AtomicMoveNotSupportedException, Files, Path, Paths, StandardCopyOption}
import scala.collection.mutable
import scala.io.Source
import scala.util.{Random, Using}

final class ConceptGloveTrainer(
    val sentenceFilename: String = "resources/sentences/sentences-tr.txt",
    val latticeFilename: String = "resources/binary/simple-lattice.bin",
    val embeddingFilename: String = "resources/embeddings/vectors.txt",
    val checkpointFilename: String = "resources/models/lattice-glove-join.bin",
    val outputLatticeFilename: String = "resources/binary/simple-glove-paired-lattice.bin",
    val outputEmbeddingFilename: String = "resources/embeddings/vectors-glove-paired.txt",
    val device: Device = Device.CUDA,
    val epochs: Int = 5,
    val learningRate: Double = 0.02,
    val batchSize: Int = 4096,
    val skipSamples: Int = 0,
    val maxSamples: Int =50000000,
    val dimension: Int = 300,
    val candidateExtents: Int = 30,
    val contextRadius: Int = 5,
    val xMax: Double = 500.0,
    val alpha: Double = 0.9,
    val maxPositiveConcepts: Int = 4,
    val countWorkers: Int = math.max(1, math.min(8, Runtime.getRuntime.availableProcessors())),
    val targetWordsOverride: Option[Set[String]] = None,
    val contextConditioned: Boolean = false
) {
  require(epochs > 0 && learningRate > 0 && batchSize > 0 && maxSamples > 0)
  require(xMax > 0 && alpha > 0 && maxPositiveConcepts >= 2 && countWorkers > 0)

  val targetWords = targetWordsOverride.getOrElse(
    readEvaluation("resources/evaluations/turkish_3cosadd_ambiguous.txt") ++
      readEvaluation("resources/evaluations/turkish_3cosadd_nonambiguous.txt"))

  val model = new ConceptGloveModel(sentenceFilename, targetWords, latticeFilename, embeddingFilename,
    device, dimension, candidateExtents, contextRadius, maxPositiveConcepts, contextConditioned)

  private val selectorMode = if contextConditioned then "occurrence" else "word-pair"
  private def selectorModePath(checkpoint: Path): Path =
    Paths.get(checkpoint.toString + ".selector-mode")

  private def verifySelectorMode(checkpoint: Path): Unit = {
    val modePath = selectorModePath(checkpoint)
    require(Files.isRegularFile(modePath), s"Checkpoint selector mode not found: $modePath")
    val savedMode = Files.readString(modePath).trim
    require(savedMode == selectorMode,
      s"Checkpoint uses '$savedMode' selection; this trainer uses '$selectorMode'. " +
        "Use a separate checkpoint for each selector mode.")
  }

  private def verifyParameterScale(): Unit = {
    val maximum = Seq(model.wordEmb, model.contextEmb, model.conceptEmb,
      model.wordBias, model.contextBias).map { parameter =>
      math.max(math.abs(parameter.max().item.toDouble),
        math.abs(torch.amin(parameter, dim = (0 until parameter.shape.size).toSeq,
          keepdim = false).item.toDouble))
    }.max
    require(maximum.isFinite && maximum < 100.0,
      f"GloVe parameters diverged (maximum absolute value $maximum%.3f); refusing to use this checkpoint")
  }

  def readEvaluation(filename: String): Set[String] =
    Using.resource(Source.fromFile(filename)) { source =>
      source.getLines().flatMap(line => line.split("\\s+"))
        .toSet
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
      Files.writeString(selectorModePath(target), selectorMode + "\n")
      target
    } finally Files.deleteIfExists(temporary)
  }

  def loadCheckpoint(): Path = {
    val source = Paths.get(checkpointFilename).toAbsolutePath
    require(Files.isRegularFile(source), s"Model checkpoint not found: $source")
    require(Files.isRegularFile(Paths.get(source.toString + ".rows.tsv")),
      s"Model checkpoint row mapping not found: $source.rows.tsv")
    verifySelectorMode(source)
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
    verifyParameterScale()
    source
  }

  private def reportCounts(samples: Seq[ConceptCooccurrenceSample]): Unit = {
    val reservoir = new Array[Double](math.min(samples.size, 100000))
    val random = new Random(42)
    val targetRows = targetWords.flatMap(word => model.index.get(word).map(_ -> word)).toMap
    val targetMass = mutable.Map.empty[Int, Double].withDefaultValue(0.0)
    val targetPairs = mutable.Map.empty[Int, Int].withDefaultValue(0)
    var belowCutoff = 0L
    samples.iterator.zipWithIndex.foreach { (sample, position) =>
      if sample.count < xMax then belowCutoff += 1
      if position < reservoir.length then reservoir(position) = sample.count
      else {
        val replacement = random.nextLong(position.toLong + 1L)
        if replacement < reservoir.length then reservoir(replacement.toInt) = sample.count
      }
      if targetRows.contains(sample.target) then {
        targetMass(sample.target) += sample.count
        targetPairs(sample.target) += 1
      }
    }
    java.util.Arrays.sort(reservoir)
    def percentile(fraction: Double): Double =
      reservoir(math.floor((reservoir.length - 1) * fraction).toInt)
    def weight(count: Double): Double =
      math.pow(math.min(count / xMax, 1.0), alpha)
    val median = percentile(0.5)
    val p90 = percentile(0.9)
    val p99 = percentile(0.99)
    println(f"Co-occurrence counts (up to 100000 sampled rows): p50=$median%.3f, " +
      f"p90=$p90%.3f, p99=$p99%.3f; $belowCutoff%d/${samples.size}%d pairs below xMax=$xMax%.3f")
    println(f"GloVe weights: count=1 -> ${weight(1.0)}%.4f, " +
      f"p50 -> ${weight(median)}%.4f, p90 -> ${weight(p90)}%.4f (alpha=$alpha%.3f)")
    if targetWords.nonEmpty then {
      val scarcest = targetWords.toVector.sortBy(word =>
        (model.index.get(word).map(targetMass).getOrElse(0.0), word)).take(20)
      println("Scarcest focus words (distance-weighted co-occurrences, distinct selected pairs):")
      scarcest.foreach { word =>
        val row = model.index.get(word)
        val mass = row.map(targetMass).getOrElse(0.0)
        val pairs = row.map(targetPairs).getOrElse(0)
        println(f"  $word%-22s $mass%12.2f $pairs%8d")
      }
    }
  }

  /** Inspect the actual selected-pair counts before choosing xMax and alpha. */
  def calibrate(): Unit = {
    val samples = model.selectedCooccurrences(skipSamples, maxSamples, countWorkers)
    require(samples.nonEmpty, "No concept-conditioned co-occurrences found for calibration")
    reportCounts(samples)
  }

  def train(): Path = {
    if Files.exists(Paths.get(checkpointFilename)) then
      println(s"Loaded GloVe model from ${loadCheckpoint()}")
    model.to(device)
    val cooccurrences = model.selectedCooccurrences(skipSamples, maxSamples, countWorkers)
    require(cooccurrences.nonEmpty, "No concept-conditioned co-occurrences found for GloVe training")
    println(s"Counted ${cooccurrences.size} distinct concept-conditioned co-occurrences")
    reportCounts(cooccurrences)
    val optimizer = new Adagrad(
      new TensorVector(model.parameters.map(_.native).toSeq*), new AdagradOptions(learningRate))
    for (epoch <- 1 to epochs) {
      var totalLoss = 0.0
      var batches = 0
      var processed = 0L
      cooccurrences.grouped(batchSize).foreach { batch =>
        optimizer.zero_grad()
        val value = Using.resource(new PointerScope()) { _ =>
          val loss = model.loss(batch, xMax, alpha)
          val result = loss.item.toDouble
          require(result.isFinite, s"Non-finite loss in epoch $epoch, batch ${batches + 1}")
          loss.backward()
          torch.nn.utils.clipGradNorm_(model.parameters, 1.0)
          optimizer.step()
          result
        }
        batches += 1
        processed += batch.size
        totalLoss += value
        if batches % 100 == 0 then {
          verifyParameterScale()
          println(f"Epoch $epoch/$epochs: $processed%d concept-conditioned pairs, loss=${totalLoss / batches}%.4f")
        }
      }
      verifyParameterScale()
      println(f"Epoch $epoch/$epochs: $processed%d concept-conditioned pairs, mean loss=${totalLoss / batches}%.4f")
      println(s"Saved GloVe model to ${saveCheckpoint()}")
    }
    Paths.get(checkpointFilename).toAbsolutePath
  }

  def exportEmbeddings(): Path = {
    val checkpoint = Paths.get(checkpointFilename).toAbsolutePath
    require(Files.isRegularFile(checkpoint), s"Train or load a checkpoint before export: $checkpoint")
    require(Files.isRegularFile(Paths.get(checkpoint.toString + ".rows.tsv")),
      s"Model checkpoint row mapping not found: $checkpoint.rows.tsv")
    verifySelectorMode(checkpoint)
    val output = new AttentionLatticeUpdater(checkpoint.toString, latticeFilename,
      outputLatticeFilename, dimension, candidateExtents, embeddingFilename,
      outputEmbeddingFilename, combineContextEmbeddings = true).update()
    println(s"Exported concept lattice to $output")
    println(s"Exported word embeddings to ${Paths.get(outputEmbeddingFilename).toAbsolutePath}")
    output
  }
}

object ConceptGloveTrainer {
  def main(args: Array[String]): Unit = {
    require(args.length <= 1 || Set("calibrate", "train-focused").contains(args.head),
      "Usage: ConceptGloveTrainer [train|export|sentence|sentence-export|calibrate [words...]|train-focused words...]")
    args.headOption.getOrElse("train") match {
      case "train" =>
        val trainer = new ConceptGloveTrainer()
        trainer.train()
        trainer.exportEmbeddings()
      case "export" => new ConceptGloveTrainer().exportEmbeddings()
      case "sentence" =>
        val trainer = new SentenceConceptTrainer()
        trainer.train()
        trainer.exportEmbeddings()
      case "sentence-export" => new SentenceConceptTrainer().exportEmbeddings()
      case "calibrate" =>
        val words = args.drop(1).toSet
        new ConceptGloveTrainer(targetWordsOverride = if words.nonEmpty then Some(words) else None)
          .calibrate()
      case "train-focused" =>
        require(args.length > 1, "train-focused requires at least one focus word")
        val trainer = new ConceptGloveTrainer(targetWordsOverride = Some(args.drop(1).toSet))
        trainer.train()
        trainer.exportEmbeddings()
      case _ => throw new IllegalArgumentException(
        "Usage: ConceptGloveTrainer [train|export|sentence|sentence-export|calibrate [words...]|train-focused words...]")
    }
  }
}
