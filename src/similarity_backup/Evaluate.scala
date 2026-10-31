package similarity

import cvafca.{CVA, CVADataset, Tokenizer}
import org.bytedeco.javacpp.PointerScope
import simple.{Instance, Lattice, Node}
import torch.Device

import java.io.{BufferedInputStream, DataInputStream, FileInputStream}
import java.nio.charset.StandardCharsets
import java.nio.file.{AtomicMoveNotSupportedException, Files, Path, Paths, StandardCopyOption}
import java.time.Instant
import scala.collection.mutable
import scala.io.Source
import scala.util.Using

case class ExtrinsicSentenceSample(
                                    target: String,
                                    sense: String,
                                    disambiguatedWord: String,
                                    similarWords: Vector[String],
                                    sentence: String
                                  ) {
  def readPrior(): String =
    val pattern = ("(?U)(?<!\\w)" + java.util.regex.Pattern.quote(target) + "(?!\\w)").r
    val occurrence = pattern.findFirstMatchIn(sentence).getOrElse(
      throw new IllegalArgumentException(s"Target '$target' is missing from sentence"))
    sentence.substring(0, occurrence.start)

}

class EvalScore(val name: String, var trueCount: Double = 0, var totalCount: Double = 0, val topK: Int = 1) {
  override def toString: String =
    s"Name : ${name}, TopK:${topK}, True Count: ${trueCount}, Total:${totalCount}, Accuracy:${trueCount * 100.0 / totalCount}"
}

final case class IntrinsicAnalogy(source: String, destination: String, item: String, expected: String)

final case class IntrinsicComparison(
    filename: String,
    totalRows: Int,
    embeddingCoveredRows: Int,
    sharedCoveredRows: Int,
    threeCosAdd: Vector[EvalScore],
    conceptSimilarity: Vector[EvalScore]
)

class Evaluate(val nodes: Vector[Node], val embeddingMap: Map[String, torch.Tensor[torch.BFloat16]],
               val window:Int = 3) {

  /** Average or concatenate known prior-token embeddings in sentence order.
   * Skip samples without known prior tokens.
   */
  def readExtrinsic(avg: Boolean = true, filename: String = Evaluate.defaultExtrinsicFilename): Array[(Set[String], Instance)] = {
    val tokenizer = new Tokenizer()
    val dataset = new CVADataset()
    val sentences = Evaluate.readExtrinsic(filename)

    val instances = sentences.flatMap(sentence => {
      val priors = tokenizer.tokenize(sentence.readPrior()).flatMap(embeddingMap.get).toVector
      if priors.isEmpty then {
        None
      }
      else if avg then {
        val vectors = dataset.concatVectors(Iterable(embeddingMap(sentence.target), dataset.averageVectors(priors)))
        Some((sentence.similarWords.toSet, Instance(sentence.target, vectors)))
      } else {
        val vectors = dataset.concatVectors(priors)
        Some((sentence.similarWords.toSet, Instance(sentence.target, vectors)))
      }
    })

    instances
  }

  /** Compare both rankings on words available to both methods. */
  def evaluateInstrinsic(): Vector[IntrinsicComparison] =
    evaluateInstrinsic(Evaluate.defaultIntrinsicFilenames)

  def evaluateInstrinsic(
      filenames: Seq[String],
      resultFilename: String = Evaluate.defaultIntrinsicResultFilename,
      checkpointFilename: String = "unspecified",
      embeddingsFilename: String = "unspecified", limit:Int = 5
  ): Vector[IntrinsicComparison] = {
    require(nodes.nonEmpty, "Intrinsic evaluation needs lattice concepts")
    require(embeddingMap.nonEmpty, "Intrinsic evaluation needs embeddings")
    println(s"[Evaluate] Intrinsic: preparing vocabulary from ${nodes.size} concepts " +
      s"and ${embeddingMap.size} embeddings...")
    val conceptSymbols = nodes.iterator.flatMap(_.extent.map(_.id)).filter(embeddingMap.contains).toSet
    require(conceptSymbols.nonEmpty, "Lattice and embeddings have no shared words")
    val threeCosAdd = new ThreeCosAddSimilarity(embeddingMap, conceptSymbols)
    val sharedSymbols = threeCosAdd.vocabulary
    require(sharedSymbols.nonEmpty, "No shared words have usable nonzero embeddings")
    val conceptSimilarity = new Concept3CosAdd(nodes)
    val cutoffs = Vector(1, 5, 20)
    val resultPath = Paths.get(resultFilename).toAbsolutePath.normalize()
    val completedResults = mutable.ArrayBuffer.empty[(IntrinsicComparison, Int)]
    val parameters = Vector(
      "checkpoint_file" -> checkpointFilename,
      "embedding_file" -> embeddingsFilename,
      "concept_count" -> nodes.size.toString,
      "embedding_count" -> embeddingMap.size.toString,
      "concept_common_dimension" -> Math.toIntExact(nodes.head.commonVector.numel).toString,
      "embedding_dimension" -> Math.toIntExact(embeddingMap.head._2.numel).toString,
      "shared_candidate_words" -> sharedSymbols.size.toString,
      "concept_index_vectors" -> conceptSimilarity.indexedVectorCount.toString,
      "concept_index_words" -> conceptSimilarity.indexedSymbolCount.toString,
      "top_k" -> cutoffs.mkString(","),
      "datasets" -> filenames.mkString(","),
      "baseline_equation" -> "destination - source + target",
      "concept_equation" -> "destination - source + target",
      "candidate_vocabulary" -> "shared nonzero embeddings and concept symbols",
      "excluded_words" -> "source,destination,target"
    )
    def saveResults(active: Option[(IntrinsicComparison, Int)] = None, complete: Boolean = false): Unit =
      Evaluate.writeIntrinsicResults(resultPath, parameters,
        completedResults.toVector ++ active.toVector, complete)

    saveResults()

    val comparisons = filenames.toVector.map { filename =>
      val name = Paths.get(filename).getFileName.toString.stripSuffix(".txt")
      val addScores = cutoffs.map(k => new EvalScore(s"3CosAdd $name", topK = k))
      val conceptScores = cutoffs.map(k => new EvalScore(s"Concept3CosAdd $name", topK = k))
      val rows = Evaluate.readIntrinsic(filename)
      val startedAt = System.nanoTime()
      var lastProgressAt = startedAt
      var embeddingCovered = 0
      var sharedCovered = 0
      println(s"[Evaluate] $name: starting ${rows.size} analogies")
      def currentResult: IntrinsicComparison =
        IntrinsicComparison(filename, rows.size, embeddingCovered, sharedCovered,
          addScores, conceptScores)
      saveResults(Some((currentResult, 0)))

      def reportProgress(
          completed: Int,
          searchedVariants: Long = 0L,
          totalVariants: Long = 0L,
          force: Boolean = false
      ): Unit =
        val now = System.nanoTime()
        if force || now - lastProgressAt >= 2000000000L then {
          val percent = if rows.isEmpty then 100.0 else completed.toDouble * 100.0 / rows.size
          val scan = if totalVariants == 0 then "" else
            s" | current query variants $searchedVariants/$totalVariants"
          val elapsed = (now - startedAt) / 1e9
          println(f"[Evaluate] $name: $completed/${rows.size} rows ($percent%.1f%%) | " +
            s"shared=$sharedCovered | embeddings=$embeddingCovered$scan | " +
            f"elapsed=$elapsed%.1fs")
          lastProgressAt = now
        }

      rows.iterator.zipWithIndex.take(limit).foreach { (row, index) =>
        val words = Vector(row.source, row.destination, row.item, row.expected)
        if words.forall(embeddingMap.contains) then {
          embeddingCovered += 1
          if words.forall(sharedSymbols.contains) then {
            sharedCovered += 1
            Using.resource(new PointerScope()) { _ =>
              val query = threeCosAdd.queryVector(row.source, row.destination, row.item)
              val excluded = Set(row.source, row.destination, row.item)
              
              val conceptWords = conceptSimilarity.projectAdd(
                row.source, row.destination, row.item, cutoffs.last, sharedSymbols,
                (searched, total) => reportProgress(index, searched, total))
              val addWords = threeCosAdd.rank(query, excluded, cutoffs.last)
              cutoffs.indices.foreach { index =>
                val k = cutoffs(index)
                val addScore = addScores(index)
                val conceptScore = conceptScores(index)
                addScore.totalCount += 1
                conceptScore.totalCount += 1
                if addWords.take(k).contains(row.expected) then addScore.trueCount += 1
                if conceptWords.take(k).contains(row.expected) then conceptScore.trueCount += 1
              }
            }
          }
        }
        reportProgress(index + 1)
        saveResults(Some((currentResult, index + 1)))
      }
      reportProgress(rows.size, force = true)
      val comparison = currentResult
      completedResults += ((comparison, rows.size))
      saveResults()
      println(s"[Evaluate] $name: $sharedCovered/${rows.size} shared rows " +
        s"($embeddingCovered embedding-covered)")
      cutoffs.indices.foreach { index =>
        val add = addScores(index)
        val concept = conceptScores(index)
        def result(score: EvalScore): String =
          if score.totalCount == 0 then "n/a"
          else f"${score.trueCount.toInt}/${score.totalCount.toInt} " +
            f"(${score.trueCount * 100.0 / score.totalCount}%.2f%%)"
        println(s"[Evaluate] top-${cutoffs(index)}: 3CosAdd ${result(add)} | " +
          s"Concept3CosAdd ${result(concept)}")
      }
      comparison
    }
    saveResults(complete = true)
    println(s"[Evaluate] Intrinsic results written to $resultPath")
    comparisons
  }

  /** Stream one sentence at a time and share its ranking across all cutoffs. */
  def evaluateExtrinsic(filename: String = Evaluate.defaultExtrinsicFilename): Vector[EvalScore] = {
    val scores = Vector(1, 5, 20).map(k => EvalScore("extrinsic", topK = k))
    val similarity = new Concept3CosAdd(nodes)
    val supportedTargets = scala.collection.mutable.HashMap.empty[String, Boolean]
    val tokenizer = new Tokenizer()
    val dataset = new CVADataset()
    val startedAt = System.nanoTime()
    var lastProgressAt = startedAt
    var completed = 0L
    var missingConcepts = 0L
    var missingPriors = 0L
    println(s"[Evaluate] Extrinsic: streaming samples against ${nodes.length} concepts...")

    Using.resource(Evaluate.foreachExtrinsic(filename)) { samples =>
      samples.foreach { sentence =>
        val priors = tokenizer.tokenize(sentence.readPrior()).reverse.take(window).reverse.flatMap(embeddingMap.get).toVector
        if priors.isEmpty then missingPriors += 1
        else if !supportedTargets.getOrElseUpdate(sentence.target, nodes.exists(_.contains(sentence.target))) then
          missingConcepts += 1
        else

          Using.resource(new PointerScope()) { _ =>
            val mean = dataset.averageVectors(priors)
            val concat = dataset.concatVectors(Iterable(embeddingMap(sentence.target), mean))
            val items = similarity.projectAdd(sentence.target, concat, scores.last.topK)
            scores.foreach { score =>
              val topSimilar = items.take(score.topK).flatten
              val targetSimilar = sentence.similarWords
              if targetSimilar.exists(similar => topSimilar.contains(similar)) then {
                score.trueCount += 1
                println(s"True count: ${score.trueCount}")
              }
              score.totalCount += 1
            }
          }
          completed += 1
        val now = System.nanoTime()
        if now - lastProgressAt >= 1000000000L then
          val elapsed = (now - startedAt) / 1e9
          println(f"[Evaluate] Extrinsic: $completed comparisons, elapsed=$elapsed%.1fs")
          lastProgressAt = now
      }
    }
    println(s"[Evaluate] Extrinsic: complete, $completed comparisons " +
      s"($missingConcepts skipped: missing lattice concepts; $missingPriors skipped: missing prior embeddings)")
    scores
  }

  def extrinsic(): Unit = {
    val scores = evaluateExtrinsic()
    val resultPath = Paths.get(s"resources/results/result-extrinsic-${window}.txt").toAbsolutePath.normalize()
    Files.createDirectories(resultPath.getParent)
    Using.resource(Files.newBufferedWriter(resultPath)) { writer =>
      scores.foreach(score => {
        writer.write(score.toString)
        writer.newLine()
      })
    }
    println(s"[Evaluate] Complete. Scores written to $resultPath")
  }


}

object Evaluate {
  val defaultExtrinsicFilename = "resources/evaluations/templates_expanded.txt"
  val defaultIntrinsicResultFilename = "resources/results/result-intrinsic.txt"
  val defaultIntrinsicFilenames = Vector(
    "resources/evaluations/turkish_3cosadd_ambiguous.txt",
    "resources/evaluations/turkish_3cosadd_nonambiguous.txt"
  )

  private[similarity] def writeIntrinsicResults(
      target: Path,
      parameters: Seq[(String, String)],
      results: Seq[(IntrinsicComparison, Int)],
      complete: Boolean
  ): Unit = {
    Files.createDirectories(target.getParent)
    val temporary = Files.createTempFile(target.getParent, ".intrinsic-result-", ".tmp")
    try {
      Using.resource(Files.newBufferedWriter(temporary, StandardCharsets.UTF_8)) { writer =>
        def line(fields: String*): Unit = {
          writer.write(fields.map(_.replace('\t', ' ').replace('\n', ' ')).mkString("\t"))
          writer.newLine()
        }
        line("status", if complete then "complete" else "running")
        line("updated_at_utc", Instant.now().toString)
        parameters.foreach { (name, value) => line(name, value) }
        writer.newLine()
        line("dataset", "processed_rows", "total_rows", "embedding_covered_rows", "shared_covered_rows")
        results.foreach { (result, processedRows) =>
          line(result.filename, processedRows.toString, result.totalRows.toString,
            result.embeddingCoveredRows.toString, result.sharedCoveredRows.toString)
        }
        writer.newLine()
        line("dataset", "method", "top_k", "correct", "evaluated", "accuracy_percent")
        results.foreach { (result, _) =>
          Vector("3CosAdd" -> result.threeCosAdd,
            "Concept3CosAdd" -> result.conceptSimilarity).foreach { (method, scores) =>
            scores.foreach { score =>
              val accuracy = if score.totalCount == 0 then "n/a"
                else f"${score.trueCount * 100.0 / score.totalCount}%.6f"
              line(result.filename, method, score.topK.toString,
                score.trueCount.toLong.toString, score.totalCount.toLong.toString, accuracy)
            }
          }
        }
      }
      try Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE,
        StandardCopyOption.REPLACE_EXISTING)
      catch {
        case _: AtomicMoveNotSupportedException =>
          Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING)
      }
    } finally Files.deleteIfExists(temporary)
  }

  def readIntrinsic(filename: String): Vector[IntrinsicAnalogy] =
    Using.resource(Source.fromFile(filename, "UTF-8")) { source =>
      source.getLines().zipWithIndex.flatMap { (raw, index) =>
        val line = raw.stripPrefix("\uFEFF").trim
        if line.isEmpty || line.startsWith("#") || line.startsWith(":") then None
        else {
          val words = line.split("\\s+")
          require(words.length == 4, s"$filename:${index + 1}: expected four analogy words")
          Some(IntrinsicAnalogy(words(0), words(1), words(2), words(3)))
        }
      }.toVector
    }

  /** Headers start with target|sense|preferred word, followed by one or more
   * related words separated by pipes or commas, with no fixed upper limit.
   * Each following template expands its [alternative|alternative] group.
   */
  def readExtrinsic(filename: String = defaultExtrinsicFilename): Array[ExtrinsicSentenceSample] =
    Using.resource(foreachExtrinsic(filename))(_.toArray)

  /** Lazily expand templates. Closes on exhaustion or parsing failure.
   * Use Using.resource or close() when stopping early or processing may throw.
   */
  def foreachExtrinsic(filename: String): Iterator[ExtrinsicSentenceSample] & AutoCloseable = {
    val source = Source.fromFile(filename, "UTF-8")
    new Iterator[ExtrinsicSentenceSample] with AutoCloseable {
      private var closed = false
      private val alternatives = "\\[([^\\[\\]]+)\\]".r
      private var current: Option[(String, String, String, Vector[String])] = None
      private val samples = source.getLines().zipWithIndex.flatMap { case (raw, index) =>
        val line = raw.stripPrefix("\uFEFF").trim
        val location = s"$filename:${index + 1}"
        if line.nonEmpty then
          if !line.contains('[') && !line.contains(']') then
            val fields = line.split("\\|", -1).map(_.trim)
            require(fields.length >= 4 && fields.forall(_.nonEmpty), s"$location: invalid sense header")
            val related = fields.iterator.drop(3).flatMap(_.split(",", -1)).map(_.trim).toVector
            require(related.forall(_.nonEmpty), s"$location: empty similar word")
            val words = (Vector(fields(2)) ++ related).distinct.filterNot(_ == fields(0))
            require(words.nonEmpty, s"$location: missing disambiguation words")
            current = Some((fields(0), fields(1), fields(2), words))
            Iterator.empty
          else
            val (target, sense, preferred, words) = current.getOrElse(
              throw new IllegalArgumentException(s"$location: sentence before sense header"))
            val groups = alternatives.findAllMatchIn(line).toVector
            require(groups.size == 1, s"$location: expected one alternatives group")
            val group = groups.head
            val prefix = line.substring(0, group.start)
            val suffix = line.substring(group.end)
            require(!(prefix + suffix).exists(c => c == '[' || c == ']' || c == '|'),
              s"$location: malformed alternatives")
            val choices = group.group(1).split("\\|", -1).map(_.trim)
            require(choices.forall(_.nonEmpty), s"$location: empty alternative")
            val targetPattern = ("(?U)(?<!\\w)" + java.util.regex.Pattern.quote(target) + "(?!\\w)").r
            choices.iterator.map { choice =>
              val sentence = prefix + choice + suffix
              require(targetPattern.findAllMatchIn(sentence).size == 1,
                s"$location: sentence must contain the target '$target' exactly once")
              ExtrinsicSentenceSample(target, sense, preferred, words, sentence)
            }
        else Iterator.empty
      }

      override def close(): Unit =
        if !closed then
          closed = true
          source.close()

      override def hasNext: Boolean =
        if closed then false
        else
          try
            val available = samples.hasNext
            if !available then close()
            available
          catch
            case scala.util.control.NonFatal(error) =>
              close()
              throw error

      override def next(): ExtrinsicSentenceSample =
        if !hasNext then throw new NoSuchElementException("No more extrinsic samples")
        else
          try samples.next()
          catch
            case scala.util.control.NonFatal(error) =>
              close()
              throw error
    }
  }

  def main(args: Array[String]): Unit = {
    val checkpoint = args.lift(1).getOrElse("resources/binary/simple-lattice.bin")
    val embeddingFilename = args.lift(2).getOrElse("resources/embeddings/vectors.txt")
    val dim = 300
    println("[Evaluate] Loading lattice...")
    val lattice = new Lattice(dim).load(checkpoint)
    println("[Evaluate] Loading embeddings...")
    val map = new CVADataset().readEmbeddingMap(embeddingFilename, Device.CPU)
    println(s"[Evaluate] Loaded ${map.size} embeddings; preparing common-vector concept similarity...")
    val evaluator = new Evaluate(nodes = lattice.getNodes.toVector, embeddingMap = map)
    evaluator.evaluateInstrinsic(Evaluate.defaultIntrinsicFilenames,
      checkpointFilename = checkpoint, embeddingsFilename = embeddingFilename)
    //evaluator.extrinsic()
  }
}
