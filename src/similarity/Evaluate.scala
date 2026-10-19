package similarity

import cvafca.{CVA, CVADataset, Instance, Lattice, Node, Tokenizer}
import org.bytedeco.javacpp.PointerScope
import torch.Device

import java.nio.file.{Files, Paths}
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

  /** Stream one sentence at a time and share its ranking across all cutoffs. */
  def evaluateExtrinsic(filename: String = Evaluate.defaultExtrinsicFilename): Vector[EvalScore] = {
    val scores = Vector(1, 5, 20).map(k => EvalScore("extrinsic", topK = k))
    val similarity = new ConceptSimilarity(nodes)
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
        else if !supportedTargets.getOrElseUpdate(sentence.target, nodes.exists(_.containsSymbol(sentence.target))) then
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

  val dim = 600

  def main(args: Array[String]): Unit = {
    println("[Evaluate] Loading lattice...")
    val windows = 3
    val lattice = new Lattice(CVA(dim), c_device = Device.CPU).load(s"resources/binary/fca-cva-${windows}.bin")
    println("[Evaluate] Loading embeddings...")
    val map = new CVADataset().readEmbeddingMap("resources/embeddings/vectors.txt", Device.CPU)
    println(s"[Evaluate] Loaded ${map.size} embeddings; preparing common-vector concept similarity...")
    val evaluator = new Evaluate(nodes = lattice.nodes, embeddingMap = map)
    evaluator.extrinsic()
  }
}
