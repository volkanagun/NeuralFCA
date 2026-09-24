package cvafca


import torch.Device

import java.io.{FilterInputStream, InputStream}
import java.nio.file.{Files, Paths}
import java.util.Locale
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.io.Source

final class Tokenizer {
  private val Token = raw"[\p{L}\p{M}\p{N}]+|[^\p{L}\p{M}\p{N}\s]".r
  private val locale = new Locale("tr")

  def tokenize(sentence: String): Array[String] = {
    require(sentence != null, "Sentence cannot be null")
    Token.findAllIn(sentence).toArray.map(token => token.toLowerCase(locale))
  }
}

class CVADataset(val size: Int = 1, val limit:Int = 1000000, val targetVocab:Set[String] = Set()) {
  require(size > 0, "Batch size must be greater than zero")

  private final class CountingInputStream(input: InputStream) extends FilterInputStream(input) {
    var bytesRead = 0L

    override def read(): Int = {
      val value = super.read()
      if value >= 0 then bytesRead += 1
      value
    }

    override def read(buffer: Array[Byte], offset: Int, length: Int): Int = {
      val count = super.read(buffer, offset, length)
      if count > 0 then bytesRead += count
      count
    }
  }

  private val EmbeddingLine =
    """^\s*\{\s*"word"\s*:\s*("(?:\\.|[^"\\])*")\s*,\s*"vector"\s*:\s*\[(.*)\]\s*\}\s*$""".r

  private def decodeJsonString(encoded: String): String = {
    val decoded = new java.lang.StringBuilder(encoded.length - 2)
    val lastIndex = encoded.length - 1
    var index = 1

    while index < lastIndex do
      val current = encoded.charAt(index)
      if current != '\\' then
        decoded.append(current)
      else
        index += 1
        if index >= lastIndex then
          throw new IllegalArgumentException("Incomplete escape in JSON string")

        encoded.charAt(index) match
          case '"' => decoded.append('"')
          case '\\' => decoded.append('\\')
          case '/' => decoded.append('/')
          case 'b' => decoded.append('\b')
          case 'f' => decoded.append('\f')
          case 'n' => decoded.append('\n')
          case 'r' => decoded.append('\r')
          case 't' => decoded.append('\t')
          case 'u' =>
            if index + 4 >= lastIndex then
              throw new IllegalArgumentException("Incomplete Unicode escape in JSON string")
            val codeUnit = Integer.parseInt(encoded.substring(index + 1, index + 5), 16)
            decoded.append(codeUnit.toChar)
            index += 4
          case escape =>
            throw new IllegalArgumentException(s"Invalid JSON escape: \\$escape")
      index += 1

    decoded.toString
  }

  private def parseEmbedding(line: String, lineNumber: Long, device: torch.Device): Instance = {
    try
      line match
        case EmbeddingLine(encodedWord, encodedVector) =>
          val vectorText = encodedVector.trim
          val values =
            if vectorText.isEmpty then Array.emptyFloatArray
            else vectorText.split(',').map(value => java.lang.Float.parseFloat(value.trim))
          Instance(decodeJsonString(encodedWord), numeric.BFloatTensors(values.toSeq, device))
        case _ =>
          val fields = line.trim.split("\\s+")
          require(fields.length >= 2, "Expected a word followed by vector components")
          val values = fields.drop(1).map(java.lang.Float.parseFloat)
          Instance(fields.head, numeric.BFloatTensors(values.toSeq, device))
    catch
      case error: IllegalArgumentException =>
        throw new IllegalArgumentException(
          s"Invalid embedding at line $lineNumber in input file",
          error
        )
  }

  def readEmbeddingMap(
                        filename: String,
                        device: torch.Device
                      ): Map[String, torch.Tensor[torch.BFloat16]] = {
    val source = Source.fromFile(filename, "UTF-8")

    try
      source.getLines().zipWithIndex.map { case (line, index) =>
        val embedding = parseEmbedding(line, index.toLong + 1, device)
        embedding.symbol -> embedding.vector
      }.toMap
    finally
      source.close()
  }

  def averageVectors(
                      vectors: Iterable[torch.Tensor[torch.BFloat16]]
                    ): torch.Tensor[torch.BFloat16] = {
    numeric.BFloatTensors.average(vectors)
  }

  def concatVectors(
                     vectors: Iterable[torch.Tensor[torch.BFloat16]]
                   ): torch.Tensor[torch.BFloat16] = {
    val tensors = vectors.toArray

    torch.cat(tensors.map(_.flatten).toSeq)
  }


  def read(embeddingFile: String, filename: String, windowSize: Int, device: torch.Device): Iterator[Array[Instance]] = {
    val tokenizer = new Tokenizer()
    val map = readEmbeddingMap(embeddingFile, device)
    val dimensions = map.head._2.shape(0)
    Source.fromFile(filename).getLines().take(limit).sliding(size, size).flatMap(sentences => {
      sentences.map(sentence=> tokenizer.tokenize(sentence))
        .filter(tokens=> tokens.length >= windowSize && tokens.forall(token=> map.contains(token)))
        .filter(tokens=> targetVocab.isEmpty || tokens.exists(token=> targetVocab.contains(token)))
        .map(tokens=> {
          tokens.sliding(windowSize, 1)
            .map(tokens => {
              val tokenIndex = (windowSize - 1)
              val tokenModifier = tokens.take(windowSize - 1)
              (tokens(tokenIndex), tokenModifier.map(token => map(token)))
            })
            .map((token, vectors) => Instance(token, concatVectors(vectors))).toArray
        })
    })
  }

  def read(embeddingFile: String, filename: String, device: torch.Device): Iterator[Array[Instance]] = {
    val tokenizer = new Tokenizer()
    val map = readEmbeddingMap(embeddingFile, device)
    val dimensions = map.head._2.shape(0)
    map.toArray.sliding(size, size).map(tokens => tokens.map((token, embedding)=>{
      Instance(token, embedding)
    }))

  }

  def read_avg(embeddingFile: String, filename: String, windowSize: Int, device: torch.Device): Iterator[Array[Instance]] = {
    val tokenizer = new Tokenizer()
    println(s"Dataset read_avg: loading embeddings from $embeddingFile...")
    val embeddings = readEmbeddingMap(embeddingFile, device)
    println(s"Dataset read_avg: loaded ${embeddings.size} embeddings; reading $filename (limit=$limit)...")
    val totalBytes = Files.size(Paths.get(filename))
    val input = new CountingInputStream(Files.newInputStream(Paths.get(filename)))
    val source = Source.fromInputStream(input, "UTF-8")
    val batches = source.getLines().sliding(size, size)

    new Iterator[Array[Instance]] {
      private val startedAt = System.nanoTime()
      private var lastProgressAt = startedAt
      private var sentenceCount = 0L
      private var instanceCount = 0L
      private var closed = false

      private def reportProgress(complete: Boolean = false): Unit = {
        val now = System.nanoTime()
        if complete || now - lastProgressAt >= 1000000L then
          val readBytes = math.min(input.bytesRead, totalBytes)
          val percentage = if totalBytes == 0 then 100.0 else readBytes * 100.0 / totalBytes
          val elapsed = (now - startedAt) / 1e9
          val status = if complete then "complete" else "reading"
          println(
            f"Dataset read_avg: $status | sentences=$sentenceCount%,d | instances=$instanceCount%,d | " +
              f"$percentage%6.2f%% | ${readBytes / 1048576.0}%.1f / ${totalBytes / 1048576.0}%.1f MiB | " +
              f"elapsed=$elapsed%.1fs"
          )
          lastProgressAt = now
      }

      private def close(): Unit =
        if !closed then
          closed = true
          source.close()

      override def hasNext: Boolean =
        if closed then false
        else
          try
            val available = batches.hasNext && sentenceCount < limit
            if !available then
              reportProgress(complete = true)
              close()
            available
          catch
            case error: Throwable =>
              close()
              throw error

      override def next(): Array[Instance] =
        if !hasNext then Iterator.empty.next()
        else
          try
            val sentences = batches.next()
            val instances = sentences.toArray.par.map(tokenizer.tokenize)
              .filter(tokens => tokens.length >= windowSize && tokens.forall(token => embeddings.contains(token)))
              .filter(tokens => targetVocab.isEmpty || tokens.exists(token => targetVocab.contains(token)))
              .flatMap(tokens => {
                tokens.sliding(windowSize, 1).map(tokens => {
                  val tokenIndex = windowSize - 1
                  val tokenModifiers = tokens.take(windowSize - 1)
                  val token = tokens(tokenIndex)
                  val tokenEmbedding = embeddings(token)
                  val averageBias = concatVectors(Iterable(tokenEmbedding, averageVectors(tokenModifiers.map(tok => embeddings(tok)))))
                  Instance(token, averageBias)
                }).toArray
              }).toArray
            sentenceCount += sentences.size
            instanceCount += instances.length
            reportProgress()
            instances
          catch
            case error: Throwable =>
              close()
              throw error
    }
  }

  def read(filename: String, device: torch.Device): Iterator[Array[Instance]] = {
    val path = Paths.get(filename)
    val totalBytes = Files.size(path)
    val input = new CountingInputStream(Files.newInputStream(path))
    val source = Source.fromInputStream(input, "UTF-8")
    val lines = source.getLines().take(limit)
    val progressInterval = math.max(1024L, size.toLong)

    val embeddings = new Iterator[Instance] {
      private var closed = false
      private var lineNumber = 0L
      private var nextProgressAt = progressInterval
      private var lastReportedLine = -1L

      private def reportProgress(force: Boolean = false): Unit =
        if lineNumber >= nextProgressAt || (force && lineNumber != lastReportedLine) then
          val readBytes = math.min(input.bytesRead, totalBytes)
          val percentage =
            if totalBytes == 0 then 100.0
            else readBytes.toDouble * 100.0 / totalBytes.toDouble
          println(
            f"Dataset read: $lineNumber%,d | $percentage%6.2f%% | " +
              f"${readBytes / 1048576.0}%.1f / ${totalBytes / 1048576.0}%.1f MiB"
          )
          lastReportedLine = lineNumber
          nextProgressAt = (lineNumber / progressInterval + 1) * progressInterval

      private def close(): Unit =
        if !closed then
          closed = true
          source.close()

      override def hasNext: Boolean =
        if closed then false
        else
          try
            val available = lines.hasNext
            if !available then
              reportProgress(force = true)
              close()
            available
          catch
            case error: Throwable =>
              close()
              throw error

      override def next(): Instance =
        if !hasNext then Iterator.empty.next()
        else
          try
            lineNumber += 1
            val embedding = parseEmbedding(lines.next(), lineNumber, device)
            reportProgress()
            embedding
          catch
            case error: Throwable =>
              close()
              throw error
    }

    embeddings.grouped(size).map(_.toArray)
  }
}
