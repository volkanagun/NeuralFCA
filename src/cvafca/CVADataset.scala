package cvafca


import java.io.{FilterInputStream, InputStream}
import java.nio.file.{Files, Paths}
import java.util.Locale
import scala.io.Source

final class Tokenizer {
  private val Token = raw"[\p{L}\p{M}\p{N}]+|[^\p{L}\p{M}\p{N}\s]".r
  private val locale = new Locale("tr")

  def tokenize(sentence: String): Array[String] = {
    require(sentence != null, "Sentence cannot be null")
    Token.findAllIn(sentence).toArray.map(token => token.toLowerCase(locale))
  }
}

class CVADataset(val size: Int = 1, val start: Int = 0, val limit: Int = 1000000) {
  require(size > 0, "Batch size must be greater than zero")
  require(start >= 0, "Start must not be negative")
  require(limit >= 0, "Limit must not be negative")

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
      source.getLines().drop(start).take(limit).zipWithIndex
        .map { case (line, index) =>
        val embedding = parseEmbedding(line, start.toLong + index + 1, device)
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


  def read(filename: String, device: torch.Device): Iterator[Array[Instance]] = {
    val path = Paths.get(filename)
    val totalBytes = Files.size(path)
    val input = new CountingInputStream(Files.newInputStream(path))
    val source = Source.fromInputStream(input, "UTF-8")
    val lines = source.getLines().drop(start).take(limit)
    val progressInterval = math.max(1024L, size.toLong)

    val embeddings = new Iterator[Instance] {
      private var closed = false
      private var lineNumber = start.toLong
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
