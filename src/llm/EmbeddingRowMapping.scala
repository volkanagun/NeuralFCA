package llm

import java.nio.charset.StandardCharsets
import java.nio.file.{AtomicMoveNotSupportedException, Files, Path, Paths, StandardCopyOption}
import java.util.Base64
import scala.io.Source
import scala.util.Using

/** Checkpoint sidecar that fixes the meanings of word and concept tensor rows. */
object EmbeddingRowMapping {
  private def sidecar(checkpoint: Path): Path = Paths.get(checkpoint.toString + ".rows.tsv")

  private def lines(model: BaseModel): Iterator[String] = {
    val encoder = Base64.getUrlEncoder.withoutPadding()
    Iterator.single("NeuralFCA-rows\t1") ++
      model.index.iterator.toVector.sortBy(_._2).iterator.map { (token, row) =>
        s"W\t$row\t${encoder.encodeToString(token.getBytes(StandardCharsets.UTF_8))}"
      } ++
      model.candidateNodes.iterator.zipWithIndex.map { (node, row) => s"C\t$row\t${node.id}" }
  }

  def save(checkpoint: Path, model: BaseModel): Path = {
    val target = sidecar(checkpoint.toAbsolutePath)
    Files.createDirectories(target.getParent)
    val temporary = Files.createTempFile(target.getParent, target.getFileName.toString + ".", ".tmp")
    try {
      Using.resource(Files.newBufferedWriter(temporary, StandardCharsets.UTF_8)) { writer =>
        lines(model).foreach { line =>
          writer.write(line)
          writer.newLine()
        }
      }
      try Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
      catch {
        case _: AtomicMoveNotSupportedException =>
          Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING)
      }
      target
    } finally Files.deleteIfExists(temporary)
  }

  def verifyIfPresent(checkpoint: Path, model: BaseModel): Unit = {
    val mapping = sidecar(checkpoint.toAbsolutePath)
    if Files.exists(mapping) then {
      Using.resource(Source.fromFile(mapping.toFile, "UTF-8")) { source =>
        val actual = source.getLines()
        lines(model).foreach { expected =>
          require(actual.hasNext && actual.next() == expected,
            s"Embedding row mapping differs from checkpoint: $mapping")
        }
        require(!actual.hasNext, s"Embedding row mapping has extra rows: $mapping")
      }
    }
  }
}
