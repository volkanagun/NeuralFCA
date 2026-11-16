package llm

import numeric.BFloatTensors
import org.bytedeco.javacpp.{FloatPointer, PointerScope}
import org.bytedeco.pytorch.{DeviceOptional, InputArchive}
import torch.{Device as TorchDevice, Float32, Tensor}

import java.nio.charset.StandardCharsets
import java.nio.file.{AtomicMoveNotSupportedException, Files, Path, Paths, StandardCopyOption}
import scala.util.Using

/** Copies the learned concept embeddings into the matching simple-lattice common vectors.
  *
  * The lattice must be the one used to train the attention model: BaseModel assigns
  * rows to candidateNodes in lattice order (at most 30 extent members by default).
  * The final conceptEmb row is the fallback concept and has no lattice node.
  */
final class AttentionLatticeUpdater(
    val modelFilename: String,
    val latticeFilename: String,
    val outputFilename: String,
    val dim: Int = 300,
    val candidateExtents: Int = 30,
    val embeddingFilename: String = "resources/embeddings/vectors.txt",
    val outputEmbeddingFilename: String = "resources/embeddings/vectors-updated.txt",
    val combineContextEmbeddings: Boolean = false
) {
  require(dim > 0, "Vector dimension must be positive")
  require(candidateExtents >= 0, "Candidate extent limit must be nonnegative")

  def update(): Path = {
    val modelPath = Paths.get(modelFilename).toAbsolutePath
    val latticePath = Paths.get(latticeFilename).toAbsolutePath
    val embeddingPath = Paths.get(embeddingFilename).toAbsolutePath
    require(Files.isRegularFile(modelPath), s"Attention model checkpoint not found: $modelPath")
    require(Files.isRegularFile(latticePath), s"Simple lattice checkpoint not found: $latticePath")
    require(Files.isRegularFile(embeddingPath), s"Embedding file not found: $embeddingPath")

    Using.resource(new PointerScope()) { _ =>
      val (conceptEmb, wordEmb, contextEmb) = Using.resource(new InputArchive()) { archive =>
        Using.resource(new org.bytedeco.pytorch.Device("cpu")) { cpu =>
          Using.resource(new DeviceOptional(cpu)) { location =>
            archive.load_from(modelPath.toString, location)
          }
        }
        def readParameter(name: String): Tensor[Float32] = {
          val native = new org.bytedeco.pytorch.Tensor()
          archive.read(name, native)
          Tensor.fromNative[Float32](native)
        }
        (readParameter("conceptEmb"), readParameter("wordEmb"),
          if combineContextEmbeddings then Some(readParameter("contextEmb")) else None)
      }

      require(conceptEmb.dtype == torch.float32, "conceptEmb must contain Float32 vectors")
      require(conceptEmb.shape.size == 2 && conceptEmb.shape(1) == dim,
        s"conceptEmb must have $dim columns; found ${conceptEmb.shape}")

      val baseModel = new BaseModel(latticePath.toString, embeddingPath.toString,
        dim, candidateExtents) {}
      EmbeddingRowMapping.verifyIfPresent(modelPath, baseModel)
      val candidates = baseModel.candidateNodes
      require(conceptEmb.shape.toSeq == Seq(candidates.size + 1, dim),
        s"conceptEmb shape ${conceptEmb.shape.toSeq} does not match " +
          s"${candidates.size} lattice concepts plus one fallback row of width $dim")
      require(wordEmb.dtype == torch.float32, "wordEmb must contain Float32 vectors")
      require(wordEmb.shape.toSeq == Seq(baseModel.vocabLength, dim),
        s"wordEmb shape ${wordEmb.shape.toSeq} does not match " +
          s"${baseModel.index.size} indexed tokens plus one unknown row of width $dim")
      contextEmb.foreach { context =>
        require(context.dtype == torch.float32 && context.shape.toSeq == wordEmb.shape.toSeq,
          "contextEmb must match the Float32 wordEmb table")
      }

      val values = conceptEmb.to(TorchDevice.CPU).contiguous
      val pointer = new FloatPointer(values.native.data_ptr())
      candidates.zipWithIndex.foreach { (node, row) =>
        val components = new Array[Float](dim)
        pointer.position(row.toLong * dim).get(components)
        require(components.forall(_.isFinite), s"Non-finite concept vector at row $row")
        val common = BFloatTensors(components.toIndexedSeq)
        require(BFloatTensors.toArray(common).forall(_.isFinite),
          s"Concept vector at row $row overflows BFloat16")
        node.intent = node.intent.copy(commonVector = common)
      }

      val output = Paths.get(outputFilename).toAbsolutePath
      baseModel.lattice.save(output.toString)
      val exportedWords = contextEmb.map(context => (wordEmb + context) * 0.5f).getOrElse(wordEmb)
      exportEmbeddings(exportedWords, baseModel.index, Paths.get(outputEmbeddingFilename).toAbsolutePath)
      output
    }
  }

  private def exportEmbeddings(wordEmb: Tensor[Float32], index: Map[String, Int], output: Path): Unit = {
    Files.createDirectories(output.getParent)
    val temporary = Files.createTempFile(output.getParent, output.getFileName.toString + ".", ".tmp")
    try {
      val values = wordEmb.to(TorchDevice.CPU).contiguous
      val pointer = new FloatPointer(values.native.data_ptr())
      Using.resource(Files.newBufferedWriter(temporary, StandardCharsets.UTF_8)) { writer =>
        index.toVector.sortBy(_._2).foreach { (token, row) =>
          val components = new Array[Float](dim)
          pointer.position(row.toLong * dim).get(components)
          require(components.forall(_.isFinite), s"Non-finite word vector for '$token' at row $row")
          require(token.nonEmpty && !token.exists(_.isWhitespace),
            s"Token '$token' cannot be written in vectors.txt format")
          writer.write(token)
          writer.write(" ")
          writer.write(components.mkString(" "))
          writer.newLine()
        }
      }
      try Files.move(temporary, output, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
      catch {
        case _: AtomicMoveNotSupportedException =>
          Files.move(temporary, output, StandardCopyOption.REPLACE_EXISTING)
      }
    } finally Files.deleteIfExists(temporary)
  }

}

object AttentionLatticeUpdater {
  def main(args: Array[String]): Unit = {
    val attentionModel = "resources/models/attention-model-causal-stable.bin"
    val latticeFilename = "resources/binary/simple-lattice.bin"
    val newLatticeFilename = "resources/binary/simple-updated-lattice.bin"
    val updater = new AttentionLatticeUpdater(attentionModel, latticeFilename, newLatticeFilename)
    val output = updater.update()
    println(s"Updated simple lattice saved to $output")
    println(s"Updated word embeddings saved to ${Paths.get(updater.outputEmbeddingFilename).toAbsolutePath}")
  }
}
