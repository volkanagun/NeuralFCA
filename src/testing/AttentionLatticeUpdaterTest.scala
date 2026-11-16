package testing

import cvafca.CVADataset
import llm.{AttentionLatticeUpdater, BaseModel}
import numeric.BFloatTensors
import org.bytedeco.pytorch.OutputArchive
import simple.{Instance, Lattice}

import java.nio.file.Files
import scala.util.Using

/** Run with testing.AttentionLatticeUpdaterTest. */
object AttentionLatticeUpdaterTest {
  def main(args: Array[String]): Unit = {
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    val directory = Files.createTempDirectory("attention-lattice-updater-")
    val latticePath = directory.resolve("lattice.bin")
    val embeddingPath = directory.resolve("embeddings.txt")
    val modelPath = directory.resolve("model.bin")
    val outputPath = directory.resolve("updated.bin")
    val outputEmbeddings = directory.resolve("vectors-updated.txt")
    val invalidPath = directory.resolve("invalid.bin")
    try {
      val lattice = new Lattice(dim = 2)
      lattice.addInstance(new Instance("first", BFloatTensors(Seq(1f, 0f))))
      lattice.addInstance(new Instance("second", BFloatTensors(Seq(0f, 1f))))
      lattice.addInstance(new Instance("third", BFloatTensors(Seq(20f, 20f))))
      lattice.save(latticePath.toString)
      Files.writeString(embeddingPath, "first 1 0\nsecond 0 1\nthird 20 20\n")
      val candidates = lattice.getNodes.filter(_.size <= 1)
      val skipped = lattice.getNodes.filter(_.size > 1)
      assert(candidates.nonEmpty && skipped.nonEmpty)
      assert(candidates.zipWithIndex.exists((node, row) => node.id != row),
        "Fixture must distinguish concept rows from lattice node IDs")
      val learned = candidates.indices.map(row => Seq(row.toFloat + 0.25f, row.toFloat + 0.5f))
      val fallback = Seq(99f, 99f)
      val baseModel = new BaseModel(latticePath.toString, embeddingPath.toString, 2, 1) {}
      val wordVectors = (0 until baseModel.vocabLength).map(row =>
        Seq(row.toFloat + 0.25f, row.toFloat + 0.5f))

      Using.resource(new OutputArchive()) { archive =>
        archive.write("conceptEmb", torch.Tensor[Float](learned.flatten ++ fallback).view(candidates.size + 1, 2).native)
        archive.write("wordEmb", torch.Tensor[Float](wordVectors.flatten).view(baseModel.vocabLength, 2).native)
        archive.save_to(modelPath.toString)
      }

      val result = new AttentionLatticeUpdater(modelPath.toString, latticePath.toString,
        outputPath.toString, dim = 2, candidateExtents = 1,
        embeddingFilename = embeddingPath.toString,
        outputEmbeddingFilename = outputEmbeddings.toString).update()
      assert(result == outputPath.toAbsolutePath)
      val exportedLines = Files.readAllLines(outputEmbeddings)
      assert(exportedLines.size == baseModel.index.size)
      assert(exportedLines.get(0).split("\\s+").length == 3)
      val exported = new CVADataset().readEmbeddingMap(outputEmbeddings.toString, torch.Device.CPU)
      baseModel.index.foreach { (token, row) =>
        assert(BFloatTensors.toArray(exported(token)).sameElements(
          BFloatTensors.toArray(BFloatTensors(wordVectors(row)))))
      }
      val restored = new Lattice(dim = 2).load(outputPath.toString)
      candidates.zipWithIndex.foreach { (before, row) =>
        val after = restored.getNodes.find(_.id == before.id).get
        assert(BFloatTensors.toArray(after.commonVector).sameElements(
          BFloatTensors.toArray(BFloatTensors(learned(row)))))
        assert(BFloatTensors.toArray(after.intent.projection).sameElements(
          BFloatTensors.toArray(before.intent.projection)))
        assert(after.intent.radius == before.intent.radius)
        assert(after.extent.map(_.symbol) == before.extent.map(_.symbol))
      }
      skipped.foreach { before =>
        val after = restored.getNodes.find(_.id == before.id).get
        assert(BFloatTensors.toArray(after.commonVector).sameElements(
          BFloatTensors.toArray(before.commonVector)))
      }

      val unchanged = new Lattice(dim = 2).load(latticePath.toString)
      assert(BFloatTensors.toArray(unchanged.getNodes.head.commonVector).sameElements(
        BFloatTensors.toArray(lattice.getNodes.head.commonVector)))

      Using.resource(new OutputArchive()) { archive =>
        archive.write("conceptEmb", torch.Tensor[Float](Seq(1f, 2f)).view(1, 2).native)
        archive.write("wordEmb", torch.Tensor[Float](wordVectors.flatten).view(baseModel.vocabLength, 2).native)
        archive.save_to(invalidPath.toString)
      }
      val rejectedPath = directory.resolve("rejected.bin")
      val rejected = try {
        new AttentionLatticeUpdater(invalidPath.toString, latticePath.toString,
          rejectedPath.toString, dim = 2, candidateExtents = 1,
          embeddingFilename = embeddingPath.toString,
          outputEmbeddingFilename = outputEmbeddings.toString).update()
        false
      } catch {
        case _: IllegalArgumentException => true
      }
      assert(rejected && !Files.exists(rejectedPath))
    } finally {
      Files.deleteIfExists(invalidPath)
      Files.deleteIfExists(outputPath)
      Files.deleteIfExists(outputEmbeddings)
      Files.deleteIfExists(modelPath)
      Files.deleteIfExists(latticePath)
      Files.deleteIfExists(embeddingPath)
      Files.deleteIfExists(directory)
    }
  }
}
