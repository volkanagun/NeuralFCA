package cvafca

import java.nio.file.Files
import numeric.BFloatTensors.*

/** Train and resume with real word embeddings and an isolated checkpoint. */
object CVAWordTrainerLatticeTest:
  def main(args: Array[String]): Unit =
    val directory = Files.createTempDirectory("cva-word-trainer-")
    val embeddings = directory.resolve("vectors.txt")
    val checkpoint = directory.resolve("lattice.bin")
    val config = LatticeConfig(closureSearchVisits = 4,
      closureSearchFrontier = 2, closureSearchBeamWidth = 1, closureSearchParallel = false)
    try
      Files.writeString(embeddings, "word 0 0\nalias 0 0\nfar 8 8\nfar-alias 8 8\n")
      val trainer = new CVAWordTrainer(0, 3, 2):
        override val binaryFilename = checkpoint.toString
      val lattice = trainer.train(embeddings.toString, minDistance = 0.01, latticeConfig = config)
      assert(lattice.config == config)
      assert(lattice.nodes.map(_.symbols).toSet == Set(Set("word", "alias"), Set("far")))
      assert(Files.isRegularFile(checkpoint))
      lattice.validate()

      val resumedTrainer = new CVAWordTrainer(3, 1, 2):
        override val binaryFilename = checkpoint.toString
      val resumed = resumedTrainer.train(embeddings.toString, minDistance = 0.01, latticeConfig = config)
      assert(resumed.nodes.map(_.symbols).toSet == Set(Set("word", "alias"), Set("far", "far-alias")))
      resumed.nodes.foreach { node =>
        val expected = resumed.cva.fit(node.instances.map(_.vector))
        assert(node.commonVector.floatValues.sameElements(expected.commonVector.floatValues))
        assert(node.projection.floatValues.sameElements(expected.projection.floatValues))
        assert(node.instances.forall(node.classifies))
      }
      resumed.validate()
      println("Word trainer FastAddIntent, search configuration and checkpoint resume checks passed.")
    finally
      Files.deleteIfExists(checkpoint)
      Files.deleteIfExists(embeddings)
      Files.deleteIfExists(directory)
