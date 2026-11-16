package testing

import cvafca.CVADataset
import llm.SentenceConceptTrainer
import numeric.BFloatTensors
import simple.{Instance, Lattice}

import java.nio.file.Files

/** Small CPU sentence training, checkpoint, and export round trip. */
object SentenceConceptTrainerTest {
  def main(args: Array[String]): Unit = {
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    val dir = Files.createTempDirectory("sentence-concept-")
    val latticePath = dir.resolve("lattice.bin")
    val embeddings = dir.resolve("vectors.txt")
    val sentences = dir.resolve("sentences.txt")
    val checkpoint = dir.resolve("model.bin")
    val outputLattice = dir.resolve("updated-lattice.bin")
    val outputEmbeddings = dir.resolve("updated-vectors.txt")
    try {
      val lattice = new Lattice(2)
      lattice.addInstance(new Instance("first", BFloatTensors(Seq(1f, 0f))))
      lattice.addInstance(new Instance("second", BFloatTensors(Seq(0f, 1f))))
      lattice.addInstance(new Instance("third", BFloatTensors(Seq(0.7f, 0.7f))))
      lattice.save(latticePath.toString)
      Files.writeString(embeddings, "first 1 0\nsecond 0 1\nthird 0.7 0.7\n")
      Files.writeString(sentences,
        "first second third first second third\nthird first second third first second\n")

      def makeTrainer() = new SentenceConceptTrainer(
        sentenceFilename = sentences.toString,
        latticeFilename = latticePath.toString,
        embeddingFilename = embeddings.toString,
        checkpointFilename = checkpoint.toString,
        outputLatticeFilename = outputLattice.toString,
        outputEmbeddingFilename = outputEmbeddings.toString,
        device = torch.Device.CPU,
        epochs = 1,
        learningRate = 0.01,
        batchSize = 2,
        maxSamples = 12,
        dimension = 2,
        candidateExtents = 3,
        contextRadius = 1,
        wordNegatives = 3,
        conceptNegatives = 3)
      val trainer = makeTrainer()
      val initialWords = trainer.model.wordEmb.detach().clone()
      val initialConcepts = trainer.model.conceptEmb.detach().clone()
      val initialAttention = trainer.model.Wq.detach().clone()
      assert(trainer.train() == checkpoint.toAbsolutePath)
      assert((trainer.model.wordEmb - initialWords).abs.sum.item.toDouble > 0,
        "Masked prediction must update word embeddings")
      assert((trainer.model.conceptEmb - initialConcepts).abs.sum.item.toDouble > 0,
        "Concept prediction must update concept embeddings")
      assert((trainer.model.Wq - initialAttention).abs.sum.item.toDouble > 0,
        "Masked prediction must train the sentence encoder")
      assert(trainer.exportEmbeddings() == outputLattice.toAbsolutePath)
      val exported = new CVADataset().readEmbeddingMap(outputEmbeddings.toString, torch.Device.CPU)
      assert(exported.keySet == trainer.model.index.keySet)
      assert(new Lattice(2).load(outputLattice.toString).getNodes.size == lattice.getNodes.size)
      assert(makeTrainer().loadCheckpoint() == checkpoint.toAbsolutePath)
    } finally {
      List(outputEmbeddings, outputLattice, dir.resolve("model.bin.rows.tsv"),
        dir.resolve("model.bin.sentence-config"), checkpoint, sentences, embeddings,
        latticePath).foreach(Files.deleteIfExists)
      Files.deleteIfExists(dir)
    }
  }
}
