package testing

import cvafca.CVADataset
import llm.ConceptGloveTrainer
import numeric.BFloatTensors
import simple.{Instance, Lattice}

import java.nio.file.Files

/** Small CPU training and export round trip. */
object LatticeContrastiveModelTest {
  def main(args: Array[String]): Unit = {
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    val testDevice = if args.headOption.contains("cuda") then torch.Device.CUDA else torch.Device.CPU
    val dir = Files.createTempDirectory("lattice-contrastive-")
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
      lattice.addInstance(new Instance("third", BFloatTensors(Seq(20f, 20f))))
      lattice.save(latticePath.toString)
      Files.writeString(embeddings, "first 1 0\nsecond 0 1\nthird 20 20\n")
      Files.writeString(sentences, "first second third first second third\nthird first second third first\n")

      val trainer = new ConceptGloveTrainer(
        sentenceFilename = sentences.toString,
        latticeFilename = latticePath.toString,
        embeddingFilename = embeddings.toString,
        checkpointFilename = checkpoint.toString,
        outputLatticeFilename = outputLattice.toString,
        outputEmbeddingFilename = outputEmbeddings.toString,
        device = testDevice,
        epochs = 1,
        learningRate = 0.01,
        batchSize = 2,
        skipSamples = 0,
        maxSamples = 4,
        dimension = 2,
        candidateExtents = 3,
        contextRadius = 1,
        xMax = 2.0,
        targetWordsOverride = Some(Set("first", "second", "third"))
      )
      val pairs = trainer.model.cooccurrences(maxSamples = 4)
      assert(pairs.exists(pair => pair.target == trainer.model.index("first") &&
        pair.context == trainer.model.index("second") && pair.count == 2.0))
      val candidateRows = trainer.model.candidateNodes.indices.map(row =>
        trainer.model.candidateNodes(row).id -> row).toMap
      val parentCases = trainer.model.index.values.toVector.flatMap { word =>
        val direct = trainer.model.conceptMap.getOrElse(word, Seq.empty)
        val parents = direct.flatMap(row => trainer.model.candidateNodes(row).parents.iterator
          .flatMap(node => candidateRows.get(node.id))).toSet
        if parents.nonEmpty then Some((word, parents)) else None
      }
      assert(parentCases.nonEmpty, "Test lattice must contain a word with a parent concept")
      assert(parentCases.exists { (word, parents) =>
        trainer.model.conceptCandidates(word).exists(parents.contains)
      }, "Parent FCA concepts must be available to pair selection")
      val initialWords = trainer.model.wordEmb.detach().clone()
      val initialContexts = trainer.model.contextEmb.detach().clone()
      val initialConcepts = trainer.model.conceptEmb.detach().clone()
      assert(trainer.train() == checkpoint.toAbsolutePath)
      val wordChange = trainer.model.wordEmb - initialWords
      assert((wordChange * wordChange).sum.item.toDouble > 0.0,
        "Training must update the word embeddings")
      val contextChange = trainer.model.contextEmb - initialContexts
      assert((contextChange * contextChange).sum.item.toDouble > 0.0,
        "GloVe training must update the context embeddings")
      val conceptChange = trainer.model.conceptEmb - initialConcepts
      assert((conceptChange * conceptChange).sum.item.toDouble > 0.0,
        "Training must update the concept embeddings")
      val mapping = dir.resolve("model.bin.rows.tsv")
      assert(Files.isRegularFile(checkpoint) && Files.isRegularFile(mapping))
      assert(trainer.exportEmbeddings() == outputLattice.toAbsolutePath)
      val exported = new CVADataset().readEmbeddingMap(outputEmbeddings.toString, torch.Device.CPU)
      assert(exported.keySet == trainer.model.index.keySet)
      assert(new Lattice(2).load(outputLattice.toString).getNodes.size == lattice.getNodes.size)

      val resumed = new ConceptGloveTrainer(
        sentenceFilename = sentences.toString,
        latticeFilename = latticePath.toString,
        embeddingFilename = embeddings.toString,
        checkpointFilename = checkpoint.toString,
        outputLatticeFilename = outputLattice.toString,
        outputEmbeddingFilename = outputEmbeddings.toString,
        device = testDevice,
        epochs = 1,
        dimension = 2,
        candidateExtents = 3,
        contextRadius = 1,
        targetWordsOverride = Some(Set("first", "second", "third"))
      )
      assert(resumed.loadCheckpoint() == checkpoint.toAbsolutePath)

      val originalMapping = Files.readString(mapping)
      Files.writeString(mapping, originalMapping.replaceFirst("W\\t0\\t", "W\\t99\\t"))
      val rejected = try {
        trainer.exportEmbeddings()
        false
      } catch {
        case _: IllegalArgumentException => true
      }
      assert(rejected, "Exporter must reject a changed row mapping")
    } finally {
      List(outputEmbeddings, outputLattice, dir.resolve("model.bin.rows.tsv"),
        dir.resolve("model.bin.selector-mode"), checkpoint,
        sentences, embeddings, latticePath).foreach(Files.deleteIfExists)
      Files.deleteIfExists(dir)
    }
  }
}
