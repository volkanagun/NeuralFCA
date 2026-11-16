package llm

import numeric.BFloatTensors
import simple.{Instance, Lattice}

import java.nio.file.Files

/** Join-based concept selection and concept-conditioned GloVe regression. */
object LatticeGraveCandidateTest {
  def main(args: Array[String]): Unit = {
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    val testDevice = if args.headOption.contains("cuda") then torch.Device.CUDA else torch.Device.CPU
    val dir = Files.createTempDirectory("lattice-grave-join-")
    val latticePath = dir.resolve("lattice.bin")
    val embeddingsPath = dir.resolve("vectors.txt")
    val sentencesPath = dir.resolve("sentences.txt")
    val checkpointPath = dir.resolve("model.bin")
    try {
      def sample(name: String, x: Float) =
        new Instance(name, BFloatTensors(Seq(x, 0f)))
      val lattice = new Lattice(dim = 2)
      lattice.beamInsert(sample("left", 0f), beamWidth = 4, maxVisits = 8)
      lattice.beamInsert(sample("right", 2.2f), beamWidth = 4, maxVisits = 8)
      val middle = lattice.beamInsert(sample("middle", 1.4f),
        beamWidth = 4, maxVisits = 8, maxParents = 2)
      assert(middle.parents.size == 2)
      middle.parents.toVector.zipWithIndex.foreach { (parent, row) =>
        parent.intent = parent.intent.copy(commonVector =
          BFloatTensors(Seq(row.toFloat + 1f, 0f)))
      }
      lattice.save(latticePath.toString)
      Files.writeString(embeddingsPath, "left 0 0\nright 2.2 0\nmiddle 1.4 0\n")
      Files.writeString(sentencesPath, "middle middle left\nmiddle middle right\nleftover right\n")
      def makeModel(conditioned: Boolean) = new ConceptGloveModel(sentencesPath.toString,
        Set("middle", "left", "right"), latticePath.toString, embeddingsPath.toString,
        device = testDevice, dimension = 2, candidateExtents = 3,
        maxPositiveConcepts = 4, contextConditioned = conditioned)
      val model = makeModel(true)
      assert(model.candidateDimension == 4)
      val m = model.index("middle")
      val l = model.index("left")
      val r = model.index("right")
      val candidates = model.conceptCandidates(m)
      assert(candidates.size >= 2)
      val join = new FcaJoinSelector(model.candidateNodes, model.lattice.getNodes,
        model.lattice.getInstances.size)
      val expectedSelf = math.log((model.lattice.getInstances.size.toDouble + 1.0) /
        (model.candidateNodes(candidates.head).size.toDouble + 1.0))
      assert(math.abs(join.informationContent(candidates.head, candidates.head) -
        expectedSelf) < 1e-9)
      assert(join.informationContent(candidates.head, candidates.head) >=
        join.informationContent(candidates.head, candidates(1)))
      val leftChoice = model.selectConceptPair(m, m, Seq(l), Seq(l))
      val rightChoice = model.selectConceptPair(m, m, Seq(r), Seq(r))
      assert(leftChoice != rightChoice,
        "Surrounding words must be able to change the selected concepts")
      val pairOnly = makeModel(false)
      assert(pairOnly.selectConceptPair(m, m, Seq(l), Seq(l)) ==
        pairOnly.selectConceptPair(m, m, Seq(r), Seq(r)))

      val selected = model.selectedCooccurrences(maxSamples = 6)
      val focused = new ConceptGloveModel(sentencesPath.toString, Set("left"),
        latticePath.toString, embeddingsPath.toString, device = testDevice,
        dimension = 2, candidateExtents = 3, maxPositiveConcepts = 4)
      val focusedPairs = focused.selectedCooccurrences(maxSamples = 6)
      assert(focusedPairs.nonEmpty && focusedPairs.forall(pair =>
        pair.target != r && pair.context != r),
        "Focused training must use exact token matches in selected co-occurrences")
      val original = model.cooccurrences(maxSamples = 6)
      val selectedTotals = selected.groupBy(sample => (sample.target, sample.context))
        .view.mapValues(_.map(_.count).sum).toMap
      assert(original.forall(sample =>
        math.abs(selectedTotals((sample.target, sample.context)) - sample.count) < 1e-9))
      val middlePairs = selected.filter(sample => sample.target == m && sample.context == m)
      assert(middlePairs.map(sample => (sample.targetConcept, sample.contextConcept)).distinct.size >= 2)
      val (targets, contexts) = model.candidateVectors(selected)
      assert(targets.shape.toSeq == Seq(selected.size, 4))
      assert(contexts.shape.toSeq == Seq(selected.size, 4))
      val loss = model.loss(selected)
      assert(loss.item.toDouble.isFinite)
      loss.backward()
      assert(model.conceptEmb.grad.exists(_.abs.sum.item.toDouble > 0.0))

      val fallback = ConceptCooccurrenceSample(m, model.conceptSize,
        model.wordId("missing"), model.conceptSize, 1.0)
      val (paddedTarget, paddedContext) = model.candidateVectors(Seq(fallback))
      assert(torch.narrow(paddedTarget, 1, 2, 2).abs.sum.item.toDouble == 0.0)
      assert(torch.narrow(paddedContext, 1, 2, 2).abs.sum.item.toDouble == 0.0)

      Files.writeString(sentencesPath, "middle middle left\nmiddle middle right\n" * 40)
      val sequential = model.selectedCooccurrences(maxSamples = 240, workers = 1)
      val parallel = model.selectedCooccurrences(maxSamples = 240, workers = 3)
      assert(sequential.map(sample =>
        (sample.target, sample.targetConcept, sample.context, sample.contextConcept)) ==
        parallel.map(sample =>
          (sample.target, sample.targetConcept, sample.context, sample.contextConcept)))
      assert(sequential.zip(parallel).forall { (left, right) =>
        math.abs(left.count - right.count) < 1e-9
      })

      val trainer = new ConceptGloveTrainer(sentenceFilename = sentencesPath.toString,
        latticeFilename = latticePath.toString, embeddingFilename = embeddingsPath.toString,
        checkpointFilename = checkpointPath.toString, device = testDevice, epochs = 1,
        dimension = 2,
        candidateExtents = 3, maxPositiveConcepts = 4, batchSize = 2,
        maxSamples = 6, targetWordsOverride = Some(Set("middle", "left", "right")))
      assert(trainer.train() == checkpointPath.toAbsolutePath)
      val otherMode = new ConceptGloveTrainer(sentenceFilename = sentencesPath.toString,
        latticeFilename = latticePath.toString, embeddingFilename = embeddingsPath.toString,
        checkpointFilename = checkpointPath.toString, device = testDevice, epochs = 1,
        dimension = 2, candidateExtents = 3,
        targetWordsOverride = Some(Set("middle", "left", "right")),
        contextConditioned = false)
      val rejected = try {
        otherMode.loadCheckpoint()
        false
      } catch {
        case _: IllegalArgumentException => true
      }
      assert(rejected, "Checkpoints must not mix selector modes")
    } finally {
      List(dir.resolve("model.bin.rows.tsv"), dir.resolve("model.bin.selector-mode"), checkpointPath,
        sentencesPath, embeddingsPath, latticePath).foreach(Files.deleteIfExists)
      Files.deleteIfExists(dir)
    }
  }
}
