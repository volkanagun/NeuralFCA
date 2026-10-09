package nn

import cvafca.{CVA, CVADataset, Lattice, Node}
import org.bytedeco.javacpp.PointerScope
import torch.{Float32, Tensor}

import java.io.{BufferedInputStream, DataInputStream, FileInputStream}
import java.nio.file.{AtomicMoveNotSupportedException, Files, Path, Paths, StandardCopyOption}
import scala.collection.mutable
import scala.util.Using
import scala.util.Random
import scala.util.control.NonFatal

class FCATrainer(
                  val embeddingFilename: String = "resources/embeddings/vectors.txt",
                  val hiddenDim: Int = 800,
                  val epochs: Int = 1,
                  val learningRate: Double = 1e-5,
                  val lambdaDirect: Float = 1.0f,
                  val lambdaGeometry: Float = 0.1f,
                  val maxGradNorm: Double = 0.1,
                  val maxGradientNormBeforeClip: Double = 1_000_000.0,
                  val maxParameterAbs: Double = 1000.0,
                  val maxResumeParameterAbs: Double = 100.0,
                  val maxLoss: Double = 1000.0,
                  val maxConsecutiveNonFiniteBatches: Int = 10,
                  val weightDecay: Double = 1e-4,
                  val outputLimit: Float = 4.0f,
                  val huberDelta: Float = 1.0f,
                  val maxTargetAbs: Double = 100.0,
                  val maxExtentSize: Int = 10960,
                  val maxPairRepeatsPerEpoch: Int = 10,
                  val batchSize: Int = 256,
                  val samplesPerEpoch: Long = 1_000_000L,
                  val randomSeed: Long = 0L,
                  val device: torch.Device = torch.Device.CUDA,
                  val progressEvery: Int = 10_000,
                  val modelFilename: String = "resources/models/fca-network.pt"
                ) {

  require(hiddenDim > 0)
  require(epochs > 0)
  require(learningRate > 0.0)
  require(maxGradNorm > 0.0)
  require(maxGradientNormBeforeClip > 0.0)
  require(maxParameterAbs > 0.0)
  require(maxResumeParameterAbs > 0.0 && maxResumeParameterAbs <= maxParameterAbs)
  require(maxLoss > 0.0)
  require(maxConsecutiveNonFiniteBatches > 0)
  require(weightDecay >= 0.0)
  require(outputLimit > 0.0f)
  require(huberDelta > 0.0f)
  require(maxTargetAbs > 0.0)
  require(maxExtentSize >= 2)
  require(maxPairRepeatsPerEpoch > 0)
  require(batchSize > 0)
  require(samplesPerEpoch > 0L)
  require(progressEvery > 0)
  require(modelFilename.nonEmpty)
  require(device.device != torch.DeviceType.CUDA || torch.cuda.isAvailable, "CUDA is not available")

  private val FileMagic = 0x43564146

  /** Reads only the checkpoint header so input and common-vector dimensions may differ. */
  private def latticeDimension(filename: String): Int =
    Using.resource(new DataInputStream(new BufferedInputStream(new FileInputStream(filename)))) { input =>
      val magic = input.readInt()
      require(magic == FileMagic, s"Not a CVA-FCA lattice file: $filename")
      val version = input.readInt()
      require(version == 1 || version == 2, s"Unsupported lattice file version: $version")
      val dimension = input.readInt()
      require(dimension > 0, s"Invalid lattice dimension: $dimension")
      dimension
    }

  private def extentVectors(
                             node: Node,
                             embeddings: Map[String, Tensor[torch.BFloat16]]
                           ): Option[Seq[Tensor[torch.BFloat16]]] =
    val vectors = node.instancesIterator.flatMap(instance => embeddings.get(instance.symbol)).toArray
    if vectors.length != node.size || vectors.isEmpty then None
    else Some(vectors.toSeq)

  /** Clips gradients without passing a potentially stale TensorVector through JNI. */
  private def clipGradientNorm(network: FCANetwork): Double =
    val gradients: Seq[Tensor[Float32]] = network.parameters
      .flatMap(_.grad)
      .map(_.asInstanceOf[Tensor[Float32]])
    if gradients.isEmpty then 0.0
    else
      val squaredNorms: Seq[Tensor[Float32]] = gradients.map(gradient => torch.square(gradient).sum)
      val norm = torch.sqrt(torch.stack(squaredNorms).sum)
      val normValue = norm.item.asInstanceOf[Number].doubleValue()
      if normValue.isFinite && normValue > maxGradNorm then
        val scale = (maxGradNorm / (normValue + 1e-6)).toFloat
        gradients.foreach(_ *= scale)
      normValue

  private def maximumParameterMagnitude(network: FCANetwork): Double =
    Using.resource(new PointerScope()) { _ =>
      network.parameters.iterator
        .map(parameter => parameter.abs.max().item.asInstanceOf[Number].doubleValue())
        .maxOption
        .getOrElse(0.0)
    }

  private def saveModel(network: FCANetwork): Path =
    val target = Paths.get(modelFilename).toAbsolutePath
    val directory = target.getParent
    Files.createDirectories(directory)
    val temporary = Files.createTempFile(directory, target.getFileName.toString + ".", ".tmp")
    try
      Using.resource(new org.bytedeco.pytorch.OutputArchive()) { archive =>
        network.save(archive)
        archive.save_to(temporary.toString)
      }
      try
        Files.move(
          temporary,
          target,
          StandardCopyOption.ATOMIC_MOVE,
          StandardCopyOption.REPLACE_EXISTING
        )
      catch
        case _: AtomicMoveNotSupportedException =>
          Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING)
      target
    finally
      Files.deleteIfExists(temporary)

  /** Loads an existing checkpoint onto CPU before the network is moved to its training device. */
  private def loadModel(network: FCANetwork): Option[Path] =
    val source = Paths.get(modelFilename).toAbsolutePath
    if Files.notExists(source) then None
    else
      require(Files.isRegularFile(source), s"Model checkpoint is not a regular file: $source")
      Using.resource(new org.bytedeco.pytorch.InputArchive()) { archive =>
        Using.resource(new org.bytedeco.pytorch.Device("cpu")) { cpuDevice =>
          Using.resource(new org.bytedeco.pytorch.DeviceOptional(cpuDevice)) { mapLocation =>
            archive.load_from(source.toString, mapLocation)
          }
        }
        network.load(archive)
      }
      Some(source)

  /**
   * Loads the FCA checkpoint and randomly samples child-ancestor training pairs.
   *
   * x/y contain the child/ancestor symbolic extents resolved through the embedding
   * dictionary. x_cva/y_cva contain the corresponding stored common vectors.
   */
  def train(filename:String): FCANetwork = {
    // Keep the complete dictionary in host memory and transfer only the current
    // batch. This avoids consuming GPU memory in proportion to vocabulary size.
    val embeddings = new CVADataset().readEmbeddingMap(embeddingFilename, device)
    require(embeddings.nonEmpty, s"No embeddings found in $embeddingFilename")
    val embeddingDim = Math.toIntExact(embeddings.head._2.numel)

    val commonVectorDim = latticeDimension(filename)
    val lattice = new Lattice(new CVA(commonVectorDim)).load(filename)

    def newNetwork(): FCANetwork =
      new FCANetwork(
        lambdaDirect,
        lambdaGeometry,
        inputDim = embeddingDim,
        hiddenDim = hiddenDim,
        outputDim = commonVectorDim,
        outputLimit = outputLimit,
        huberDelta = huberDelta
      )

    val nodesById = lattice.nodes.iterator.map(node => node.id -> node).toMap


    def getParents(child:Node, depth:Int = 0):Vector[Node]={
      if depth > 5 then return Vector()
      else {
        val parents = child.parents.map(id=> nodesById(id))
          .filter(p => p.size >= 2)
          .toVector
        val children = parents.flatMap(p => getChildren(p, depth+1))
        parents ++ parents.flatMap(p=> {getParents(p, depth+1)}) ++ children
      }
    }

    def getChildren(child:Node, depth:Int = 0):Vector[Node]={
      if depth > 5 then return Vector()
      else {
        val children = child.children.map(id=> nodesById(id))
          .filter(p => p.size >= 2)
          .toVector
        val parents = children.flatMap(p => getParents(p, depth+1))
        children ++ children.flatMap(p=> {getChildren(p, depth+1)}) ++ parents
      }
    }

    /** Parents through great-grandparents, with each ancestor included once. */
    def trainingAncestors(child: Node): Vector[Node] =

      val parents = getParents(child)
      val children = getChildren(child)
      parents++children

    def targetVectorIsSafe(node: Node): Boolean =
      val values = numeric.BFloatTensors.toArray(node.commonVector)
      values.nonEmpty && values.forall(value => value.isFinite && math.abs(value.toDouble) <= maxTargetAbs)

    def pairIsSafe(child: Node, ancestor: Node): Boolean =
      child.size <= maxExtentSize &&
        ancestor.size <= maxExtentSize &&
        targetVectorIsSafe(child) &&
        targetVectorIsSafe(ancestor)

    val rawCandidatePairs = lattice.nodes.iterator.flatMap { child =>
      trainingAncestors(child).iterator.map(ancestor => (child, ancestor))
    }.toVector
    val candidatePairs = rawCandidatePairs.filter(pair => pairIsSafe(pair._1, pair._2))
    val filteredPairCount = rawCandidatePairs.size - candidatePairs.size
    require(candidatePairs.nonEmpty, "The lattice has no child-ancestor pairs to train")
    val candidatesByShape = candidatePairs.groupBy(pair => (pair._1.size, pair._2.size))
    val validationPairs = candidatesByShape.values.maxBy(_.size).take(math.min(batchSize, 256)).toVector
    require(validationPairs.nonEmpty, "The lattice has no same-shape validation pairs")

    def lossValueFor(network: FCANetwork, pairs: Seq[(Node, Node)]): Option[Double] =
      Using.resource(new PointerScope()) { _ =>
        val resolved = pairs.flatMap { case (child, ancestor) =>
          for
            childExtent <- extentVectors(child, embeddings)
            ancestorExtent <- extentVectors(ancestor, embeddings)
          yield (child, ancestor, childExtent, ancestorExtent)
        }
        if resolved.isEmpty then None
        else
          val wasTraining = network.isTraining
          network.eval()
          try
            val x = torch.stack(resolved.map(sample => torch.stack(sample._3))).to(device, torch.float32)
            val y = torch.stack(resolved.map(sample => torch.stack(sample._4))).to(device, torch.float32)
            val childCommon = torch.stack(resolved.map(_._1.commonVector)).to(device, torch.float32)
            val ancestorCommon = torch.stack(resolved.map(_._2.commonVector)).to(device, torch.float32)
            Some(network(x, y, childCommon, ancestorCommon).item.asInstanceOf[Number].doubleValue())
          finally
            network.train(wasTraining)
      }

    val checkpointNetwork = newNetwork()
    var network =
      try
        loadModel(checkpointNetwork) match
          case Some(path) =>
            checkpointNetwork.to(device)
            val checkpointMaximum = maximumParameterMagnitude(checkpointNetwork)
            val checkpointLoss = lossValueFor(checkpointNetwork, validationPairs).getOrElse(Double.PositiveInfinity)
            if checkpointMaximum.isFinite &&
              checkpointMaximum <= maxResumeParameterAbs &&
              checkpointLoss.isFinite &&
              checkpointLoss <= maxLoss
            then
              println(
                f"FCA network: resumed from $path " +
                  f"(maximum parameter magnitude=$checkpointMaximum%.6f, validation loss=$checkpointLoss%.12f)"
              )
              checkpointNetwork
            else
              println(
                f"FCA network: ignored unstable checkpoint $path " +
                  f"(maximum parameter magnitude=$checkpointMaximum%.6f, resume limit=$maxResumeParameterAbs%.6f, " +
                  f"validation loss=$checkpointLoss%.12f, loss limit=$maxLoss%.12f)"
              )
              val fresh = newNetwork()
              fresh.to(device)
              fresh
          case None =>
            checkpointNetwork.to(device)
            checkpointNetwork
      catch
        case NonFatal(error) =>
          println(
            s"FCA network: ignored incompatible checkpoint ${Paths.get(modelFilename).toAbsolutePath}: " +
              error.getMessage
          )
          val fresh = newNetwork()
          fresh.to(device)
          fresh

    val initialLoss = lossValueFor(network, validationPairs).getOrElse(Double.PositiveInfinity)
    if !initialLoss.isFinite || initialLoss > maxLoss then
      println(
        f"FCA network: fresh initialization validation loss=$initialLoss%.12f exceeds limit=$maxLoss%.12f; " +
          "continuing because no stable checkpoint is available."
      )

    val optimizer = new torch.optim.AdamW(network.parameters, lr = learningRate, weightDecay = weightDecay)

    val epochSampleLimit = Math.multiplyExact(candidatePairs.size.toLong, maxPairRepeatsPerEpoch.toLong)
    val effectiveSamplesPerEpoch = math.min(samplesPerEpoch, epochSampleLimit)
    val totalPairs = Math.multiplyExact(effectiveSamplesPerEpoch, epochs.toLong)
    val random = new scala.util.Random(randomSeed)

    var trainedPairs = 0L
    var skippedPairs = 0L
    var nonFinitePairs = 0L
    var processedPairs = 0L
    var totalLoss = 0.0
    var lastLoss = Double.NaN
    var lastGradNorm = Double.NaN
    var consecutiveNonFiniteBatches = 0

    def abortDiverged(message: String): Nothing =
      optimizer.zeroGrad()
      throw new IllegalStateException(s"$message The unstable network will not be saved.")

    def reportProgress(epoch: Int): Unit =
      val parameterMaximum = maximumParameterMagnitude(network)
      if !parameterMaximum.isFinite || parameterMaximum > maxParameterAbs then
        abortDiverged(
          f"Training diverged: maximum parameter magnitude=$parameterMaximum%.6f " +
            f"exceeds limit=$maxParameterAbs%.6f."
        )
      val percentage = if totalPairs == 0 then 100.0 else processedPairs * 100.0 / totalPairs
      val averageLoss = if trainedPairs == 0 then Double.NaN else totalLoss / trainedPairs
      println(
        f"FCA network: device=$device%s | batch size=$batchSize%d | epoch=$epoch%d/$epochs%d | " +
          f"progress=$processedPairs%,d/$totalPairs%,d ($percentage%6.2f%%) | trained=$trainedPairs%,d | " +
          f"skipped=$skippedPairs%,d | nonfinite=$nonFinitePairs%,d | loss=$lastLoss%.12f | " +
          f"gradient norm before clipping=$lastGradNorm%.6f | max parameter=$parameterMaximum%.6f"
      )

    network.train()
    println(
      s"FCA network: randomly sampling $effectiveSamplesPerEpoch child-ancestor pairs per epoch " +
        s"from ${candidatePairs.size} candidates (filtered $filteredPairCount unsafe pairs, " +
        s"max repeats per pair=$maxPairRepeatsPerEpoch, seed=$randomSeed)."
    )
    try
      for epoch <- 1 to epochs do
        var epochPairs = 0L
        var epochLoss = 0.0
        var nextProgress = Math.min(
          (processedPairs / progressEvery + 1L) * progressEvery,
          totalPairs
        )
        def trainBatch(pairs: Seq[(Node, Node)]): Unit =
          Using.resource(new PointerScope()) { _ =>
            val resolved = pairs.flatMap { case (child, ancestor) =>
              for
                childExtent <- extentVectors(child, embeddings)
                ancestorExtent <- extentVectors(ancestor, embeddings)
              yield (child, ancestor, childExtent, ancestorExtent)
            }
            skippedPairs += pairs.size - resolved.size
            processedPairs += pairs.size

            if resolved.nonEmpty then
              val x = torch.stack(resolved.map(sample => torch.stack(sample._3))).to(device, torch.float32)
              val y = torch.stack(resolved.map(sample => torch.stack(sample._4))).to(device, torch.float32)
              val childCommon = torch.stack(resolved.map(_._1.commonVector)).to(device, torch.float32)
              val ancestorCommon = torch.stack(resolved.map(_._2.commonVector)).to(device, torch.float32)
              optimizer.zeroGrad()
              val loss = network(x, y, childCommon, ancestorCommon)
              val lossValue = loss.item.asInstanceOf[Number].doubleValue()
              val samples = resolved.size.toLong
              if lossValue.isFinite && lossValue <= maxLoss then
                loss.backward()
                val gradNorm = clipGradientNorm(network)
                if gradNorm.isFinite && gradNorm <= maxGradientNormBeforeClip then
                  optimizer.step()
                  val parameterMaximum = maximumParameterMagnitude(network)
                  if !parameterMaximum.isFinite || parameterMaximum > maxParameterAbs then
                    abortDiverged(
                      f"Training diverged after optimizer step: maximum parameter magnitude=$parameterMaximum%.6f " +
                        f"exceeds limit=$maxParameterAbs%.6f."
                    )
                  trainedPairs += samples
                  epochPairs += samples
                  epochLoss += lossValue * samples
                  totalLoss += lossValue * samples
                  lastLoss = lossValue
                  lastGradNorm = gradNorm
                  consecutiveNonFiniteBatches = 0
                else
                  abortDiverged(
                    f"Training diverged: gradient norm before clipping=$gradNorm%.6f " +
                      f"exceeds limit=$maxGradientNormBeforeClip%.6f."
                  )
              else
                nonFinitePairs += samples
                consecutiveNonFiniteBatches += 1
                optimizer.zeroGrad()
                if lossValue.isFinite then
                  abortDiverged(
                    f"Training diverged: loss=$lossValue%.12f exceeds limit=$maxLoss%.12f."
                  )
                if consecutiveNonFiniteBatches >= maxConsecutiveNonFiniteBatches then
                  abortDiverged(
                    s"Training diverged: encountered $consecutiveNonFiniteBatches consecutive non-finite batches."
                  )

            if processedPairs >= nextProgress || processedPairs == totalPairs then
              reportProgress(epoch)
              while nextProgress <= processedPairs && nextProgress < totalPairs do
                nextProgress = Math.min(nextProgress + progressEvery, totalPairs)
          }

        var sampledPairs = 0L
        while sampledPairs < effectiveSamplesPerEpoch do
          // Selecting the seed pair uniformly makes the shape bucket probability
          // proportional to its candidate count. Sampling the rest from that same
          // bucket preserves uniform pair sampling while guaranteeing stackable,
          // full batches instead of flushing thousands of undersized buckets.
          val seedPair = candidatePairs(random.nextInt(candidatePairs.size))
          val bucket = candidatesByShape((seedPair._1.size, seedPair._2.size))
          val currentBatchSize = math.min(batchSize.toLong, effectiveSamplesPerEpoch - sampledPairs).toInt
          val batch = Vector.fill(currentBatchSize)(bucket(random.nextInt(bucket.size)))
          trainBatch(batch)
          sampledPairs += currentBatchSize
        val epochAverage = if epochPairs == 0 then Double.NaN else epochLoss / epochPairs
        println(
          f"FCA network epoch complete: $epoch%d/$epochs%d | trained=$epochPairs%,d | " +
            f"average loss=$epochAverage%.6f | nonfinite=$nonFinitePairs%,d"
        )
    finally
      optimizer.zeroGrad()

    val savedModel = saveModel(network)
    println(
      s"FCA network training complete: $trainedPairs sampled child-ancestor pairs trained, " +
        s"$skippedPairs skipped because an extent symbol was absent from the embedding dictionary. " +
        s"Model saved to $savedModel"
    )
    network
  }
}

object FCATrainer {
  def main(args: Array[String]): Unit = {
    new FCATrainer(
      epochs = 10,
      progressEvery = 5000,
      modelFilename = "resources/models/fca-network.pt"
    ).train("resources/binary/fca-cva.bin")
  }
}
