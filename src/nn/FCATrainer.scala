package nn

import cvafca.{CVA, CVADataset, Lattice, Node}
import org.bytedeco.javacpp.PointerScope
import torch.{Float32, Tensor}

import java.io.{BufferedInputStream, DataInputStream, FileInputStream}
import java.nio.file.{AtomicMoveNotSupportedException, Files, Path, Paths, StandardCopyOption}
import scala.collection.mutable
import scala.util.Using
import scala.util.Random

class FCATrainer(
                  val embeddingFilename: String = "resources/embeddings/vectors.txt",
                  val hiddenDim: Int = 300,
                  val epochs: Int = 1,
                  val learningRate: Double = 1e-4,
                  val lambdaDirect: Float = 1.0f,
                  val lambdaGeometry: Float = 0.1f,
                  val maxGradNorm: Double = 1.0,
                  val batchSize: Int = 1024,
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
    val embeddings = new CVADataset().readEmbeddingMap(embeddingFilename, torch.Device.CPU)
    require(embeddings.nonEmpty, s"No embeddings found in $embeddingFilename")
    val embeddingDim = Math.toIntExact(embeddings.head._2.numel)

    val commonVectorDim = latticeDimension(filename)
    val lattice = new Lattice(new CVA(commonVectorDim)).load(filename)

    val network = new FCANetwork(
      lambdaDirect,
      lambdaGeometry,
      inputDim = embeddingDim,
      hiddenDim = hiddenDim,
      outputDim = commonVectorDim
    )
    loadModel(network).foreach(path => println(s"FCA network: resumed from $path"))
    network.to(device)
    val optimizer = new torch.optim.Adam(network.parameters, learningRate)
    val nodesById = lattice.nodes.iterator.map(node => node.id -> node).toMap

    /** Parents through great-grandparents, with each ancestor included once. */
    def trainingAncestors(child: Node): Vector[Node] =
      val ancestorIds = mutable.LinkedHashSet.empty[Long]
      val parents = child.parents.filter(p => nodesById(p).size >= 2)
      val grandParents = child.parents.map(p => nodesById(p)).flatMap(p=> p.parents).filter(p => nodesById(p).size >= 2)
      val children = child.children.filter(p => nodesById(p).size >= 2)
      val grandChildren = child.children.map(p => nodesById(p)).flatMap(p=> p.children).filter(p => nodesById(p).size >= 2)
      ancestorIds.addAll(parents)
      ancestorIds.addAll(grandParents)
      ancestorIds.addAll(children)
      ancestorIds.addAll(grandChildren)
      ancestorIds.iterator.flatMap(nodesById.get).toVector

    val candidatePairs = lattice.nodes.iterator.flatMap { child =>
      trainingAncestors(child).iterator.map(ancestor => (child, ancestor))
    }.toVector
    require(candidatePairs.nonEmpty, "The lattice has no child-ancestor pairs to train")

    val totalPairs = Math.multiplyExact(samplesPerEpoch, epochs.toLong)
    val random = new scala.util.Random(randomSeed)

    var trainedPairs = 0L
    var skippedPairs = 0L
    var nonFinitePairs = 0L
    var processedPairs = 0L
    var totalLoss = 0.0
    var lastLoss = Double.NaN
    var lastGradNorm = Double.NaN

    def reportProgress(epoch: Int): Unit =
      val percentage = if totalPairs == 0 then 100.0 else processedPairs * 100.0 / totalPairs
      val averageLoss = if trainedPairs == 0 then Double.NaN else totalLoss / trainedPairs
      println(
        f"FCA network: device=$device%s | batch size=$batchSize%d | epoch=$epoch%d/$epochs%d | " +
          f"progress=$processedPairs%,d/$totalPairs%,d ($percentage%6.2f%%) | trained=$trainedPairs%,d | " +
          f"skipped=$skippedPairs%,d | nonfinite=$nonFinitePairs%,d | loss=$lastLoss%.12f | " +
          f"gradient norm before clipping=$lastGradNorm%.6f"
      )

    network.train()
    println(
      s"FCA network: randomly sampling $samplesPerEpoch child-ancestor pairs per epoch " +
        s"from ${candidatePairs.size} candidates (seed=$randomSeed)."
    )
    try
      for epoch <- 1 to epochs do
        var epochPairs = 0L
        var epochLoss = 0.0
        var nextProgress = Math.min(
          (processedPairs / progressEvery + 1L) * progressEvery,
          totalPairs
        )
        val pending = mutable.LinkedHashMap.empty[(Int, Int), mutable.ArrayBuffer[(Node, Node)]]

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
              if lossValue.isFinite then
                loss.backward()
                val gradNorm = clipGradientNorm(network)
                if gradNorm.isFinite then
                  optimizer.step()
                  trainedPairs += samples
                  epochPairs += samples
                  epochLoss += lossValue * samples
                  totalLoss += lossValue * samples
                  lastLoss = lossValue
                  lastGradNorm = gradNorm
                else
                  nonFinitePairs += samples
                  optimizer.zeroGrad()
              else
                nonFinitePairs += samples
                optimizer.zeroGrad()

            if processedPairs >= nextProgress || processedPairs == totalPairs then
              reportProgress(epoch)
              while nextProgress <= processedPairs && nextProgress < totalPairs do
                nextProgress = Math.min(nextProgress + progressEvery, totalPairs)
          }

        var sampledPairs = 0L
        while sampledPairs < samplesPerEpoch do
          val pair = candidatePairs(random.nextInt(candidatePairs.size))
          val key = (pair._1.size, pair._2.size)
          val bucket = pending.getOrElseUpdate(key, mutable.ArrayBuffer.empty)
          bucket += pair
          sampledPairs += 1L
          if bucket.size >= batchSize then
            pending.remove(key)
            trainBatch(bucket.toSeq)
        pending.valuesIterator.foreach(bucket => trainBatch(bucket.toSeq))
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
      epochs = 3,
      progressEvery = 1000,
      modelFilename = "resources/models/fca-network.pt"
    ).train("resources/binary/fca-cva.bin")
  }
}
