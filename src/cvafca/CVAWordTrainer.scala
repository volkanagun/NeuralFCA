package cvafca

import concept.CVALattice
import similarity.Evaluate

import java.nio.file.{Files, Paths}
import java.util.concurrent.ForkJoinPool
import java.util.concurrent.atomic.LongAdder
import org.bytedeco.pytorch.global.torch as nativeTorch

import scala.collection.mutable
import scala.collection.parallel.CollectionConverters.{ArrayIsParallelizable, ImmutableSeqIsParallelizable}
import scala.collection.parallel.ForkJoinTaskSupport
import torch.{BFloat16, Tensor}

final case class Instance(symbol: String, vector: Tensor[BFloat16])

class CVAWordTrainer(val limit: Int, val dim: Int) {
  val batchSize = 8
  val parallelSize = 4
  val cpuParallelism: Int = 48
  val device = torch.Device.CPU
  val tokenizer = new Tokenizer()
  val contextVocabulary = Evaluate.readExtrinsic()
    .flatMap(sample => tokenizer.tokenize(sample.sentence)).toSet
  val cvaDataset = new CVADataset(batchSize, limit, contextVocabulary)
  val binaryFilename = "resources/binary/"


  def binaryFilename(doContext: Boolean, windowSize: Int): String =
    if doContext then s"${binaryFilename}fca-cva-${windowSize}.bin"
    else s"${binaryFilename}fca-cva.bin"


  private def configureCpu(): Unit =
    // addIntent is a graph mutation transaction, while its tensor operations
    // use PyTorch's native CPU pool across all available processors.
    nativeTorch.set_num_threads(cpuParallelism)


  def train(embeddingFilename: String, noCVA: Boolean = false, minDistance: Double = 0.8): Lattice =
    val loads = cvaDataset.read(embeddingFilename, device)
    val config = CvaConfig(minDistance, minDistance, noCVA)
    val lattice = new Lattice(CVA(dim, config))
    if Files.exists(Paths.get(binaryFilename(false, 0))) then
      lattice.load(binaryFilename(false, 0))
    val startedAt = System.nanoTime()
    var processed = 0L


    loads.sliding(parallelSize, parallelSize).foreach(parallelBatch => {
      parallelBatch.foreach(batch => {
        batch.foreach(embedding => {
          lattice.addIntent(embedding)
        })
        //lattice.compute()
      })


      processed += parallelBatch.iterator.map(_.size.toLong).sum
      //lattice.printRandomNodeSymbols()
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1_000_000_000L
      val hours = elapsedSeconds / 3600
      val minutes = (elapsedSeconds % 3600) / 60
      val seconds = elapsedSeconds % 60
      println(
        f"Processed: $processed%,d | Time: $hours%02d:$minutes%02d:$seconds%02d | " +
          f"Nodes: ${lattice.size}%,d"
      )
    })
    Files.createDirectories(Paths.get(binaryFilename(false, 0)).getParent)
    lattice.save(binaryFilename(false, 0))

  def train(embeddingFilename: String, filename: String, windowSize: Int, noCVA: Boolean, avg: Boolean, minDistance: Double): Lattice =
    val loads = if avg then cvaDataset.read_avg(embeddingFilename, filename, windowSize, device)
    else cvaDataset.read(embeddingFilename, filename, windowSize, device)
    val ddim = if avg then dim else (windowSize - 1) * dim
    val config = CvaConfig(minDistance, minDistance, noCVA)
    val lattice = new Lattice(CVA(ddim, config))
    if Files.exists(Paths.get(binaryFilename(true, windowSize))) then
      lattice.load(binaryFilename(true, windowSize))
    val startedAt = System.nanoTime()
    var processed = 0L

    configureCpu()
    val workerPool = new ForkJoinPool(cpuParallelism)
    val taskSupport = new ForkJoinTaskSupport(workerPool)
    println(s"Training with $cpuParallelism CPU workers")
    try
      loads.sliding(parallelSize, parallelSize).foreach(parallelBatch => {
        parallelBatch.toArray.par.foreach(batch => {
          batch.foreach(embedding => {
            lattice.addIntent(embedding)
          })
        })
        //lattice.save(binaryFilename)
        processed += parallelBatch.iterator.map(_.size.toLong).sum
        //lattice.printRandomNodeSymbols()
        val elapsedSeconds = (System.nanoTime() - startedAt) / 1_000_000_000L
        val hours = elapsedSeconds / 3600
        val minutes = (elapsedSeconds % 3600) / 60
        val seconds = elapsedSeconds % 60


        println(
          f"Processed: $processed%,d | Time: $hours%02d:$minutes%02d:$seconds%02d | " +
            f"Nodes: ${lattice.size}%,d"
        )
      })
    finally workerPool.shutdown()
    Files.createDirectories(Paths.get(binaryFilename(true, windowSize)).getParent)
    lattice.save(binaryFilename(true, windowSize))
}


object CVAWordTrainer extends CVAWordTrainer(1000000, 300) {

  val folder = "resources/embeddings/"
  val embeddingFilename = folder + "vectors.txt"
  val sentenceFilename = "resources/sentences/sentences-tr.txt"
  val windowSize = 3;
  val avg = true
  val noCVA = false
  val startDistance = 0.75
  val endDistance = 0.85

  def train(crrDistance: Double): Unit = {
    train(embeddingFilename, noCVA, crrDistance)
  }


  def incremental(startDistance: Double, endDistance: Double): Unit = {
    var crrDistance = startDistance
    while (crrDistance < endDistance) {
      train(crrDistance)
      crrDistance += 0.01
    }
  }

  def main(args: Array[String]): Unit = {
    incremental(startDistance, endDistance)
  }

}
