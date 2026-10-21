package cvafca

import java.nio.file.{Files, Paths}
import org.bytedeco.pytorch.global.torch as nativeTorch
import torch.{BFloat16, Tensor}

final case class Instance(symbol: String, vector: Tensor[BFloat16])

class CVAWordTrainer(val start: Int, val limit: Int, val dim: Int) {
  val batchSize = 4
  val parallelSize = 4
  val device = torch.Device.CPU
  val cvaDataset = new CVADataset(batchSize, start, limit)
  val binaryFilename = "resources/binary/fca-cva.bin"


  def printNodes(lattice: Lattice, sz: Int, cnt: Int): Unit = {
    scala.util.Random.shuffle(lattice.nodes.filter(node => node.size > sz))
      .take(cnt).foreach(node => println(node.symbols.mkString("[", ",", "]")))


  }

  def train(embeddingFilename: String, noCVA: Boolean = false, minDistance: Double = 0.8): Lattice =
    // Lattice mutation is serialized; scoring owns the worker parallelism.
    nativeTorch.set_num_threads(1)
    val loads = cvaDataset.read(embeddingFilename, device)
    val config = CvaConfig(minDistance, minDistance, noCVA)
    val lattice = new Lattice(CVA(dim, config))
    if Files.exists(Paths.get(binaryFilename)) then
      lattice.load(binaryFilename)

    val startedAt = System.nanoTime()
    var processed = 0L
    loads.sliding(parallelSize, parallelSize).foreach { parallelBatch =>
      parallelBatch.foreach(batch => batch.foreach(instance => {
        lattice.fastAddIntent(instance)
      }))

      printNodes(lattice, 3, 2)
      processed += parallelBatch.iterator.map(_.size.toLong).sum

      val elapsedSeconds = (System.nanoTime() - startedAt) / 1_000_000_000L
      val hours = elapsedSeconds / 3600
      val minutes = (elapsedSeconds % 3600) / 60
      val seconds = elapsedSeconds % 60
      println(
        f"Processed: $processed%,d | Time: $hours%02d:$minutes%02d:$seconds%02d | " +
          f"Nodes: ${lattice.size}%,d"
      )
    }
    Files.createDirectories(Paths.get(binaryFilename).getParent)
    lattice.save(binaryFilename)
}

class CVAWordIncremental(val start: Int, val limit: Int, val dim: Int = 300) {

  val folder = "resources/embeddings/"
  val embeddingFilename = folder + "vectors.txt"
  val noCVA = false
  val startDistance = 0.52
  val endDistance = 0.63

  def train(crrDistance: Double): Unit = {
    val trainer = new CVAWordTrainer(start, limit, dim)
    trainer.train(embeddingFilename, noCVA, crrDistance)
  }

  def incremental(startDistance: Double, endDistance: Double): Unit = {
    var crrDistance = startDistance

    while (crrDistance < endDistance) {
      train(crrDistance)
      crrDistance += 0.01
    }

  }

  def train(): Unit = {
    incremental(startDistance, endDistance)
  }
}


object CVAWordMain {
  def main(args: Array[String]): Unit = {
    val max = 500000
    new CVAWordIncremental(0, max).train()

  }
}
