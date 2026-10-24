package simple

class Main {

  /** Stream embeddings into the simple lattice and report every insertion. */
  def train(
      embeddingFilename: String,
      start: Int = 0,
      limit: Int = 1000000,
      dim: Int = 300,
      beamWidth: Int = 16,
      maxVisits: Int = 256
  ): Lattice = {
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    val dataset = new cvafca.CVADataset(size = 1, start = start, limit = limit)
    val lattice = new Lattice(dim = dim)
    val startedAt = System.nanoTime()
    var processed = 0L

    dataset.read(embeddingFilename, torch.Device.CPU).foreach { batch =>
      batch.foreach { embedding =>
        val instance = new Instance(embedding.symbol, embedding.vector)
        val concept = lattice.beamInsert(instance, beamWidth, maxVisits)
        processed += 1

        val elapsedSeconds = (System.nanoTime() - startedAt) / 1000000000L
        val hours = elapsedSeconds / 3600
        val minutes = (elapsedSeconds % 3600) / 60
        val seconds = elapsedSeconds % 60
        println(
          f"Processed: $processed%,d | Time: $hours%02d:$minutes%02d:$seconds%02d | " +
            s"Instance: ${instance.id} | Concept: ${concept.id}"
        )
      }
    }

    lattice
  }

}

object Main extends Main {
  val folder = "resources/embeddings/"
  val embeddingFilename = folder + "vectors.txt"
  def main(args: Array[String]): Unit = {
    train(embeddingFilename)
  }
}
