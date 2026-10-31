package similarity

import java.util.concurrent.ForkJoinPool
import scala.collection.mutable
import scala.collection.parallel.CollectionConverters.VectorIsParallelizable
import scala.collection.parallel.ForkJoinTaskSupport

/** Score independent Torch matrix batches without sharing mutable ranking state. */
private[similarity] object BatchScoring:
  // A max-heap with this ordering keeps the worst retained score at its head.
  val ordering: Ordering[(String, Double)] = Ordering.by[(String, Double), (Double, String)] {
    case (symbol, score) => (-score, symbol)
  }

  private val workers = math.min(8, Runtime.getRuntime.availableProcessors())
  private lazy val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(workers))

  def offer(best: mutable.PriorityQueue[(String, Double)], entry: (String, Double), topK: Int): Unit =
    if best.size < topK then best.enqueue(entry)
    else if ordering.lt(entry, best.head) then
      best.dequeue()
      best.enqueue(entry)

  def map[A, B](batches: Vector[A], enabled: Boolean)(score: A => B): Vector[B] =
    if enabled && workers > 1 && batches.size >= 4 then
      val parallel = batches.par
      parallel.tasksupport = taskSupport
      parallel.map(score).toVector
    else batches.map(score)
