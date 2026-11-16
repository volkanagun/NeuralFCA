package llm

import simple.Node

import java.util.concurrent.ConcurrentHashMap
import scala.collection.mutable

/** Structural information content of the most specific shared superconcept.
  * This join-based rule adapts Resnik's shared-ancestor idea to FCA extent order;
  * it is not a claim about the model in Resnik (1995).
  * https://www.ijcai.org/Proceedings/95-1/Papers/059.pdf
  */
final class FcaJoinSelector(nodes: Vector[Node], allNodes: Seq[Node], objectCount: Int) {
  require(objectCount >= 0)
  private val ancestorCache = new ConcurrentHashMap[Int, Set[Int]]()
  private val scoreCache = new ConcurrentHashMap[Long, java.lang.Double]()
  private val sizes = allNodes.iterator.map(node => node.id -> node.size).toMap
  private val cacheLimit = 100000

  private def ancestors(row: Int): Set[Int] = {
    val cached = ancestorCache.get(row)
    if cached != null then return cached
    val found = mutable.HashSet.empty[Int]
    val stack = mutable.ArrayBuffer(nodes(row))
    while stack.nonEmpty do {
      val node = stack.remove(stack.size - 1)
      if found.add(node.id) then stack ++= node.parents
    }
    val result = found.toSet
    if ancestorCache.size < cacheLimit then ancestorCache.putIfAbsent(row, result)
    result
  }

  /** Returns zero if the graph has no shared ancestor or either row is padding. */
  def informationContent(left: Int, right: Int): Double = {
    if left < 0 || right < 0 || left >= nodes.size || right >= nodes.size then return 0.0
    val low = math.min(left, right)
    val high = math.max(left, right)
    val key = (low.toLong << 32) | (high.toLong & 0xffffffffL)
    val cached = scoreCache.get(key)
    if cached != null then return cached.doubleValue()
    val common = ancestors(left).intersect(ancestors(right))
    val score = if common.isEmpty then 0.0 else {
      val joinSize = common.iterator.map(sizes).min
      math.log((objectCount.toDouble + 1.0) / (joinSize.toDouble + 1.0))
    }
    if scoreCache.size < cacheLimit then scoreCache.putIfAbsent(key, score)
    score
  }
}
