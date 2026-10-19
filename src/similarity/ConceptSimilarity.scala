package similarity

import cvafca.Node
import org.bytedeco.javacpp.PointerScope

import scala.collection.mutable
import scala.util.Using

class ConceptSimilarity(val nodes: Vector[Node]) {
  /** Return the closest topK node extents, with input order breaking distance ties. */
  def projectAdd(target:String, x: torch.Tensor[torch.BFloat16], topK: Int): Array[Set[String]] = {
    if topK <= 0 then return Array.empty[Set[String]]
    // PriorityQueue is a max-heap: the farthest retained node is replaced first.
    val ordering = Ordering.by[(Node, Double, Int), (Double, Int)](entry => (entry._2, entry._3))
    val best = mutable.PriorityQueue.empty[(Node, Double, Int)](using ordering)
    nodes.iterator.zipWithIndex.foreach { case (node, index) =>
      val distance = Using.resource(new PointerScope()) { _ => node.distance(x) }
      val entry = (node, distance, index)
      if node.size == 1 && node.symbols.contains(target) then
      {
        //no single target accepted
      }
      else if best.size < topK then best.enqueue(entry)
      else if ordering.lt(entry, best.head) then
        best.dequeue()
        best.enqueue(entry)
    }
    best.dequeueAll.reverseIterator.map(_._1.symbols).toArray
  }

}
