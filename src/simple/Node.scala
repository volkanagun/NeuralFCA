package simple

import concept.CVAUtil
import cvafca.CvaModel

import scala.collection.mutable

class Node(
            val id: Int,
            val intent: CvaModel,
            var extent: Set[Instance] = Set.empty[Instance]
          ) {

  def this(id:Int, intent:CvaModel, instance: Instance) = this(id, intent, Set(instance))

  // More general concepts (immediate upper neighbors)
  val parents: mutable.Set[Node] =
    mutable.LinkedHashSet.empty[Node]

  // More specific concepts (immediate lower neighbors)
  val children: mutable.Set[Node] =
    mutable.LinkedHashSet.empty[Node]

  override def toString: String = {

    val i = intent.commonVector.toString
    val e = extent.toSeq.mkString("{", ",", "}")

    s"C$id: extent=$e, intent=$i"
  }

  def size = extent.size
  def common = intent.commonVector

  override def equals(node: Any): Boolean = {
    val obj = node.asInstanceOf[Node]
    obj.size == size && obj.extent.forall(instance=> extent.contains(instance))
  }

  def subset_of(node: Node, ratio: Float, threshold: Float): Boolean = {
    size <= node.size && {
      val count = extent.count(instance => {
        node.classifies(instance, threshold)
      })
      (count.toFloat / size) >= ratio
    }
  }


  def intersect(node: Node):Node = {
    val symbolics = node.extent ++ (extent)
    val (projection, commonVector) = CVAUtil.compute(symbolics.map(_.attributes).toArray)
    val provisional = CvaModel(commonVector, projection, radius = 0.0)
    Node(0, provisional, symbolics)
  }
  def classifies(instance: Instance, threshold: Float): Boolean = {
    val distance = intent.distance(instance.attributes)
    distance <= threshold
  }
}