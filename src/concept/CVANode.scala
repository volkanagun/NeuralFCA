package concept

import torch.{Device, BFloat16, Tensor}

import scala.util.Random

class CVANode(
    val id:Int,
    val cv: Array[Tensor[BFloat16]] = Array.empty,
    val projections: Array[Tensor[BFloat16]] = Array.empty
) extends Serializable {
  var instances: Array[Array[Tensor[BFloat16]]] = Array.empty
  var items = Array[String]()
  var parents:Array[CVANode] = Array.empty


  override def hashCode(): Int = id

  override def equals(obj: Any): Boolean = {
    obj.asInstanceOf[CVANode].id == id
  }

  def items(symbols:Array[String]):CVANode={
    this.items = symbols
    this
  }

  def randomSymbols(count: Int, random: Random = Random): Array[String] = {
    require(count >= 0, "Symbol count cannot be negative")
    random.shuffle(items.toIndexedSeq).take(count).toArray
  }

  def instances(instances: Array[Array[Tensor[BFloat16]]]):CVANode = {
    this.instances = instances
    this
  }

  def remove(parent:CVANode):CVANode={
    parents = parents.filter(current => parent != current)
    this
  }

  def membership(test: Array[Tensor[BFloat16]]): Boolean = {
    test.length == cv.length &&
    cv.zip(projections).zip(test).forall {
      case ((commonVector, projectionMatrix), instance) =>
        val (_, accept) = CVAUtil.test(instance, projectionMatrix, commonVector)
        accept
    }
  }

  def add(symbol:String, test: Array[Tensor[BFloat16]]):Boolean={
    if items.contains(symbol) then false
    else{
      instances = instances :+ test
      items = items :+ symbol
      true
    }
  }

  def compute():CVANode={
    if instances.nonEmpty then {
      for (i <- projections.indices) {
        val items = instances.map(instance=> instance(i))
        val (projection, cva) = CVAUtil.compute(items)
        cv(i) = cva
        projections(i) = projection
      }
    }
    this
  }

}

object CVANode {

  /** Creates the universal root node, which accepts every correctly shaped instance.
    *
    * For each attribute, P = 0 and c = 0, so the membership condition P*x = c
    * holds for every attribute vector x.
    */
  def empty(id:Int, size: Int, dim: Int, device: Device = Device.CUDA): CVANode = {
    val cvs = Array.fill[Tensor[BFloat16]](size) {
      torch.zeros(Seq(dim), dtype = torch.bfloat16, device = device)
    }

    val projections = Array.fill[Tensor[BFloat16]](size) {
      torch.zeros(Seq(dim, dim), dtype = torch.bfloat16, device = device)
    }

    new CVANode(id, cvs, projections)

  }


}
