package cvafca

import scala.collection.mutable
import torch.{BFloat16, Tensor}

/**
 * A CVA-FCA node has the four requested components:
 *
 *   intent          : commonVector
 *   vector extent   : instances.values.map(_.vector)
 *   CVA projection  : projection
 *   symbolic extent : symbols / instance keys
 *
 * Parents classify each sample of their children's extents.
 * Children are more specific classifiers under the learned CVA relation.
 */
final class Node private[cvafca] (
    val id: Long,
    initialInstances: IterableOnce[Instance],
    private val cva: CVA,
    private val do_avg:Boolean = false,
    restoredModel: Option[CvaModel] = None
):

  private val extent = mutable.LinkedHashMap.empty[String, Instance]
  private var extentSymbols = Set.empty[String]
  private[cvafca] var extentChanged = Option.empty[() => Unit]
  initialInstances.iterator.foreach(addWithoutRefit)
  require(extent.nonEmpty, "A node must contain at least one instance")

  private var currentModel: CvaModel = restoredModel.getOrElse {
    if do_avg then cva.fit_avg(extent.valuesIterator.map(_.vector).toVector)
    else cva.fit(extent.valuesIterator.map(_.vector).toVector)
  }

  val parents: mutable.LinkedHashSet[Long] = mutable.LinkedHashSet.empty
  val children: mutable.LinkedHashSet[Long] = mutable.LinkedHashSet.empty

  /** Intent of this node. */
  def commonVector: Tensor[BFloat16] = currentModel.commonVector

  /** Read-only CVA projection; loaded identity matrices may be shared across nodes.
   * Refit replaces the model. Clone this tensor before modifying it externally.
   */
  def projection: Tensor[BFloat16] = currentModel.projection

  def classificationRadius: Double = currentModel.radius

  /** Vector extent. */
  def instances: Vector[Instance] = extent.values.toVector

  /** Allocation-free traversal for read-only training code. */
  def instancesIterator: Iterator[Instance] = extent.valuesIterator

  /** Symbolic extent. */
  def symbols: Set[String] = extentSymbols

  def containsSymbol(symbol: String): Boolean = extent.contains(symbol)

  def size: Int = extent.size

  /**
   * FCA-style subset/membership operation in the proposed model:
   * this node's common-vector model classifies the incoming instance.
   */
  def classifies(instance: Instance): Boolean = classifies(instance.vector)

  def classifies(vector: Tensor[BFloat16]): Boolean =
    cva.classifies(currentModel, vector)

  def distance(vector: Tensor[BFloat16]): Double =
    cva.distance(currentModel, vector)

  /** Append a new stored sample and refresh its common vector and projection. */
  def addInstance(instance: Instance): Boolean =
    addInstances(Vector(instance))

  /** Merge another extent into this node, then perform one CVA rebuild. */
  def addInstances(newInstances: Iterable[Instance]): Boolean =
    var changed = false
    newInstances.foreach { instance =>
      if !extent.contains(instance.symbol) then
        addWithoutRefit(instance)
        changed = true
    }
    if changed then
      extentChanged.foreach(_.apply())
      refit()
    changed

  /** Append a complete intersection extent without changing the model. */
  private[cvafca] def addInstancesWithoutRefit(newInstances: IterableOnce[Instance]): Boolean =
    var changed = false
    newInstances.iterator.foreach { instance =>
      if !extent.contains(instance.symbol) then
        extent.put(instance.symbol, instance)
        extentSymbols += instance.symbol
        changed = true
    }
    if changed then extentChanged.foreach(_.apply())
    changed

  private[cvafca] def model: CvaModel = currentModel

  private def addWithoutRefit(instance: Instance): Unit =
    extent.put(instance.symbol, instance)
    extentSymbols += instance.symbol

  /** Refresh the complete extent through the configured CVA.update path. */
  def refit(): Unit =
    currentModel = cva.update(extent.valuesIterator.map(_.vector).toVector)

  override def toString: String =
    val cv = numeric.BFloatTensors.toArray(commonVector).map(v => f"$v%.4f").mkString("[", ", ", "]")
    s"Node($id, extent=${symbols.mkString("{", ",", "}")}, common=$cv)"
