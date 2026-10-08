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
 * parents are more general nodes (larger symbolic extents).
 * children are more specific nodes (smaller symbolic extents).
 */
final class Node private[cvafca] (
    val id: Long,
    initialInstances: IterableOnce[Instance],
    private val cva: CVA,
    private val do_avg:Boolean = true,
    restoredModel: Option[CvaModel] = None
):

  private val extent = mutable.LinkedHashMap.empty[String, Instance]
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
  def symbols: Set[String] = extent.keySet.toSet

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

  /** Add a new extent item and incrementally refresh common vector + projection. */
  def addInstance(instance: Instance): Boolean =
    if extent.contains(instance.symbol) then false
    else
      addWithoutRefit(instance)
      true

  /** Adds one instance and updates the model without rebuilding the full extent. */
  def addInstanceFast(instance: Instance): Boolean =
    if extent.contains(instance.symbol) then false
    else
      addWithoutRefit(instance)
      currentModel = cva.update(currentModel, instance.vector, extent.size)
      true

  /** Merge another extent into this node, then perform one CVA rebuild. */
  def addInstances(newInstances: Iterable[Instance]): Boolean =
    var changed = false
    newInstances.foreach { instance =>
      if !extent.contains(instance.symbol) then
        addWithoutRefit(instance)
        changed = true
    }
    if changed then refit()
    changed

  private[cvafca] def model: CvaModel = currentModel

  private def addWithoutRefit(instance: Instance): Unit =
    extent.put(instance.symbol, instance)

  def refit(): Unit =
    currentModel = cva.update(extent.valuesIterator.map(_.vector).toVector)

  override def toString: String =
    val cv = numeric.BFloatTensors.toArray(commonVector).map(v => f"$v%.4f").mkString("[", ", ", "]")
    s"Node($id, extent=${symbols.mkString("{", ",", "}")}, common=$cv)"
