package cvafca

import numeric.BFloatTensors.*
import concept.CVAUtil
import org.bytedeco.javacpp.PointerScope
import torch.{BFloat16, Float32, Tensor}

import scala.collection.mutable.ArrayBuffer
import scala.util.Using

/** Append-only replay of CVAUtil.compute's two-pass Gram-Schmidt algorithm.
  *
  * Reuses the accepted BFloat16 directions and the unrounded Float32 projector.
  * The operation order and tolerance match a complete refit. If a larger extent
  * or difference scale invalidates a previously accepted direction, replay starts
  * at that sample and preserves the unchanged basis prefix. Sample tensors follow
  * the lattice's read-only contract.
  */
private[cvafca] final class CvaBasisCache extends AutoCloseable:
  private var previous = Array.empty[Tensor[BFloat16]]
  private val basis = ArrayBuffer.empty[Tensor[BFloat16]]
  private val acceptedNorms = ArrayBuffer.empty[Double]
  private val acceptedSamples = ArrayBuffer.empty[Int]
  private var projection32 = Option.empty[Tensor[Float32]]
  private var scale = 0.0

  override def close(): Unit =
    basis.foreach(_.native.close())
    projection32.foreach(_.native.close())
    basis.clear()
    acceptedNorms.clear()
    acceptedSamples.clear()
    projection32 = None
    previous = Array.empty
    scale = 0.0

  def compute(samples: Array[Tensor[BFloat16]]): (Tensor[BFloat16], Tensor[BFloat16]) =
    require(samples.nonEmpty)
    val prefixMatches = previous.length <= samples.length &&
      previous.indices.forall(index => previous(index) eq samples(index))
    if !prefixMatches then close()

    // Result handles belong to the caller; cache handles are detached from the
    // local scope and explicitly released on invalidation.
    val resultProjection = new org.bytedeco.pytorch.Tensor()
    val resultCommon = new org.bytedeco.pytorch.Tensor()
    var pendingProjector = Option.empty[org.bytedeco.pytorch.Tensor]
    try
      Using.resource(new PointerScope()) { scope =>
        val reference = samples.head
        val flatReference = reference.flatten
        val appendFrom = math.max(1, previous.length)
        val newDifferences = samples.drop(appendFrom)
          .map(_.flatten - flatReference)
        if newDifferences.nonEmpty then
          val newScale = newDifferences.iterator.map(v => math.sqrt((v * v).sum.scalar.toDouble)).max
          scale = math.max(scale, newScale)
        val tolerance = math.max(samples.length - 1, reference.numel.toInt) *
          java.lang.Math.ulp(1.0f) * scale

        val firstInvalid = acceptedNorms.indexWhere(norm => !(norm > tolerance))
        val replayFrom = if firstInvalid < 0 then appendFrom else acceptedSamples(firstInvalid)
        val differences =
          if firstInvalid >= 0 then
            basis.iterator.drop(firstInvalid).foreach(_.native.close())
            basis.dropRightInPlace(basis.length - firstInvalid)
            acceptedNorms.dropRightInPlace(acceptedNorms.length - firstInvalid)
            acceptedSamples.dropRightInPlace(acceptedSamples.length - firstInvalid)
            projection32.foreach(_.native.close())
            projection32 = None
            samples.drop(replayFrom).map(_.flatten - flatReference)
          else newDifferences

        val workingProjector = new org.bytedeco.pytorch.Tensor()
        workingProjector.put(projection32.getOrElse(
          torch.eye(reference.numel.toInt, dtype = torch.float32, device = reference.device)).native)
        val projector = Tensor.fromNative[Float32](workingProjector)
        def removeDirection(direction: Tensor[BFloat16]): Unit =
          val direction32 = direction.to(torch.float32)
          val updated = projector.to(torch.float32) -
            direction32.unsqueeze(1).matmul(direction32.unsqueeze(0))
          workingProjector.put(updated.native)

        // Reconstruct the valid prefix in its original Float32 operation order;
        // adding directions back to the old projector would change rounding.
        if firstInvalid >= 0 then
          basis.foreach { direction =>
            Using.resource(new PointerScope()) { _ => removeDirection(direction) }
          }
        if scale != 0.0 then
          differences.zipWithIndex.foreach { (difference, offset) =>
            val retained = new org.bytedeco.pytorch.Tensor()
            val accepted = Using.resource(new PointerScope()) { _ =>
              val (residual, norm) = CVAUtil.orthogonalResidual(difference, basis)
              if norm > tolerance then
                val direction = residual / norm.toFloat
                retained.put(direction.native)
                removeDirection(direction)
                Some(norm)
              else None
            }
            accepted.foreach { norm =>
              retained.retainReference()
              scope.detach(retained)
              basis += Tensor.fromNative[BFloat16](retained)
              acceptedNorms += norm
              acceptedSamples += replayFrom + offset
            }
          }

        val retainedProjector = new org.bytedeco.pytorch.Tensor()
        retainedProjector.put(projector.native)
        retainedProjector.retainReference()
        scope.detach(retainedProjector)
        pendingProjector = Some(retainedProjector)
        val common =
          if samples.length == 1 || scale == 0.0 then reference.clone()
          else projector.matmul(flatReference.to(torch.float32))
            .reshape(reference.shape *).to(torch.bfloat16)
        resultProjection.put(projector.to(torch.bfloat16).native)
        resultCommon.put(common.native)
        projection32.foreach(_.native.close())
        projection32 = Some(Tensor.fromNative[Float32](retainedProjector))
        pendingProjector = None
        previous = samples.clone()
      }
      (Tensor.fromNative[BFloat16](resultProjection), Tensor.fromNative[BFloat16](resultCommon))
    catch
      case error: Throwable =>
        resultProjection.close()
        resultCommon.close()
        pendingProjector.foreach(_.close())
        close()
        throw error
