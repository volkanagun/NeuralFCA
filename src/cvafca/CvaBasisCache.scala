package cvafca

import numeric.BFloatTensors.*
import org.bytedeco.javacpp.PointerScope
import torch.{BFloat16, Float32, Tensor}

import scala.collection.mutable.ArrayBuffer
import scala.util.Using

/** Append-only replay of CVAUtil.compute's two-pass Gram-Schmidt algorithm.
  *
  * Reuses the accepted BFloat16 directions and the unrounded Float32 projector.
  * The operation order and tolerance match a complete refit. If a larger extent
  * or difference scale invalidates a previously accepted direction, replay starts
  * from the beginning. Sample tensors follow the lattice's read-only contract.
  */
private[cvafca] final class CvaBasisCache extends AutoCloseable:
  private var previous = Array.empty[Tensor[BFloat16]]
  private val basis = ArrayBuffer.empty[Tensor[BFloat16]]
  private val acceptedNorms = ArrayBuffer.empty[Double]
  private var projection32 = Option.empty[Tensor[Float32]]
  private var scale = 0.0

  override def close(): Unit =
    basis.foreach(_.native.close())
    projection32.foreach(_.native.close())
    basis.clear()
    acceptedNorms.clear()
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
        val newDifferences = samples.drop(math.max(1, previous.length))
          .map(_.flatten - flatReference)
        if newDifferences.nonEmpty then
          val newScale = newDifferences.iterator.map(v => math.sqrt((v * v).sum.scalar.toDouble)).max
          scale = math.max(scale, newScale)
        val tolerance = math.max(samples.length - 1, reference.numel.toInt) *
          java.lang.Math.ulp(1.0f) * scale

        val replay = acceptedNorms.exists(norm => !(norm > tolerance))
        val differences =
          if replay then
            close()
            val all = samples.tail.map(_.flatten - flatReference)
            scale = if all.isEmpty then 0.0
              else all.iterator.map(v => math.sqrt((v * v).sum.scalar.toDouble)).max
            all
          else newDifferences

        var projector = projection32.getOrElse(
          torch.eye(reference.numel.toInt, dtype = torch.float32, device = reference.device))
        if scale != 0.0 then
          differences.foreach { difference =>
            var residual = difference
            for _ <- 0 until 2; direction <- basis do
              val coefficient = (residual * direction).sum.scalar
              residual = residual - direction * coefficient
            val norm = math.sqrt((residual * residual).sum.scalar.toDouble)
            if norm > tolerance then
              val direction = residual / norm.toFloat
              val retained = new org.bytedeco.pytorch.Tensor()
              retained.put(direction.native)
              retained.retainReference()
              scope.detach(retained)
              basis += Tensor.fromNative[BFloat16](retained)
              acceptedNorms += norm
              val direction32 = direction.to(torch.float32)
              projector = projector.to(torch.float32) -
                direction32.unsqueeze(1).matmul(direction32.unsqueeze(0))
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
