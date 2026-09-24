package concept

import numeric.BFloatTensors.*
import torch.{BFloat16, Float32, Tensor}

import scala.collection.mutable.ArrayBuffer

object CVAUtil {
  
  /** Returns an identity projection and the element-wise sample mean. */
  def compute_avg(tensors: Array[Tensor[BFloat16]]): (Tensor[BFloat16], Tensor[BFloat16]) = {
    val reference = tensors.head
    
    val average = numeric.BFloatTensors.average(tensors.toSeq)
    val identity = torch.eye(
      reference.numel.toInt,
      dtype = torch.float32,
      device = reference.device
    )
    (identity.to(torch.bfloat16), average)
  }
  
  def compute(
      tensors: Array[Tensor[BFloat16]]
  ): (Tensor[BFloat16], Tensor[BFloat16]) = {

    val reference = tensors.head
    val sampleShape = reference.shape

    // Every sample is x_0 plus a vector in the within-class difference
    // subspace. Removing the projection onto that subspace therefore leaves
    // the same vector for every sample.
    val flatReference = reference.flatten
    val differences = tensors.tail.map(_.flatten - flatReference)
    val identity = torch.eye(
      flatReference.numel.toInt,
      dtype = torch.float32,
      device = flatReference.device
    )

    if (differences.isEmpty) return (identity.to(dtype = torch.bfloat16), reference.clone())

    val differenceNorms = differences.map(v => math.sqrt((v * v).sum.scalar.toDouble))
    val scale = differenceNorms.max
    if (scale == 0.0) return (identity.to(dtype = torch.bfloat16), reference.clone())

    // This is the usual SVD-style numerical-rank tolerance, adapted to the
    // modified Gram-Schmidt factorisation used below.
    val rankTolerance =
      math.max(differences.length, flatReference.numel.toInt) * java.lang.Math.ulp(1.0f) * scale
    val basis = ArrayBuffer.empty[Tensor[BFloat16]]

    differences.foreach { difference =>
      // Re-orthogonalising substantially improves stability for nearly
      // linearly-dependent differences while retaining tensor/device locality.
      var residual = difference
      for (_ <- 0 until 2; direction <- basis) {
        val coefficient = (residual * direction).sum.scalar
        residual = residual - direction * coefficient
      }

      val residualNorm = math.sqrt((residual * residual).sum.scalar.toDouble)
      if (residualNorm > rankTolerance) {
        basis += residual / residualNorm.toFloat
      }
    }

    // P projects onto the orthogonal complement of the difference subspace:
    // P = I - sum(q q^T), for the orthonormal directions q in basis.
    val projectionMatrix = basis.foldLeft(identity) { (projection, direction) =>
      val direction32 = direction.to(dtype = torch.float32)
      projection.to(dtype = torch.float32) - direction32.unsqueeze(1).matmul(direction32.unsqueeze(0))
    }
    val commonVector = projectionMatrix.matmul(flatReference.to(torch.float32))
      .reshape(sampleShape *).to(torch.bfloat16)
    (projectionMatrix.to(dtype = torch.bfloat16), commonVector)
  }

  def update(
      instance: Tensor[BFloat16],
      projectionMatrix: Tensor[BFloat16],
      commonVector: Tensor[BFloat16]
  ): (Tensor[BFloat16], Tensor[BFloat16]) = numeric.BFloatTensors.scopedPair {

    // Keep all update arithmetic in Float32; only the stored result is BFloat16.
    val flatInstance = instance.flatten.to(dtype = torch.float32)
    val flatCommon = commonVector.flatten.to(dtype = torch.float32)
    val dimension = flatCommon.numel.toInt
    val projection32: Tensor[Float32] = projectionMatrix.to(dtype = torch.float32)

    // Only the part of the new member that survives the current projector can
    // reveal a variation direction not represented by the existing model.
    // Since commonVector = P*x_0, this is P*(instance - x_0) without retaining x_0.
    val projectedInstance = projection32.matmul(flatInstance)
    val residual = projectedInstance - flatCommon
    val residualNorm = math.sqrt((residual * residual).sum.item.toDouble)
    val commonNorm = math.sqrt((flatCommon * flatCommon).sum.item.toDouble)
    val projectedNorm = math.sqrt((projectedInstance * projectedInstance).sum.item.toDouble)
    val scale = math.max(residualNorm, math.max(commonNorm, projectedNorm))
    val tolerance = dimension * java.lang.Math.ulp(1.0f) * scale

    if (residualNorm <= tolerance) {
      (projectionMatrix, commonVector)
    } else {
      // Incrementally remove the newly observed unit direction q from P.
      val direction = residual / residualNorm.toFloat
      val updatedProjection =
        projection32 - direction.unsqueeze(1).matmul(direction.unsqueeze(0))
      val updatedCommon =
        (flatCommon - direction * (flatCommon * direction).sum.item).reshape(commonVector.shape *)

      (updatedProjection.to(torch.bfloat16), updatedCommon.to(torch.bfloat16))
    }
  }

  def test(
      instance: Tensor[BFloat16],
      projectionMatrix: Tensor[BFloat16],
      commonVector: Tensor[BFloat16],
      relativeTolerance: Double = 0.6d,
      absoluteTolerance: Double = 1d
  ): (Double, Boolean) = {
    val flatInstance = instance.flatten
    val flatCommon = commonVector.flatten

    // Members satisfy P*x = c. Use a scale-aware Euclidean tolerance because
    // floating-point factorisation and online rank-one updates are approximate.
    val projectedInstance = projectionMatrix.matmul(flatInstance)
    val residual = projectedInstance - flatCommon
    val residualNorm = math.sqrt((residual * residual).sum.scalar.toDouble)
    val projectedNorm = math.sqrt((projectedInstance * projectedInstance).sum.scalar.toDouble)
    val commonNorm = math.sqrt((flatCommon * flatCommon).sum.scalar.toDouble)
    val tolerance =
      absoluteTolerance + relativeTolerance * math.max(projectedNorm, commonNorm)

    (residualNorm, residualNorm <= tolerance)
  }

  /** Fuzzy subset test for common vectors.
    *
    * `left` is contained in `right` when it has no component substantially
    * greater than the corresponding component of `right`. The returned
    * decision uses the [0, 1] implication score
    * `1 - max(0, max(left - right))`.
    */
  def commonVectorSubset(
      left: Tensor[BFloat16],
      right: Tensor[BFloat16],
      threshold: Float
  ): Boolean = {
    require(left.numel == right.numel, "Common vectors must have equal sizes")
    val maximumViolation = math.max(
      0.0f,
      (left.flatten - right.flatten).max().scalar
    )
    1.0f - maximumViolation > threshold
  }
}
