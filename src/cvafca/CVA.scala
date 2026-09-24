package cvafca

import numeric.BFloatTensors.*
import concept.CVAUtil
import org.bytedeco.javacpp.PointerScope
import torch.{BFloat16, Tensor}

import scala.util.Using

final case class CvaConfig(
                            classificationThreshold: Double = 0.80,
                            trainingRadiusScale: Double = 0.80,
                            noCVA:Boolean = false
)

final case class CvaModel(
    commonVector: Tensor[BFloat16],
    projection: Tensor[BFloat16],
    radius: Double
)

/** Common Vector Approach backed entirely by Storch tensors. */
final class CVA(val dimension: Int, val config: CvaConfig = CvaConfig()):

  require(dimension > 0, "dimension must be positive")
  require(config.classificationThreshold >= 0.0, "classificationThreshold cannot be negative")
  require(config.trainingRadiusScale >= 0.0, "trainingRadiusScale cannot be negative")

  def fit(samples: Iterable[Tensor[BFloat16]]): CvaModel =
    if config.noCVA then fit_avg(samples)
    else fit_cva(samples)

  def fit_cva(samples: Iterable[Tensor[BFloat16]]): CvaModel =
    val tensors = samples.toArray
    require(tensors.nonEmpty, "CVA.fit requires at least one sample")
    tensors.foreach(validate)

    val (projection, commonVector) = CVAUtil.compute(tensors)
    val provisional = CvaModel(commonVector, projection, radius = 0.0)
    val maxTrainingResidual = tensors.iterator.map(distance(provisional, _)).max
    val radius = math.max(
      config.classificationThreshold,
      maxTrainingResidual * config.trainingRadiusScale
    )

    CvaModel(commonVector, projection, radius)

  def fit_avg(samples: Iterable[Tensor[BFloat16]]): CvaModel =
    val tensors = samples.toArray
    require(tensors.nonEmpty, "CVA.fit requires at least one sample")
    tensors.foreach(validate)

    val (projection, commonVector) = CVAUtil.compute_avg(tensors)
    val provisional = CvaModel(commonVector, projection, radius = 0.0)
    val maxTrainingResidual = tensors.iterator.map(distance(provisional, _)).max
    val radius = math.max(
      config.classificationThreshold,
      maxTrainingResidual * config.trainingRadiusScale
    )

    CvaModel(commonVector, projection, radius)

  /** Refit after an extent change. */
  def update(samples: Iterable[Tensor[BFloat16]]): CvaModel =
    if config.noCVA then update_avg(samples)
    else update_cva(samples)

  def update_cva(samples: Iterable[Tensor[BFloat16]]): CvaModel = fit(samples)
  def update_avg(samples: Iterable[Tensor[BFloat16]]): CvaModel = fit_avg(samples)

  /** Fit the common-vector model of the union of two extents. */
  def intersect(
      a: Iterable[Tensor[BFloat16]],
      b: Iterable[Tensor[BFloat16]]
  ): CvaModel = fit(a ++ b)

  def project(model: CvaModel, x: Tensor[BFloat16]): Tensor[BFloat16] =

    model.projection.matmul(x.flatten).reshape(x.shape *)

  def distance(model: CvaModel, value: Tensor[BFloat16]):Double=
    if config.noCVA then distance_average(model, value)
    else distance_cva(model, value)

  // Only a scalar escapes these scopes. Release projections, views and residuals
  // on every call, including training callers without their own PointerScope.
  def distance_cva(model: CvaModel, x: Tensor[BFloat16]): Double =
    Using.resource(new PointerScope()) { _ =>
      val residual = project(model, x).flatten - model.commonVector.flatten
      val euclideanDistance = math.sqrt((residual * residual).sum.scalar.toDouble)

      if euclideanDistance.isPosInfinity then 1.0
      else euclideanDistance / (1.0 + euclideanDistance)
    }

  def distance_average(model: CvaModel, x: Tensor[BFloat16]): Double =
    Using.resource(new PointerScope()) { _ =>
      validate(x)
      val residual = x.flatten - model.commonVector.flatten
      val euclideanDistance = math.sqrt((residual * residual).sum.scalar.toDouble)

      if euclideanDistance.isPosInfinity then 1.0
      else euclideanDistance / (1.0 + euclideanDistance)
    }

  def classifies(model: CvaModel, x: Tensor[BFloat16]): Boolean =
    distance(model, x) <= model.radius

  private def validate(vector: Tensor[BFloat16]): Unit =
    require(
      vector.numel == dimension,
      s"Expected vector dimension $dimension but got ${vector.numel}"
    )
