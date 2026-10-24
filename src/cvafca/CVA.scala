package cvafca

import numeric.BFloatTensors.*
import concept.CVAUtil
import org.bytedeco.javacpp.PointerScope
import torch.{BFloat16, Device, Float32, Tensor}

import scala.util.Using

final case class CvaConfig(
                            classificationThreshold: Double = 0.80,
                            trainingRadiusScale: Double = 0.80,
                            noCVA: Boolean = false
                          )

final case class CvaModel(
                           commonVector: Tensor[BFloat16],
                           projection: Tensor[BFloat16],
                           radius: Double
                         ){

  def project(x: Tensor[BFloat16]): Tensor[BFloat16] =
    projection.matmul(x.flatten).reshape(x.shape *)

  def distance(x: Tensor[BFloat16]): Double =
    Using.resource(new PointerScope()) { _ =>
      val residual = project(x).flatten - commonVector.flatten
      val euclideanDistance = math.sqrt((residual * residual).sum.scalar.toDouble)

      if euclideanDistance.isPosInfinity then 1.0
      else euclideanDistance / (1.0 + euclideanDistance)
    }
}

/** Common Vector Approach backed entirely by Storch tensors. */
final class CVA(val dimension: Int, val config: CvaConfig = CvaConfig()):

  require(dimension > 0, "dimension must be positive")
  require(config.classificationThreshold >= 0.0, "classificationThreshold cannot be negative")
  require(config.trainingRadiusScale >= 0.0, "trainingRadiusScale cannot be negative")

  // Average and singleton models never mutate their identity projection. Keep
  // one native tensor per CVA instead of dimension² BFloat16 values per node.
  private var cachedIdentityProjection = Option.empty[Tensor[BFloat16]]

  private def identityProjection(reference: Tensor[BFloat16]): Tensor[BFloat16] = this.synchronized {
    cachedIdentityProjection.getOrElse {
      val identity = numeric.BFloatTensors.scoped {
        torch.eye(
          dimension,
          dtype = torch.float32,
          device = reference.device
        ).to(torch.bfloat16)
      }
      cachedIdentityProjection = Some(identity)
      identity
    }
  }

  def emptyModel():CvaModel =
    val ze = Array.tabulate(dimension){i=>0f}.toSeq
    val tensor = numeric.BFloatTensors(ze.toSeq, Device.CPU)
    fit_cva(Iterable(tensor))


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
    val radius = calibratedRadius(maxTrainingResidual)

    CvaModel(commonVector, projection, radius)

  def fit_avg(samples: Iterable[Tensor[BFloat16]]): CvaModel =
    val tensors = samples.toArray
    require(tensors.nonEmpty, "CVA.fit requires at least one sample")
    tensors.foreach(validate)

    val projection = identityProjection(tensors.head)
    val commonVector = numeric.BFloatTensors.average(tensors)
    val provisional = CvaModel(commonVector, projection, radius = 0.0)
    val maxTrainingResidual = tensors.iterator.map(distance(provisional, _)).max
    val radius = calibratedRadius(maxTrainingResidual)

    CvaModel(commonVector, projection, radius)

  /** Restore persisted model tensors while deriving the runtime radius from the
    * active configuration and the node's stored training samples.
    */
  def restore(
      commonVector: Tensor[BFloat16],
      projection: Tensor[BFloat16],
      samples: Iterable[Tensor[BFloat16]]
  ): CvaModel =
    val tensors = samples.toArray
    require(tensors.nonEmpty, "CVA.restore requires at least one sample")
    tensors.foreach(validate)
    validate(commonVector)
    val provisional = CvaModel(commonVector, projection, radius = 0.0)
    val maxTrainingResidual = tensors.iterator.map(distance(provisional, _)).max
    val radius = calibratedRadius(maxTrainingResidual)
    CvaModel(commonVector, projection, radius)

  /** A data-calibrated open radius includes the samples used to set its bound
    * when trainingRadiusScale is at least one. Keep the configured threshold
    * exact whenever it determines the radius.
    */
  private def calibratedRadius(maxTrainingResidual: Double): Double =
    val calibrated = maxTrainingResidual * config.trainingRadiusScale
    if calibrated > config.classificationThreshold then Math.nextUp(calibrated)
    else config.classificationThreshold

  /** Refit after an extent change. */
  def update(samples: Iterable[Tensor[BFloat16]]): CvaModel =
    if config.noCVA then update_avg(samples)
    else update_cva(samples)

  def update_cva(samples: Iterable[Tensor[BFloat16]]): CvaModel = fit(samples)

  def update_avg(samples: Iterable[Tensor[BFloat16]]): CvaModel = fit_avg(samples)

  def project(model: CvaModel, x: Tensor[BFloat16]): Tensor[BFloat16] =

    model.projection.matmul(x.flatten).reshape(x.shape *)

  def distance(model: CvaModel, value: Tensor[BFloat16]): Double =
    val score = if config.noCVA then distance_average(model, value)
    else distance_cva(model, value)
    score

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
      val residual = x.flatten - model.commonVector.flatten
      val euclideanDistance = math.sqrt((residual * residual).sum.scalar.toDouble)

      if euclideanDistance.isPosInfinity then 1.0
      else euclideanDistance / (1.0 + euclideanDistance)
    }

  def classifies(model: CvaModel, x: Tensor[BFloat16]): Boolean =
    distance(model, x) < model.radius

  private def validate(vector: Tensor[BFloat16]): Unit =
    require(
      vector.numel == dimension,
      s"Expected vector dimension $dimension but got ${vector.numel}"
    )
