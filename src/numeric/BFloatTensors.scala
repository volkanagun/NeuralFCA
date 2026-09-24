package numeric

import org.bytedeco.javacpp.{PointerScope, ShortPointer}
import org.bytedeco.pytorch.ScalarTypeOptional
import org.bytedeco.pytorch.global.torch as nativeTorch
import torch.{BFloat16, Device, Tensor}

import scala.util.Using

/** Construct BFloat16 model tensors and release temporary conversion storage. */
object BFloatTensors:
  def apply(values: Seq[Float], device: Device = Device.CPU): Tensor[BFloat16] =
    scoped {
      torch.Tensor[Float](values).to(dtype = torch.bfloat16).to(device)
    }

  /** Keep the result in the caller's scope and release construction intermediates. */
  def scoped(operation: => Tensor[BFloat16]): Tensor[BFloat16] =
    val retained = new org.bytedeco.pytorch.Tensor()
    try
      Using.resource(new PointerScope()) { _ =>
        retained.put(operation.native)
      }
      Tensor.fromNative[BFloat16](retained)
    catch
      case error: Throwable =>
        retained.close()
        throw error

  /** Release temporary tensors while preserving both returned model tensors. */
  def scopedPair(operation: => (Tensor[BFloat16], Tensor[BFloat16])): (Tensor[BFloat16], Tensor[BFloat16]) =
    val first = new org.bytedeco.pytorch.Tensor()
    val second = new org.bytedeco.pytorch.Tensor()
    try
      Using.resource(new PointerScope()) { _ =>
        val (a, b) = operation
        first.put(a.native)
        second.put(b.native)
      }
      (Tensor.fromNative[BFloat16](first), Tensor.fromNative[BFloat16](second))
    catch
      case error: Throwable =>
        first.close()
        second.close()
        throw error

  def average(vectors: Iterable[Tensor[BFloat16]]): Tensor[BFloat16] =
    val tensors = vectors.toSeq
    require(tensors.nonEmpty, "Cannot average an empty collection")
    // Native mean accumulates before rounding the result. Repeated BFloat16
    // additions would stop changing after 256 equal positive values.
    scoped {
      Tensor.fromNative[BFloat16](torch.stack(tensors).native.mean(
        Array(0L), false, new ScalarTypeOptional(nativeTorch.ScalarType.BFloat16)
      ))
    }

  extension (tensor: Tensor[BFloat16])
    // Storch's Scala scalar match type omits BFloat16; the native conversion supports it.
    def scalar: Float = tensor.native.item_float()
    def floatValues: Array[Float] = toArray(tensor)

  /** Decode the 16-bit payload directly, without allocating a full-precision tensor. */
  def toArray(tensor: Tensor[BFloat16]): Array[Float] =
    Using.resource(new PointerScope()) { _ =>
      val cpu = tensor.to(Device.CPU).contiguous
      val bits = new Array[Short](Math.toIntExact(cpu.numel))
      if bits.nonEmpty then new ShortPointer(cpu.native.data_ptr()).get(bits)
      bits.map(value => java.lang.Float.intBitsToFloat((value & 0xffff) << 16))
    }
