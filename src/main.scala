
import torch.DType.bfloat16
import torch.Device
import torch.DeviceType

@main
def main(): Unit = {
  val data = Seq(0f, 1f, 2f, 3f)
  // data: Seq[Float] = List(0, 1, 2, 3)
  val t1 = numeric.BFloatTensors(data)
  // t1: Tensor[BFloat16] = dtype=bfloat16, shape=[4], device=CPU
  // [0, 1, 2, 3]
  t1.equal(torch.arange(0, 4, dtype = bfloat16))
  // res0: Boolean = true
  val t2 = t1.to(dtype = bfloat16)
  // t2: Tensor[BFloat16] = dtype=bfloat16, shape=[4], device=CPU
  // [0,0000, 1,0000, 2,0000, 3,0000]
  val t3 = t1 + t2
  // t3: Tensor[BFloat16] = dtype=bfloat16, shape=[4], device=CPU
  // [0,0000, 2,0000, 4,0000, 6,0000]

  val shape = Seq(2, 3)
  // shape: Seq[Long] = List(2, 3)
  val randTensor = torch.rand(shape, dtype = bfloat16)
  // randTensor: Tensor[BFloat16] = dtype=bfloat16, shape=[2, 3], device=CPU
  // [[0,4341, 0,9738, 0,9305],
  //  [0,8987, 0,1122, 0,3912]]
  val zerosTensor = torch.zeros(shape, dtype = bfloat16)
  // zerosTensor: Tensor[BFloat16] = dtype=bfloat16, shape=[2, 3], device=CPU
  // [[0, 0, 0],
  //  [0, 0, 0]]

  val x = torch.ones(Seq(5), dtype = bfloat16)
  // x: Tensor[BFloat16] = dtype=bfloat16, shape=[5], device=CPU
  // [1,0000, 1,0000, 1,0000, 1,0000, 1,0000]
  val w = torch.randn(Seq(5, 3), dtype = bfloat16, requiresGrad = true)
  // w: Tensor[BFloat16] = dtype=bfloat16, shape=[5, 3], device=CPU
  // [[0,8975, 0,5484, 0,2307],
  //  [0,2689, 0,7430, 0,6446],
  //  [0,9503, 0,6342, 0,7523],
  //  [0,5332, 0,7497, 0,3665],
  //  [0,3376, 0,6040, 0,5033]]
  val b = torch.randn(Seq(3), dtype = bfloat16, requiresGrad = true)
  // b: Tensor[BFloat16] = dtype=bfloat16, shape=[3], device=CPU
  // [0,2638, 0,9697, 0,3664]
  val z = (x matmul w) + b
}
