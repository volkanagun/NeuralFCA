package testing

import concept.CVAUtil
import cvafca.CVADataset
import numeric.BFloatTensors
import numeric.BFloatTensors.*
import org.bytedeco.javacpp.{Pointer, PointerScope}

import java.nio.file.Files
import scala.util.Using

object BFloat16Test:
  def main(args: Array[String]): Unit =
    Using.resource(new PointerScope()) { _ =>
      val values = BFloatTensors(Seq(0.1f, -0.0f, Float.PositiveInfinity, Float.NaN))
      assert(values.dtype == torch.bfloat16 && values.native.element_size() == 2)
      val decoded = values.floatValues
      assert(decoded(0) == 0.10009765625f)
      assert(java.lang.Float.floatToRawIntBits(decoded(1)) == 0x80000000)
      assert(decoded(2).isPosInfinity && decoded(3).isNaN)
      val matrix = BFloatTensors(Seq(1f, 2f, 3f, 4f)).view(2, 2).transpose(0, 1)
      assert(matrix.floatValues.sameElements(Array(1f, 3f, 2f, 4f)))

      val ones = BFloatTensors(Seq(1f, 1f))
      val average = BFloatTensors.average(Vector.fill(1024)(ones))
      assert(average.dtype == torch.bfloat16)
      assert(average.floatValues.sameElements(Array(1f, 1f)))

      val projection = BFloatTensors(Seq(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)).view(3, 3)
      val common = BFloatTensors(Seq(1f, 1f, 1f))
      val instance = BFloatTensors(Seq(1f, 2f, 3f))
      val before = Pointer.totalCount()
      Using.resource(new PointerScope()) { _ =>
        val (updatedProjection, updatedCommon) = CVAUtil.update(instance, projection, common)
        assert(Pointer.totalCount() - before <= 2, "Float32 update temporaries survived")
        assert(updatedProjection.dtype == torch.bfloat16 && updatedCommon.dtype == torch.bfloat16)
        // Analytically: q = (0, 1, 2)/sqrt(5), P = I - q*q^T, c = (1, .4, -.2).
        // These are the exact BFloat16 roundings of that Float32 result.
        assert(updatedProjection.floatValues.sameElements(Array(
          1f, 0f, 0f, 0f, 0.80078125f, -0.400390625f, 0f, -0.400390625f, 0.2001953125f)))
        assert(updatedCommon.floatValues.sameElements(Array(1f, 0.400390625f, -0.2001953125f)))
      }
      assert(Pointer.totalCount() <= before)
      val (unchangedProjection, unchangedCommon) = CVAUtil.update(common, projection, common)
      assert(unchangedProjection.floatValues.sameElements(projection.floatValues))
      assert(unchangedCommon.floatValues.sameElements(Array(1f, 1f, 1f)))
      assert(projection.floatValues.sameElements(Array(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)))

      val file = Files.createTempFile("bfloat-embeddings-", ".txt")
      try
        Files.writeString(file, "word 0.1 2.0\n")
        val embeddings = new CVADataset().readEmbeddingMap(file.toString, torch.Device.CPU)
        assert(embeddings("word").dtype == torch.bfloat16)
        assert(embeddings("word").floatValues.sameElements(Array(0.10009765625f, 2f)))
      finally Files.deleteIfExists(file)
    }
    println("BFloat16 checks passed: storage, conversions, averaging, Float32 update, native cleanup, embeddings.")
