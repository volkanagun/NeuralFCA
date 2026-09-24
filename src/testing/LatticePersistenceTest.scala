package testing

import numeric.BFloatTensors.*
import cvafca.{CVA, CvaConfig, Lattice}
import java.io.{DataOutputStream, FileOutputStream}
import java.nio.file.Files
import scala.util.Using

/** Run with testing.LatticePersistenceTest to check both persistence versions. */
object LatticePersistenceTest:
  def main(args: Array[String]): Unit =
    val source = Files.createTempFile("lattice-model-", ".bin")
    val saved = Files.createTempFile("lattice-roundtrip-", ".bin")
    val cva = new CVA(2, CvaConfig(noCVA = true))

    def fixture(version: Int, firstValue: Float = 7f): Unit =
      Using.resource(new DataOutputStream(new FileOutputStream(source.toFile))) { out =>
        def tensor(shape: Seq[Int], values: Seq[Float]): Unit =
          out.writeInt(shape.size)
          shape.foreach(out.writeInt)
          out.writeInt(values.size)
          values.foreach(out.writeFloat)
        out.writeInt(0x43564146)
        out.writeInt(version)
        out.writeInt(2)
        out.writeLong(1L)
        out.writeInt(1)
        out.writeLong(0L)
        if version == 2 then
          tensor(Seq(2), Seq(firstValue, 8f))
          tensor(Seq(2, 2), Seq(0.25f, 0f, 0f, 0.5f))
          out.writeDouble(0.125)
        out.writeInt(1)
        out.writeUTF("sample")
        tensor(Seq(2), Seq(1f, 2f))
        out.writeInt(0)
        out.writeInt(0)
      }

    try
      fixture(2)
      val original = new Lattice(cva).load(source.toString)
      original.save(saved.toString)
      val restored = new Lattice(cva).load(saved.toString)
      val node = restored.node(0L)
      assert(node.commonVector.floatValues.sameElements(Array(7f, 8f)))
      assert(node.projection.shape.sameElements(Array(2, 2)))
      assert(node.projection.floatValues.sameElements(Array(0.25f, 0f, 0f, 0.5f)))
      assert(node.classificationRadius == 0.125)
      assert(node.instances.head.vector.floatValues.sameElements(Array(1f, 2f)))
      assert(Files.readAllBytes(source).sameElements(Files.readAllBytes(saved)))
      fixture(2, 0.1f)
      val quantized = new Lattice(cva).load(source.toString)
      assert(quantized.node(0L).commonVector.dtype == torch.bfloat16)
      assert(quantized.node(0L).commonVector.floatValues.head == 0.10009765625f)
      quantized.save(saved.toString)
      assert(new Lattice(cva).load(saved.toString).node(0L).commonVector.floatValues.head == 0.10009765625f)
      fixture(1)
      val legacy = new Lattice(cva).load(source.toString)
      assert(legacy.node(0L).commonVector.floatValues.sameElements(Array(1f, 2f)))
      new Lattice(cva).save(saved.toString)
      assert(new Lattice(cva).load(saved.toString).size == 0)
      println("Lattice persistence checks passed")
    finally
      Files.deleteIfExists(source)
      Files.deleteIfExists(saved)
