package testing

import numeric.BFloatTensors.*
import cvafca.{CVA, Instance, Lattice}
import org.bytedeco.javacpp.{Pointer, PointerScope}

import java.io.{BufferedInputStream, BufferedOutputStream, DataInputStream, DataOutputStream, FileInputStream}
import java.nio.file.Files
import scala.util.Using

/** With a checkpoint argument, load it read-only; otherwise check sharing and native lifetimes. */
object LatticeLoadMemoryTest:
  def main(args: Array[String]): Unit =
    if args.nonEmpty then
      val (dimension, count) = Using.resource(new DataInputStream(new BufferedInputStream(new FileInputStream(args(0))))) { in =>
        require(in.readInt() == 0x43564146)
        require(in.readInt() == 2)
        val dimension = in.readInt()
        in.readLong()
        (dimension, in.readInt())
      }
      val lattice = new Lattice(new CVA(dimension)).load(args(0))
      assert(lattice.size == count)
      lattice.nodes.foreach { node =>
        assert(node.commonVector.dtype == torch.bfloat16)
        assert(node.projection.dtype == torch.bfloat16)
        assert(node.projection.native.element_size() == 2)
        assert(node.instances.forall(_.vector.dtype == torch.bfloat16))
      }
      lattice.nodes.take(10).foreach { node =>
        val distance = node.distance(node.instances.head.vector)
        assert(!distance.isNaN && distance >= 0.0 && distance <= 1.0)
      }
      println(f"Loaded $count%,d nodes; resident memory=${Pointer.physicalBytes() / 1048576.0}%.1f MiB")
    else
      val source = Files.createTempFile("lattice-shared-projection-", ".bin")
      val saved = Files.createTempFile("lattice-shared-roundtrip-", ".bin")
      val truncated = Files.createTempFile("lattice-truncated-", ".bin")
      val dimension = 32
      val count = 128
      try
        Using.resource(new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(source)))) { out =>
          def tensor(shape: Seq[Int], values: Array[Float]): Unit =
            out.writeInt(shape.size)
            shape.foreach(out.writeInt)
            out.writeInt(values.length)
            values.foreach(out.writeFloat)
          out.writeInt(0x43564146)
          out.writeInt(2)
          out.writeInt(dimension)
          out.writeLong(count.toLong)
          out.writeInt(count)
          for index <- 0 until count do
            out.writeLong(index.toLong)
            tensor(Seq(dimension), Array.fill(dimension)(index.toFloat))
            val projection = Array.tabulate(dimension * dimension)(i => if i / dimension == i % dimension then 1f else 0f)
            // Neither a nearly-identity matrix nor signed zero may be silently changed.
            if index == count - 2 then projection(1) = 0.00006103515625f
            if index == count - 1 then projection(1) = -0.0f
            tensor(Seq(dimension, dimension), projection)
            out.writeDouble(0.125)
            out.writeInt(1)
            out.writeUTF(s"word$index")
            tensor(Seq(dimension), Array.fill(dimension)(index.toFloat))
            out.writeInt(0)
            out.writeInt(0)
        }

        // Warm up native libraries, then ensure only final tensor handles survive loading.
        Using.resource(new PointerScope()) { _ => new Lattice(new CVA(dimension)).load(source.toString) }
        val before = Pointer.totalCount()
        Using.resource(new PointerScope()) { _ =>
          val lattice = new Lattice(new CVA(dimension)).load(source.toString)
          val retained = Pointer.totalCount() - before
          val expectedHandles = count * 2 + 3 // common + instance per node; three projections
          assert(retained <= expectedHandles, s"Loader retained temporary pointers: $retained > $expectedHandles")
          assert(lattice.node(0).projection.asInstanceOf[AnyRef] eq lattice.node(1).projection)
          assert(!(lattice.node(0).projection.asInstanceOf[AnyRef] eq lattice.node(count - 2).projection))
          assert(!(lattice.node(0).projection.asInstanceOf[AnyRef] eq lattice.node(count - 1).projection))
          assert(lattice.nodes.forall(_.classificationRadius == 0.125))
          assert(lattice.node(1).distance(lattice.node(1).instances.head.vector) == 0.0)
          lattice.save(saved.toString)
          assert(Files.readAllBytes(source).sameElements(Files.readAllBytes(saved)))

          val untouched = lattice.node(1).projection.floatValues
          lattice.node(0).addInstances(List(Instance("extra", numeric.BFloatTensors(Array.fill(dimension)(2f).toSeq))))
          assert(lattice.node(1).projection.floatValues.sameElements(untouched))
          assert(lattice.node(1).distance(lattice.node(1).instances.head.vector) == 0.0)
        }
        assert(Pointer.totalCount() <= before, "Loaded tensors escaped the caller's pointer scope")

        // EOF inside a node must stop recovery without publishing that partial node.
        val completeBytes = Files.readAllBytes(source)
        Files.write(truncated, completeBytes.dropRight(4))
        val recovered = new Lattice(new CVA(dimension)).load(truncated.toString)
        assert(recovered.size == count - 1)
        assert(recovered.nodes.forall(_.id < count - 1))

        // Without an outer scope, returned tensors must remain alive after construction cleanup.
        val independent = new Lattice(new CVA(dimension)).load(source.toString)
        assert(independent.node(2).commonVector.floatValues.forall(_ == 2f))
        assert(independent.node(2).distance(independent.node(2).instances.head.vector) == 0.0)
        println("Lattice loading checks passed: shared identity, exact round-trip, refit isolation, and native lifetimes.")
      finally
        Files.deleteIfExists(source)
        Files.deleteIfExists(saved)
        Files.deleteIfExists(truncated)
