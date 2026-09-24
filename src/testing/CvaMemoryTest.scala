package testing

import numeric.BFloatTensors.*
import cvafca.{CVA, CvaConfig, Instance, Lattice, Node}
import org.bytedeco.javacpp.{Pointer, PointerScope}

import scala.collection.mutable
import scala.util.Using

/** Exercise training's scalar distance path without an enclosing pointer scope. */
object CvaMemoryTest:
  def main(args: Array[String]): Unit =
    val samples = Vector(
      numeric.BFloatTensors(Seq(1f, 2f, 5f)),
      numeric.BFloatTensors(Seq(3f, 2f, 5f)),
      numeric.BFloatTensors(Seq(1f, 4f, 5f)))
    val query = numeric.BFloatTensors(Seq(1f, 2f, 6f))
    for noCVA <- Seq(false, true) do
      val cva = new CVA(3, CvaConfig(noCVA = noCVA))
      val model = cva.fit(samples)
      val expected = cva.distance(model, query)
      if !noCVA then assert(math.abs(expected - 0.5) < 1e-6)
      val lattice = new Lattice(cva)
      val node = lattice.targetNode(mutable.HashMap.empty[Set[String], Node],
        samples.zipWithIndex.map((sample, index) => Instance(s"word$index", sample)).toList)
      val incoming = Instance("query", query)
      val expectedIncoming = node.distance(query)
      val cache = mutable.HashMap.empty[Long, Double]
      val before = Pointer.totalCount()
      val bytesBefore = Pointer.totalBytes()
      for _ <- 0 until 1000 do
        assert(cva.distance(model, query) == expected)
        cva.classifies(model, query)
        cache.clear()
        assert(lattice.incomingDistance(cache, node, incoming) == expectedIncoming)
        assert(lattice.incomingDistance(cache, node, incoming) == expectedIncoming)
      val after = Pointer.totalCount()
      assert(after <= before, s"Distance pointers accumulated (noCVA=$noCVA): $before -> $after")
      assert(Pointer.totalBytes() <= bytesBefore)
      // Nested callers still release their own temporaries and leave the model alive.
      Using.resource(new PointerScope()) { _ =>
        assert(cva.distance(model, query + 0f) == expected)
      }
      assert(Pointer.totalCount() <= before)
      assert(query.floatValues.sameElements(Array(1f, 2f, 6f)))
      assert(model.commonVector.floatValues.length == 3)
      assert(model.projection.floatValues.length == 9)
      assert(samples.head.floatValues.sameElements(Array(1f, 2f, 5f)))
      println(s"CVA distance checks passed (noCVA=$noCVA): $before -> $after pointers.")
