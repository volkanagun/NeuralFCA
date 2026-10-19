package testing

import numeric.BFloatTensors.*
import cvafca.{CVA, CvaConfig, Instance, Lattice}
import org.bytedeco.javacpp.{Pointer, PointerScope}

import scala.util.{Random, Using}

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
      val node = lattice.fastAddIntent(
        samples.zipWithIndex.map((sample, index) => Instance(s"word$index", sample)))
      val incoming = Instance("query", query)
      val expectedIncoming = node.distance(query)
      val before = Pointer.totalCount()
      val bytesBefore = Pointer.totalBytes()
      for _ <- 0 until 1000 do
        assert(cva.distance(model, query) == expected)
        cva.classifies(model, query)
        assert(node.distance(incoming.vector) == expectedIncoming)
        node.classifies(incoming)
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

    // Full refits must release Gram-Schmidt temporaries before another sample
    // is inserted, even when the caller does not supply a PointerScope.
    val random = new Random(1947L)
    val refitSamples = Vector.fill(128) {
      numeric.BFloatTensors(Vector.fill(160)(random.nextFloat()))
    }
    val refitCva = new CVA(160)
    def checkRefit(count: Int): Unit =
      val before = Pointer.totalCount()
      val bytesBefore = Pointer.totalBytes()
      val model = refitCva.fit_cva(refitSamples.take(count))
      try
        assert(Pointer.totalCount() <= before + 2,
          s"Refit retained native temporaries for $count samples: $before -> ${Pointer.totalCount()}")
        assert(model.commonVector.floatValues.length == 160)
        assert(model.projection.floatValues.length == 160 * 160)
        assert(!refitCva.distance(model, refitSamples.head).isNaN)
      finally
        model.commonVector.native.close()
        model.projection.native.close()
      assert(Pointer.totalCount() <= before, "Refit handles survived model disposal")
      assert(Pointer.totalBytes() <= bytesBefore)

    for count <- Seq(1, 2, 16, 64, 100, 128) do checkRefit(count)
    Using.resource(new PointerScope()) { _ => checkRefit(128) }
    assert(refitSamples.head.floatValues.length == 160, "Refit released an input tensor")
    println("CVA refits release native temporaries, including beyond 100 samples.")
