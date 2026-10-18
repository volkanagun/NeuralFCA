package testing

import cvafca.{CVA, CvaConfig, Instance, Lattice, LatticeConfig}
import numeric.BFloatTensors.*
import scala.io.Source
import scala.util.Using

/** Run testing.LatticeIndexedInsertionBenchmark [embedding-file] [sample-count].
  * Reports medians after warming both engines; parsing and comparison are untimed.
  */
object LatticeIndexedInsertionBenchmark:
  def main(args: Array[String]): Unit =
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    val filename = args.headOption.getOrElse("resources/embeddings/vectors.txt")
    val count = args.lift(1).map(_.toInt).getOrElse(192)
    require(count > 0)
    val words = Using.resource(Source.fromFile(filename)) { source =>
      source.getLines().take(count).map { line =>
        val fields = line.trim.split("\\s+")
        Instance(fields.head, numeric.BFloatTensors(fields.tail.map(_.toFloat).toSeq))
      }.toVector
    }
    val cluster = (0 until 64).map { index =>
      Instance(s"cluster$index", numeric.BFloatTensors((0 until 64).map { dimension =>
        1f + (if dimension == index then 0.03125f else 0f)
      }))
    }.toVector

    for (label, samples) <- Seq("embeddings" -> words, "growing-cluster" -> cluster) do
      val dimension = samples.head.vector.numel.toInt
      def run(indexed: Boolean, inputs: Vector[Instance]): (Double, Lattice) =
        val lattice = new Lattice(new CVA(dimension, CvaConfig(0.3, 0.3)),
          LatticeConfig(fastBasisCacheNodes = if indexed then 64 else 0))
        val start = System.nanoTime()
        inputs.foreach { sample =>
          if indexed then lattice.addIntentFastIndexed(sample) else lattice.addIntentFast(sample)
        }
        ((System.nanoTime() - start) / 1e6, lattice)

      run(false, samples.take(16))
      run(true, samples.take(16))
      val timings = (0 until 3).map { round =>
        // Alternate order to reduce warmup bias.
        val results = if round % 2 == 0 then Seq(false, true).map(run(_, samples))
          else Seq(true, false).map(run(_, samples)).reverse
        val reference = results.head._2
        val indexed = results.last._2
        assert(reference.nodes.map(_.id) == indexed.nodes.map(_.id))
        reference.nodes.zip(indexed.nodes).foreach { (left, right) =>
          assert(left.symbols == right.symbols && left.parents == right.parents && left.children == right.children)
          assert(left.commonVector.floatValues.sameElements(right.commonVector.floatValues))
          assert(left.projection.floatValues.sameElements(right.projection.floatValues))
          assert(left.classificationRadius == right.classificationRadius)
        }
        (results.head._1, results.last._1)
      }
      val referenceMs = timings.map(_._1).sorted.apply(1)
      val indexedMs = timings.map(_._2).sorted.apply(1)
      println(f"$label: ${samples.size}%d samples, $dimension%d dimensions | " +
        f"fast=$referenceMs%.1f ms | indexed=$indexedMs%.1f ms | speedup=${referenceMs / indexedMs}%.2fx")
