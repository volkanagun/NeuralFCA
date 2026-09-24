package testing

import numeric.BFloatTensors.*
import cvafca.{CVA, CvaConfig, Lattice}
import similarity.{Evaluate, ExtrinsicSentenceSample}
import java.nio.file.Files

object ExtrinsicPriorsTest {
  def main(args: Array[String]): Unit = {
    val embeddings = Map(
      "kasada" -> numeric.BFloatTensors(Seq(2f, 4f)),
      "tam" -> numeric.BFloatTensors(Seq(6f, 8f)),
      "yüz" -> numeric.BFloatTensors(Seq(10f, 12f))
    )
    val evaluator = new Evaluate(new Lattice(new CVA(2, CvaConfig(noCVA = true))).nodes, embeddings, 1)
    val file = Files.createTempFile("extrinsic-priors-", ".txt")
    try {
      Files.writeString(file, "yüz|sayı|100|elli,bir\n" +
        "Kasada tam yüz [lira|kuruş] kaldı.\n" +
        "Bilinmeyen tam yüz [lira|kuruş] kaldı.\n" +
        "yüz [lira|kuruş] kaldı.\n" +
        "Bilinmeyen yüz [lira|kuruş] kaldı.\n")
      val averaged = evaluator.readExtrinsic(avg = true, filename = file.toString)
      val concatenated = evaluator.readExtrinsic(avg = false, filename = file.toString)
      assert(averaged.length == 4 && concatenated.length == 4)
      assert(averaged.take(2).forall(_._2.vector.floatValues.sameElements(Array(10f, 12f, 4f, 6f))))
      assert(concatenated.take(2).forall(_._2.vector.floatValues.sameElements(Array(2f, 4f, 6f, 8f))))
      assert(averaged.drop(2).forall(_._2.vector.floatValues.sameElements(Array(10f, 12f, 6f, 8f))))
      assert(concatenated.drop(2).forall(_._2.vector.floatValues.sameElements(Array(6f, 8f))))
      
      val sample = ExtrinsicSentenceSample("el", "organ", "uzuv", Vector("avuç"),
        "Gelen işçi el parmaklarını yıkadı.")
      assert(sample.readPrior() == "Gelen işçi ")
      println("Extrinsic prior checks passed: mean, concatenation, unknown/empty prefixes, target boundary.")
    } finally {
      Files.deleteIfExists(file)
    }
  }
}
