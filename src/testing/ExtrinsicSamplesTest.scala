package testing

import similarity.Evaluate
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import scala.util.Using

/** Run with testing.ExtrinsicSamplesTest; no embeddings or lattice need to be loaded. */
object ExtrinsicSamplesTest {
  def main(args: Array[String]): Unit = {
    val samples = Evaluate.readExtrinsic("resources/evaluations/templates.txt")
    assert(samples.length == 1000)
    assert(samples.map(_.sentence).distinct.length == 1000)
    assert(samples.map(_.target).distinct.length == 25)
    val senses = samples.groupBy(s => (s.target, s.sense))
    assert(senses.size == 50 && senses.values.forall(_.length == 20))
    val number = samples.find(_.sentence == "Kasada tam yüz lira kaldı.").get
    assert(number.target == "yüz" && number.disambiguatedWord == "100")
    assert(number.similarWords == Vector("100", "elli", "beş", "bir", "on", "doksan", "sayı"))
    val face = samples.find(s => s.target == "yüz" && s.sense == "surat").get
    assert(face.similarWords == Vector("surat", "çehre", "sima"))
    assert(samples.forall(s => !s.similarWords.contains(s.target)))
    assert(samples.forall(s => !s.sentence.exists(c => c == '[' || c == ']' || c == '|')))
    assert(samples.exists(_.similarWords.contains("kaleme almak")))
    Using.resource(Evaluate.foreachExtrinsic("resources/evaluations/templates.txt")) { iterator =>
      assert(iterator.toArray.sameElements(samples))
      assert(!iterator.hasNext)
      assert(scala.util.Try(iterator.next()).failed.get.isInstanceOf[NoSuchElementException])
    }

    val expanded = Evaluate.readExtrinsic()
    assert(expanded.length == 1000)
    val expandedNumber = expanded.find(_.sentence == "Kasada tam yüz lira kaldı.").get
    assert(expandedNumber.disambiguatedWord == "100")
    assert(expandedNumber.similarWords.take(6) == Vector("100", "elli", "beş", "bir", "on", "doksan"))
    assert(expandedNumber.similarWords.contains("nicelik"))
    val flower = expanded.find(s => s.target == "gül" && s.sense == "çiçek").get
    assert(flower.similarWords.contains("lale") && flower.similarWords.contains("menekşe"))
    assert(flower.similarWords.contains("taç yaprak") && flower.similarWords.contains("koku"))
    assert(expanded.forall(s => s.similarWords.distinct == s.similarWords && !s.similarWords.contains(s.target)))
    Using.resource(Evaluate.foreachExtrinsic(Evaluate.defaultExtrinsicFilename)) { iterator =>
      assert(iterator.toArray.sameElements(expanded))
    }

    val file = Files.createTempFile("extrinsic-templates-", ".txt")
    def read(text: String) = {
      Files.writeString(file, text, StandardCharsets.UTF_8)
      Evaluate.readExtrinsic(file.toString)
    }
    def rejects(text: String): Unit = {
      val error = try {
        read(text)
        None
      } catch {
        case e: IllegalArgumentException => Some(e)
      }
      assert(error.exists(_.getMessage.contains(file.toString)))
    }
    try {
      val header = "yüz|sayı|100|elli,bir\n"
      val rows = read("\uFEFF\n" + header + "\nTam yüz [lira|kuruş] kaldı.\n")
      assert(rows.map(_.sentence).toVector == Vector("Tam yüz lira kaldı.", "Tam yüz kuruş kaldı."))
      val related = (1 to 512).map(index => s"word$index,phrase $index").mkString("|")
      val manyWords = read(s"yüz|sayı|100|$related|100,yüz,word1\nTam yüz [lira|kuruş] kaldı.\n")
      assert(manyWords.length == 2)
      assert(manyWords.forall(_.similarWords.length == 1025))
      assert(manyWords.head.similarWords.take(3) == Vector("100", "word1", "phrase 1"))
      assert(manyWords.head.similarWords.takeRight(2) == Vector("word512", "phrase 512"))
      rejects("Tam yüz [lira|kuruş] kaldı.")
      rejects("yüz|sayı|100\n")
      rejects("yüz|sayı|100|elli||bir\n")
      rejects("yüz|sayı|100|elli|bir,\n")
      rejects(header + "Tam yüz [lira|] kaldı.")
      rejects(header + "Tam yüz [lira|kuruş kaldı.")
      rejects(header + "Tam yüz [lira|kuruş]] kaldı.")
      rejects(header + "Tam yüzlerce [lira|kuruş] kaldı.")
      rejects(header + "yüz yüz [lira|kuruş] kaldı.")

      // Neither creation nor reading an earlier sample parses a malformed later line.
      Files.writeString(file, header + "Tam yüz [lira|kuruş] kaldı.\ninvalid header\n")
      Using.resource(Evaluate.foreachExtrinsic(file.toString)) { iterator =>
        assert(iterator.hasNext && iterator.hasNext)
        assert(iterator.next().sentence == "Tam yüz lira kaldı.")
        assert(iterator.next().sentence == "Tam yüz kuruş kaldı.")
        val error = scala.util.Try(iterator.hasNext).failed.get
        assert(error.isInstanceOf[IllegalArgumentException])
        assert(error.getMessage.contains(s"$file:3"))
        assert(!iterator.hasNext)
      }

      // Validation inside a lazy alternative also closes the iterator on failure.
      Files.writeString(file, header + "Tam [yüz|el] kaldı.\n")
      Using.resource(Evaluate.foreachExtrinsic(file.toString)) { iterator =>
        assert(iterator.next().sentence == "Tam yüz kaldı.")
        assert(scala.util.Try(iterator.next()).failed.get.isInstanceOf[IllegalArgumentException])
        assert(!iterator.hasNext)
      }

      Files.writeString(file, header + "Tam yüz [lira|kuruş] kaldı.\n")
      val partial = Evaluate.foreachExtrinsic(file.toString)
      partial.next()
      partial.close()
      partial.close()
      assert(!partial.hasNext)
      assert(scala.util.Try(partial.next()).failed.get.isInstanceOf[NoSuchElementException])

      Files.writeString(file, "\n" + header)
      Using.resource(Evaluate.foreachExtrinsic(file.toString)) { iterator =>
        assert(!iterator.hasNext)
      }
      println("Extrinsic sample checks passed: legacy/expanded templates, unbounded related words, lazy iteration, validation, and early close.")
    } finally {
      Files.deleteIfExists(file)
    }
  }
}
