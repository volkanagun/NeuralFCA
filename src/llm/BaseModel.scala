package llm

import cvafca.CVADataset
import torch.{Float32, Tensor}
import simple.Lattice
import zemberek.core.turkish.SecondaryPos
import zemberek.morphology.TurkishMorphology
import zemberek.tokenization.{Token, TurkishTokenizer}

import java.util.Locale
import scala.jdk.CollectionConverters.*

trait BaseModel(val latticeFilename: String, val embeddingFilename: String) {
  //load lattice concept dictionary
  val dim = 300
  val candidateExtends = 30
  val lattice = new Lattice(dim).load(latticeFilename)
  val candidateNodes = lattice.getNodes.filter(node => node.size <= candidateExtends).toVector
  val conceptSize = candidateNodes.size
  val commonMap = candidateNodes.iterator.zipWithIndex.map { case (node, row) =>
    row -> node.commonVector
  }.toMap.updated(conceptSize, torch.zeros(Seq(dim), dtype = torch.bfloat16))
  
  val embeddingMap = new CVADataset().readEmbeddingMap(embeddingFilename, torch.Device.CPU)
  val index: Map[String, Int] = Seq("num", "punct", "prob").foldLeft(
    embeddingMap.keysIterator.zipWithIndex.toMap
  ) { (ids, token) =>
    if ids.contains(token) then ids else ids.updated(token, ids.size)
  }

  val conceptMap = candidateNodes.zipWithIndex
    .flatMap { case (node, row) => node.extent.map(instance => (instance.symbol, row)) }
    .groupBy(tuple => tuple._1).view.mapValues(tuples => {
      val concepts = tuples.map(_._2)
      concepts
    }).map((token, seq) => (index(token), seq))
    .toMap


  private val tokenizer = TurkishTokenizer.DEFAULT
  private lazy val morphology = TurkishMorphology.createWithDefaults()
  private val turkishLocale = Locale.forLanguageTag("tr")

  def vocabLength: Int = index.size + 1


  def wordId(token: String): Int =
    index.getOrElse(token, index.size)
  
  
  def embedding(word: String): torch.Tensor[torch.BFloat16] = {
    embeddingMap.getOrElse(word, torch.zeros(Seq(dim), dtype = torch.bfloat16))
  }

  def concepts(words: Seq[Int]): Seq[Seq[Int]] =
    words.map(word => conceptMap.getOrElse(word, Seq(conceptSize)))
  
  def tokenize(sentence: String):Seq[Int]={
    process(sentence).map(token=> wordId(token))
  }

  def process(sentence: String): Seq[String] = {
    require(sentence != null, "Sentence cannot be null")
    tokenizer.tokenize(sentence).asScala.toVector.map { token =>
      token.getType match {
        case Token.Type.Number | Token.Type.RomanNumeral | Token.Type.PercentNumeral |
             Token.Type.Time | Token.Type.Date => "num"
        case Token.Type.Punctuation => "punct"
        case _ if isProperName(token) => "prob"
        case _ => token.getText.toLowerCase(turkishLocale)
      }
    }
  }

  private def isProperName(token: Token): Boolean = {
    val wordType = token.getType match {
      case Token.Type.Word | Token.Type.WordWithSymbol | Token.Type.Abbreviation |
           Token.Type.AbbreviationWithDots | Token.Type.UnknownWord => true
      case _ => false
    }
    val text = token.getText
    // Zemberek does not label every standalone name (for example, Ali) as a proper noun.
    wordType && text.nonEmpty && (
      Character.isUpperCase(text.codePointAt(0)) ||
        ((text.contains("'") || text.contains("’")) &&
          morphology.analyze(token).iterator().asScala.exists { analysis =>
            !analysis.isUnknown && analysis.getDictionaryItem.secondaryPos == SecondaryPos.ProperNoun
          })
      )
  }

}
