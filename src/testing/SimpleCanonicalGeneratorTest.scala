package simple

/** Run with simple.SimpleCanonicalGeneratorTest. */
object SimpleCanonicalGeneratorTest:
  private def sample(id: String, x: Float): Instance =
    new Instance(id, numeric.BFloatTensors(Seq(x, 0f)))

  def main(args: Array[String]): Unit =
    org.bytedeco.pytorch.global.torch.set_num_threads(1)
    val lattice = new Lattice(dim = 2)
    val cva = new cvafca.CVA(2)

    def node(id: Int, x: Float): Node =
      val member = sample(s"parent-$id", x)
      new Node(id, cva.fit_cva(Iterable(member.attributes)), member)

    val query = node(100, 0f)
    val generator = node(101, 40f)
    val decoys = (0 until 10).map(i => node(i + 1, 20f + i))
    val firstMatch = node(11, 0.1f)
    val laterMatch = node(12, 0.2f)
    generator.parents ++= decoys
    generator.parents += firstMatch
    generator.parents += laterMatch

    assert(lattice.findCanonicalGenerator(query, generator) eq firstMatch)
    assert(lattice.findCanonicalGeneratorParallel(query, generator) eq firstMatch)

    val ancestor = node(13, 0f)
    firstMatch.parents += node(14, 30f)
    firstMatch.parents += ancestor
    assert(lattice.findCanonicalGenerator(query, generator) eq ancestor)
    assert(lattice.findCanonicalGeneratorParallel(query, generator) eq ancestor)

    val narrow = node(15, 40f)
    narrow.parents += decoys.head
    narrow.parents += laterMatch
    assert(lattice.findCanonicalGeneratorParallel(query, narrow) eq laterMatch)

    val unmatched = node(16, 40f)
    unmatched.parents ++= decoys
    assert(lattice.findCanonicalGeneratorParallel(query, unmatched) eq unmatched)
    println("Parallel canonical generator search passed.")
