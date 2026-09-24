package cvafca

@main def runCvaFcaDemo(): Unit =
  val cva = new CVA(
    dimension = 3,
    config = CvaConfig(
      classificationThreshold = 0.12
    )
  )

  val lattice = new Lattice(
    cva,
    LatticeConfig(
      equalityCoverage = 1.0,
      requireMutualCommonClassification = true
    )
  )

  val data = Vector(
    Instance("come-home",   numeric.BFloatTensors(Seq(1.00f, 0.10f, 0.90f))),
    Instance("come-mall",   numeric.BFloatTensors(Seq(0.96f, 0.14f, 0.88f))),
    Instance("go-school",   numeric.BFloatTensors(Seq(0.12f, 0.98f, 0.82f))),
    Instance("go-station",  numeric.BFloatTensors(Seq(0.15f, 0.94f, 0.80f))),
    Instance("arrive-park", numeric.BFloatTensors(Seq(0.42f, 0.45f, 1.00f)))
  )

  data.foreach { x =>
    val objectConcept = lattice.addIntent(x)
    println(s"\nInserted: ${x.symbol}; object concept = ${objectConcept.id}")
    println(lattice.describe())
    lattice.validate()
  }

  println("\nFinal lattice:")
  println(lattice.describe())
