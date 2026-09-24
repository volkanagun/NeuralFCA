package testing

import numeric.BFloatTensors.*
import concept.{CVANode, CVAUtil}

object Tester {
  def main(args: Array[String]): Unit = {
    val top = CVANode.empty(0,size = 2, dim = 4, device = torch.Device.CPU)
    val arbitraryInstance = Array(
      numeric.BFloatTensors(Seq(-100.0f, 2.5f, 9.0f, 42.0f)),
      numeric.BFloatTensors(Seq(7.0f, -3.0f, 0.0f, 1.0f))
    )
    assert(top.membership(arbitraryInstance))
    assert(!top.membership(arbitraryInstance.take(1)))

    val smallerCommon = numeric.BFloatTensors(Seq(0.1f, 0.2f))
    val largerCommon = numeric.BFloatTensors(Seq(0.3f, 0.4f))
    assert(CVAUtil.commonVectorSubset(smallerCommon, largerCommon, 0.9f))
    assert(!CVAUtil.commonVectorSubset(largerCommon, smallerCommon, 0.9f))

    val tensorCva = new cvafca.CVA(
      dimension = 3,
      config = cvafca.CvaConfig(classificationThreshold = 1e-4)
    )
    val tensorSamples = Vector(
      numeric.BFloatTensors(Seq(1.0f, 2.0f, 5.0f)),
      numeric.BFloatTensors(Seq(3.0f, 2.0f, 5.0f)),
      numeric.BFloatTensors(Seq(1.0f, 4.0f, 5.0f))
    )
    val tensorModel = tensorCva.fit(tensorSamples)
    assert(tensorModel.commonVector.dtype == torch.bfloat16)
    assert(tensorModel.projection.dtype == torch.bfloat16)
    assert(tensorModel.projection.shape.sameElements(Array(3, 3)))
    assert(tensorSamples.forall(tensorCva.classifies(tensorModel, _)))

    // Direct AddIntent places each new intent above its maximal generator and
    // recursively reuses the candidate/Y intersection.
    val lattice = new concept.CVALattice(torch.Device.CPU, 1, 3)
    val a = Array(numeric.BFloatTensors(Seq(0.2f, 0.2f, 0.2f)))
    val b = Array(numeric.BFloatTensors(Seq(1.0f, 0.2f, 0.2f)))
    val c = Array(numeric.BFloatTensors(Seq(0.2f, 1.0f, 0.2f)))
    lattice.intent("a", a).intent("b", b).intent("c", c)

    assert(lattice.concepts.length == 4)
    assert(lattice.concepts.forall(node => node.items.length == node.instances.length))
    assert(lattice.bottom.items.toSet == Set("a", "b", "c"))
    assert(lattice.bottom.parents.length == 1)
    assert(lattice.concepts.map(_.parents.length).sum == 3)
    assert(lattice.concepts.forall(node => !node.parents.contains(node)))

    // An existing intent must be found through the diagram without creating a
    // duplicate concept.
    lattice.intent("a-copy", a)
    assert(lattice.concepts.length == 4)

    // The recursive call must materialise a genuinely new intersection without
    // revisiting either candidate indefinitely.
    val recursive = new concept.CVALattice(torch.Device.CPU, 1, 3)
    val ra = Array(numeric.BFloatTensors(Seq(0.2f, 0.2f, 0.2f)))
    val rb = Array(numeric.BFloatTensors(Seq(1.0f, 0.5f, 0.2f)))
    val rc = Array(numeric.BFloatTensors(Seq(0.5f, 1.0f, 0.2f)))
    recursive.intent("ra", ra).intent("rb", rb).intent("rc", rc)
    assert(
      recursive.concepts.exists { concept =>
        concept.cv.head.floatValues.sameElements(Array(0.5f, 0.5f, 0.2001953125f))
      }
    )
    assert(recursive.concepts.forall(node => !node.parents.contains(node)))

    // Exact object intents and intersections are canonical: reinsertion must
    // update extents without allocating any duplicate concepts.
    val recursiveConceptCount = recursive.concepts.length
    recursive.intent("ra-copy", ra).intent("rb-copy", rb).intent("rc-copy", rc)
    assert(recursive.concepts.length == recursiveConceptCount)
  }
}
