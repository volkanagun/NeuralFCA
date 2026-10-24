package simple
import cvafca.{CVA, CvaModel}
import torch.{BFloat16, Tensor}

import scala.collection.mutable

class Lattice(val dim:Int = 300, val ratio:Float=0.7, val threshold:Float = 0.6) {

  // All inserted objects
  private val instances =
    mutable.LinkedHashMap.empty[String, Instance]

  // All formal concepts
  private val nodes =
    mutable.ArrayBuffer.empty[Node]

  // Intent -> Concept index
  private val intentIndex =
    mutable.HashMap.empty[torch.Tensor[torch.BFloat16], Node]

  private var nextNodeId = 0
  private val cva = new CVA(dim)
  private val bottomModel = cva.emptyModel()

  // Initial lattice contains only the bottom concept:
  // (empty extent, all attributes)
  private val bottomNode =
    createNode(bottomModel, Set.empty[Instance])


  // ==================================================
  // CREATE NODE
  // ==================================================

  private def createNode(instance: Instance): Node = {
    val model = cva.fit_cva(mutable.Iterable(instance.attributes))
    Node(0, model, instance)
  }

  private def createNode(
                          intent: CvaModel,
                          extent: Set[Instance]
                        ): Node = {

    require(
      !intentIndex.contains(intent.commonVector),
      s"Concept already exists: $intent"
    )

    val node = new Node(
      nextNodeId,
      intent,
      extent
    )

    nextNodeId += 1

    nodes += node
    intentIndex(intent.commonVector) = node

    node
  }


  // ==================================================
  // GRAPH OPERATIONS
  // ==================================================

  // upper is more general than lower
  private def setLink(
                       upper: Node,
                       lower: Node
                     ): Unit = {

    upper.children += lower
    lower.parents += upper
  }


  private def removeLink(
                          upper: Node,
                          lower: Node
                        ): Unit = {

    upper.children -= lower
    lower.parents -= upper
  }


  // ==================================================
  // FIND CANONICAL GENERATOR
  // ==================================================

  /**
   * Finds the most general existing concept whose
   * intent contains the requested intent.
   *
   * The search moves upward through parent links.
   *
   * Precondition:
   *   intent subsetOf generator.intent
   */
  private def findCanonicalGenerator(
                                      intent: Node,
                                      generator: Node
                                    ): Node = {

/*    require(
      intent.subsetOf(generator.intent),
      "The initial generator must contain the intent"
    )*/

    var current = generator
    var searching = true

    while (searching) {

      val next = current.parents.find { parent =>
        intent.subset_of(parent, ratio, threshold)
      }

      next match {

        case Some(parent) =>
          current = parent

        case None =>
          searching = false
      }
    }

    current
  }


  // ==================================================
  // SELECT IMMEDIATE PARENTS
  // ==================================================

  /**
   * Keeps the least general concepts among candidates.
   *
   * For two intents A and B:
   *
   * A subset B  => concept(A) is above concept(B)
   *
   * Therefore, among candidate parents, we retain
   * maximal intents under set inclusion.
   */
  private def selectImmediateParents(
                                      candidates: Seq[Node]
                                    ): Set[Node] = {

    val unique = candidates.distinct

    unique.filter { candidate =>

      !unique.exists { other =>

        (other ne candidate) &&
          candidate.subset_of(other, ratio, threshold) &&
          candidate.intent != other.intent
      }

    }.toSet
  }


  // ==================================================
  // UPDATE COVERING RELATIONS
  // ==================================================

  /**
   * Inserts a newly created concept between its
   * canonical generator and its new parents.
   *
   * All parent-child relationships are updated locally.
   */
  private def updateCoveringRelations(
                                       newNode: Node,
                                       generator: Node,
                                       parents: Set[Node]
                                     ): Unit = {

    // Connect the new concept to its immediate parents.
    parents.foreach { parent =>

      // The old edge is no longer an immediate cover
      // if it connected this parent to the generator.
      removeLink(parent, generator)

      setLink(parent, newNode)
    }

    // The new concept lies immediately above generator.
    setLink(newNode, generator)
  }


  // ==================================================
  // FAST ADD INTENT
  // ==================================================

  /**
   * Recursive intent insertion.
   *
   * Returns the concept having exactly the specified
   * intent. Creates it if necessary.
   *
   * This is an optimized AddIntent-style procedure.
   */
  private def fastAddIntent(
                             intent: Node,
                             generator: Node
                           ): Node = {

   /* require(
      intent.subsetOf(attributes),
      "Unknown attributes"
    )*/

    /*// O(1) expected indexed lookup for an existing intent.
    intentIndex.get(intent) match {

      case Some(existing) =>
        return existing

      case None =>
    }*/

    // Step 1: locate canonical generator.
    val canonicalGenerator =
      findCanonicalGenerator(intent, generator)

    // Step 2: existing exact concept?
    if (canonicalGenerator == intent) {
      return canonicalGenerator
    }

    // Step 3: examine parents of the generator.
    // Snapshot prevents concurrent graph modifications
    // from changing the iteration.
    val generatorParents =
      canonicalGenerator.parents.toVector

    val parentCandidates =
      mutable.ArrayBuffer.empty[Node]

    for (candidate <- generatorParents) {

      val parentNode =

        if (candidate.subset_of(intent, ratio, threshold)) {

          // Existing parent can be reused.
          candidate

        } else {

          // Recursively insert their common intent.
          val intersection =
            candidate.intersect(intent)

          fastAddIntent(intersection, candidate)
        }

      parentCandidates += parentNode
    }

    // Step 4: remove non-immediate parent candidates.
    val newParents =
      selectImmediateParents(parentCandidates.toVector)

    // Another recursive branch might have created this
    // intent, so check the index again.
    intentIndex.get(intent.common) match {

      case Some(existing) =>
        return existing

      case None =>
    }

    // Step 5: create a new concept.
    //
    // Its old extent is initially the extent of its
    // canonical generator.
    val newNode = createNode(
      intent.intent,
      canonicalGenerator.extent
    )

    // Step 6: update only affected lattice edges.
    updateCoveringRelations(
      newNode,
      canonicalGenerator,
      newParents
    )

    newNode
  }


  // ==================================================
  // ADD INSTANCE
  // ==================================================

  /**
   * Incrementally inserts an object into the context.
   */
  def addInstance(instance: Instance): Node = {

/*    require(
      !instances.contains(instance.id),
      s"Duplicate instance: ${instance.id}"
    )

    require(
      instance.attributes.subsetOf(attributes),
      "Instance contains unknown attributes"
    )*/

    // Insert structural concept before changing extents.
    val node = createNode(instance)
    val objectConcept =
      fastAddIntent(node, bottomNode)

    instances(instance.id) = instance

    // Add the object to the extent of its object concept
    // and to all of its ancestors.
    val visited =
      mutable.HashSet.empty[Node]

    val stack =
      mutable.Stack[Node](objectConcept)

    while (stack.nonEmpty) {

      val node = stack.pop()

      if (visited.add(node)) {

        node.extent = node.extent + instance

        node.parents.foreach { parent =>
          stack.push(parent)
        }
      }
    }

    objectConcept
  }


  // ==================================================
  // ACCESSORS
  // ==================================================

  def getNodes: Seq[Node] =
    nodes.toVector

  def getInstances: Seq[Instance] =
    instances.values.toVector

  def findNode(
                intent: Tensor[BFloat16]
              ): Option[Node] =
    intentIndex.get(intent)

  def top: Node =
    nodes.find(_.parents.isEmpty).get

  def bottom: Node =
    bottomNode


  // ==================================================
  // PRINT CONCEPTS
  // ==================================================

  def printConcepts(): Unit = {

    println("\nCONCEPTS")

    nodes
      .sortBy(n => (n.extent.size, n.id))
      .foreach(println)
  }


  // ==================================================
  // PRINT EDGES
  // ==================================================

  def printEdges(): Unit = {

    println("\nCOVERING RELATIONS")

    nodes.sortBy(_.id).foreach { node =>

      node.children.toSeq.sortBy(_.id).foreach { child =>

        println(
          s"C${node.id} -> C${child.id}"
        )
      }
    }
  }


  def printLattice(): Unit = {

    printConcepts()
    printEdges()
  }


  // ==================================================
  // VALIDATE FORMAL CONCEPTS AND LATTICE EDGES
  // ==================================================

  def validate(): Boolean = {

    val allNodes = nodes.toVector

    // Every concept must satisfy A' = B and B' = A.
    val conceptsValid = true
    // Check immediate covering relations.
    val edgesValid = allNodes.forall { upper =>

      upper.children.forall { lower =>

        val ordered =
          upper.subset_of(lower, ratio, threshold) &&
            upper.intent != lower.intent

        val reciprocal =
          lower.parents.contains(upper)

        val intermediateExists =
          allNodes.exists { middle =>

            (middle ne upper) &&
              (middle ne lower) &&
            upper.subset_of(middle, ratio, threshold) &&
              upper.intent != middle.intent &&
            middle.subset_of(lower, ratio, threshold) &&
              middle.intent != lower.intent
          }

        ordered && reciprocal && !intermediateExists
      }
    }

    // Every strict comparable pair must have a path.
    val reachabilityValid = allNodes.forall { upper =>

      val reachable =
        mutable.HashSet.empty[Node]

      val stack =
        mutable.Stack.empty[Node]

      upper.children.foreach(n => stack.push(n))

      while (stack.nonEmpty) {

        val current = stack.pop()

        if (reachable.add(current)) {
          current.children.foreach(n => stack.push(n))
        }
      }

      allNodes.forall { lower =>

        if (
          upper.subset_of(lower, ratio, threshold) &&
            upper.intent != lower.intent
        ) {
          reachable.contains(lower)
        } else {
          true
        }
      }
    }

    val reverseEdgesValid = allNodes.forall { lower =>
      lower.parents.forall(_.children.contains(lower))
    }

    conceptsValid &&
      edgesValid &&
      reachabilityValid &&
      reverseEdgesValid
  }
}