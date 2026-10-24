package simple

import concept.CVAUtil

class Instance(
                val id: String,
                val attributes: torch.Tensor[torch.BFloat16]
              ) {

  override def equals(instance: Any): Boolean = {
    val obj = instance.asInstanceOf[Instance]
    id == obj.id
  }

  override def toString: String =
    s"$id"
}