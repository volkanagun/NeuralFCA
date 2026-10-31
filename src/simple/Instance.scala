package simple

import concept.CVAUtil

class Instance(
                val symbol: String,
                val attributes: torch.Tensor[torch.BFloat16]
              ) {

  override def equals(instance: Any): Boolean = {
    val obj = instance.asInstanceOf[Instance]
    symbol == obj.symbol
  }

  override def toString: String =
    s"$symbol"
}