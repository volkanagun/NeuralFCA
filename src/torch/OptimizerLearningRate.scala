package torch

import torch.optim.AdamW

object OptimizerLearningRate {
  // Storch exposes the native optimizer only within the torch package.
  def set(optimizer: AdamW, rate: Double): Unit =
    optimizer.native.param_groups().get(0).options().set_lr(rate)

}
