"""Guide RNA optimisation engine: cross-attention generator + frozen DeepCRISPR scorer.

Pipeline::

    seed guide ─┐
                ├─> GuideOptimizer ─> logits ─> PAM mask ─> ST Gumbel-softmax
    target DNA ─┘                                                    │
                                                              hard one-hot
                                                                     │
                            gradient  <─── frozen DeepCRISPRScorer <──┘

Only the generator trains; the scorer is frozen but stays differentiable so its
predicted efficacy can be backpropagated into the generator.
"""

__all__ = [
    "config",
    "constraints",
    "convert_checkpoint",
    "data",
    "deepcrispr_torch",
    "encoding",
    "generator",
    "gumbel",
    "logging_utils",
    "scorer",
    "tf_checkpoint",
    "trainer",
    "validity",
]
