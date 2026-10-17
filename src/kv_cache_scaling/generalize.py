"""A5 / Person 1: architecture-specific config adapters and transfer uncertainty.

Predict Qwen2.5-3B and Qwen2.5-7B on A100-40GB and RTX 4090-24GB without
loading weights. Published comparison must match the measured quantity.
B6 / Person 2 supplies pinned config provenance and published-source evidence.
"""


def generalize(configs, budgets, assumptions):
    raise NotImplementedError("A5: transfer scenarios and external verification pending")
