"""Config-derived Qwen2 counts and explicitly hypothetical peak scenarios."""

from dataclasses import dataclass
from math import isfinite

GIB = 1024**3


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Architecture:
    layers: int
    hidden: int
    intermediate: int
    query_heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    tied_embeddings: bool
    max_positions: int

    @classmethod
    def from_config(cls, config):
        if config.get("model_type") != "qwen2":
            raise ValueError("Parameter counting currently supports Qwen2 only")
        keys = ("num_hidden_layers", "hidden_size", "intermediate_size",
                "num_attention_heads", "num_key_value_heads", "vocab_size",
                "max_position_embeddings")
        for key in keys:
            positive_int(config[key], key)
        h, q, k = (config[key] for key in
                   ("hidden_size", "num_attention_heads", "num_key_value_heads"))
        if h % q or q % k:
            raise ValueError("Invalid hidden/head dimensions or GQA grouping")
        d = config.get("head_dim") or h // q
        positive_int(d, "head_dim")
        tied = config.get("tie_word_embeddings", False)
        if not isinstance(tied, bool):
            raise ValueError("tie_word_embeddings must be boolean")
        return cls(config["num_hidden_layers"], h, config["intermediate_size"],
                   q, k, d, config["vocab_size"], tied,
                   config["max_position_embeddings"])

    def parameter_count(self):
        """Unique Qwen2 parameters: Q/K/V biases, two layer norms, final norm."""
        h, q, k = self.hidden, self.query_heads * self.head_dim, self.kv_heads * self.head_dim
        embeddings = self.vocab * h * (1 if self.tied_embeddings else 2)
        attention = h * q + 2 * h * k + q * h + q + 2 * k
        mlp = 3 * h * self.intermediate
        return embeddings + self.layers * (attention + mlp + 2 * h) + h

    def kv_bytes_per_token(self, batch_size=1, element_bytes=2, naive=False):
        positive_int(batch_size, "batch_size")
        positive_int(element_bytes, "element_bytes")
        heads = self.query_heads if naive else self.kv_heads
        return 2 * batch_size * self.layers * heads * self.head_dim * element_bytes


@dataclass(frozen=True)
class PeakScenario:
    name: str
    activation_bytes_per_token: int
    fixed_headroom_bytes: int


def prefill_scenarios(architecture):
    """Shape-inspired sensitivity bounds, not measured confidence intervals.

    Full final KV + linear activation envelope is deliberately conservative;
    see docs/DERIVATION.md for phase overlap and omitted workspace caveats.
    """
    h, i = architecture.hidden, architecture.intermediate
    return (
        PeakScenario("optimistic", 2 * (2 * i + 6 * h), GIB // 4),
        PeakScenario("central", 2 * (3 * i + 8 * h), GIB // 2),
        PeakScenario("conservative", 2 * (4 * i + 16 * h), GIB),
    )


def peak_budget_bytes(architecture, length, scenario):
    positive_int(length, "length")
    return (2 * architecture.parameter_count() + scenario.fixed_headroom_bytes
            + length * (architecture.kv_bytes_per_token()
                        + scenario.activation_bytes_per_token))


def predict_limit(architecture, usable_vram_bytes, scenario, cap=None):
    if not isfinite(usable_vram_bytes) or usable_vram_bytes <= 0:
        raise ValueError("usable VRAM must be finite and positive")
    if cap is not None:
        positive_int(cap, "cap")
    available = usable_vram_bytes - 2 * architecture.parameter_count() - scenario.fixed_headroom_bytes
    slope = architecture.kv_bytes_per_token() + scenario.activation_bytes_per_token
    limit = max(0, int(available // slope))
    return min(limit, cap) if cap is not None else limit
