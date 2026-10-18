"""Config-derived Qwen2 counts and explicitly hypothetical peak scenarios."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import isfinite

GIB = 1024**3


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def vram_gib_to_bytes(value):
    """Convert a positive GiB input to integer bytes without binary float drift."""
    try:
        gib = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("usable VRAM must be a finite positive GiB value") from error
    if not gib.is_finite() or gib <= 0:
        raise ValueError("usable VRAM must be a finite positive GiB value")
    result = int(gib * GIB)
    if result < 1:
        raise ValueError("usable VRAM must be at least one byte")
    return result


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
        if not isinstance(config, dict):
            raise ValueError("model config must be a JSON object")
        if config.get("model_type") != "qwen2":
            raise ValueError("Parameter counting currently supports Qwen2 only")
        keys = ("num_hidden_layers", "hidden_size", "intermediate_size",
                "num_attention_heads", "num_key_value_heads", "vocab_size",
                "max_position_embeddings")
        for key in keys:
            positive_int(config.get(key), key)
        h, q, k = (config[key] for key in
                   ("hidden_size", "num_attention_heads", "num_key_value_heads"))
        if h % q or q % k:
            raise ValueError("Invalid hidden/head dimensions or GQA grouping")
        d = config.get("head_dim", h // q)
        positive_int(d, "head_dim")
        if d != h // q:
            raise ValueError("This Qwen2 counter requires head_dim = hidden_size / query_heads")
        if config.get("use_sliding_window", False) is not False:
            raise ValueError("Sliding-window attention needs a different retained-cache model")
        if config.get("hidden_act", "silu") != "silu":
            raise ValueError("This parameter/activation model supports Qwen2 SwiGLU only")
        tied = config.get("tie_word_embeddings", False)
        if not isinstance(tied, bool):
            raise ValueError("tie_word_embeddings must be boolean")
        return cls(config["num_hidden_layers"], h, config["intermediate_size"],
                   q, k, d, config["vocab_size"], tied,
                   config["max_position_embeddings"])

    def parameter_count(self):
        """Unique Qwen2 parameters: Q/K/V biases, two layer norms, final norm."""
        return sum(self.parameter_breakdown().values())

    def parameter_breakdown(self):
        """Disjoint parameter components; counts exclude buffers and allocator slack."""
        h, q, k = self.hidden, self.query_heads * self.head_dim, self.kv_heads * self.head_dim
        return {
            "input_embedding": self.vocab * h,
            "untied_output_embedding": 0 if self.tied_embeddings else self.vocab * h,
            "attention_projection_weights": self.layers * (h * q + 2 * h * k + q * h),
            "attention_qkv_biases": self.layers * (q + 2 * k),
            "mlp_weights": self.layers * 3 * h * self.intermediate,
            "layer_norm_weights": self.layers * 2 * h,
            "final_norm_weight": h,
        }

    def weight_bytes(self, element_bytes=2):
        return self.parameter_count() * positive_int(element_bytes, "element_bytes")

    def kv_bytes_per_token(self, batch_size=1, element_bytes=2, naive=False):
        positive_int(batch_size, "batch_size")
        positive_int(element_bytes, "element_bytes")
        heads = self.query_heads if naive else self.kv_heads
        return 2 * batch_size * self.layers * heads * self.head_dim * element_bytes

    def kv_bytes(self, length, batch_size=1, element_bytes=2, naive=False):
        return positive_int(length, "length") * self.kv_bytes_per_token(
            batch_size, element_bytes, naive)


@dataclass(frozen=True)
class PeakScenario:
    name: str
    activation_bytes_per_token: int
    fixed_headroom_bytes: int

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("scenario name must be nonempty")
        positive_int(self.activation_bytes_per_token, "activation_bytes_per_token")
        if (isinstance(self.fixed_headroom_bytes, bool)
                or not isinstance(self.fixed_headroom_bytes, int)
                or self.fixed_headroom_bytes < 0):
            raise ValueError("fixed headroom must be a nonnegative integer byte count")


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


def peak_tensor_bytes(architecture, length, scenario, batch_size=1):
    """FP16 sensitivity model: fixed weights plus batch-scaled KV/activations."""
    positive_int(length, "length")
    positive_int(batch_size, "batch_size")
    return (architecture.weight_bytes()
            + length * (architecture.kv_bytes_per_token(batch_size)
                        + batch_size * scenario.activation_bytes_per_token))


def peak_budget_bytes(architecture, length, scenario, batch_size=1):
    return (peak_tensor_bytes(architecture, length, scenario, batch_size)
            + scenario.fixed_headroom_bytes)


def predict_limit(architecture, usable_vram_bytes, scenario, cap=None, batch_size=1):
    if (isinstance(usable_vram_bytes, bool)
            or not isinstance(usable_vram_bytes, (int, float))
            or not isfinite(usable_vram_bytes) or usable_vram_bytes <= 0):
        raise ValueError("usable VRAM must be finite and positive")
    positive_int(batch_size, "batch_size")
    if cap is not None:
        positive_int(cap, "cap")
    available = usable_vram_bytes - architecture.weight_bytes() - scenario.fixed_headroom_bytes
    slope = architecture.kv_bytes_per_token(batch_size) + batch_size * scenario.activation_bytes_per_token
    limit = max(0, int(available // slope))
    return min(limit, cap) if cap is not None else limit
