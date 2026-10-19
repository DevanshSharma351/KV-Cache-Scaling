# Config-derived transfer predictions

No target checkpoint was loaded and no target GPU was measured.

| Model | GPU | Assumed usable GiB (low/central/high) | Raw central L | Config-capped L | Capped assumption range |
| --- | --- | --- | ---: | ---: | --- |
| Qwen/Qwen2.5-3B | A100-40GB | 37/38/39 | 251278 | 32768 | 32768–32768 |
| Qwen/Qwen2.5-3B | RTX-4090-24GB | 22/23/23.5 | 132571 | 32768 | 32768–32768 |
| Qwen/Qwen2.5-7B | A100-40GB | 37/38/39 | 109629 | 109629 | 72387–131072 |
| Qwen/Qwen2.5-7B | RTX-4090-24GB | 22/23/23.5 | 39097 | 39097 | 22613–55262 |

Activation/VRAM ranges are assumptions, not statistical confidence intervals. Config censoring is not an observed OOM.

## Published KV-only comparison

Source: [Inside vLLM’s New KV Offloading Connector: Smarter Memory Transfer for Maximizing Inference Throughput](https://vllm-project.github.io/2026/01/08/kv-offloading-connector.html).

The publication uses MB without defining decimal versus binary units. Both errors are reported; the 2-byte cache dtype is an explicit compatibility assumption.

| Base / matching Instruct KV architecture | Predicted 16-token block bytes | Published MB | Error if MiB | Error if decimal MB |
| --- | ---: | ---: | ---: | ---: |
| Qwen/Qwen2.5-3B | 589824 | 0.56 | 0.4464% | 5.3257% |
| Qwen/Qwen2.5-7B | 917504 | 0.87 | 0.5747% | 5.4602% |

This checks KV-byte scaling only. It does not verify transferred HF single-prefill peak context limits.
