# Pinned Colab preparation: directly observed

Notebook: https://colab.research.google.com/drive/1GEytpHAffBJog7yi77FJjk-A9t4PedLS
Cells 17–21 append a new audit run without replacing historical outputs.
Runtime selector: 2025.07, Tesla T4. Initial image Python 3.11.13,
PyTorch 2.6.0+cu124, Transformers 4.53.2. Recorded pip installation applies
`requirements-gpu.txt`, including Transformers 4.51.3, without reinstalling torch.

Source checkout: `072293c3150f3479e853ebfdd874fdd3031d50cb`.
All 77 tests from that checkout passed in Colab. They are CPU tests, including
synthetic GPU-counter and receipt fixtures; they are not model measurements.
The fresh isolated context inventory has no live allocated/reserved tensors;
the parent notebook owns no CUDA context. This is a different GPU UUID from the
historical run. Its baseline must not be mixed with that run's identity.

`preparation.zip` is the original archive printed in cell 21, transferred through
visible cell output and SHA-256 checked locally. It contains command journals,
full stdout/stderr, full package inventory, fresh inventory and installed API
inspection. Inspection unwraps Qwen's forward and hashes modeling_qwen2.py,
cache_utils.py and sdpa_attention.py. No model was instantiated or forwarded.
The installed SDPA adapter has no enable_gqa parameter. This source observation
is not evidence of successful kernel execution.

The runtime mismatch is resolved for preparation. Kernel, allocator and model
checks remain gated by actual review and formal preregistration. Drive remains
disconnected. Colab staging is ephemeral; the local verified archive persists.
