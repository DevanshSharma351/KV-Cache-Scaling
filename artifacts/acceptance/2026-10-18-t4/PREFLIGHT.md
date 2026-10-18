# Preserved preflight failure and environment correction

The first receipt-gated attempt used prediction 7baff51 and subsequent receipt
976158c. Run b8163b23-e1ce-40af-854f-0cc16e0dbfdd stopped at preflight with
`Worker inherited an initialized CUDA context`. No checkpoint weights were
loaded, no model forward ran, no peak measurement or OOM was observed. The
stopped run is preserved and will not be resumed/retried automatically.

`failed-preflight-01.zip` SHA-256:
`5fe389834881e49498f800d0c35970ff6be9639d37e7129c360289cfa4669d76`.

A fresh-process import trace identifies installed optional torchao 0.10.0
capability checks (and its Triton imports) initializing CUDA during Transformers
class imports. Removing this unused optional package is sufficient: a fresh
`_import_runtime()` then reports cuda_initialized_after_import=false, with
unchanged torch 2.6.0+cu124 and Transformers 4.51.3. The cold-worker guard and all
scientific/execution source remain unchanged. No quantized model is used.

`preflight-environment-correction.zip` SHA-256:
`6b793263522ee005ad9e03c8a15360fcdafac54a5a880425acef5e77170be524`.
It preserves the full import trace, explicit pip uninstall command, cold-import
check, refreshed context observation, package inventory and original failure.
The refreshed UUID, free/total bytes, driver and direct scientific versions
match the frozen baseline exactly. Therefore the advance numerical prediction
is unchanged. Every command is visible in notebook cells 26–32, including a
parent-import diagnostic mistake that did not invoke the intended child.

The user authorized delegated technical review and execution. This environment
correction was reviewed on that basis before further model work. It avoids
weakening the guard or changing the registered measurement semantics. It is a
pre-model environment correction, not a calibrated prediction or a new model
measurement. All future workers in this runtime must retain torchao's absence.
The source requirements are direct pins, not a transitive environment lock; the
new raw inventory is the authoritative installed-package record.

These archives were read from visible notebook output, SHA-256 checked and
fsynced to the user's local filesystem. Every MANIFEST.json member was verified.
Google Drive remains disconnected. Colab alone is not persistent storage.
