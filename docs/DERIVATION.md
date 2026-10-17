# Derivation and draft uncertainty model

Status: analytical starting point, **not formal preregistration**. No empirical
coefficient is claimed. `kvcache predict` implements these formulas on CPU.

## Config-derived weights and KV

Let N be layer count, H hidden size, I intermediate size, Q query heads, K KV
heads, D head dimension, V vocabulary, B batch and s bytes per cache element.
For the pinned Qwen2 implementation, with q=QD and k=KD:

```text
embedding parameters = VH                           (tied input/output)
attention per layer  = Hq + Hk + Hk + qH + q + k + k
MLP per layer        = 3HI
norms per layer      = 2H
final norm           = H
P = VH + N[Hq + 2Hk + qH + q + 2k + 3HI + 2H] + H
W_fp16 = 2P
KV(L,B) = 2BNKDLs
```

Untied output embeddings add VH parameters. The parameter count is architecture
specific: do not apply it unchanged to every model family. Rotary buffers are not
parameters. Compare to deduplicated loaded parameter storage, then account for
buffers/rounding separately.

For this snapshot: P=1543714304, W=3087428608 bytes; s=2 gives k(1)=28672
bytes/token=28 KiB/token. Using Q instead of K gives 172032 bytes/token: a 6x
error. The tied-matrix duplicate would add 466747392 FP16 bytes.

| L | Correct retained KV (MiB) | Naive query-head KV (MiB) |
| ---: | ---: | ---: |
| 512 | 14 | 84 |
| 1024 | 28 | 168 |
| 2048 | 56 | 336 |
| 4096 | 112 | 672 |
| 8192 | 224 | 1344 |

## Activation envelope and uncertainty

Peak inference memory is a maximum over phases, not the sum of every layer's
activation tensors. At layer j the cache is only partly populated; the final
full cache need not coexist with the largest MLP temporary. A conservative
starting approximation is full final KV plus a linear activation envelope.

Reduced logits have B*1*V elements. MLP gate/up/product tensors scale with BLI;
residuals, normalization/rotary intermediates, projections, and repeated grouped
K/V scale with BLH or BLQD. Efficient SDPA avoids the full L² score allocation,
but still uses workspace. Float32 normalization/rotary temporaries and unknown
kernel workspaces make an exact shape-only upper bound difficult.

The CPU scaffold proposes the following **hypothetical sensitivity scenarios**:

| Scenario | a: activation bytes/token | H0: fixed allowance |
| --- | --- | --- |
| Optimistic | `2*(2*I + 6*H)` = 54272 | 0.25 GiB |
| Central | `2*(3*I + 8*H)` = 78336 | 0.50 GiB |
| Conservative | `2*(4*I + 16*H)` = 120832 | 1.00 GiB |

The coefficients encode different temporary overlap/precision assumptions; they
are not proven implementation bounds or a 95% confidence interval. H0 includes
allocator slack, constant logits, buffer/workspace uncertainty and a safety
allowance; it excludes context already removed from V_usable. Do not describe
H0 as measured fragmentation. A1 must defend/revise these assumptions by source
and tensor-lifetime reasoning before freezing the prediction.

```text
P_tensor_scenario(L) = W + [k(1)+a]L
P_budget_scenario(L) = P_tensor_scenario(L) + H0
L_uncapped(V) = max(0, floor((V - W - H0) / [k(1)+a]))
L_config_capped(V) = min(131072, L_uncapped(V))
L_quality_domain(V) = min(32768, L_uncapped(V))
```

When V<W+H0, return 0 and label the model infeasible under that scenario. If
L_uncapped exceeds a ceiling, report censoring, not a measured threshold.
For a candidate L, the safe coordinator needs an additional conservative
admission margin; the theory-only CLI does not authorize running that candidate.

Use `kvcache predict --usable-vram-gib V` for each chosen V. The command emits
tensor-peak and budget bytes at the five required lengths, scenario coefficients
and capped and uncapped limits. The formal prediction must include both columns
in its reviewed tables so analysis compares like metrics. Save outputs under a name that
states DRAFT until A1 is frozen. Do not calibrate these coefficients against GPU
forwards before preregistration.

### Worked draft preview

These numbers reproduce the current CPU assumptions; they are not a formal
prediction or observed limits. The wide range shows how strongly the answer
depends on the activation model.

| Usable VRAM (GiB) | Conservative L | Central L | Optimistic L, uncapped |
| ---: | ---: | ---: | ---: |
| 12.0 | 58351 | 86541 | 114885 |
| 13.0 | 65533 | 96575 | 127830 |
| 14.0 | 72715 | 106609 | 140775 |
| 14.5 | 76306 | 111626 | 147248 |

At 14.5 GiB the config-capped draft range is 76306–131072, central 111626.
The quality-domain summary is capped at 32768 in all these scenarios. This is
why a memory-only stress result must be reported separately from supported
context. It also explains why an OOM might not occur inside the chosen domain.

| L | Central tensor peak (MiB) | Central budget (MiB) |
| ---: | ---: | ---: |
| 512 | 2996.651 | 3508.651 |
| 1024 | 3048.901 | 3560.901 |
| 2048 | 3153.401 | 3665.401 |
| 4096 | 3362.401 | 3874.401 |
| 8192 | 3780.401 | 4292.401 |

MiB values are display-rounded; implementation uses exact byte arithmetic.

## Calibration after preregistration

Retain the original scenarios, fit a separate post-measurement peak model, and
use held-out lengths/repeats to assess its error. Add a quadratic term only when
the actual pinned execution allocates a quadratic mask/score tensor; discovering
a math/eager fallback invalidates the primary protocol instead of silently
changing its model. Transfer fits to other architectures/hardware with wider
assumption bands, not unjustified statistical precision.
