# 2026-07-24 — mflux Mage-Flow Edit Audit

## Findings

- `/Users/tilman/projects/mflux` `mage-flow-mlx` is the authoritative reference.
- mflux uses one native `MageFlowTextEncoder` for both txt2img and edit.
- The worker's successful outputs were cache hits from legacy embeddings and did not validate the current native encoder.
- The current port previously allocated KV-cache tensors during every non-cache text-encoder forward. mflux allocates KV cache only when `use_cache=True`.

## Changes made

- Matched native attention cache behavior to mflux: no cache allocation on normal conditioning passes.
- Matched decoder no-cache return behavior: returns hidden states directly; cache tuple only when requested.
- Added `test_edit_contracts.py` using stdlib `unittest` only.

## Verification

- `python3 -m py_compile mage_mlx/*.py` passed.
- `test_edit_contracts.py` passed: 3 tests.
- Multi-image RoPE contract: `(1,64,64)+(1,64,64)` -> `(8192,64)` cos/sin.
- Edit latent sequence contract: target + reference -> `(1,8192,128)`.
- No model loading, txt2img generation, or edit generation was run during this audit.

## Remaining

- Full numeric comparison against mflux still requires controlled model execution.
- Edit peak memory remains intrinsically higher because target/reference tokens increase attention sequence length; real-resolution testing requires explicit approval.
