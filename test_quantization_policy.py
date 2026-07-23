"""Regression tests for Mage-Flow DiT runtime quantization selection."""

import mlx.nn as nn

from mage_mlx.pipeline import should_quantize_dit_layer


def test_quantization_policy_selects_block_attention_and_mlp_layers():
    layer = nn.Linear(3072, 3072)

    assert should_quantize_dit_layer("transformer_blocks.0.attn.to_q", layer)
    assert should_quantize_dit_layer("transformer_blocks.5.img_mlp.fc1", layer)
    assert should_quantize_dit_layer("transformer_blocks.11.txt_mlp.fc2", layer)


def test_quantization_policy_preserves_sensitive_layers_in_bf16():
    layer = nn.Linear(3072, 3072)

    assert not should_quantize_dit_layer("img_in", layer)
    assert not should_quantize_dit_layer("transformer_blocks.0.img_mod", layer)
    assert not should_quantize_dit_layer("transformer_blocks.0.txt_mod", layer)
    assert not should_quantize_dit_layer(
        "transformer_blocks.11.img_mlp.fc1", layer
    )


def test_quantization_policy_rejects_unsupported_input_dimension():
    small_layer = nn.Linear(16, 3072)

    assert not should_quantize_dit_layer(
        "transformer_blocks.0.attn.to_q", small_layer
    )