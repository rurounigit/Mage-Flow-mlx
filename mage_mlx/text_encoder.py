"""Qwen3-VL text encoder for Mage-Flow (MLX port).

Uses mlx-lm's Qwen3-VL model (which delegates to Qwen3) with the custom
Mage-Flow text encoder config. The text encoder produces 2560-dim embeddings
that are projected to 3072-dim by the DiT's txt_in layer.

Weight mapping (PyTorch → MLX):
  model.language_model.X → language_model.model.X

The tokenizer uses the HuggingFace Qwen3-VL tokenizer with the Mage-Flow
chat template:
  <im_start>system\nDescribe the image...<im_start>user\n{prompt}<im_start>assistant
  (first 34 tokens are dropped as they're the system prompt)
"""

from __future__ import annotations

import json
import os
from typing import Any

import mlx.core as mx

# Mage-Flow text encoder config (from microsoft/Mage-Flow-Turbo/text_encoder/config.json)
MAGE_FLOW_TEXT_CONFIG = {
    "model_type": "qwen3_vl",
    "hidden_size": 2560,
    "num_hidden_layers": 36,
    "intermediate_size": 9728,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 262144,
    "rope_theta": 5000000,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
    "rope_scaling": {
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_type": "default",
    },
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "initializer_range": 0.02,
    "use_cache": True,
}

# Mage-Flow chat template (from utils.py PROMPT_TEMPLATE)
MAGE_FLOW_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
MAGE_FLOW_START_IDX = 34


class Qwen3VLTextEncoder:
    """Qwen3-VL text encoder for Mage-Flow using mlx-lm.

    Loads the Qwen3-VL model with the Mage-Flow text encoder config and
    provides text encoding via the model's last hidden state.

    Args:
        model_path: Path to the MLX-converted text encoder weights
        hf_repo_id: HuggingFace repo ID for tokenizer download
    """

    def __init__(
        self,
        model_path: str | None = None,
        hf_repo_id: str = "microsoft/Mage-Flow-Turbo",
    ):
        self.hf_repo_id = hf_repo_id
        self.model_path = model_path

        # Lazy-load model and tokenizer
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> Any:
        """Load the Qwen3-VL model from mlx-lm with Mage-Flow config."""
        if self._model is not None:
            return self._model

        from mlx_lm.models.qwen3_vl import Model, ModelArgs

        args = ModelArgs(
            model_type="qwen3_vl",
            text_config=MAGE_FLOW_TEXT_CONFIG,
        )
        self._model = Model(args)

        # Load weights if path is provided
        if self.model_path and os.path.exists(self.model_path):
            self._load_weights()

        return self._model

    def _load_weights(self) -> None:
        """Load MLX-converted weights into the model."""
        weights = mx.load(self.model_path)
        if any(key.endswith((".scales", ".biases")) for key in weights):
            raise ValueError("Quantized text encoders are unsupported; reconvert in BF16")
        self._model.load_weights(list(weights.items()), strict=False)

    def unload(self) -> None:
        """Release model weights after prompt embeddings have been computed."""
        self._model = None
        mx.clear_cache()

    def _load_tokenizer(self):
        """Load the Qwen3-VL tokenizer from HuggingFace."""
        if self._tokenizer is not None:
            return self._tokenizer

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.hf_repo_id,
            subfolder="text_encoder",
        )
        return self._tokenizer

    def __call__(self, prompt: str) -> mx.array:
        """Encode a text prompt into embeddings.

        Applies the Mage-Flow chat template, tokenizes, runs through the
        Qwen3-VL text encoder, and returns the last hidden state.

        Args:
            prompt: Text prompt string

        Returns:
            [1, seq_len, 2560] text embeddings
        """
        model = self._load_model()
        tokenizer = self._load_tokenizer()

        # Apply Mage-Flow chat template
        formatted = MAGE_FLOW_TEMPLATE.format(prompt)

        # Tokenize
        inputs = tokenizer(
            formatted,
            return_tensors="np",
            truncation=True,
            max_length=2048,
        )
        input_ids = mx.array(inputs["input_ids"][0])[None, :]  # Add batch dim [1, seq_len]

        # Run through the backbone model (model.language_model.model) to get hidden states before lm_head projection
        last_hidden = model.language_model.model(input_ids)

        # Drop the first START_IDX tokens (system prompt)
        last_hidden = last_hidden[:, MAGE_FLOW_START_IDX:, :]

        return last_hidden

    @property
    def hidden_size(self) -> int:
        return MAGE_FLOW_TEXT_CONFIG["hidden_size"]

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters in the text encoder."""
        model = self._load_model()
        return sum(p.size for _, p in model.parameters())
