from typing import Optional
import hashlib
from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mage_mlx.mflux_src.mflux.models.common.config.config import Config
from mage_mlx.mflux_src.mflux.models.common.config.model_config import ModelConfig
from mage_mlx.mflux_src.mflux.models.common.weights.saving.model_saver import ModelSaver
from mage_mlx.mflux_src.mflux.models.mage_flow.latent_creator import MageFlowLatentCreator
from mage_mlx.mflux_src.mflux.models.mage_flow.mage_flow_initializer import MageFlowInitializer
from mage_mlx.mflux_src.mflux.models.mage_flow.model.mage_flow_text_encoder import MageFlowTextEncoder
from mage_mlx.mflux_src.mflux.models.mage_flow.model.mage_flow_text_encoder.policy import (
    FilterVerdict,
    make_refusal_image,
)
from mage_mlx.mflux_src.mflux.models.mage_flow.model.mage_flow_transformer import MageFlowTransformer
from mage_mlx.mflux_src.mflux.models.mage_flow.model.mage_flow_vae import MageVAE
from mage_mlx.mflux_src.mflux.models.mage_flow.variants.conditioning import MageFlowConditioning
from mage_mlx.mflux_src.mflux.models.mage_flow.variants.edit.util import MageFlowEditUtil
from mage_mlx.mflux_src.mflux.models.mage_flow.variants.pipeline_helpers import (
    make_velocity_predictor,
    resolve_generation_parameters,
    resolve_seed,
)
from mage_mlx.mflux_src.mflux.models.mage_flow.weights import MageFlowWeightDefinition
from mage_mlx.mflux_src.mflux.utils.exceptions import StopImageGenerationException
from mage_mlx.mflux_src.mflux.utils.generated_image import GeneratedImage
from mage_mlx.mflux_src.mflux.utils.image_util import ImageUtil

ImageInput = Path | str | Image.Image
ReferenceCacheKey = tuple[tuple[tuple[int, int], bytes], ...]


class MageFlowEdit(nn.Module):
    """Native MLX multi-reference image-edit pipeline for Mage-Flow Edit."""

    vae: MageVAE
    transformer: MageFlowTransformer
    text_encoder: MageFlowTextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        load_dit_vae: bool = True,
        text_encoder: Optional[MageFlowTextEncoder] = None,
    ):
        super().__init__()
        self._model_config = model_config or ModelConfig.mage_flow_edit()
        self._quantize = quantize
        self._model_path = model_path
        MageFlowInitializer.init(
            model=self,
            model_config=self._model_config,
            quantize=quantize,
            model_path=model_path,
            load_dit_vae=load_dit_vae,
            text_encoder=text_encoder,
        )

    def load_dit_vae(self, model_dir: str | None = None, quantize: int | None = None, profiler=None) -> None:
        """Lazy-load DiT and VAE after text encoding is complete.

        When ``text_encoder`` is already loaded (pre-loaded by the worker),
        the text_encoder component is skipped during weight loading since
        the edit model does not include text_encoder weights (they are
        shared from the base model).
        """
        if profiler is not None:
            profiler.start("dit_load")
        from mage_mlx.mflux_src.mflux.models.common.weights.loading.weight_applier import WeightApplier

        model_path = model_dir or self._model_path
        root_path = MageFlowInitializer._resolve_model_path(model_path)
        # Skip text_encoder if already loaded (shared from base model)
        skip_components = {"text_encoder"} if self.text_encoder is not None else None
        weights = MageFlowInitializer._load_weights(root_path, skip_components=skip_components)
        self.vae = MageVAE(sample_posterior=True)
        self.transformer = MageFlowTransformer(**self._model_config.transformer_overrides)
        vae_weights = weights.components["vae"]
        MageFlowWeightDefinition.prepare_vae_for_loading(self.vae, vae_weights)
        self.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize if quantize is not None else self._quantize,
            weight_definition=MageFlowWeightDefinition,
            models={
                "vae": self.vae,
                "transformer": self.transformer,
            },
        )
        MageFlowWeightDefinition.finalize_vae_after_loading(self.vae, vae_weights)
        del weights
        mx.eval(self.vae, self.transformer)
        mx.clear_cache()
        if profiler is not None:
            profiler.stop("dit_load")
            profiler.start("vae_load")
            profiler.stop("vae_load")

    def generate_image(
        self,
        seed: int,
        prompt: str,
        image_paths: ImageInput | list[ImageInput],
        num_inference_steps: int | None = None,
        height: int | None = None,
        width: int | None = None,
        max_size: int | None = None,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        renormalization: bool = False,
        gaussian_shading_key: int | str | None = None,
        scheduler: str = "mage_flow",
        profiler=None,
    ) -> GeneratedImage:
        seed = resolve_seed(seed)
        num_inference_steps, guidance = resolve_generation_parameters(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
        )
        references = MageFlowEditUtil.load_references(image_paths)
        width, height = MageFlowEditUtil.resolve_target_size(
            references[0],
            width=width,
            height=height,
            max_size=max_size,
        )

        raw_paths = image_paths if isinstance(image_paths, list) else [image_paths]
        metadata_paths = [path for path in raw_paths if isinstance(path, (str, Path))]
        primary = raw_paths[0] if raw_paths else None
        primary_path = primary if isinstance(primary, (str, Path)) else None
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=primary_path,
            scheduler=scheduler,
        )
        reference_key = self._reference_cache_key(references)
        verdict = self._screen_edit(
            prompt=prompt,
            references=references,
            reference_key=reference_key,
        )
        if verdict.violates:
            print(verdict.banner())
            return self._refusal_result(
                verdict=verdict,
                config=config,
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                primary_path=primary_path,
                metadata_paths=metadata_paths,
            )

        # Encode text FIRST (Qwen only), then unload, then load DiT+VAE
        text_embeddings, text_attention_mask = self._encode_prompt_pair(
            prompt=prompt,
            negative_prompt=negative_prompt,
            references=references,
            reference_key=reference_key,
            guidance=guidance,
        )
        mx.eval(text_embeddings, text_attention_mask)

        # Unload Qwen — it's only needed for prompt encoding
        self.text_encoder.unload()
        import gc
        gc.collect()
        mx.clear_cache()

        # Lazy-load DiT+VAE (after Qwen is unloaded)
        if self.vae is None or self.transformer is None:
            self.load_dit_vae(profiler=profiler)

        target_latents = MageFlowLatentCreator.create_noise(
            seed=seed,
            height=config.height,
            width=config.width,
            gaussian_shading_key=gaussian_shading_key,
            dtype=ModelConfig.precision,
        )
        reference_latents = MageFlowEditUtil.encode_references(
            self.vae,
            references,
            width=config.width,
            height=config.height,
            seed=seed,
        )
        mx.eval(target_latents, reference_latents, text_embeddings, text_attention_mask)

        latent_height = config.height // 16
        latent_width = config.width // 16
        target_length = target_latents.shape[1]
        image_shapes = [(1, latent_height, latent_width)] * (1 + len(references))
        predict = make_velocity_predictor(
            transformer=self.transformer,
            text_embeddings=text_embeddings,
            text_attention_mask=text_attention_mask,
            image_shapes=image_shapes,
            guidance=guidance,
            target_length=target_length,
            renormalization=renormalization,
        )

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(target_latents)
        for step in config.time_steps:
            try:
                if profiler is not None:
                    profiler.start(f"edit_step_{step + 1}")
                model_input = mx.concatenate([target_latents, reference_latents], axis=1)
                velocity = predict(model_input, config.scheduler.sigmas[step])
                target_latents = config.scheduler.step(
                    noise=velocity,
                    timestep=step,
                    latents=target_latents,
                    sigmas=config.scheduler.sigmas,
                )
                ctx.in_loop(step, target_latents)
                mx.eval(target_latents)
                if profiler is not None:
                    profiler.stop(f"edit_step_{step + 1}")
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(step, target_latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {step + 1}/{config.num_inference_steps}"
                )
        # The predictor closes over the transformer. Release it and the final
        # evaluated denoising graph before low-RAM callbacks evict the model.
        del predict, velocity, model_input
        ctx.after_loop(target_latents)

        if profiler is not None:
            profiler.start("vae_decode")
        decoded = self.vae.decode(
            MageFlowLatentCreator.unpack_latents(
                target_latents,
                height=config.height,
                width=config.width,
            )
        )
        mx.eval(decoded)
        if profiler is not None:
            profiler.stop("vae_decode")
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            image_path=primary_path,
            image_paths=metadata_paths or None,
            generation_time=config.time_steps.format_dict["elapsed"],
        )

    def _encode_prompt_pair(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        references: list[Image.Image],
        reference_key: ReferenceCacheKey | None = None,
        guidance: float,
    ) -> tuple[mx.array, mx.array]:
        normalized_negative = negative_prompt if negative_prompt and negative_prompt.strip() else " "
        reference_key = reference_key or self._reference_cache_key(references)
        cache_key = (prompt, normalized_negative, guidance, reference_key)
        cached = self.prompt_cache.get(cache_key)
        if cached is not None:
            return cached

        prompts = [normalized_negative, prompt] if guidance > 1.0 else [prompt]
        image_groups = [references] * len(prompts)
        result = MageFlowConditioning.encode_edit(
            prompts=prompts,
            images_per_prompt=image_groups,
            tokenizer=self.tokenizers["mage"],
            text_encoder=self.text_encoder,
            max_sequence_length=self.model_config.max_sequence_length or 2048,
        )
        mx.eval(*result)
        self.prompt_cache[cache_key] = result
        self.prompt_cache[prompt] = result
        return result

    def _screen_edit(
        self,
        *,
        prompt: str,
        references: list[Image.Image],
        reference_key: ReferenceCacheKey,
    ) -> FilterVerdict:
        cache = getattr(self, "policy_cache", None)
        if cache is None:
            cache = self.policy_cache = {}
        cache_key = ("edit", prompt, reference_key)
        verdict = cache.get(cache_key)
        if verdict is None:
            verdict = self.text_encoder.screen_edit(
                prompt,
                references,
                self.tokenizers["mage"],
            )
            cache[cache_key] = verdict
            mx.clear_cache()
        return verdict

    @staticmethod
    def _reference_cache_key(references: list[Image.Image]) -> ReferenceCacheKey:
        return tuple((reference.size, hashlib.sha256(reference.tobytes()).digest()) for reference in references)

    def _refusal_result(
        self,
        *,
        verdict: FilterVerdict,
        config: Config,
        seed: int,
        prompt: str,
        negative_prompt: str | None,
        primary_path: Path | str | None,
        metadata_paths: list[Path | str],
    ) -> GeneratedImage:
        return GeneratedImage(
            image=make_refusal_image(verdict, height=config.height, width=config.width),
            model_config=config.model_config,
            seed=seed,
            prompt=prompt,
            steps=config.num_inference_steps,
            guidance=config.guidance,
            precision=config.precision,
            quantization=self.bits,
            generation_time=0.0,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            height=config.height,
            width=config.width,
            image_path=primary_path,
            image_paths=metadata_paths or None,
            negative_prompt=negative_prompt,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=MageFlowWeightDefinition,
        )
