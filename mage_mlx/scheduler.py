"""FlowMatchEulerDiscreteScheduler for Mage-Flow (MLX port).

Port of diffusers' FlowMatchEulerDiscreteScheduler with static shift=6.0,
following the mflux MLX pattern.

The scheduler computes sigmas using diffusers' static rational shift:
  sigma = shift * t / (1 + (shift - 1) * t)

For Mage-Flow-Turbo (4-step distilled), the schedule is:
  base_sigmas = linspace(1.0, 1/4, 4)
  shifted = shift * base_sigmas / (1 + (shift - 1) * base_sigmas)
  sigmas = [shifted..., 0.0]  (append terminal 0)
"""

from __future__ import annotations

import math
from functools import partial

import mlx.core as mx


@partial(mx.compile, shapeless=True)
def _step(noise: mx.array, latents: mx.array, s1: mx.array, s2: mx.array) -> mx.array:
    """Euler step matching mflux's FP32 update before dtype restoration."""
    updated = latents.astype(mx.float32)
    updated = updated + (s1 - s2).astype(mx.float32) * noise.astype(mx.float32)
    return updated.astype(latents.dtype)


class FlowMatchEulerDiscreteScheduler:
    """Flow matching Euler discrete scheduler with static time-shift.

    Args:
        num_train_timesteps: Training timesteps (1000 for Mage-Flow)
        shift: Static shift value (6.0 for Mage-Flow)
        num_inference_steps: Number of denoising steps (4 for turbo)
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 6.0,
        num_inference_steps: int = 4,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self._num_inference_steps = num_inference_steps
        self._sigmas, self._timesteps = self._compute_timesteps_and_sigmas(
            num_inference_steps
        )

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    @staticmethod
    def _time_shift(shift: float, t: mx.array) -> mx.array:
        """Diffusers static flow shift."""
        return shift * t / (1.0 + (shift - 1.0) * t)

    def _compute_timesteps_and_sigmas(
        self, num_steps: int
    ) -> tuple[mx.array, mx.array]:
        """Compute the sigma and timestep schedules.

        Mirrors the diffusers FlowMatchEulerDiscreteScheduler with static shift:
        1. Linear sigmas: linspace(1, 1/num_steps, num_steps)
        2. Apply the static rational time-shift
        3. Append terminal 0 to sigmas
        4. Timesteps = sigmas * num_train_timesteps
        """
        # Linear base sigmas
        sigmas_linear = mx.linspace(
            1.0, 1.0 / num_steps, num_steps, dtype=mx.float32
        )

        sigmas_shifted = self._time_shift(self.shift, sigmas_linear)

        # Timesteps = sigmas * num_train_timesteps
        timesteps = sigmas_shifted * self.num_train_timesteps

        # Append terminal 0 to sigmas
        sigmas_with_zero = mx.concat(
            [sigmas_shifted, mx.zeros((1,), dtype=sigmas_shifted.dtype)], axis=0
        )

        return sigmas_with_zero, timesteps

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Recompute the schedule for a different number of inference steps."""
        self._num_inference_steps = num_inference_steps
        self._sigmas, self._timesteps = self._compute_timesteps_and_sigmas(
            num_inference_steps
        )

    def step(
        self, noise: mx.array, timestep_idx: int, latents: mx.array
    ) -> mx.array:
        """Perform one Euler step.

        x_{t+1} = x_t + (sigma_{t+1} - sigma_t) * v_t

        Args:
            noise: Model output (velocity prediction v_t)
            timestep_idx: Current step index
            latents: Current latent sample x_t

        Returns:
            Next latent sample x_{t+1}
        """
        sigmas = self._sigmas
        return _step(noise, latents, sigmas[timestep_idx + 1], sigmas[timestep_idx])

    def scale_model_input(self, latents: mx.array, t: int) -> mx.array:
        """Scale model input (no-op for flow matching)."""
        return latents
