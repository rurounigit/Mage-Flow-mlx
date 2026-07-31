"""Offline shape contracts; does not load model weights."""
import unittest
import mlx.core as mx
from mage_mlx.rope import MageFlowEmbedRope
from mage_mlx.vae import MageVAE

class EditContractTests(unittest.TestCase):
    def test_multi_image_rope_covers_every_image(self):
        cos, sin = MageFlowEmbedRope()([(1, 64, 64), (1, 64, 64)])
        self.assertEqual(cos.shape, (8192, 64))
        self.assertEqual(sin.shape, (8192, 64))

    def test_vae_pack_unpack_round_trip_shapes(self):
        latents = mx.zeros((1, 64, 64, 128), dtype=mx.bfloat16)
        packed = MageVAE.pack_latents(None, latents)
        unpacked = MageVAE.unpack_latents(None, packed, 64, 64)
        self.assertEqual(packed.shape, (1, 4096, 128))
        self.assertEqual(unpacked.shape, latents.shape)

    def test_edit_image_sequence_contract(self):
        combined = mx.concatenate([mx.zeros((1,4096,128)), mx.zeros((1,4096,128))], axis=1)
        self.assertEqual(combined.shape, (1, 8192, 128))

    def test_canonical_mageflow_noise_shape(self):
        from mage_mlx.latent_creator import MageFlowLatentCreator
        noise = MageFlowLatentCreator.create_noise(seed=42, height=256, width=256)
        self.assertEqual(noise.shape, (1, 256, 128))

if __name__ == '__main__':
    unittest.main()
