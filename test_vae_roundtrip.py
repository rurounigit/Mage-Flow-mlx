"""Quick VAE round-trip test: encode a simple image, decode it, compare."""
import mlx.core as mx
import numpy as np
from PIL import Image
from mage_mlx.pipeline import MageFlowPipeline

pipe = MageFlowPipeline.from_pretrained('models/mage_flow_mlx')

# Create a simple test image: gradient with a recognizable pattern
H, W = 512, 512
img = np.zeros((H, W, 3), dtype=np.float32)
# Horizontal gradient (red increases left to right)
img[:, :, 0] = np.linspace(0, 1, W)[None, :]
# Vertical gradient (green increases top to bottom)
img[:, :, 1] = np.linspace(0, 1, H)[:, None]
# Blue channel: a white square in the center
img[128:384, 128:384, 2] = 1.0

# Convert to [-1, 1]
img_norm = (img * 2 - 1).astype(np.float32)
img_mx = mx.array(img_norm)[None, ...]  # [1, H, W, 3]

# Encode
print("Encoding...")
latent = pipe.vae.encode(img_mx)
print(f"Latent shape: {latent.shape}, mean={float(latent.mean()):.4f}, std={float(latent.std()):.4f}")

# Decode
print("Decoding...")
decoded = pipe.vae.decode(latent)
print(f"Decoded shape: {decoded.shape}, mean={float(decoded.mean()):.4f}, std={float(decoded.std()):.4f}")

# Convert to [0, 255]
decoded_np = (np.array(decoded[0]) + 1.0) * 127.5
decoded_np = np.clip(decoded_np, 0, 255).astype(np.uint8)

# Save both images for comparison
Image.fromarray((img * 255).astype(np.uint8)).save('/tmp/test_input.png')
Image.fromarray(decoded_np).save('/tmp/test_output.png')

# Compute similarity metrics - use img_norm directly (no batch dim)
input_flat = img_norm.flatten()
output_flat = np.array(decoded[0]).flatten()
mse = np.mean((input_flat - output_flat) ** 2)
psnr = 10 * np.log10(4.0 / mse) if mse > 0 else float('inf')
print(f"\nMSE: {mse:.4f}")
print(f"PSNR: {psnr:.2f} dB")

# Check spatial structure: does the output have a gradient?
print(f"\nOutput row 256 (should show gradient): min={decoded_np[256].min()}, max={decoded_np[256].max()}")
print(f"Output col 256 (should show gradient): min={decoded_np[:, 256].min()}, max={decoded_np[:, 256].max()}")
print(f"Center square (should be bright blue): mean={decoded_np[256, 256, :].mean():.0f}")

# Check if output is just noise (std should be high for noise, lower for structured)
print(f"\nOutput std: {decoded_np.std():.2f}")
print(f"Input std: {img.std():.2f}")
