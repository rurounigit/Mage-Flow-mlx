"""Check which VAE weights fail to load (strict=False hides missing weights)."""
import mlx.core as mx
from mage_mlx.vae import MageVAE

# Load the VAE weights file
vae_path = 'models/Comfy-Org_Mage-Flow-Turbo/vae.safetensors'

weights = mx.load(vae_path)

print(f"Total weights in file: {len(weights)}")

# Create the VAE model
vae = MageVAE(vae_path, sample_posterior=False)

# Get all expected parameter paths
from mlx.utils import tree_flatten
expected = dict(tree_flatten(vae.parameters()))
print(f"Total parameters in model: {len(expected)}")

# Check which weights are loaded vs missing
loaded_keys = set()
missing_keys = set()
for key, _ in expected.items():
    if key in weights:
        loaded_keys.add(key)
    else:
        missing_keys.add(key)

# Check for extra weights in file (not in model)
extra_keys = set(weights.keys()) - set(expected.keys())

print(f"\nLoaded: {len(loaded_keys)}")
print(f"Missing: {len(missing_keys)}")
print(f"Extra (in file but not model): {len(extra_keys)}")

if missing_keys:
    print("\n--- MISSING WEIGHTS ---")
    for k in sorted(missing_keys):
        print(f"  {k}")

if extra_keys:
    print("\n--- EXTRA WEIGHTS (in file but not in model) ---")
    for k in sorted(extra_keys):
        print(f"  {k}")
