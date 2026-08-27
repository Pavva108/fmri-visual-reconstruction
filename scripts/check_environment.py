import importlib

packages = [
    "numpy", "pandas", "scipy", "sklearn", "skimage", "torch",
    "torchvision", "nibabel", "nilearn", "h5py", "open_clip",
    "transformers", "diffusers"
]

for name in packages:
    try:
        importlib.import_module(name)
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
