
import importlib

REQUIRED = [
    "numpy", "pandas", "scipy", "sklearn", "skimage",
    "nibabel", "nilearn", "h5py", "PIL", "yaml", "torch"
]

OPTIONAL = [
    "huggingface_hub", "open_clip", "transformers", "diffusers", "timm"
]

def check(names):
    for name in names:
        try:
            mod = importlib.import_module(name)
            print(f"[OK] {name}: {getattr(mod, '__version__', 'installed')}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

if __name__ == "__main__":
    print("Required:")
    check(REQUIRED)
    print("\nOptional / reconstruction:")
    check(OPTIONAL)
