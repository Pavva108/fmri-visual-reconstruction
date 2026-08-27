#!/usr/bin/env python
import argparse

from fmri_reconstruction.io import inspect_hdf5, load_npy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True)
    p.add_argument("--annot")
    args = p.parse_args()

    print("HDF5 datasets:")
    for name, shape, dtype in inspect_hdf5(args.hdf5):
        print(f"{name}: shape={shape}, dtype={dtype}")

    if args.annot:
        x = load_npy(args.annot)
        print(f"Annotation: shape={x.shape}, dtype={x.dtype}")


if __name__ == "__main__":
    main()
