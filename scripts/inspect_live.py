#!/usr/bin/env python
import argparse

from fmri_reconstruction.data.live import load_live_bold, validate_bold_shape


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bold", required=True)
    p.add_argument("--no-shape-check", action="store_true")
    args = p.parse_args()

    img, data = load_live_bold(args.bold)
    print("Shape:", data.shape)
    print("TR:", img.header.get_zooms()[3])

    if not args.no_shape_check:
        validate_bold_shape(data)


if __name__ == "__main__":
    main()
