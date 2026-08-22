#!/usr/bin/env python3
"""Extract the paper's figures from the arXiv PDF as transparent-background PNGs.

Usage:  python extract_figures.py paper.pdf [--dpi 600]

Rendering with alpha=True leaves unpainted regions transparent, but each figure
also paints its own white background rectangle. So we additionally unmix opaque,
bright, near-neutral pixels (white fill, grey gridlines, text antialiasing) into
black-with-proportional-alpha, which composites cleanly over any page colour.
Saturated content (plot lines, confidence bands, pale panel fills) is untouched.
"""
import argparse

import numpy as np
import pymupdf
from PIL import Image

# name, page index, clip box in PDF points (y0, y1, x0, x1), white threshold.
# fig1's "Weight Noise" panel is an intentional flat grey (229), so only
# near-white counts as background there; the plots can use a looser threshold.
FIGURES = [
    ("fig1_teaser",    1, 60, 200, 100, 512, 240),
    ("fig2_method",    4, 60, 282, 100, 512, 170),
    ("fig3_mechanism", 6, 60, 189, 100, 512, 170),
    ("fig4_code",      7, 60, 192,  95, 348, 170),
    ("fig5_ablations", 8, 60, 177, 100, 512, 170),
]


def extract(doc, name, page, y0, y1, x0, x1, thr, dpi, out_dir):
    pix = doc[page].get_pixmap(dpi=dpi, clip=pymupdf.Rect(x0, y0, x1, y1), alpha=True)
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 4).astype(np.int16)

    # The painted background is sliced mid-pixel at the clip boundary, leaving a
    # ring of semi-transparent white that would survive as a thin grey border.
    edge = max(2, round(dpi / 75))
    im[:edge] = 0
    im[-edge:] = 0
    im[:, :edge] = 0
    im[:, -edge:] = 0

    rgb, alpha = im[..., :3], im[..., 3]
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    neutral = (alpha == 255) & (spread <= 14) & (rgb.min(axis=2) >= thr)

    out = im.astype(np.uint8).copy()
    out[..., 3] = np.where(neutral, (255 - rgb.mean(axis=2)).astype(np.uint8), alpha.astype(np.uint8))
    for c in range(3):
        out[..., c] = np.where(neutral, 0, rgb[..., c].astype(np.uint8))

    # Drop faint neutral partial-alpha residue left over from the clip edge.
    ghost = (alpha > 0) & (alpha < 255) & (spread <= 14) & (rgb.max(axis=2) >= alpha - 20)
    out[..., 3] = np.where(ghost, 0, out[..., 3])

    img = Image.fromarray(out, "RGBA")
    bbox = img.getchannel("A").getbbox()
    if bbox:
        pad = round(dpi / 25)
        img = img.crop((max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                        min(img.width, bbox[2] + pad), min(img.height, bbox[3] + pad)))
    path = f"{out_dir}/{name}.png"
    img.save(path, optimize=True)
    return img, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out", default=f"{__file__.rsplit('/', 2)[0]}/static/images")
    args = ap.parse_args()

    doc = pymupdf.open(args.pdf)
    for name, *box, thr in FIGURES:
        img, path = extract(doc, name, *box, thr, args.dpi, args.out)
        a = np.asarray(img)[..., 3]
        frame = np.concatenate([a[:10].ravel(), a[-10:].ravel(), a[:, :10].ravel(), a[:, -10:].ravel()])
        print(f"{name}: {img.size[0]}x{img.size[1]}, transparent {(a == 0).mean():.0%}, "
              f"edge max alpha {frame.max()}")


if __name__ == "__main__":
    main()
