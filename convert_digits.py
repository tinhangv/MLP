"""Convert a folder of handwritten digits to 28 x 28 grayscale PNGs."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from train_mnist import IMAGE_SIZE


def prepare_digit(path):
    """Return a centered 28 x 28 uint8 image with a dark background."""
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        gray = Image.alpha_composite(background, rgba).convert("L")

    pixels = np.asarray(gray)
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    if np.median(border) > 127:
        gray = ImageOps.invert(gray)

    # Remove the background before finding the stroke bounds.
    pixels = np.asarray(gray).astype(np.float32)
    pixels = np.maximum(pixels - np.median(np.concatenate(
        (pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1])
    )), 0)
    peak = pixels.max()
    if peak < 16:
        raise ValueError("no digit found")
    pixels[pixels < peak * 0.15] = 0
    ink = Image.fromarray(np.rint(pixels * (255 / peak)).astype(np.uint8))
    ink = ink.crop(ink.getbbox())
    scale = 20 / max(ink.size)
    size = tuple(max(1, round(side * scale)) for side in ink.size)
    ink = ink.resize(size, Image.Resampling.LANCZOS)

    mass = np.asarray(ink, dtype=np.float32)
    rows, cols = np.indices(mass.shape)
    center_x = (cols * mass).sum() / mass.sum()
    center_y = (rows * mass).sum() / mass.sum()
    center = (IMAGE_SIZE - 1) / 2
    left = max(0, min(round(center - center_x), IMAGE_SIZE - ink.width))
    top = max(0, min(round(center - center_y), IMAGE_SIZE - ink.height))
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE))
    canvas.paste(ink, (left, top))
    return np.asarray(canvas)


def main():
    root = Path(__file__).resolve().parent
    input_dir = root / "handwritten_digits"
    output_dir = root / "processed_digits"
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    supported = Image.registered_extensions()
    images = sorted(path for path in input_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in supported)
    if not images:
        print(f"Put one image per digit in {input_dir}.")
        return 0

    failed = False
    for path in images:
        output = output_dir / f"{path.name}_mnist.png"
        if output.is_file():
            print(f"Skipping {path.name} (already converted)")
            continue
        try:
            Image.fromarray(prepare_digit(path)).save(output)
        except (OSError, ValueError) as error:
            print(f"{path.name}: {error}")
            failed = True
            continue
        print(f"{path.name} -> {output}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
