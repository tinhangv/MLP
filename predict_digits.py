"""Classify converted 28 x 28 grayscale digit images using the scratch MLP."""

from pathlib import Path

import numpy as np
from PIL import Image

from mlp import CrossEntropy, MLP, ReLU, SoftMax
from train_mnist import IMAGE_SIZE, model_inputs


def load_model(path):
    with np.load(path, allow_pickle=False) as checkpoint:
        sizes = checkpoint["sizes"].tolist()
        model = MLP(sizes, ReLU(), SoftMax(), CrossEntropy())
        for i in range(len(model.W)):
            model.W[i] = checkpoint[f"W{i}"]
            model.b[i] = checkpoint[f"b{i}"]
    return model


def main():
    root = Path(__file__).resolve().parent / "mnist"
    digits = root / "processed_digits"
    paths = sorted(digits.glob("*.png"))
    if not paths:
        print(f"No converted PNGs found in {digits}. Run mnist/convert_digits.py first.")
        return 1

    model = load_model(root / "mnist_model.npz")
    failed = False
    for path in paths:
        try:
            with Image.open(path) as image:
                if image.mode != "L" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
                    raise ValueError("expected a converted 28 x 28 grayscale image")
                pixels = np.array(image)
        except (OSError, ValueError) as error:
            print(f"{path}: {error}")
            failed = True
            continue

        # use the trained MLP model to predict the digit
        digit = model.predict(model_inputs(pixels[None]))[0]
        print(f"{path.name}: {digit}")

    return int(failed)


if __name__ == "__main__":
    main()
