"""Classify converted 28 x 28 grayscale digit images with the scratch MLP and CNN."""

from pathlib import Path

import numpy as np
from PIL import Image

from cnn import CNN
from mlp import CrossEntropy, MLP, ReLU, SoftMax
from train_mnist import IMAGE_SIZE, cnn_inputs, mlp_inputs


def load_mlp(path):
    with np.load(path, allow_pickle=False) as checkpoint:
        sizes = checkpoint["sizes"].tolist()
        model = MLP(sizes, ReLU(), SoftMax(), CrossEntropy())
        for i in range(len(model.W)):
            model.W[i] = checkpoint[f"W{i}"]
            model.b[i] = checkpoint[f"b{i}"]
    return model


def load_cnn(path):
    with np.load(path, allow_pickle=False) as checkpoint:
        filters = tuple(int(value) for value in checkpoint["filters"])
        model = CNN(
            in_channels=int(checkpoint["in_channels"]),
            filters=filters,
            classes=int(checkpoint["classes"]),
            image_size=int(checkpoint["image_size"]),
            hidden_activation=ReLU(),
            output_activation=SoftMax(),
            loss_fn=CrossEntropy(),
            kernel_size=int(checkpoint["kernel_size"]),
            stride=int(checkpoint["stride"]),
            padding=int(checkpoint["padding"]),
        )
        for i in range(len(filters)):
            model.K[i] = checkpoint[f"K{i}"]
            model.b[i] = checkpoint[f"b{i}"]
        model.W = checkpoint["W"]
        model.b_out = checkpoint["b_out"]
    return model


def main():
    root = Path(__file__).resolve().parent / "mnist"
    digits = root / "processed_digits"
    paths = sorted(digits.glob("*.png"))
    if not paths:
        print(f"No converted PNGs found in {digits}. Run mnist/convert_digits.py first.")
        return 1

    mlp = load_mlp(root / "mlp_model.npz")
    cnn = load_cnn(root / "cnn_model.npz")
    failed = False
    print(f"{'file':<28} {'MLP':>6} {'CNN':>6}")
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

        row = pixels[None]
        mlp_digit = int(mlp.predict(mlp_inputs(row))[0])
        cnn_digit = int(cnn.predict(cnn_inputs(row))[0])
        print(f"{path.name:<28} {mlp_digit:6d} {cnn_digit:6d}")

    return int(failed)


if __name__ == "__main__":
    main()
