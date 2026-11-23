"""Train the scratch MLP to recognize MNIST digits."""

from pathlib import Path

import numpy as np

from mlp import CrossEntropy, MLP, ReLU, SoftMax


IMAGE_SIZE = 28


def model_inputs(images):
    """Flatten each image to 784 pixels and scale intensities to [0, 1]."""
    return images.reshape(-1, IMAGE_SIZE ** 2).astype(np.float32) / 255.0


def main():
    folder = Path(__file__).resolve().parent / "mnist"
    dataset = folder / "mnist.npz"
    if not dataset.is_file():
        raise SystemExit(f"Place the downloaded MNIST file at {dataset} and run again.")

    with np.load(dataset) as mnist:
        # 60,000 training images, 10,000 test images
        x_train = model_inputs(mnist["x_train"])
        y_train = mnist["y_train"]
        x_test = model_inputs(mnist["x_test"])
        y_test = mnist["y_test"]

    # One input per pixel, 128 hidden units, and one output per digit.
    sizes = (IMAGE_SIZE ** 2, 128, 10)
    model = MLP(sizes, ReLU(), SoftMax(), CrossEntropy())

    # Scale the initial weights so wider layers don't start with huge activations.
    for weights in model.W:
        weights *= np.sqrt(2 / weights.shape[0])

    print("Training for 10 epochs...", flush=True)
    model.fit(x_train, y_train, epochs=10, lr=0.1, batch_size=128)

    predictions = model.predict(x_test)
    print(f"MNIST test accuracy: {np.mean(predictions == y_test):.2%}")

    # Save both layers so predict_digits.py can reuse the trained weights.
    np.savez(folder / "mnist_model.npz", sizes=sizes,
             W0=model.W[0], b0=model.b[0], W1=model.W[1], b1=model.b[1])


if __name__ == "__main__":
    main()
