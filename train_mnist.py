"""Train the scratch MLP and CNN to recognize MNIST digits."""

import time
from pathlib import Path

import numpy as np

from cnn import CNN
from mlp import CrossEntropy, MLP, ReLU, SoftMax


IMAGE_SIZE = 28


def mlp_inputs(images):
    """Flatten each image to 784 pixels and scale intensities to [0, 1]."""
    return images.reshape(-1, IMAGE_SIZE ** 2).astype(np.float32) / 255.0


def cnn_inputs(images):
    """One grayscale channel per image, scaled to [0, 1]. Shape (N, 1, 28, 28)."""
    return images.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32) / 255.0


def main():
    folder = Path(__file__).resolve().parent / "mnist"
    dataset = folder / "mnist.npz"
    if not dataset.is_file():
        raise SystemExit(f"Place the downloaded MNIST file at {dataset} and run again.")

    with np.load(dataset) as mnist:
        # 60,000 training images, 10,000 test images
        x_train = mnist["x_train"]
        y_train = mnist["y_train"]
        x_test = mnist["x_test"]
        y_test = mnist["y_test"]

    # One input per pixel, 128 hidden units, and one output per digit.
    sizes = (IMAGE_SIZE ** 2, 128, 10)
    mlp = MLP(sizes, ReLU(), SoftMax(), CrossEntropy())

    # Scale the initial weights so wider layers don't start with huge activations.
    for weights in mlp.W:
        weights *= np.sqrt(2 / weights.shape[0])

    print("Training MLP for 10 epochs...", flush=True)
    started = time.perf_counter()
    mlp.fit(mlp_inputs(x_train), y_train, epochs=10, lr=0.1, batch_size=128)
    mlp_time = time.perf_counter() - started
    mlp_pred = mlp.predict(mlp_inputs(x_test))

    filters = (8, 16)
    cnn = CNN(in_channels=1, filters=filters, classes=10, image_size=IMAGE_SIZE,
              hidden_activation=ReLU(), output_activation=SoftMax(),
              loss_fn=CrossEntropy(),
              kernel_size=3, stride=1, padding=1)
    print("Training CNN for 10 epochs...", flush=True)
    started = time.perf_counter()
    cnn.fit(cnn_inputs(x_train), y_train, epochs=10, lr=0.1, batch_size=128)
    cnn_time = time.perf_counter() - started
    test_images = cnn_inputs(x_test)
    cnn_pred = np.concatenate([
        cnn.predict(test_images[start:start + 512])
        for start in range(0, len(test_images), 512)
    ])

    print()
    print(f"{'MLP accuracy':<16} {np.mean(mlp_pred == y_test):.2%}")
    print(f"{'CNN accuracy':<16} {np.mean(cnn_pred == y_test):.2%}")
    print(f"{'MLP time':<16} {mlp_time:.1f}s")
    print(f"{'CNN time':<16} {cnn_time:.1f}s")

    # Save both networks so predict_digits.py can reuse the trained weights.
    np.savez(folder / "mlp_model.npz", sizes=sizes,
             W0=mlp.W[0], b0=mlp.b[0], W1=mlp.W[1], b1=mlp.b[1])
    saved = {
        "in_channels": np.array(1),
        "filters": np.array(filters),
        "classes": np.array(10),
        "image_size": np.array(IMAGE_SIZE),
        "kernel_size": np.array(cnn.kernel_size),
        "stride": np.array(cnn.stride),
        "padding": np.array(cnn.padding),
        "W": cnn.W,
        "b_out": cnn.b_out,
    }
    for i, (kernel, bias) in enumerate(zip(cnn.K, cnn.b)):
        saved[f"K{i}"] = kernel
        saved[f"b{i}"] = bias
    np.savez(folder / "cnn_model.npz", **saved)


if __name__ == "__main__":
    main()
