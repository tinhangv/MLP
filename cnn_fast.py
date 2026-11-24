"""The CNN from cnn.py, with the same arithmetic and less overhead.

cnn.py copies each 3 x 3 neighborhood into a matrix and multiplies it. That
matrix product is the fast way to do this in NumPy. This version keeps that
product and drops the costs around it:

- parameters stay float32, and the backward pass no longer promotes them
- the kernel gradient is one matrix product instead of np.einsum
- windows built during forward are reused during backward
- the pool sends its gradient back with put_along_axis

The training run matches train_mnist.py: 60,000 images, 10 epochs,
learning rate 0.1, batches of 128.
"""

import time
from pathlib import Path

import numpy as np

from cnn import CNN
from mlp import CrossEntropy, ReLU, SoftMax
from train_mnist import cnn_inputs


class FastCNN(CNN):
    def __init__(self, in_channels, filters, classes, image_size,
                 hidden_activation, output_activation, loss_fn,
                 kernel_size=3, stride=1, padding=1):
        super().__init__(in_channels, filters, classes, image_size,
                         hidden_activation, output_activation, loss_fn,
                         kernel_size, stride, padding)
        self.K = [kernel.astype(np.float32) for kernel in self.K]
        self.b = [bias.astype(np.float32) for bias in self.b]
        self.W = self.W.astype(np.float32)
        self.b_out = self.b_out.astype(np.float32)

    def forward(self, x):
        self._saved_windows = []
        return super().forward(x)

    def conv(self, x, kernel, bias):
        windows = self._windows(x, kernel.shape[-1])          # (N, H_out, W_out, C, k, k)
        self._saved_windows.append(windows)
        patches = windows.reshape(windows.shape[0], windows.shape[1], windows.shape[2], -1)
        out = patches @ kernel.reshape(len(kernel), -1).T + bias
        return out.transpose(0, 3, 1, 2)                      # (N, filters, H_out, W_out)

    def _conv_backward(self, x, kernel, grad):
        windows = self._saved_windows.pop()
        n, out_h, out_w, channels, k, _ = windows.shape
        patches = windows.reshape(n, out_h, out_w, -1)        # (N, H_out, W_out, C * k * k)
        upstream = np.ascontiguousarray(grad.transpose(0, 2, 3, 1))
        flat_kernel = kernel.reshape(len(kernel), -1)         # (filters, C * k * k)
        # (C*k*k, N*H*W) @ (N*H*W, filters) -> (filters, C*k*k)
        dkernel = (patches.reshape(-1, patches.shape[-1]).T @ upstream.reshape(-1, grad.shape[1]))
        dkernel = dkernel.T.reshape(kernel.shape)
        dbias = grad.sum(axis=(0, 2, 3))
        scattered = (upstream @ flat_kernel).reshape(n, out_h, out_w, channels, k, k)
        return self._scatter(scattered, x.shape), dkernel, dbias

    def _scatter(self, patches, in_shape):
        n, _, height, width = in_shape
        _, out_h, out_w, channels, k, _ = patches.shape
        stride, pad = self.stride, self.padding
        ordered = np.ascontiguousarray(patches.transpose(0, 3, 4, 5, 1, 2))
        acc = np.zeros((n, channels, height + 2 * pad, width + 2 * pad), dtype=patches.dtype)
        for row in range(k):
            for col in range(k):
                acc[:, :, row:row + stride * out_h:stride, col:col + stride * out_w:stride] += (
                    ordered[:, :, row, col]
                )
        if pad:
            return acc[:, :, pad:pad + height, pad:pad + width]
        return acc

    def _unpool(self, grad, index, shape):
        n, channels, half_h, half_w = index.shape
        flat = np.zeros(index.shape + (4,), dtype=grad.dtype)
        np.put_along_axis(flat, index[..., None], grad[..., None], axis=-1)
        restored = flat.reshape(n, channels, half_h, half_w, 2, 2).transpose(0, 1, 2, 4, 3, 5)
        restored = restored.reshape(n, channels, half_h * 2, half_w * 2)
        height, width = shape
        if restored.shape[2:] == (height, width):
            return restored
        full = np.zeros((n, channels, height, width), dtype=grad.dtype)
        full[:, :, :restored.shape[2], :restored.shape[3]] = restored
        return full


def main():
    folder = Path(__file__).resolve().parent / "mnist"
    dataset = folder / "mnist.npz"
    if not dataset.is_file():
        raise SystemExit(f"Place the downloaded MNIST file at {dataset} and run again.")

    with np.load(dataset) as mnist:
        x_train = cnn_inputs(mnist["x_train"])
        y_train = mnist["y_train"]
        x_test = cnn_inputs(mnist["x_test"])
        y_test = mnist["y_test"]

    model = FastCNN(in_channels=1, filters=(8, 16), classes=10, image_size=28,
                    hidden_activation=ReLU(), output_activation=SoftMax(),
                    loss_fn=CrossEntropy(),
                    kernel_size=3, stride=1, padding=1)
    print("Training CNN for 10 epochs...", flush=True)
    started = time.perf_counter()
    model.fit(x_train, y_train, epochs=10, lr=0.1, batch_size=128)
    elapsed = time.perf_counter() - started

    predictions = np.concatenate([
        model.predict(x_test[start:start + 512])
        for start in range(0, len(x_test), 512)
    ])
    print(f"{'CNN accuracy':<16} {np.mean(predictions == y_test):.2%}")
    print(f"{'CNN time':<16} {elapsed:.1f}s")


if __name__ == "__main__":
    main()
