"""The CNN from cnn.py, with the convolution written as a direct sliding sum.

Each output activation reads the input under its window and multiplies by
the filter in place. The windows are not stored as a matrix. One input
pixel updates every filter together, which is the same sum a tuned
convolution runs. Same data, architecture, epochs, learning rate, and
batch size as train_mnist.py.
"""

import time
from pathlib import Path

import numpy as np
from numba import njit, prange

from cnn import CNN
from mlp import CrossEntropy, ReLU, SoftMax
from train_mnist import cnn_inputs


@njit(parallel=True, fastmath=True, cache=True)
def _conv_forward(x, weight, bias, stride, pad):
    # One output activation is the window under it dotted with the filter, plus bias.
    n, channels, height, width = x.shape
    filters, _, k_h, k_w = weight.shape
    out_h = (height + 2 * pad - k_h) // stride + 1
    out_w = (width + 2 * pad - k_w) // stride + 1
    y0 = (pad + stride - 1) // stride
    x0 = (pad + stride - 1) // stride
    y1 = (height - k_h + pad) // stride
    x1 = (width - k_w + pad) // stride
    out = np.zeros((n, filters, out_h, out_w), dtype=np.float32)
    for image in prange(n):
        for filt in range(filters):
            for oy in range(y0, y1 + 1):
                ih0 = oy * stride - pad
                for ox in range(x0, x1 + 1):
                    iw0 = ox * stride - pad
                    acc = bias[filt]
                    for channel in range(channels):
                        for ky in range(k_h):
                            for kx in range(k_w):
                                acc += x[image, channel, ih0 + ky, iw0 + kx] * weight[filt, channel, ky, kx]
                    out[image, filt, oy, ox] = acc
            for oy in range(out_h):
                ih0 = oy * stride - pad
                for ox in range(out_w):
                    if y0 <= oy <= y1 and x0 <= ox <= x1:
                        continue
                    iw0 = ox * stride - pad
                    acc = bias[filt]
                    for channel in range(channels):
                        for ky in range(k_h):
                            ih = ih0 + ky
                            if ih < 0 or ih >= height:
                                continue
                            for kx in range(k_w):
                                iw = iw0 + kx
                                if 0 <= iw < width:
                                    acc += x[image, channel, ih, iw] * weight[filt, channel, ky, kx]
                    out[image, filt, oy, ox] = acc
    return out


@njit(parallel=True, fastmath=True, cache=True)
def _conv_weight_grad(x, grad, stride, pad, k_h, k_w):
    # Each filter weight sums upstream gradient times the input it was multiplied by.
    n, channels, height, width = x.shape
    _, filters, out_h, out_w = grad.shape
    y0 = (pad + stride - 1) // stride
    x0 = (pad + stride - 1) // stride
    y1 = (height - k_h + pad) // stride
    x1 = (width - k_w + pad) // stride
    partial = np.zeros((n, filters, channels, k_h, k_w), dtype=np.float32)
    for image in prange(n):
        for filt in range(filters):
            for oy in range(y0, y1 + 1):
                ih0 = oy * stride - pad
                for ox in range(x0, x1 + 1):
                    iw0 = ox * stride - pad
                    upstream = grad[image, filt, oy, ox]
                    for channel in range(channels):
                        for ky in range(k_h):
                            for kx in range(k_w):
                                partial[image, filt, channel, ky, kx] += (
                                    upstream * x[image, channel, ih0 + ky, iw0 + kx])
            for oy in range(out_h):
                ih0 = oy * stride - pad
                for ox in range(out_w):
                    if y0 <= oy <= y1 and x0 <= ox <= x1:
                        continue
                    iw0 = ox * stride - pad
                    upstream = grad[image, filt, oy, ox]
                    for channel in range(channels):
                        for ky in range(k_h):
                            ih = ih0 + ky
                            if ih < 0 or ih >= height:
                                continue
                            for kx in range(k_w):
                                iw = iw0 + kx
                                if 0 <= iw < width:
                                    partial[image, filt, channel, ky, kx] += upstream * x[image, channel, ih, iw]
    return partial.sum(axis=0)


@njit(parallel=True, fastmath=True, cache=True)
def _conv_input_grad(weight, grad, stride, pad, height, width):
    # Scatter each upstream gradient back through the filter. Overlaps add.
    n, filters, out_h, out_w = grad.shape
    _, channels, k_h, k_w = weight.shape
    y0 = (pad + stride - 1) // stride
    x0 = (pad + stride - 1) // stride
    y1 = (height - k_h + pad) // stride
    x1 = (width - k_w + pad) // stride
    dx = np.zeros((n, channels, height, width), dtype=np.float32)
    for image in prange(n):
        for filt in range(filters):
            for oy in range(y0, y1 + 1):
                ih0 = oy * stride - pad
                for ox in range(x0, x1 + 1):
                    iw0 = ox * stride - pad
                    upstream = grad[image, filt, oy, ox]
                    for channel in range(channels):
                        for ky in range(k_h):
                            for kx in range(k_w):
                                dx[image, channel, ih0 + ky, iw0 + kx] += (
                                    upstream * weight[filt, channel, ky, kx])
            for oy in range(out_h):
                ih0 = oy * stride - pad
                for ox in range(out_w):
                    if y0 <= oy <= y1 and x0 <= ox <= x1:
                        continue
                    iw0 = ox * stride - pad
                    upstream = grad[image, filt, oy, ox]
                    for channel in range(channels):
                        for ky in range(k_h):
                            ih = ih0 + ky
                            if ih < 0 or ih >= height:
                                continue
                            for kx in range(k_w):
                                iw = iw0 + kx
                                if 0 <= iw < width:
                                    dx[image, channel, ih, iw] += upstream * weight[filt, channel, ky, kx]
    return dx


@njit(parallel=True, fastmath=True, cache=True)
def _conv_forward_vec(x, weight, bias):
    # weight is (in_channels, 3, 3, filters): one input pixel updates every filter together.
    # 3x3, stride 1, padding 1. Interior pixels skip the edge checks.
    n, channels, height, width = x.shape
    filters = bias.shape[0]
    out_h = height
    out_w = width
    out = np.empty((n, filters, out_h, out_w), dtype=np.float32)
    for image in prange(n):
        acc = np.empty(filters, dtype=np.float32)
        for oy in range(1, out_h - 1):
            ih0 = oy - 1
            for ox in range(1, out_w - 1):
                iw0 = ox - 1
                for filt in range(filters):
                    acc[filt] = bias[filt]
                for channel in range(channels):
                    r0 = x[image, channel, ih0]
                    r1 = x[image, channel, ih0 + 1]
                    r2 = x[image, channel, ih0 + 2]
                    c00 = weight[channel, 0, 0]
                    c01 = weight[channel, 0, 1]
                    c02 = weight[channel, 0, 2]
                    c10 = weight[channel, 1, 0]
                    c11 = weight[channel, 1, 1]
                    c12 = weight[channel, 1, 2]
                    c20 = weight[channel, 2, 0]
                    c21 = weight[channel, 2, 1]
                    c22 = weight[channel, 2, 2]
                    x00 = r0[iw0]
                    x01 = r0[iw0 + 1]
                    x02 = r0[iw0 + 2]
                    x10 = r1[iw0]
                    x11 = r1[iw0 + 1]
                    x12 = r1[iw0 + 2]
                    x20 = r2[iw0]
                    x21 = r2[iw0 + 1]
                    x22 = r2[iw0 + 2]
                    for filt in range(filters):
                        acc[filt] += (
                            x00 * c00[filt] + x01 * c01[filt] + x02 * c02[filt]
                            + x10 * c10[filt] + x11 * c11[filt] + x12 * c12[filt]
                            + x20 * c20[filt] + x21 * c21[filt] + x22 * c22[filt]
                        )
                for filt in range(filters):
                    out[image, filt, oy, ox] = acc[filt]
        for oy in range(out_h):
            ih0 = oy - 1
            for ox in range(out_w):
                if 1 <= oy < out_h - 1 and 1 <= ox < out_w - 1:
                    continue
                iw0 = ox - 1
                for filt in range(filters):
                    acc[filt] = bias[filt]
                for channel in range(channels):
                    for ky in range(3):
                        ih = ih0 + ky
                        if ih < 0 or ih >= height:
                            continue
                        row = x[image, channel, ih]
                        for kx in range(3):
                            iw = iw0 + kx
                            if iw < 0 or iw >= width:
                                continue
                            xv = row[iw]
                            coeff = weight[channel, ky, kx]
                            for filt in range(filters):
                                acc[filt] += xv * coeff[filt]
                for filt in range(filters):
                    out[image, filt, oy, ox] = acc[filt]
    return out


@njit(parallel=True, fastmath=True, cache=True)
def _conv_weight_grad_vec(x, grad, channels):
    # grad is (N, H, W, filters). Each weight sums gradient times the input it multiplied.
    n, _, height, width = x.shape
    _, out_h, out_w, filters = grad.shape
    partial = np.zeros((n, channels, 3, 3, filters), dtype=np.float32)
    for image in prange(n):
        for oy in range(1, out_h - 1):
            ih0 = oy - 1
            for ox in range(1, out_w - 1):
                iw0 = ox - 1
                upstream = grad[image, oy, ox]
                for channel in range(channels):
                    r0 = x[image, channel, ih0]
                    r1 = x[image, channel, ih0 + 1]
                    r2 = x[image, channel, ih0 + 2]
                    p00 = partial[image, channel, 0, 0]
                    p01 = partial[image, channel, 0, 1]
                    p02 = partial[image, channel, 0, 2]
                    p10 = partial[image, channel, 1, 0]
                    p11 = partial[image, channel, 1, 1]
                    p12 = partial[image, channel, 1, 2]
                    p20 = partial[image, channel, 2, 0]
                    p21 = partial[image, channel, 2, 1]
                    p22 = partial[image, channel, 2, 2]
                    x00 = r0[iw0]
                    x01 = r0[iw0 + 1]
                    x02 = r0[iw0 + 2]
                    x10 = r1[iw0]
                    x11 = r1[iw0 + 1]
                    x12 = r1[iw0 + 2]
                    x20 = r2[iw0]
                    x21 = r2[iw0 + 1]
                    x22 = r2[iw0 + 2]
                    for filt in range(filters):
                        gv = upstream[filt]
                        p00[filt] += gv * x00
                        p01[filt] += gv * x01
                        p02[filt] += gv * x02
                        p10[filt] += gv * x10
                        p11[filt] += gv * x11
                        p12[filt] += gv * x12
                        p20[filt] += gv * x20
                        p21[filt] += gv * x21
                        p22[filt] += gv * x22
        for oy in range(out_h):
            ih0 = oy - 1
            for ox in range(out_w):
                if 1 <= oy < out_h - 1 and 1 <= ox < out_w - 1:
                    continue
                iw0 = ox - 1
                upstream = grad[image, oy, ox]
                for channel in range(channels):
                    for ky in range(3):
                        ih = ih0 + ky
                        if ih < 0 or ih >= height:
                            continue
                        row = x[image, channel, ih]
                        for kx in range(3):
                            iw = iw0 + kx
                            if iw < 0 or iw >= width:
                                continue
                            xv = row[iw]
                            slot = partial[image, channel, ky, kx]
                            for filt in range(filters):
                                slot[filt] += upstream[filt] * xv
    return partial.sum(axis=0)


@njit(parallel=True, fastmath=True, cache=True)
def _conv_input_grad_vec(weight, grad, height, width):
    # grad is (N, H, W, filters). Scatter it back through the filter; overlaps add.
    n, out_h, out_w, filters = grad.shape
    channels = weight.shape[0]
    dx = np.zeros((n, channels, height, width), dtype=np.float32)
    for image in prange(n):
        for oy in range(1, out_h - 1):
            ih0 = oy - 1
            for ox in range(1, out_w - 1):
                iw0 = ox - 1
                upstream = grad[image, oy, ox]
                for channel in range(channels):
                    for ky in range(3):
                        for kx in range(3):
                            coeff = weight[channel, ky, kx]
                            acc = np.float32(0)
                            for filt in range(filters):
                                acc += upstream[filt] * coeff[filt]
                            dx[image, channel, ih0 + ky, iw0 + kx] += acc
        for oy in range(out_h):
            ih0 = oy - 1
            for ox in range(out_w):
                if 1 <= oy < out_h - 1 and 1 <= ox < out_w - 1:
                    continue
                iw0 = ox - 1
                upstream = grad[image, oy, ox]
                for channel in range(channels):
                    for ky in range(3):
                        ih = ih0 + ky
                        if ih < 0 or ih >= height:
                            continue
                        for kx in range(3):
                            iw = iw0 + kx
                            if iw < 0 or iw >= width:
                                continue
                            coeff = weight[channel, ky, kx]
                            acc = np.float32(0)
                            for filt in range(filters):
                                acc += upstream[filt] * coeff[filt]
                            dx[image, channel, ih, iw] += acc
    return dx


@njit(parallel=True, fastmath=True, cache=True)
def _pool2(x):
    # Largest of each 2x2. The index is 0..3 in row-major order inside the window.
    n, channels, height, width = x.shape
    half_h = height // 2
    half_w = width // 2
    out = np.empty((n, channels, half_h, half_w), dtype=np.float32)
    index = np.empty((n, channels, half_h, half_w), dtype=np.int64)
    for image in prange(n):
        for channel in range(channels):
            for i in range(half_h):
                i2 = i * 2
                for j in range(half_w):
                    j2 = j * 2
                    best = x[image, channel, i2, j2]
                    which = 0
                    v = x[image, channel, i2, j2 + 1]
                    if v > best:
                        best = v
                        which = 1
                    v = x[image, channel, i2 + 1, j2]
                    if v > best:
                        best = v
                        which = 2
                    v = x[image, channel, i2 + 1, j2 + 1]
                    if v > best:
                        best = v
                        which = 3
                    out[image, channel, i, j] = best
                    index[image, channel, i, j] = which
    return out, index


@njit(parallel=True, fastmath=True, cache=True)
def _unpool2(grad, index, height, width):
    # Put each pooled gradient on the winner and leave the other three at zero.
    n, channels, half_h, half_w = grad.shape
    out = np.zeros((n, channels, height, width), dtype=np.float32)
    for image in prange(n):
        for channel in range(channels):
            for i in range(half_h):
                for j in range(half_w):
                    which = index[image, channel, i, j]
                    out[image, channel, i * 2 + which // 2, j * 2 + which % 2] = grad[image, channel, i, j]
    return out


def _filters_last(kernel):
    # (filters, channels, 3, 3) -> (channels, 3, 3, filters), contiguous along filters.
    return np.ascontiguousarray(kernel.transpose(1, 2, 3, 0))


class FastCNN(CNN):
    def conv(self, x, kernel, bias):
        # Slide the filter across the maps without storing the windows.
        if kernel.shape[-2:] == (3, 3) and self.stride == 1 and self.padding == 1:
            return _conv_forward_vec(x, _filters_last(kernel), bias)
        return _conv_forward(x, kernel, bias, self.stride, self.padding)

    def _conv_backward(self, x, kernel, grad):
        # Filter, bias, and input gradients from the same sliding sum.
        k_h, k_w = kernel.shape[-2], kernel.shape[-1]
        dbias = grad.sum(axis=(0, 2, 3))
        if (k_h, k_w) == (3, 3) and self.stride == 1 and self.padding == 1:
            weight = _filters_last(kernel)
            upstream = np.ascontiguousarray(grad.transpose(0, 2, 3, 1))
            dkernel = _conv_weight_grad_vec(x, upstream, x.shape[1])
            dkernel = np.ascontiguousarray(dkernel.transpose(3, 0, 1, 2))
            dx = _conv_input_grad_vec(weight, upstream, x.shape[2], x.shape[3])
            return dx, dkernel, dbias
        dkernel = _conv_weight_grad(x, grad, self.stride, self.padding, k_h, k_w)
        dx = _conv_input_grad(kernel, grad, self.stride, self.padding, x.shape[2], x.shape[3])
        return dx, dkernel, dbias

    def pool(self, x):
        # Keep the largest activation in each 2 x 2 and record which one won.
        self.pool_shape.append((x.shape[2], x.shape[3]))
        pooled, index = _pool2(x)
        self.pool_index.append(index)
        return pooled

    def _unpool(self, grad, index, shape):
        # Write the gradient onto the winning activation and leave the other three at zero.
        return _unpool2(grad, index, shape[0], shape[1])


def main():
    dataset = Path(__file__).resolve().parent / "mnist" / "mnist.npz"
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
    model.forward(x_train[:128])  # compile the sliding loops before the timer

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
