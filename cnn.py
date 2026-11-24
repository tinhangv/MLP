import numpy as np

rng = np.random.default_rng(0)


class CNN:
    def __init__(self, in_channels, filters, classes, image_size,
                 hidden_activation, output_activation, loss_fn,
                 kernel_size=3, stride=1, padding=1):
        # Build the filters, their biases, and the final linear layer.
        # filters: (8, 16), the output channel count of each convolution.
        # stride and padding belong to the convolutions. Each pool is 2 x 2 with stride 2.
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.K = []                                  # K[i] is (out_channels, in_channels, k, k)
        self.b = []                                  # b[i] is (out_channels,), one bias per filter
        fan_channels = in_channels
        for out_channels in filters:
            fan_in = fan_channels * kernel_size * kernel_size
            kernel = rng.standard_normal(
                (out_channels, fan_channels, kernel_size, kernel_size), dtype=np.float32)
            kernel *= np.sqrt(2 / fan_in)
            self.K.append(kernel)
            self.b.append(np.zeros(out_channels, dtype=np.float32))
            fan_channels = out_channels

        side = image_size
        for _ in filters:
            side = (side + 2 * padding - kernel_size) // stride + 1
            side //= 2
        if side < 1:
            raise ValueError("kernel, stride, and padding shrink the image below one pixel")
        flat = filters[-1] * side * side
        self.W = rng.standard_normal((flat, classes), dtype=np.float32)
        self.W *= np.sqrt(2 / flat)
        self.b_out = np.zeros(classes, dtype=np.float32)

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss_fn = loss_fn

    def _windows(self, x, kernel_size):
        # Cut one window per output position.
        x = np.ascontiguousarray(x)
        if self.padding:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        n, channels, height, width = x.shape
        stride = self.stride
        out_h = (height - kernel_size) // stride + 1
        out_w = (width - kernel_size) // stride + 1
        view = np.lib.stride_tricks.as_strided(
            x,
            shape=(n, channels, kernel_size, kernel_size, out_h, out_w),
            strides=(x.strides[0], x.strides[1], x.strides[2], x.strides[3],
                     stride * x.strides[2], stride * x.strides[3]),
        )
        return np.ascontiguousarray(view.transpose(0, 4, 5, 1, 2, 3))

    def conv(self, x, kernel, bias):
        # Dot every window with every filter and add that filter's bias.
        windows = self._windows(x, kernel.shape[-1])              # (N, H_out, W_out, C, k, k)
        self._saved_windows.append(windows)
        patches = windows.reshape(windows.shape[0], windows.shape[1], windows.shape[2], -1)
        out = patches @ kernel.reshape(len(kernel), -1).T + bias  # (N, H_out, W_out, filters)
        return out.transpose(0, 3, 1, 2)                          # (N, filters, H_out, W_out)

    def _conv_backward(self, x, kernel, grad):
        # Gradient of each filter, its bias, and the input feature maps.
        windows = self._saved_windows.pop()
        n, out_h, out_w, channels, k, _ = windows.shape
        patches = windows.reshape(n, out_h, out_w, -1)            # (N, H_out, W_out, C * k * k)
        upstream = np.ascontiguousarray(grad.transpose(0, 2, 3, 1))
        flat_kernel = kernel.reshape(len(kernel), -1)             # (filters, C * k * k)
        # Sum upstream gradient times the input window over every image and position.
        dkernel = (patches.reshape(-1, patches.shape[-1]).T @ upstream.reshape(-1, grad.shape[1])).T
        dkernel = dkernel.reshape(kernel.shape)
        dbias = grad.sum(axis=(0, 2, 3))
        scattered = (upstream @ flat_kernel).reshape(n, out_h, out_w, channels, k, k)
        return self._scatter(scattered, x.shape), dkernel, dbias

    def _scatter(self, patches, in_shape):
        # Add each window's input gradient back onto the positions it covered.
        n, _, height, width = in_shape
        _, out_h, out_w, channels, k, _ = patches.shape
        stride, pad = self.stride, self.padding
        acc = np.zeros((n, channels, height + 2 * pad, width + 2 * pad), dtype=patches.dtype)
        for row in range(k):
            for col in range(k):
                acc[:, :, row:row + stride * out_h:stride, col:col + stride * out_w:stride] += (
                    patches[:, :, :, :, row, col].transpose(0, 3, 1, 2)
                )
        if pad:
            return acc[:, :, pad:pad + height, pad:pad + width]
        return acc

    def pool(self, x):
        # Keep the largest activation in each 2 x 2 and record which one won.
        n, channels, height, width = x.shape
        self.pool_shape.append((height, width))
        half_h, half_w = height // 2, width // 2
        cropped = x[:, :, :half_h * 2, :half_w * 2]
        windows = cropped.reshape(n, channels, half_h, 2, half_w, 2).transpose(0, 1, 2, 4, 3, 5)
        flat = windows.reshape(n, channels, half_h, half_w, 4)
        index = flat.argmax(axis=-1)
        self.pool_index.append(index)
        return flat.max(axis=-1)

    def _unpool(self, grad, index, shape):
        # Write the gradient onto the winning activation and leave the other three at zero.
        n, channels, half_h, half_w = index.shape
        flat = np.zeros((n, channels, half_h, half_w, 4), dtype=grad.dtype)
        rows, cols, ys, xs = np.indices(index.shape)
        flat[rows, cols, ys, xs, index] = grad
        restored = flat.reshape(n, channels, half_h, half_w, 2, 2).transpose(0, 1, 2, 4, 3, 5)
        restored = restored.reshape(n, channels, half_h * 2, half_w * 2)
        height, width = shape
        if restored.shape[2:] == (height, width):
            return restored
        full = np.zeros((n, channels, height, width), dtype=grad.dtype)
        full[:, :, :restored.shape[2], :restored.shape[3]] = restored
        return full

    def forward(self, x):
        # Conv, ReLU, and pool for each layer, then softmax of the linear layer.
        self.hidden_activation.zs = []
        self.conv_inputs = []
        self._saved_windows = []
        self.pool_index = []
        self.pool_shape = []
        for kernel, bias in zip(self.K, self.b):
            self.conv_inputs.append(x)                             # feature maps entering this convolution
            x = self.hidden_activation.apply(self.conv(x, kernel, bias))
            x = self.pool(x)
        self.flat = x.reshape(len(x), -1)                         # (N, flat)
        return self.output_activation.apply(self.flat @ self.W + self.b_out)

    def backward(self):
        # Walk the forward path in reverse and store a gradient for every parameter.
        grad = self.loss_fn.gradient()                            # (N, classes)
        grad = self.output_activation.chain(grad)                 # (N, classes), w.r.t. the output sum
        self.dW = self.flat.T @ grad                              # (flat, classes)
        self.db_out = grad.sum(axis=0)                            # (classes,)
        grad = (grad @ self.W.T).reshape(self.pool_index[-1].shape)

        self.dK = [None] * len(self.K)
        self.db = [None] * len(self.K)
        for i in range(len(self.K) - 1, -1, -1):
            grad = self._unpool(grad, self.pool_index.pop(), self.pool_shape.pop())
            grad = self.hidden_activation.chain(grad)             # through ReLU, before the kernel gradient
            grad, self.dK[i], self.db[i] = self._conv_backward(self.conv_inputs.pop(), self.K[i], grad)

    def step(self, lr):
        # Subtract learning rate times each gradient.
        for i in range(len(self.K)):
            self.K[i] -= lr * self.dK[i]
            self.b[i] -= lr * self.db[i]
        self.W -= lr * self.dW
        self.b_out -= lr * self.db_out

    def fit(self, x, y, epochs, lr, batch_size=None):
        # Repeat forward, loss, backward, and step over epochs and batches.
        n = len(x)
        for _ in range(epochs):
            if batch_size is None:
                slices = [np.arange(n)]
            else:
                order = rng.permutation(n)
                slices = [order[s:s + batch_size] for s in range(0, n, batch_size)]
            for idx in slices:
                y_hat = self.forward(x[idx])
                self.loss_fn.loss(y_hat, y[idx])
                self.backward()
                self.step(lr)

    def predict(self, x):
        # Return the class with the highest probability.
        return np.argmax(self.forward(x), axis=1)
