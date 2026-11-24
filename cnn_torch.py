"""The same MNIST run as cnn.py, written with PyTorch modules.

scikit-learn's neural_network module only has the dense MLP, so the CNN is
built here. The MLP is the 784-128-10 network from mlp.py.
"""

import numpy as np
import torch
from torch import nn

torch.manual_seed(0)
mnist = np.load("mnist/mnist.npz")
n_train = 10000
x_train = torch.tensor(mnist["x_train"][:n_train], dtype=torch.float32).unsqueeze(1) / 255
y_train = torch.tensor(mnist["y_train"][:n_train].astype(np.int64))
x_test = torch.tensor(mnist["x_test"], dtype=torch.float32).unsqueeze(1) / 255
y_test = mnist["y_test"]

mlp = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
cnn = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 7 * 7, 10),
)

loss_fn = nn.CrossEntropyLoss()

def fit(model, epochs=5, lr=0.1, batch_size=128):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        for start in range(0, len(x_train), batch_size):
            batch_x = x_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

print("Training MLP...", flush=True)
fit(mlp)
print("Training CNN...", flush=True)
fit(cnn)

with torch.no_grad():
    mlp_pred = mlp(x_test).argmax(dim=1).numpy()
    cnn_pred = cnn(x_test).argmax(dim=1).numpy()

print()
print(f"{'image':>8} {'true':>6} {'MLP':>6} {'CNN':>6}")
for i in range(10):
    print(f"{i:8d} {int(y_test[i]):6d} {int(mlp_pred[i]):6d} {int(cnn_pred[i]):6d}")
print()
print(f"{'MLP accuracy':<16} {np.mean(mlp_pred == y_test):.1%}")
print(f"{'CNN accuracy':<16} {np.mean(cnn_pred == y_test):.1%}")
