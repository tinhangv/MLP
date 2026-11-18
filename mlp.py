import numpy as np
from abc import ABC, abstractmethod
rng = np.random.default_rng

class MLP:
    def __init__(self, sizes, hidden_activation, output_activation, loss_fn):
        # sizes: (in_features, hidden, ..., out_features)
        self.W = []                              # one array per layer, W[i] is (sizes[i], sizes[i + 1])
        self.b = []                              # one array per layer, b[i] is (sizes[i + 1],)
        # initialize weights and biases
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.standard_normal((in_dim, out_dim)))    # randomize weights  between (-1,1)            
            self.b.append(np.zeros(out_dim))                         # zero biases
        self.loss_fn = loss_fn

    def forward(self, x):                       
        for W, b in zip(self.W[:-1], self.b[:-1]):   # every layer except the last
            x = self.hidden_activation.forward(x @ W + b)
        W, b = self.W[-1], self.b[-1]                # the output layer
        return self.output_activation.forward(x @ W + b)
    
    def loss(self, logits, y): ...
    def backward(self): ...                      # same loop, reversed
    def step(self, lr): ...                      # update every W and b
    def fit(self, x, y, epochs, lr): ...
    def predict(self, x): ...

class ReLU:
    def forward(self, x): ...
    def backward(self, grad):...

class SoftMax:
    def forward(self, z):...
    def backward(self, grad):...

class CrossEntropy:
    def forward(self, y_hat, y):...
    def backward(self):...

if __name__ == "__main__":
    # example usage
    x = np.array([[0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0]])           
    y = np.array([0, 1, 1, 0])          

    x_train = x[:3]
    y_train = y[:3]
    x_val = x[3:]

    model = MLP(sizes=(2, 4, 2), 
                hiddden_activation = ReLU(), 
                output_activation = SoftMax(), 
                loss_fn = CrossEntropy())        # 2 inputs, hidden layer of 4, 2 classes
    model.fit(x, y, epochs=200, lr=0.1)
    pred = model.predict(x_val)             
