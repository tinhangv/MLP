import numpy as np
rng = np.random.default_rng(0)

class MLP:
    def __init__(self, sizes, hidden_activation, output_activation, loss_fn):
        # sizes: (in_features, hidden, ..., out_features)
        self.W = []                              # one array per layer, W[i] is (sizes[i], sizes[i + 1])
        self.b = []                              # one array per layer, b[i] is (sizes[i + 1],)
        # initialize weights and biases
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.standard_normal((in_dim, out_dim)))    # randomize weights, between (-1,1)            
            self.b.append(np.zeros(out_dim))                         # zero biases
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss_fn = loss_fn

    def forward(self, x):                       
        for W, b in zip(self.W[:-1], self.b[:-1]):   # every layer except the last
            x = self.hidden_activation.apply(x @ W + b)
        W, b = self.W[-1], self.b[-1]                # the output layer
        return self.output_activation.apply(x @ W + b)
    
    def backward(self): ...                      # same loop, reversed
    def step(self, lr): ...                      # update every W and b

    def fit(self, x, y, epochs, lr, batch_size=None): 
        n = len(x)  # number of samples in training set
        for _ in range(epochs): # main training loop
            if batch_size is None: # regular, entire dataset gradient descent
                slices = [np.arange(n)]
            else: # stochastic gradient descent, process one epoch in batches
                order = rng.permutation(n) # shuffle the samples

                # organize into batches
                # each slice is an array of indices
                slices = [order[s:s + batch_size] for s in range(0, n, batch_size)] 
            for idx in slices:
                # predicted outputs
                # x[idx] selects samples of x with the corresponding indices, forming a batch
                y_hat = self.forward(x[idx])     
                self.loss_fn.loss(y_hat, y[idx])    
                self.backward()
                self.step(lr)

    def predict(self, x): 
        y_hat = self.forward(x)
        return np.argmax(y_hat, axis = 1)

class ReLU:
    def apply(self, x): ...
    def chain(self, grad):...

class SoftMax:
    def apply(self, z):...
    def chain(self, grad):...

class CrossEntropy:
    def loss(self, y_hat, y):...
    def gradient(self):...

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
