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
        self.inputs = []                                 # x that entered each layer
        self.hidden_activation.zs = []                   # pre-activations, consumed in reverse
        for W, b in zip(self.W[:-1], self.b[:-1]):       # every layer except the last
            self.inputs.append(x)                        # (N, in) entering this layer
            x = self.hidden_activation.apply(x @ W + b)
        self.inputs.append(x)                            # (N, in) entering the output layer
        W, b = self.W[-1], self.b[-1]
        return self.output_activation.apply(x @ W + b)
    
    def backward(self): 
        grad = self.loss_fn.gradient()                         # (N, C)
        grad = self.output_activation.chain(grad)              # (N, C), now w.r.t. the output sum
        last = len(self.W) - 1
        self.dW = [None] * len(self.W)
        self.db = [None] * len(self.W)
        for i in range(last, -1, -1):                          # last, last-1, ..., 0
            if i < last:
                grad = self.hidden_activation.chain(grad)      # through ReLU, before the weight gradient
            x = self.inputs[i]                                 # (N, in)
            self.dW[i] = x.T @ grad                            # (in, out), same shape as W[i]
            self.db[i] = grad.sum(axis=0)                      # (out,), same shape as b[i]
            grad = grad @ self.W[i].T                         # (N, in), gradient for the previous layer

    def step(self, lr):
        for i in range(len(self.W)):
            self.W[i] -= lr * self.dW[i]        # (in, out)
            self.b[i] -= lr * self.db[i]        # (out,)
                
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
    def __init__(self):
        self.zs = []

    def apply(self, z):
        self.zs.append(z)                                # (N, H), saved for chain
        return np.maximum(0, z)

    def chain(self, grad):
        z = self.zs.pop()                                # hidden layers are visited in reverse
        return grad * (z > 0)                            # (N, H); blocked where z was negative

class SoftMax:
    def apply(self, z):
        # exp(z_k - m) / sum_j exp(z_j - m)  =  exp(z_k) / sum_j exp(z_j)
        shifted = z - z.max(axis=1, keepdims=True)       
        exp = np.exp(shifted)                           
        self.probs = exp / exp.sum(axis=1, keepdims=True)
        return self.probs

    def chain(self, grad):
        # dL/dz_j = p_j * (dL/dp_j - sum_k dL/dp_k * p_k)
        dot = np.sum(grad * self.probs, axis=1, keepdims=True)  # (N, 1)
        return self.probs * (grad - dot)                        # (N, C)

class CrossEntropy:
    def loss(self, y_hat, y):
        self.y_hat = y_hat                              # (N, C) probabilities
        self.y = y                                      # (N,) true class indexes
        chosen = y_hat[np.arange(len(y)), y]            # (N,) probability of the true class
        return -np.mean(np.log(chosen + 1e-12))         # scalar

    def gradient(self):
        grad = np.zeros_like(self.y_hat)                # (N, C)
        n = len(self.y)
        rows = np.arange(n)
        p = np.maximum(self.y_hat[rows, self.y], 1e-12)
        grad[rows, self.y] = -1.0 / (n * p)              # d(-mean log p)/dp for the true class
        return grad

if __name__ == "__main__":
    # example usage
    x = np.array([[0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0]])           
    y = np.array([0, 1, 1, 0])          

    model = MLP(sizes=(2, 4, 2),
                hidden_activation=ReLU(),
                output_activation=SoftMax(),
                loss_fn=CrossEntropy())          # 2 inputs, hidden layer of 4, 2 classes
    model.fit(x, y, epochs=2000, lr=0.1)
    pred = model.predict(x)                      # class index for each of the four rows
    print(y)
    print(pred)             
