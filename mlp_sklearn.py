"""Train the XOR example from mlp.py using scikit-learn."""

import numpy as np
from sklearn.neural_network import MLPClassifier


if __name__ == "__main__":
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])

    # The classifier infers input width and class labels from x and y.
    model = MLPClassifier(
        hidden_layer_sizes=(4,),
        activation="relu",
        solver="sgd",
        learning_rate_init=0.1,
        batch_size=len(x),
        shuffle=False,
        momentum=0.0,
        alpha=0.0,
        max_iter=2000,
        random_state=0,
    )
    model.fit(x, y)

    print(y)
    print(model.predict(x))
