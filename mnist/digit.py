from pathlib import Path

import numpy as np
from PIL import Image

folder = Path(__file__).resolve().parent

# show an example input
mnist = np.load(folder / "mnist.npz")
image = mnist["x_train"][0]          # shape (28, 28)
print(mnist["y_train"][0])            # 5
Image.fromarray(image).resize((280, 280), Image.Resampling.NEAREST).save(folder / "digit.png")