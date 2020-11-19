# import the necessary package
from pyimagesearch.nn import Perceptron
import numpy as np

# Construct the 'OR' dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

# define our perceptron and train it
print('[INFO]: Training....')
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

# now that our perceptron is trained we can evaluate it
print('[INFO]: Testing....')

# now that our network is trained, loop over the data points
for (x, target) in zip(X, y):
    # make a prediction and display the result
    # to our console
    pred = p.predict(x)
    print('[INFO]: Data={}, Ground Truth={}, Prediction={}'.format(x, target[0], pred))
