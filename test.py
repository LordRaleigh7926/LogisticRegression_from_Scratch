# Importing our model and numpy
from LogisticRegression import LogisticRegression
import numpy as np

# Our training features. Shape = (n_sample, n_features) which is (12, 2)
x = np.array([
    [1, 1],
    [1.5, 1.5],
    [2.4, 2.4],
    [2.5, 2.5],
    [4, 4],
    [4.89, 4.89],
    [5, 5],
    [6, 6],
    [7, 7],
    [1,1],
    [0,0],
    [3,3]
])

# Training Labels
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1])

# Initializing Model
model = LogisticRegression()

# Training model
model.train(x, y)

# Making predictions
pred = model.predict(np.array([[1,1],[5,5]]))

# Printing predictions and weights and bias
print("Predictions - ", pred, "\n\nModel Weights - ", model.weights, "\n\nModel Bias - ", model.bias)



