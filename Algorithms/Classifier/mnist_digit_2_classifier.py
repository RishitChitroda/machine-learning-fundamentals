# Importing essential libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Fetching MNIST dataset
mnist = fetch_openml('mnist_784')

# Converting features and labels to NumPy arrays
x = mnist.data.to_numpy()
y = mnist.target.to_numpy()

# Displaying a digit
some_digit = x[36001]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

print(y[36001])

# Splitting data
x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]

# Shuffling training data
shuffled = np.random.permutation(60000)

x_train = x_train[shuffled]
y_train = y_train[shuffled]

# Creating a "2" detector
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train == 2)
y_test_2 = (y_test == 2)

# Creating and training the model
cls = LogisticRegression()
cls.fit(x_train, y_train_2)

# Testing
cls.predict([some_digit])

# Cross validation
a = cross_val_score(cls, x_train, y_train_2, cv=3, scoring='accuracy')

a.mean()