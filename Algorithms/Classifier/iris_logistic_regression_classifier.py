# Training a Logistic Regression Classifier To Check Wheather The Flower Is Verginica Or Not

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

x = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(int)

# Training Classifier
clf = LogisticRegression()
clf.fit(x,y)

#Test

print(clf.predict([[1.6]]))
# Predicted 0 : Not an verginica

print(clf.predict([[4]]))
# Predicted 1 : an verginica

# Visualizing using matplotlib
x_new = np.linspace(0,3,100).reshape(-1,1)
y_prob = clf.predict_proba(x_new)

plt.plot(x_new, y_prob[:,1], "g-", label = "Verginica")
plt.show