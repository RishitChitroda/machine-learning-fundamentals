# Training a Logistic Regression Classifier To Check Wheather The Flower Is Verginica Or Not

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

features = iris.data
lables = (iris["target"] == 2).astype(int)

# Training Classifier
clf = LogisticRegression()
clf.fit(features,lables)

#Test

print(clf.predict([[1.6,2,2,0]]))
# Predicted 0 : Not an verginica

print(clf.predict([[4,4,4,4]]))
# Predicted 1 : an verginica