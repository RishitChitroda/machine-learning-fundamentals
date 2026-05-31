# Loading required modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Loading Dataset
iris = datasets.load_iris()

# Loading features and lables
features = iris.data
lables = iris.target

# Training classifier
clf = KNeighborsClassifier()
clf.fit(features, lables)

# Testing classifier
pred = clf.predict([[1,1,1,1]])
print(pred)
# Predicted 0 : Setosa