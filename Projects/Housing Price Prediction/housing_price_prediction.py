# Importing the essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# Loading the dataset
housing = pd.read_csv('housing.csv')

# Checking some sample entries from the dataset
housing.head()

# Checking data types, total entries and missing values
housing.info()

# Getting some basic statistics about the dataset
housing.describe()

# Plotting histograms to understand the data distribution
housing.hist(bins=50, figsize=(20, 15))
plt.show()

# Performing stratified train test split based on CHAS
split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Working on a copy of the training data for analysis
housing = strat_train_set.copy()

# Checking correlation of other features with MEDV
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

# Plotting some important features to see relationships
attributes = ['MEDV', 'RM', 'ZN', 'LSTAT']
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()

# RM has a strong positive correlation with MEDV
housing.plot(kind='scatter', x='RM', y='MEDV', alpha=0.8)
plt.show()

# Creating our own feature
# Tax per room may be more useful than TAX alone
housing['TAXRM'] = housing['TAX'] / housing['RM']

# Checking correlation again after adding the new feature
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

housing.plot(kind='scatter', x='TAXRM', y='MEDV', alpha=0.8)
plt.show()

# Adding the same feature to both train and test sets
strat_train_set['TAXRM'] = strat_train_set['TAX'] / strat_train_set['RM']
strat_test_set['TAXRM'] = strat_test_set['TAX'] / strat_test_set['RM']

# Separating features and labels
housing = strat_train_set.drop('MEDV', axis=1)
housing_labels = strat_train_set['MEDV'].copy()

# Creating a pipeline
# Missing values -> Scaling
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

# Preparing the training data
housing_prepared = my_pipeline.fit_transform(housing)

# Training the model
model = RandomForestRegressor(random_state=42)
model.fit(housing_prepared, housing_labels)

# Checking training error
housing_predictions = model.predict(housing_prepared)

mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

print("Training RMSE :", rmse)

# Using cross validation for a better estimate
scores = cross_val_score(
    model,
    housing_prepared,
    housing_labels,
    scoring='neg_mean_squared_error',
    cv=10
)

rmse_scores = np.sqrt(-scores)

# Function to print score details
def print_scores(scores):
    print("Scores :", scores)
    print("Mean :", scores.mean())
    print("Standard Deviation :", scores.std())

print_scores(rmse_scores)

# Saving the trained model
dump(model, 'HousingPricePredictor.joblib')

# Testing on unseen data
x_test = strat_test_set.drop('MEDV', axis=1)
y_test = strat_test_set['MEDV'].copy()

x_test_prepared = my_pipeline.transform(x_test)

final_predictions = model.predict(x_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print("Final Test RMSE :", final_rmse)