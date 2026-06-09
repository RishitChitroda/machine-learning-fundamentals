# Importing essentials
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Loading dataset
df = pd.read_csv('mail_data.csv')

# Handling null values
data = df.where((pd.notnull(df)), '')

# Dataset anaylzing
print(data.head())
print(data.info())
print(data.describe())

# Converting "Spam"/"Notspam" to 0 and 1
data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category',] = 1

# Assiging values
x = data['Message']
y = data['Category']
print(x)
print(y)

# Splitting training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

y_train.head()

# Feature extraction
feature_extraction = TfidfVectorizer(min_df = 1, stop_words= 'english', lowercase=True)
x_train_feature = feature_extraction.fit_transform(x_train)
x_test_feature = feature_extraction.transform(x_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Training Model
model = LogisticRegression()
model.fit(x_train_feature, y_train)

# Testing
prediction = model.predict(x_test_feature)
accuracy = accuracy_score(y_test, prediction)
print(accuracy)

# Implemeting
input_mail = ["Congratulations! You have won a free iPhone. Click here now."]

input_mail_feature = feature_extraction.transform(input_mail)

predict = model.predict(input_mail_feature)

if predict == 0:
    print("Spam Mail")
else:
    print("Ham Mail")

# Predicited : Spam Mail