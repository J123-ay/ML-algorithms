# IMPORTING THE LIBRARIES

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Read Dataset
dataset = pd.read_csv("../Position_Salaries.csv")

# Split in X features and Y label
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# SPlit data in training data and test data
X_train, X_test, Y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Initiating KNeighborsRegressor and Fitting training data
clf = KNeighborsRegressor(n_neighbors=2)
clf.fit(X, y)

# Predicting test data
y_predicted = clf.predict(X_test)

# Evaluating accuracy and score
score = clf.score(X_train, Y_train)
print('Score: {}'.format(score))

mae = mean_absolute_error(y_test, y_predicted)
print('Mean Absolute Error: {}'.format(mae))

mse = mean_squared_error(y_test, y_predicted)
print('Mean Absolute Error: {}'.format(mse))
