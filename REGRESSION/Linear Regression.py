import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("/content/Salary_Data.csv")

X=dataset.iloc[:,:-1].values

print(X)

Y=dataset.iloc[:,-1].values

print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

y_pred=regressor.predict
print(y_pred)

