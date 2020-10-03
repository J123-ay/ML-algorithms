                    IMPLEMENTATION OF SIMPLE LINEAR REGRESSION
DATA PREPROCESSING PART

IMPORTING LIBRARIES

  import numpy as np
  import matplotlib.pyplot as plt
  import pandas as pd
  
READ THE DATATSET:
  
  dataset=pd.read_csv("/content/Salary_Data.csv")

INDEPENDENT FEATURES

  X=dataset.iloc[:,:-1].values
  print(X)

DEPENDENT FEATURE
  
  Y=dataset.iloc[:,-1].values
  print(Y)

SPLITING THE DATASET INTO TRAINING AND TESTING
  
  from sklearn.model_selection import train_test_split
  X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
  print(X_train)
  print(X_test)
  print(Y_train)
  print(Y_test)

FITTING SIMPLE LINEAR REGRESSION MODEL TO THE TRAINING SET

  from sklearn.linear_model import LinearRegression
  regressor=LinearRegression()
  regressor.fit(X_train,Y_train)
  
PREEDICTING THE TEST SET
  
  y_pred=regressor.predict
  print(y_pred)

