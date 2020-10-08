# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset

dataset=pd.read_csv("/content/50_Startups.csv")
print(dataset)

 X=dataset.iloc[:, :-1].values
 print(X)
 
 Y=dataset.iloc[:,4].values
 print(Y)
 
 # Encoding Categorical Data

from sklearn.preprocessing import LabelEncoder
label_encoder_X=LabelEncoder()
X[:,3]=label_encoder_X.fit_transform(X[:,3])

print(X)

#Dummy Encoding 

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Avoiding the Dummy Variable Trap

X=X[:,1:]
print(X)

# SPLITING THE DATASET INTO TRAINING AND TESTING

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

# FITTING THE MULTIPLE REGRESSION TO THE TRAINING SET

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

# PREDICTING THE TEST SET safasf

Y_pred=regressor.predict(X_test)
print(Y_pred)

# BUILDING THE OPTIMAL MODEL BY BACKWARD ELIMINATION

import statsmodels.api as sm
X=np.append(arr=np.ones([50,1]).astype(int),values=X,axis=1)
print(X)
X_opt=X[: ,[0,1,2,3,4,5]]
print(X_opt)

regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[: ,[0,1,2,3,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,2,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

# AUTOMATIC BACKWARD ELIMINATION

import statsmodels.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
print(X_Modeled)


