# IMPORTING THE LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset

dataset=pd.read_csv("/content/Position_Salaries.csv")
print(dataset)

X=dataset.iloc[: ,1:2].values
print(X)

Y=dataset.iloc[: ,2].values
print(Y)

# Fitting Dataset to Polynomial regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,Y)
print(X_poly)





