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


