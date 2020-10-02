# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# importer les librairies
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as mat

# importer le dataset 
dataset = pd.read_csv("../input/train.csv")

# Detecting Nan

def isnan(dataframe, column):
    for i in range(0, column):
            if dataframe.iloc[:,i].isnull().any() == True:
                print("Column ", i, "has Nan")
                

isnan(dataset, 81)

# Replacing nan for strings
                
for i in range(0, 81):
    if type(dataset.iloc[0, i]) == str or np.isnan(dataset.iloc[0,i]):
        if dataset.iloc[:,i].isnull().any() == True:
            dataset.iloc[:, i] = dataset.iloc[:, i].replace(np.nan, "None", regex = True)
        

# Spliting Dataset into independant & Dependant Variables
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, 80:81].values


# Replacing Missing Values for numbers
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy = "mean", axis = 0)
imputer.fit(X[:, [2, 25, 58]])
X[:,[2, 25, 58]] = imputer.transform(X[:, [2, 25, 58]])


# Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
k = []
for i in range(0,79):
    if type(X[0,i]) == str:
        k += [i]
        X[:,i] = labelencoder.fit_transform(X[:,i])
        
onehotencoder = OneHotEncoder(categorical_features = [k])
X = onehotencoder.fit_transform(X).toarray()

# Séparer entre training set et test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor_rd = RandomForestRegressor(n_estimators = 2000)
y_train = np.ravel(y_train)
regressor_rd.fit(X_train, y_train)

y_pred_rd = regressor_rd.predict(X_test)


# Testing accuracy of each model

accuracy_rd = []

for i in range (0, 292):
    if y_test[i] - y_pred_rd[i] < 0:
        accuracy_rd.append(y_pred_rd[i] - y_test[i])
    else:
        accuracy_rd.append(y_test[i] - y_pred_rd[i])
        
accuracy_rd = np.asarray(accuracy_rd)
accuracy_rd = accuracy_rd.mean()


# Applying model to test.csv
test = pd.read_csv("../input/test.csv")

# Preprocessing test file

# Replacing nan for strings
                
for i in range(0, 80):
    if type(test.iloc[0, i]) == str or np.isnan(test.iloc[0,i]):
        if test.iloc[:,i].isnull().any() == True:
            test.iloc[:, i] = test.iloc[:, i].replace(np.nan, "None", regex = True)
                   
isnan(test, 80)

# Replacing Missing Values for numbers
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy = "median", axis = 0)
imputer.fit(test.iloc[:, [3, 26, 34, 36, 37, 38]])
test.iloc[:,[3, 26, 34, 36, 37, 38]] = imputer.transform(test.iloc[:, [3, 26, 34, 36, 37, 38]])

imputer.fit(test.iloc[:, [47, 48, 59, 61, 62]])
test.iloc[:, [47, 48, 59, 61, 62]] = imputer.transform(test.iloc[:,[47, 48, 59, 61, 62]])


# Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
p = []
for i in range(0,80):
    if type(test.iloc[0,i]) == str:
        p += [i]
        test.iloc[:,i] = labelencoder.fit_transform(test.iloc[:,i])

onehotencoder = OneHotEncoder(categorical_features = [p])
test = onehotencoder.fit_transform(test).toarray()

test = test[:,1:]

y_pred_rd_testset = regressor_rd.predict(test)

y_pred_to_df = pd.DataFrame(y_pred_rd_testset)
y_pred_to_df.to_csv("submissions_final.csv")

# Any results you write to the current directory are saved as output.