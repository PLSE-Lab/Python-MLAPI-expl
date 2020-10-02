# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

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
imputer = Imputer(missing_values="NaN", strategy = "median", axis = 0)

k = []
for i in range(0, 79):
    if type(X[:,i].any()) == int or type(X[:,i].any()) == float:
        if X[:,i].sum() != X[:,i].sum():
            k += [i]
imputer.fit(X[:, k])
X[:,k] = imputer.transform(X[:, k])


# Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for i in range(0,79):
    if type(X[0,i]) == str:        
        X[:,i] = labelencoder.fit_transform(X[:,i])
    

# Séparer entre training set et test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

# linear Regression Model

from sklearn.linear_model import LinearRegression
regressor_lr = LinearRegression()
regressor_lr.fit(X_train, y_train)

y_pred_lr = regressor_lr.predict(X_test)


# Testing accuracy of each model
accuracy_lr = []
        
for i in range (0, 292):
    if y_test[i] - y_pred_lr[i] < 0:
        accuracy_lr.append(y_pred_lr[i] - y_test[i])
    else:
        accuracy_lr.append(y_test[i] - y_pred_lr[i])


accuracy_lr = np.asarray(accuracy_lr)
accuracy_lr.mean()

X = X.astype(float)

# OLS 
import statsmodels.formula.api as sm
X = np.append(X, np.ones((1460,1)).astype(int), axis=1) 
regressor_OLS = sm.OLS(endog = Y, exog = X).fit()
regressor_OLS.summary()

# Applying model to test.csv
test = pd.read_csv("../input/test.csv")

# Preprocessing test file

# Replacing nan for strings
                
for i in range(0, 80):
    if type(test.iloc[0, i]) == str or np.isnan(test.iloc[0,i]):
        if test.iloc[:,i].isnull().any() == True:
            test.iloc[:, i] = test.iloc[:, i].replace(np.nan, "None", regex = True)
                      

kaggle = test.iloc[:, 1:].values

# Replacing Missing Values for numbers
k = []
for i in range(0, 79):
    if type(kaggle[:,i].any()) == int or type(kaggle[:,i].any()) == float:
        if kaggle[:,i].sum() != kaggle[:,i].sum():
            k += [i]
        
imputer.fit(kaggle[:, k])
kaggle[:,k] = imputer.transform(kaggle[:, k])


# Encoding Categorical Variables
k = []
for i in range(0,79):
    if type(kaggle[0,i]) == str:        
        kaggle[:,i] = labelencoder.fit_transform(kaggle[:,i])
        k += [i]
        

y_pred_kaggle = regressor_lr.predict(kaggle)
y_pred_kaggle = pd.DataFrame(y_pred_kaggle)
y_pred_kaggle.to_csv("submissions_final.csv")
