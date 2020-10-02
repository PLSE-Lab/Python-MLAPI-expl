# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../input/usa-cers-dataset/USA_cars_datasets.csv')
X = dataset.iloc[:, [2,3,4,5,6,7,10]].values
y = dataset.iloc[:, 1].values 
 
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() 
X[:,0]=le.fit_transform(X[:,0]) 
X[:,1]=le.fit_transform(X[:,1])
X[:,3]=le.fit_transform(X[:,3])
X[:,5]=le.fit_transform(X[:,5])
X[:,6]=le.fit_transform(X[:,6])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0   )
regressor.fit( X_train, y_train)

y_pred = regressor.predict(X_test)


from sklearn.metrics import r2_score
r2_score(y_test,y_pred) 