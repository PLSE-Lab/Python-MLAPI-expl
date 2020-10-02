#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasets
dataset = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
X = dataset.iloc[:, 7].values.reshape(-1,1)
y = dataset.iloc[:, 12].values.reshape(-1,1)

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#fitting the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#table for actual and predicted results
dataset = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
dataset

#visualizing the training set results
plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'black')
plt.title('Training set for Simple Linear Regression Model')
plt.xlabel('degreep')
plt.ylabel('mbap')
plt.show()

#visualizing the test set results
plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'black')
plt.title('Test set for Simple Linear Regression Model')
plt.xlabel('degreep')
plt.ylabel('mbap')
plt.show()

#Evaluation 
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
                       