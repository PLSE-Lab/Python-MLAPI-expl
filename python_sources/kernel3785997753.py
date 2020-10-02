# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

dataset=pd.read_csv('../input/insurance/insurance.csv')
dataset.head()
X=dataset[['age','bmi','smoker']]
y=dataset[['charges']]
#converting categorical data into continuous 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()

X[['smoker']]=labelencoder_X.fit_transform(X[['smoker']])

#splitting the data into dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#fitting to the Linear Model

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#calculating the R2 square
y_pred=regressor.predict(X_test)
from sklearn.metrics import  r2_score,mean_squared_error
mean_squared_error(y_test,y_pred)
r2_score(y_test,y_pred)

#plotting the graph
plt.scatter(X_test.iloc[:,2].values,y_test,color='red')
plt.title("Insurance")
plt.xlabel("X_Value")
plt.ylabel("Y_Value")
plt.plot(X_test, y_pred, color='blue')
plt.show()
