# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
df=pd.read_csv("../input/cattle.csv")# read data
print(df.head())
print(df.describe())#about data
print(df.info())
X=df[["year"]] # reshape for algorithm
y=df.number
plt.scatter(X,y) # real data it is blue circle
X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=0) #seperate data educaion and train
regfirst=linear_model.LinearRegression() # model  's object
print(regfirst.fit(X_train,y_train))# educaiton
print("we did education now we will test:\n")
y_predict=regfirst.predict(x_test)
print("y test :\n",y_predict)
print("our estimate's correctly:\n")
print(r2_score(y_test,y_predict))
plt.xlabel("years")
plt.ylabel("count")
plt.plot(X,regfirst.predict(X),color="red")
plt.plot(X_train,regfirst.predict(X_train),"y^")