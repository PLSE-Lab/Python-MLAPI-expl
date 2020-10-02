#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

correlation = dataset.corr()
fig = plt.subplots(figsize=(10,10))
sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='Reds')


# Although correlation predicts that certain features(alcohol and pH) have a relationship with quality. The data does not establish a linear relationship between any of the features.
# 

# In[ ]:


features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
for feature in features:
    sns.set()
    sns.relplot(data = dataset,x = feature,y = Y, kind = 'line', height = 7, aspect = 1)


# As shown in the above graphs no feature has a correlation with the quality of the wine. Hence the Linear regression model would not fair well in making predictions.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=4)
linear_regressor = LinearRegression()
linear_regressor.fit(x_train,y_train)
y_pred = linear_regressor.predict(x_test)

accuracy = linear_regressor.score(x_test,y_test)
print("Linear Accuracy: {}".format(accuracy))

rmse_linear = mean_squared_error(y_test,y_pred)
print("Linear RMSE: {} ".format(rmse_linear))


# We estimated the linear regression model using the equation y = a0 + x1a1 + x2a2 + x3a3+...+x11a11 where x- feature values a- model parameter. In this model, a0 is bias term and a1,...a11 are feature weights. 

# In[ ]:


from sklearn.linear_model import Lasso

model2 = Lasso()
model2.fit(X=x_train, y=y_train)
y_pred2 = model2.predict(x_test)
rmse_lasso = mean_squared_error(y_pred2, y_test)
print("Lasso RMSE: {}".format(rmse_lasso))
accuracy_lasso = model2.score(x_test,y_test)
print("Accuracy Lasso: {}".format(accuracy_lasso))


# Lasso(Least Absolute Shrinkage and Selection Operator) Regression involves regularization of data by typically constraining weights. It is used when data is overfitted.
# 

# In[ ]:


from sklearn.linear_model import Ridge

model3 = Ridge(alpha = 1 , solver = "cholesky")
model3.fit(X=x_train, y=y_train)
y_pred3 = model3.predict(x_test)
rmse_ridge = mean_squared_error(y_pred3, y_test)
print("Ridge RMSE: {}".format(rmse_ridge))
accuracy_ridge = model3.score(x_test,y_test)
print("Accuracy Ridge: {}".format(accuracy_ridge))


# Summary:
# As estimated by the correlation plot of this model, it can be established that the model does not have any linear relationship between input features and quality of wine. Although 'Alcohol' has a stron relationship with the quality of wine, it cannot be concluded that the relationship between them is linear. This model is clearly a classification model. Classifiers like K-Means and Random Forest would do well with this data.
# 
# Note:
# As the model is already scaled no scaling was needed for regularization. Regularization is to be performed when using Lasso, Ridge and other regularization models.

# 
