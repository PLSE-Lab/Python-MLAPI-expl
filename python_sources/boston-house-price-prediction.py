#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('../input/boston-house-prices/housing.csv', header=None, delimiter=r"\s+", names=column_names)
print(data.head(5))


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


corrmat = data.corr()
corrmat


# In[ ]:


def getCorrelatedFeature(corrmat, threshold):
  feature = []
  value = []

  for i, index in enumerate(corrmat.index):
    if abs(corrmat[index])> threshold:
      feature.append(index)
      value.append(corrmat[index])

  df = pd.DataFrame(data = value, index = feature, columns = ['Corr value'])
  return df


# In[ ]:


threshold = 0.50
corr_value = getCorrelatedFeature(corrmat['MEDV'], threshold)
corr_value


# In[ ]:


corr_value.index.values


# In[ ]:


correlated_data = data[corr_value.index]
correlated_data.head()


# In[ ]:


sns.pairplot(correlated_data)
plt.tight_layout()


# In[ ]:


sns.heatmap(correlated_data.corr(), annot=True, fmt=".2")


# In[ ]:


from sklearn.model_selection import train_test_split

x = correlated_data.drop(labels=['MEDV'], axis = 1)
y = correlated_data['MEDV']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)


# In[ ]:


X_train.shape, X_test.shape


# Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)


# In[ ]:


y_predict = model.predict(X_test)


# In[ ]:


df = pd.DataFrame(data = [y_predict, y_test])
df.T


# In[ ]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

score = r2_score(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)

print('r2_score: ', score)
print('mae: ', mae)


# In[ ]:


from math import sqrt

rms_LR = sqrt(mean_squared_error(y_test, y_predict))
mse_LR = mean_squared_error(y_test, y_predict)

print('RMSE: ',rms_LR)
print('MSE: ',mse_LR)


# Decision Tree Regressor

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

DT = DecisionTreeRegressor()
DT.fit(X_train, y_train)
DT_pred = DT.predict(X_test)


# In[ ]:


rms_DT = sqrt(mean_squared_error(y_test, DT_pred))
mse_DT = mean_squared_error(y_test, DT_pred)

print('RMSE: ',rms_DT)
print('MSE: ',mse_DT)


# RandomForest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor()
RF.fit(X_train, y_train)
RF_pred = RF.predict(X_test)


# In[ ]:


rms_RF = sqrt(mean_squared_error(y_test, RF_pred))
mse_RF = mean_squared_error(y_test, RF_pred)

print('RMSE: ',rms_RF)
print('MSE: ',mse_RF)


# In[ ]:


import pandas as pd

cars = {'Model': ['Linear Regression','Decision Tree Regressor','RandomForest Regressor'],
        'RMSE': [6.2868,5.7399,5.1294], 'MSE': [39.5243,32.9464,26.3107]}

df = pd.DataFrame(cars, columns = ['Model', 'RMSE', 'MSE'])

print (df)


# In[ ]:




