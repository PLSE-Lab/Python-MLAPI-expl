#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/energy-molecule/roboBohr.csv')


# In[ ]:


df. shape


# In[ ]:


df.columns


# In[ ]:


df = df.drop(['Unnamed: 0','pubchem_id'], axis=1)


# In[ ]:


df.isnull().sum().sum()


# In[ ]:


df.Eat.describe()


# In[ ]:


sns.distplot(df['Eat'], kde=True, color="r")
plt.xlabel('Atomization Energy (kcal/mol)')
plt.ylabel ('Frequency Density')
plt.title('Atomization Energy Distribution')


# In[ ]:


Y = df['Eat']
df = df.drop(['Eat'], axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df, Y, test_size=0.3)


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso

# XGB
xgb_model = XGBRegressor(objective='reg:linear', random_state=42)
xgb_model.fit(df, Y)
predict = xgb_model.predict(df)
mse = mean_squared_error(Y, predict)
print (float(np.sqrt(mse)))

plt.figure()
sns.distplot(predict, kde=True, color='b')
plt.xlabel('Data')
plt.ylabel('Pred')
plt.title('XGB Distribution')


# In[ ]:


#RandomForest
rf_model = RandomForestRegressor()
rf_model.fit(df, Y)
predict1 = rf_model.predict(df)
mse1 = mean_squared_error(Y, predict1)
print(np.sqrt(mse1))

plt.figure()
sns.distplot(predict1, kde=True, color='y')
plt.xlabel('Data')
plt.ylabel('Pred')
plt.title('Random Forest Distribution')


# In[ ]:


#KNeighbors
kn_model = KNeighborsRegressor(weights='distance')
kn_model.fit(df, Y)
predict2 = kn_model.predict(df)
mse2 = mean_squared_error(Y, predict2)
print(np.sqrt(mse2))

plt.figure()
sns.distplot(predict2, kde=True, color='r')
plt.xlabel('Data')
plt.ylabel('Pred')
plt.title('K-Neighbors Distrubtion')


# In[ ]:


#Ridge
ridge_model = Ridge()
ridge_model.fit(df, Y)
predict3 = ridge_model.predict(df)
mse3 = mean_squared_error(Y, predict3)
print (np.sqrt(mse3))

plt.figure()
sns.distplot(predict3, kde=True, color='g')
plt.xlabel('Data')
plt.ylabel('Pred')
plt.title("RIDGE Distrubtion")


# In[ ]:


#Lasso
lasso_model = Lasso()
lasso_model.fit(df, Y)
predict4 = lasso_model.predict(df)
mse4 = mean_squared_error(Y, predict4)
print (np.sqrt(mse4))

plt.figure()
sns.distplot(predict4, kde=True, color='b')
plt.xlabel('Data')
plt.ylabel('Pred')
plt.title("LASSO Distribution")

