#!/usr/bin/env python
# coding: utf-8

# 
# ## The Boston Housing Dataset
# 
# 
# 
# The Boston Housing Dataset is a derived from information collected by the U.S. Census Service concerning housing in the area of [ Boston MA](http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html). The following describes the dataset columns:
# 
#  - CRIM - per capita crime rate by town
# 
# - ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# - INDUS - proportion of non-retail business acres per town.
# - CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# - NOX - nitric oxides concentration (parts per 10 million)
# - RM - average number of rooms per dwelling
# - AGE - proportion of owner-occupied units built prior to 1940
# - DIS - weighted distances to five Boston employment centres
# - RAD - index of accessibility to radial highways
# - TAX - full-value property-tax rate per \$10,000
# - PTRATIO - pupil-teacher ratio by town
# - B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# - LSTAT - % lower status of the population
# - MEDV - Median value of owner-occupied homes in \$1000's

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:



column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']


# In[ ]:


df = pd.read_csv('../input/boston-house-prices/housing.csv',header=None, delimiter=r"\s+",names=column_names)


# In[ ]:


df.head()


# In[ ]:


X = df.drop('MEDV',axis=1).values
y = df['MEDV'].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=1)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler  = MinMaxScaler()


# In[ ]:


X_train = scaler.fit_transform(X_train)


# In[ ]:


X_test = scaler.transform(X_test)


# In[ ]:


from tensorflow.keras.models import Sequential


# In[ ]:


from tensorflow.keras.layers import Dense


# In[ ]:


model = Sequential()


# In[ ]:


model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[ ]:


model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test), epochs=600)


# In[ ]:


loss_data = pd.DataFrame(model.history.history)


# In[ ]:


loss_data.plot()


# In[ ]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[ ]:


predict= model.predict(X_test)


# In[ ]:


mean_absolute_error(y_test,predict)


# In[ ]:


np.sqrt(mean_squared_error(y_test,predict))


# In[ ]:


plt.figure(figsize=(12,6))
plt.scatter(y_test,predict)
plt.plot(y_test,y_test,'r')


# In[ ]:


single_house = df.drop('MEDV',axis=1).iloc[0]


# In[ ]:


single_house = scaler.transform(single_house.values.reshape(-1,13))


# In[ ]:


model.predict(single_house)

