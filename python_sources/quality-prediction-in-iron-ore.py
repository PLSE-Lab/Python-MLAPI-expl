#!/usr/bin/env python
# coding: utf-8

# # Quality Prediction in Iron Ore Mining

# Our Aim is to predict the percentage of silica in the end of the mining process of the iron ore

# ### Importing the Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Importing the Dataset

# In[ ]:


df = pd.read_csv("../input/MiningProcess_Flotation_Plant_Database.csv",decimal=",",parse_dates=["date"],infer_datetime_format=True).drop_duplicates()


# ### A basic analysis of dataset

# In[ ]:


df.head()


# * In the dataset we have to predict the  **% Silica Concentrate**
# * Silica Concentrate is the impurity in the iron ore which needs to be removed
# * The current process of detecting silica takes many hours.
# * With the help of some analysis and modelling of data we can give a good approximation of silica concentrate which will reduce a lot of time and effort required for processing iron ore

# In[ ]:


df.describe()


# In[ ]:


df = df.dropna()
df.shape


# Great! So we can see that there are no null values in the dataset.

# In[ ]:


plt.figure(figsize=(30, 25))
p = sns.heatmap(df.corr(), annot=True)


# Lets reduce the number of variables 

# In[ ]:


silic_corr = df.corr()['% Silica Concentrate']
silic_corr = abs(silic_corr).sort_values()
silic_corr


# In[ ]:


drop_index= silic_corr.index[:5].tolist()+["date"]#+["% Iron Concentrate"]
print (drop_index)


# Above plot shows the correaltions between the features.
# From the list we can find out the features which affects the % Silica Concentrate the most and discard the least important ones. 
# Note that 

# ### Preparing the Dataset

# Now we will have to drop those features which are not useful for us

# In[ ]:


df = df.drop(drop_index, axis=1)


# In[ ]:


df.head()


# In[ ]:


Y = df['% Silica Concentrate']
X = df.drop(['% Silica Concentrate'], axis=1)


# ### Scaling the features

# In[ ]:


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()


# In[ ]:


X_scaled = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)


# ### Splitting the Data

# Now we will split data into train and test set

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)


# ### Training a Model

# #### Using Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


reg = LinearRegression()


# In[ ]:


_ = reg.fit(X_train, Y_train)


# In[ ]:


predictions = reg.predict(X_test)
predictions


# Finding Mean Squared Error

# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


error = mean_squared_error(Y_test, predictions)
error


# #### Using Stochastic Gradient Descent

# In[ ]:


from sklearn.linear_model import SGDRegressor


# In[ ]:


reg_sgd = SGDRegressor(max_iter=1000, tol=1e-3)


# In[ ]:


_ = reg_sgd.fit(X_train, Y_train)


# In[ ]:


predicitons_sgd = reg_sgd.predict(X_test)


# Finding Mean Squared Error

# In[ ]:


error_sgd = mean_squared_error(Y_test, predicitons_sgd)
error_sgd


# In[ ]:


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from IPython.display import display
from sklearn import metrics


# In[ ]:


model = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, max_features=None, n_jobs=-1)
model.fit(X_train,Y_train)


# In[ ]:


predicitons_rdmforest = model.predict(X_test)
error_rdf = mean_squared_error(Y_test, predicitons_rdmforest)
error_rdf


# In[ ]:


from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
ax = plt.subplots()


# In[ ]:


pred_1, pred_2, Y_1, Y_2 = train_test_split(predicitons_rdmforest, Y_test, test_size=0.9, random_state=42)


# In[ ]:


plt.figure(figsize=(20,20))
plt.scatter(Y_1, pred_1)
plt.plot([Y_1.min(), Y_1.max()], [Y_1.min(), Y_1.max()], 'k--', lw=4)
plt.show()


# In[ ]:


plt.show(ax)


# In[ ]:




