#!/usr/bin/env python
# coding: utf-8

# In[24]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Table of Contents
# 1. [Loading and Exploring the data](#section1)
# 2. [Searching for missing data](#section2)
# 3. [Selecting data for our model](#section3)
# 4. [Spliting data into train and test](#section4)
# 5. [Let's play with models](#section5)
#    - [Random Forest](#rf)
#    - [Decision Tree](#dt)
#    - [MLP](#mlp)
#    - [XGBoost](#xgboost)
# 6. [Dead or Alive? Let's predict who will survive](#section6)

# [](https://tenor.com/wvO4.gif)

# <img src="https://media.giphy.com/media/Hw8vYF4DNRCKY/giphy.gif" width="350px" height="250px"/>

# # Loading and Exploring the data
# <a id="section1"></a>

# In[25]:


df = pd.read_csv("../input/train.csv")
df.sample(2)


# In[26]:


df.groupby('Sex').count()['Survived'].plot.bar()
plt.title("Distribution by sex")
plt.show()


# In[27]:


df.groupby('Age').count()['Survived'].plot.hist(color='r', alpha=0.65, bins=30, figsize=(8, 5))
#df.groupby('Age').mean()['Survived'].plot(color='g', alpha=0.9)
#plt.legend(["ages mean", "ages"])
plt.title("Distribution by age")
plt.show()


# # Searching for missing data
# <a id="section2"></a>

# In[28]:


df.isnull().sum()


# ### deleting some missing data

# In[29]:


sex_missing = df[df.Age.isnull()].index
print(len(sex_missing))
print(sex_missing)
df.drop(sex_missing, inplace=True)


# # Selecting data for our model
# <a id="section3"></a>

# In[30]:


print(df.columns)
labels = ["Pclass", "Sex", "Age"]
print(labels)


# In[31]:


df.Sex = pd.get_dummies(df.Sex)
#df.head()


# <img src="https://media.giphy.com/media/af34tVk53Li4E/giphy.gif" width="250px" height="250px"/>

# # Spliting data into train and test
# <a id="section3"></a>

# In[32]:


from math import ceil

x = df[labels]
y = df.Survived

n_train = ceil(0.75 * df.shape[0])
n_test = ceil(0.25 * df.shape[0])

print("n_train: ", n_train)
print("n_test: ", n_test)


# In[33]:


x_train = x.iloc[:n_train,:].values
y_train = y[:n_train].values

x_test = x.iloc[n_train:,:].values
y_test = y[n_train:].values


# # Let's play with models
#  <img src="https://media.giphy.com/media/ghvWn8S0jiI0M/giphy.gif" width="480" height="203"/>
#  <a id="section5"></a>

# ### importing necessary libraries, defining our architecture and making predictions

# - Random Forest
# <a id="rf"></a>

# In[34]:


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=20, random_state=0)  

print(rf_model)

rf_model.fit(x_train, y_train)  
pred_rf = rf_model.predict(x_test)  


# - Decision Tree
# <a id="dt"></a>

# In[35]:


from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()

print(dt_model)

dt_model = dt_model.fit(x_train,y_train)
pred_dt = dt_model.predict(x_test)


# - MLP
# <a id="mlp"></a>

# In[36]:


from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 8), random_state=1, max_iter=100)

print(mlp_model)

mlp_model.fit(x_train, y_train)                         
pred_mlp =  mlp_model.predict(x_test)


# - XGBoost
# <a id="xgboost"></a>

# In[37]:


from xgboost import XGBClassifier


xgb_model = XGBClassifier()

print(xgb_model)

xgb_model.fit(x_train, y_train)
pred_xgb =  xgb_model.predict(x_test)


# ### Testing predictions accuracy for the 3 models

# In[38]:


from sklearn import metrics

print("Testing using Random Forest")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred_rf))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred_rf))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred_rf))) 

print("\n\nTesting using Decision Tree")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred_dt))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred_dt))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred_dt))) 

print("\n\nTesting using MLP")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred_mlp))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred_mlp))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred_mlp))) 

print("\n\nTesting using XGB")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred_xgb))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred_xgb))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred_xgb))) 


# In[39]:


def step(x, threshold=0.6):
    if x >= threshold:
        return 1
    else: 
        return 0 
    
def binarize(preds):
    bin_preds = []
    for p in preds:
        bin_preds.append(step(p))
    return bin_preds    
    
pred_rf = binarize(pred_rf)
pred_dt = binarize(pred_dt)
pred_mlp = binarize(pred_mlp)
pred_xgb = binarize(pred_xgb)


# In[40]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Random Forest")
print("--- confusion_matrix ---")
print(confusion_matrix(y_test, pred_rf))  
print("\n--- classification report ---")
print(classification_report(y_test, pred_rf))  
print("\nmodel accuracy: ", accuracy_score(y_test, pred_rf))  

print("\n\nDecision Tree")
print("--- confusion_matrix ---")
print(confusion_matrix(y_test, pred_dt))  
print("\n--- classification report ---")
print(classification_report(y_test, pred_dt))  
print("\nmodel accuracy: ", accuracy_score(y_test, pred_dt))

print("\n\nMLP")
print("--- confusion_matrix ---")
print(confusion_matrix(y_test, pred_mlp))  
print("\n--- classification report ---")
print(classification_report(y_test, pred_mlp))  
print("\nmodel accuracy: ", accuracy_score(y_test, pred_mlp))

print("\n\nXGB")
print("--- confusion_matrix ---")
print(confusion_matrix(y_test, pred_xgb))  
print("\n--- classification report ---")
print(classification_report(y_test, pred_xgb))  
print("\nmodel accuracy: ", accuracy_score(y_test, pred_xgb))


# # Dead or Alive? Let's predict who will survive
# <a id="section6"></a>

# - loading and preparing the data to predict

# In[41]:


test = pd.read_csv("../input/test.csv")
ids = test.PassengerId
test = test[labels]
test.sample(2)


# In[42]:


test.Sex = pd.get_dummies(test.Sex)
test.sample(1)


# In[43]:


test.isnull().index
test.fillna(test.Age.mean(), inplace=True)
test.dtypes


# In[44]:


test.isnull().sum()


# In[45]:


preds = rf_model.predict(test.values)
preds = binarize(preds)


# In[46]:


d = {"PassengerId" : ids.values, "Survived" : preds}
survivors = pd.DataFrame(data=d) 
survivors.to_csv("predictions.csv", index=False)


# <h1 align=center> That's all folks!!! </h1>
# <img src="https://media.giphy.com/media/WoIPW37yvkcXS/giphy.gif" width="353" height="480"/>
