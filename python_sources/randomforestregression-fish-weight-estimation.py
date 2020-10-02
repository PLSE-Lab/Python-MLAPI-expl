#!/usr/bin/env python
# coding: utf-8

# # RANDOM FOREST REGRESSION MODEL FOR WEIGHT ESTIMATION FROM MEASUREMENTS OF THE FISH

# This is my first commit to Kaggle. Inspired by great commit "MultipleLinear Regression - Fish Weight Estimation" from Melih Akdag.
# Just change the model from Linear Regression to Random Forest Regression.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor # RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **1. UNDERSTANDING THE DATASET**

# In[ ]:


data = pd.read_csv('../input/fish-market/Fish.csv')
df = data.copy()
df.sample(10)


# **Changing the Column Names:**

# In[ ]:


df.rename(columns= {'Length1':'LengthVer', 'Length2':'LengthDia', 'Length3':'LengthCro'}, inplace=True)
df.head()


# In[ ]:


df.info()


# **Investigating Missing Values:**

# In[ ]:


print(str('Is there any NaN value in the dataset: '), df.isnull().values.any())


# **Different Species and Their Counts:**

# In[ ]:


sp = df['Species'].value_counts()
sp = pd.DataFrame(sp)
sp.T


# In[ ]:


sns.barplot(x=sp.index, y=sp['Species']);
plt.xlabel('Species')
plt.ylabel('Counts of Species')
plt.show()


# **2. PREPARING THE TRAINING-DATA AND TEST-DATA**

# In[ ]:


# Converting Qualitative data into Quantitative Data
df.loc[df["Species"] == "Perch",["Species"]] = 0
df.loc[df["Species"] == "Bream",["Species"]] = 1
df.loc[df["Species"] == "Roach",["Species"]] = 2
df.loc[df["Species"] == "Pike",["Species"]] = 3
df.loc[df["Species"] == "Smelt",["Species"]] = 4
df.loc[df["Species"] == "Parkki",["Species"]] = 5
df.loc[df["Species"] == "Whitefish",["Species"]] = 6


# In[ ]:


# Check the converting result
df


# In[ ]:


# data set
X = df.iloc[:,[0,2,3,4,5,6]]
y = df['Weight']

# split data to train and test
from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=1)


# In[ ]:


X_test2


# **3. MAKE THE MODEL -Random Forest Regressor-**

# In[ ]:


reg_rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1,random_state=1)
reg_rf.fit(X_train2,y_train2)


# **4. EVALUATE THE MODEL **

# **SCORE**

# In[ ]:


#test
reg_rf.score(X_test2,y_test2)


# In[ ]:


#train
reg_rf.score(X_train2,y_train2)


# **MSE**

# In[ ]:


test_pred = reg_rf.predict(X_test2)
test_mse = mean_squared_error(y_true=y_test2 , y_pred = test_pred)
print(test_mse)


# **RMSE**

# In[ ]:


np.sqrt(test_mse)


# In[ ]:


np.average(y_test2)


# **5. PREDICTION**

# In[ ]:


y_pred2 = reg_rf.predict(X_test2)


# In[ ]:


plt.scatter(X_test2['LengthCro'], y_test2, color='red', alpha=0.4)
plt.scatter(X_test2['LengthCro'], y_pred2, color='blue', alpha=0.4)
plt.xlabel('Cross Length in cm')
plt.ylabel('Weight of the fish')
plt.title('RandomForestRegression Model for Weight Estimation');


# In[ ]:


plt.scatter(X_test2['LengthVer'], y_test2, color='purple', alpha=0.5)
plt.scatter(X_test2['LengthVer'], y_pred2, color='orange', alpha=0.5)
plt.xlabel('Vertical Length in cm')
plt.ylabel('Weight of the fish')
plt.title('RandomForestRegression Model for Weight Estimation');


# In[ ]:


plt.scatter(X_test2['LengthDia'], y_test2, color='purple', alpha=0.4)
plt.scatter(X_test2['LengthDia'], y_pred2, color='green', alpha=0.4)
plt.xlabel('Diagonal Length in cm')
plt.ylabel('Weight of the fish')
plt.title('RandomForestRegression Model for Weight Estimation');


# In[ ]:


plt.scatter(X_test2['Height'], y_test2, color='orange', alpha=0.5)
plt.scatter(X_test2['Height'], y_pred2, color='blue', alpha=0.5)
plt.xlabel('Height in cm')
plt.ylabel('Weight of the fish')
plt.title('RandomForestRegression Model for Weight Estimation');


# In[ ]:


plt.scatter(X_test2['Width'], y_test2, color='gray', alpha=0.5)
plt.scatter(X_test2['Width'], y_pred2, color='red', alpha=0.5)
plt.xlabel('Width in cm')
plt.ylabel('Weight of the fish')
plt.title('RandomForestRegression Model for Weight Estimation');


# In[ ]:


y_pred2 = pd.DataFrame(y_pred2, columns=['Estimated Weight'])
y_test2 = pd.DataFrame(y_test2)
y_test2 = y_test2.reset_index(drop=True)
ynew2 = pd.concat([y_test2, y_pred2], axis=1)
ynew2


# In[ ]:




