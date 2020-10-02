#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


pd.set_option('display.max_columns', None)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


family_income_data = pd.read_csv('../input/family-income-and-expenditure/Family Income and Expenditure.csv')


# In[ ]:


family_income_data.head()


# Let's see if we can predict `Total Household Income` through the expenditures each family makes. I initiate this by taking all the column names with 'expenditures' in it.

# In[ ]:


expenditures = [column for column in family_income_data.columns if 'Expenditure' in column]


# Checking the values I have:

# In[ ]:


expenditures


# I set my features as the splice of the original dataset where the column names are expenditures, and set the target as the `Total Household Income` column

# In[ ]:


X = family_income_data.loc[:, expenditures]
y = family_income_data['Total Household Income']


# Importing the necessary libraries for fitting.

# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import Normalizer, PolynomialFeatures, MinMaxScaler, StandardScaler
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.decomposition import IncrementalPCA, SparsePCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_absolute_error


# But before I go to training a model, I wanna see first how the the features appear in a scatterplot when treated as a function of the label.

# In[ ]:


plt.figure(figsize=(20, 20))
i = 1
for exp in expenditures :
    plt.subplot(6,3,i)
    sns.regplot(x=X[exp], y=y)
    i += 1


# So earlier I tried to build a regression model, but the best i got from that was 0.76 score. I did a lot of manipulation to make the regression line fit but still wouldn't go above that score. Now, I understand why. The points are all over the place and the correlation between the expenses is challenging to describe!

# Okay, I wanna see the correlation of the features and the label:

# In[ ]:


Xy = X.copy()
Xy['THI'] = y


# In[ ]:


Xy_corr = Xy.corr()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(Xy_corr, square=True)


# From here, it seems like the best correlation I have is 0.8, which is `Total Rice Expenditure` and `Bread and Cereals Expenditure`. Next to that, we have `Total Food Expenditure` has a somehow high correlation value with `Meat Expenditure` and `Vegetables Expenditure`. Makes sense especially that **Pinoys** are a culture of *rice and ulam*. It's not surprising that these are related.

# In[ ]:


plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
sns.regplot(x=family_income_data['Total Rice Expenditure'], y = family_income_data['Bread and Cereals Expenditure'])
plt.subplot(3,1,2)
sns.regplot(x=family_income_data['Total Food Expenditure'], y = family_income_data['Vegetables Expenditure'])
plt.subplot(3,1,3)
sns.regplot(x=family_income_data['Total Food Expenditure'], y = family_income_data['Meat Expenditure'])


# In[ ]:


plt.figure(figsize=(15, 25))
i = 1
for exp in expenditures :
    plt.subplot(6,3,i)
    sns.distplot(X[exp])
    i += 1


# Okay, so I tried to see how the data points are distributed and there is a heavy skew to the left. This tells me that more or less, everyone spends about the same amount for expenses. The KDE's, however, tell a different story. A lot of the dataplots distort the probability line in such a way that some would describe the distribution inaccurately.

# In[ ]:


plt.figure(figsize=(10, 10))
sns.distplot(y, bins=20)


# Lmao this is an extremely ridiculous distribution curve. I set 20 bins and still the most distinguishable is still three.
# 
# What this tells me is that almost everyone in the dataset has a Total Household Income of between 5000-10000. It's quite disturbing.

# Okay, now let's train a model. I'm going to pick `RandomForestRegressor` and `KNeighborsRegressor`, since these are great picks for chaotic scatterplots.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)


# I'm setting verbose to True so that I can see what's happening under the model as it happens.

# Moreover, I'm adding a comparative regplot for the actual values for Total Household Income and the ones predicted by the model. Ideally, we want to see them fit inside the regression line to say that "Okay, this is a good model."

# ### Random Forest Regressor

# In[ ]:


rfr = RandomForestRegressor(verbose=True, n_jobs=-1, n_estimators=1000)
rfr.fit(X_train, y_train)


# In[ ]:


rfr.score(X_test, y_test)


# In[ ]:


y_rfr_predict = rfr.predict(X)
mean_absolute_error(y, y_rfr_predict)
plt.figure(figsize=(20,5))
ax = sns.regplot(x=y, y = y_rfr_predict)
ax.set(xlabel='Total Household Income', ylabel='Predicted TIH')


#  ### K-Nearest Neighbors Regressor

# In[ ]:


knr = KNeighborsRegressor(n_neighbors=15, n_jobs=-1, leaf_size=50)
knr.fit(X_train, y_train)


# In[ ]:


knr.score(X_test, y_test)


# In[ ]:


y_knr_predict = knr.predict(X)
mean_absolute_error(y, y_knr_predict)
plt.figure(figsize=(20,5))
ax = sns.regplot(x=y, y = y_knr_predict)
ax.set(xlabel='Total Household Income', ylabel='Predicted TIH')


# ### XGBRegressor

# In[ ]:


xgbr = XGBRegressor(nthread = -1, eta=0.1, subsample=0.5)
xgbr.fit(X_train, y_train)


# In[ ]:


xgbr.score(X_test, y_test)


# In[ ]:


y_xgbr_predict = xgbr.predict(X)
mean_absolute_error(y, y_xgbr_predict)
plt.figure(figsize=(20,5))
sns.regplot(x=y, y = y_xgbr_predict)
plt.figure(figsize=(10,5))


# In[ ]:


p = Pipeline([
     ('mms', MinMaxScaler()),
     ('rfr', RandomForestRegressor(verbose=True, n_jobs=-1, n_estimators=1000))
])

p.fit(X_train, y_train)
p.score(X_test, y_test)

