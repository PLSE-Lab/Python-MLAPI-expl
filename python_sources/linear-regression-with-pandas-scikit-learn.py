#!/usr/bin/env python
# coding: utf-8

# # Linear Regression with Pandas & Scikit-Learn
# 
# by https://becominghuman.ai/linear-regression-in-python-with-pandas-scikit-learn-72574a2ec1a5

# In[ ]:


# libraries import
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# loading the data
df = pd.read_csv("../input/building1retail.csv", index_col='Timestamp', date_parser=lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M'))

# show first 5 rows
df.head()


# In[ ]:


df.shape


# In[ ]:


# show column types
df.dtypes


# In[ ]:


# exploring the data
df.plot(figsize=(18,5))


# In[ ]:


# check if there is no missing values
df.isnull().values.any()


# In[ ]:


# histogram of the data
df.hist()


# In[ ]:


# filter records that are greater than 3 std, to remove outliers
df = df[(np.abs(stats.zscore(df)) < 3.).all(axis=1)]
df.shape


# In[ ]:


# graph without outliers
df.plot(figsize=(18,5))


# In[ ]:


# scatter plot to see there are linear relationship
plt.scatter(df['OAT (F)'], df['Power (kW)'])


# In[ ]:


# checking timezone on a daytime per column
df.loc['2010-01-01', ['OAT (F)']].plot()


# In[ ]:


# checking timezone on a daytime per column
df.loc['2010-01-01', ['Power (kW)']].plot()


# In[ ]:


# linear regression model
X = pd.DataFrame(df['OAT (F)'])
y = pd.DataFrame(df['Power (kW)'])
model = LinearRegression()
scores = []

# split the records into 3 folds and train 3 times the model, 
# test and get the score of each training
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
for i, (train, test) in enumerate(kfold.split(X, y)):
  model.fit(X.iloc[train,:], y.iloc[train,:])
  score = model.score(X.iloc[test,:], y.iloc[test,:])
  scores.append(score)
print(scores)


# In[ ]:


# To archieve a better model, let's consider the hour of the day
X['tod'] = X.index.hour
# drop_first = True removes multi-collinearity
add_var = pd.get_dummies(X['tod'], prefix='tod', drop_first=True)
# Add all the columns to the model data
X = X.join(add_var)
# Drop the original column that was expanded
X.drop(columns=['tod'], inplace=True)
print(X.head())


# In[ ]:


# training again with the new dummie columns
model = LinearRegression()
scores = []
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
for i, (train, test) in enumerate(kfold.split(X, y)):
 model.fit(X.iloc[train,:], y.iloc[train,:])
 scores.append(model.score(X.iloc[test,:], y.iloc[test,:]))
print(scores)

