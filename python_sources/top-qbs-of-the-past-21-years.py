#!/usr/bin/env python
# coding: utf-8

# # Methodology
# This is a simple ridge cross validation regression to find a more robust model for quarterback contribution to a game with no max or min value.
# 
# # Return Values
# Returned values are the ridge regression results along with a printout of the top 60 quarterback seasons in the past 21 years. Additionally, returned are the top quarterback averages over the past 21 years.
# 
# Aaron Rodgers, Tom Brady, Drew Brees, and Peyton Manning all do quite well overall.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output


# In[ ]:


df_all = pd.read_csv("../input/QBStats_all.csv")


# In[ ]:


# Create a list of QBs that have thrown for at least 150 to remove WRs, RBs, 
# and backups who havet attempted fewer than 150 passes
scoring_qbs = df_all.groupby('qb').sum()
scoring_qbs = scoring_qbs[scoring_qbs.att > 250]

# Munge the longest throw to remove "t" and drop any quarterback that does not have at least 250 attempts
for index, row in df_all.iterrows():
    if "t" in str(row.lg):
        lg = str(row.lg[:-1])
        df_all.set_value(index,'lg',lg)
    if row.qb not in scoring_qbs.index:
        df_all = df_all.drop(index)


# In[ ]:


df_all = df_all[df_all.ypa < 25]


# In[ ]:


df_all = df_all[df_all.ypa < 25]


# In[ ]:


df_all = df_all.dropna()
X = df_all.drop(['qb','rate','game_points','year','home_away'],axis=1)
y = df_all.game_points
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 41)


# In[ ]:


df_all.describe()


# In[ ]:


# Run a standard ridge without any feature engineering
alphas = 10**np.linspace(-1,8,100)

coefs = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train,y_train)
    coefs.append(ridge.coef_)

ridgecv = RidgeCV(alphas=alphas,normalize=False)
ridgecv.fit(X_train,y_train)
print(ridgecv.alpha_)


# In[ ]:


plt.plot(alphas,np.asarray(coefs))
plt.axvline(ridgecv.alpha_)
plt.xscale("log")


# In[ ]:


y_pred = ridgecv.predict(X_test)
mean_squared_error(y_test,y_pred)


# ## Ridge Regression Coefficients 

# In[ ]:


for i in range(len(ridgecv.coef_)):
    print(str(X.columns[i]) + ": " + str(ridgecv.coef_[i]))


# In[ ]:


# Set the quarterback scores on the main dataframe
for index, row in df_all.iterrows():
    row = row.drop(['qb','rate','game_points','year','home_away'])
    qbscore = np.dot(row.values.astype(float),ridgecv.coef_)
    df_all.set_value(index,'qbscore',qbscore)


# #Top 20 QBs of all time

# In[ ]:


df_all.groupby('qb').mean().sort('qbscore',ascending=False)[['qbscore','rate']].head(20)


# #Top QBs year over year

# In[ ]:


df_best_by_year = df_all.groupby(['year','qb']).mean().copy()


# In[ ]:


df_best_by_year.sort('qbscore',ascending=False)[['qbscore','rate']].head(60)


# #Quick Analysis - Matt Ryan for MVP in 2016
# As a Packers fan, I hate to admit this, but given that Matt Ryan's performance this year is 8th overall in the past 21 seasons, he frankly gets the MVP.

# In[ ]:




