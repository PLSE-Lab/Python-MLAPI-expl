#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("/kaggle/input/credit_scoring_sample.csv")
df.head()


# In[ ]:


df.dtypes


# In[ ]:


df.isna().sum()


# In[ ]:


df["MonthlyIncome"] = df["MonthlyIncome"].fillna(np.mean(df["MonthlyIncome"]))
#df["MonthlyIncome"] = pd.Series(df["MonthlyIncome"]).astype(int)
df["MonthlyIncome"].astype(int)


# In[ ]:


print(pd.Series(df["NumberOfDependents"]).value_counts())
print(df["NumberOfDependents"].isna().sum())


# In[ ]:


df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0).astype(int)


# In[ ]:


df.head()


# In[ ]:


df["DebtRatio"] = df["DebtRatio"].apply(lambda x : 2 if x > 1 else x)

#df["DebtRatio"] = df["DebtRatio"].reset_index(drop = True)
df = df[df["DebtRatio"] != 2]
df.reset_index(drop = True,inplace = True)
df.head()


# In[ ]:


df["credit_within_90"] = df["NumberOfTimes90DaysLate"].apply(lambda x : 1 if x != 0 else 0 )
df.head()


# In[ ]:


df["past_due_within_30-59"] = df["NumberOfTime30-59DaysPastDueNotWorse"].apply(lambda x : 1 if x > 0 else 0)
df["past_due_within_60-89"] = df["NumberOfTime60-89DaysPastDueNotWorse"].apply(lambda x : 1 if x > 0 else 0)


# In[ ]:


plt.figure(figsize=(8, 9))
corr = df.corr()
sns.heatmap(corr, annot = True,cmap=sns.diverging_palette(20, 220, n=200),square=True)
#corr


# In[ ]:


def plotBarChart(df,col,label):
    g = sns.FacetGrid(df, col=col)
    g.map(plt.hist, label, bins=10)

for val in df.columns:
    plotBarChart(df,'credit_within_90',val)   


# In[ ]:


from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression

features = df.drop(["credit_within_90", "DebtRatio", "MonthlyIncome", "NumberOfDependents"], axis = 1)
x_train, x_test, y_train, y_test = tts(features, df["credit_within_90"], random_state = 0)
reg = LogisticRegression()
reg.fit(x_train, y_train)
#y_test = y_test.values.reshape(-1,1)
predicted = reg.predict(x_test)
#x_test
from sklearn import metrics
from sklearn.metrics import accuracy_score
#r_square = metrics.score(y_test, predicted)
accuracy_score(y_test, predicted)

