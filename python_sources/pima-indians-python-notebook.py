#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#For sake of reproducibility I am also adding stack overflow links which I found useless


# In[ ]:


import os as os


# In[ ]:


os.getcwd()


# In[ ]:


os.chdir('../input')


# In[ ]:


os.getcwd()


# In[ ]:


os.listdir()


# In[ ]:


pima=pd.read_csv('PimaIndians.csv')


# In[ ]:


pima.head()


# In[ ]:


pima.info()


# In[ ]:


pd.value_counts(pima.test)


# In[ ]:


pima.describe()


# from https://stackoverflow.com/questions/19913659/pandas-conditional-creation-of-a-series-dataframe-column

# In[ ]:


pima['outcome'] = ['0' if x == 'negatif' else '1' for x in pima['test']]


# In[ ]:


pd.value_counts(pima.outcome)


# In[ ]:


pd.crosstab(pima.outcome,pima.test)


# In[ ]:


del pima['test']


# In[ ]:


pima.head()


# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import seaborn as sns


# In[ ]:


pima.corr()


# http://www.statisticsviews.com/details/feature/8868901/A-Tutorial-on-Python.html

# In[ ]:


pd.crosstab(pima.pregnant,pima.outcome,margins='TRUE')


# In[ ]:


e=pima.groupby(['pregnant', "outcome"]).insulin.median().reset_index()
e.pivot(index='pregnant', columns='outcome', values='insulin')


# In[ ]:



plt.bar(pima.outcome,pima.pregnant,label='bar1',color='blue')


# In[ ]:


sns.distplot(pima.pregnant,kde=True,rug=True)


# In[ ]:


sns.distplot(pima.age,kde=True,rug=True)


# In[ ]:


sns.regplot(x='age',y='insulin',data=pima)


# In[ ]:


sns.lmplot(x='age',y='insulin',hue='outcome',data=pima)


# In[ ]:


sns.stripplot(y='insulin',x='outcome',data=pima)


# In[ ]:


import statsmodels.formula.api as sm


# In[ ]:


pima.columns


# In[ ]:


len(pima)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


target_attribute = pima['outcome']


# https://stackoverflow.com/questions/29763620/how-to-select-all-columns-except-one-column-in-pandas/39192113

# In[ ]:


data=pima.loc[:, pima.columns != 'outcome']


# In[ ]:


data.head()


# https://stackoverflow.com/questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data, target_attribute, test_size=0.2)


# In[ ]:


X_train.head()


# In[ ]:



X_test.head()


# In[ ]:


y_train.head()


# In[ ]:


y_test.head()


# In[ ]:


len(X_train)


# In[ ]:


len(X_test)


# In[ ]:


len(y_train) 


# In[ ]:


len(y_test)


# https://stats.stackexchange.com/questions/146804/difference-between-statsmodel-ols-and-scikit-linear-regression/146809

# https://stackoverflow.com/questions/33833832/building-multi-regression-model-throws-error-pandas-data-cast-to-numpy-dtype-o

# In[ ]:


results = sm.OLS(y_train.astype(float), X_train.astype(float)).fit()


# In[ ]:



results.summary()


# In[ ]:


predictions = results.predict(X_test)


# In[ ]:


predictions[1:10]


# In[ ]:


type(predictions)


# In[ ]:


predictions.dtypes


# In[ ]:


y_test[1:10]


# In[ ]:


predictions2=pd.DataFrame(predictions)


# In[ ]:


predictions2.columns=['pred']


# In[ ]:


predictions2.head()


# In[ ]:


predictions2['newpred'] = [1 if x>= 0.5 else 0 for x in predictions2['pred']]


# https://stackoverflow.com/questions/36921951/truth-value-of-a-series-is-ambiguous-use-a-empty-a-bool-a-item-a-any-o

# In[ ]:


predictions2.head()


# In[ ]:


predictions2['y_test']=y_test


# In[ ]:


predictions2.head()


# Confusion Matrix

# In[ ]:


pd.crosstab(predictions2.newpred,predictions2.y_test,margins=True)

