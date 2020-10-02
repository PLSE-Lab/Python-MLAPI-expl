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


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pandas as pd


# In[ ]:


bottle = pd.read_csv('../input/calcofi/bottle.csv')


# In[ ]:


bottle.head()


# In[ ]:


bottle.info()


# In[ ]:


bottle.describe()


# In[ ]:


sns.heatmap(bottle.corr(),annot= True)


# In[ ]:


bottle.columns


# In[ ]:


bottle = bottle[['Depthm', 'T_degC', 'Salnty']]
bottle = bottle[:][:1000]


# In[ ]:


sns.lmplot(x='Salnty',y='T_degC',data = bottle)


# In[ ]:


sns.lmplot(x='Salnty',y='T_degC',data = bottle)


# In[ ]:


sns.lmplot(x='Depthm',y='T_degC',data = bottle)


# In[ ]:


bottle.fillna(method='ffill', inplace=True)
bottle.dropna(inplace=True)


# In[ ]:


X = bottle[['Depthm','Salnty','T_degC']]
y = bottle['Salnty']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape,y_train.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)


# In[ ]:


accuracy = lm.score(X_test, y_test)
print(accuracy*100)


# In[ ]:


y_pred = lm.predict(X_test)
for i in range(10):
    print('Actual value: {:.3f} Predicted Value: {:.3f}'.format(y_test.values[i],y_pred[i]))


# In[ ]:




