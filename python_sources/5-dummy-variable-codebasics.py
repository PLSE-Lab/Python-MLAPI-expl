#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("/kaggle/input/carprices.csv")
df.head()


# In[ ]:


dummies = pd.get_dummies(df['Car Model'])
merged = pd.concat([df, dummies], axis = 'columns')


# In[ ]:


merged


# In[ ]:


final = merged.drop(['Car Model'], axis = 'columns')
final


# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = final.drop('Sell Price($)', axis = 'columns')
y = final['Sell Price($)']
model.fit(X, y)


# In[ ]:



X


# In[ ]:


model.predict([[45000, 4, 0, 1, 0]])


# In[ ]:


model.predict([[86000, 7, 1, 0, 0]])


# In[ ]:


model.score(X, y)


# In[ ]:





# In[ ]:





# In[ ]:




