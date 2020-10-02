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


movies = pd.read_csv("../input/train.csv")
movies.head()


# In[ ]:


import seaborn as sns


# In[ ]:


movies.columns


# In[ ]:


movies = movies[['id', 'belongs_to_collection', 'budget', 'genres',
       'original_language',
       'popularity','production_companies',
       'production_countries', 'release_date', 'runtime', 'spoken_languages',
       'Keywords', 'cast', 'crew', 'revenue']]


# In[ ]:


movies.runtime = movies.runtime.fillna(method = "ffill")
x = movies[['popularity', 'runtime', 'budget']]


# In[ ]:


y = movies['revenue']
from sklearn.metrics import mean_squared_log_error as msle


# In[ ]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
def rmsle(y,y0): return np.sqrt(np.mean(np.square(np.log1p(y)-np.log1p(y0))))
model = reg.fit(x,y)
y_pred = reg.predict(x)
rmsle = rmsle(y_pred, y)
print("The linear model has intercept : {}, and coefficients : {}, and the rmsle is {} ".format(model.intercept_, model.coef_, rmsle) )


# In[ ]:


test = pd.read_csv('../input/test.csv')
test.runtime = test.runtime.fillna(method = "ffill")
x_test = test[['popularity', 'runtime', 'budget']]
pred = reg.predict(x_test)


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['revenue'] = pred
sub.to_csv('sub.csv', index=False)


# In[ ]:


sub.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




