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


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[ ]:


df=pd.read_csv('/kaggle/input/married-at-first-sight/mafs.csv')


# In[ ]:


df


# In[ ]:


df.info()


# In[ ]:


X=df.drop('Decision',axis=1)


# In[ ]:


y=df['Decision']


# In[ ]:


X['Occupation'].nunique()


# Since every one has a different job, we will not consider job during next few steps.

# Change data type

# In[ ]:


def map_values():
    X['Location']=X.Location.map({'New York City and Northern New Jersey':1
                                   ,'South Florida':2
                                   ,'Chicago, Illinois':3
                                   ,'Boston, Massachusetts':4
                                   ,'Dallas, Texas':5
                                   ,'Philadelphia, Pennsylvania':6
                                   ,'Charlotte, North Carolina':7
                                   ,'Washington D.C.':8
                                   ,'Atlanta, Georgia':9})
    X['Gender']=X.Gender.map({'F':1
                               ,'M':2})
    return 'Done!'
    


# In[ ]:


map_values()


# In[ ]:


X=X.drop(['Name','Occupation','Status'],axis=1)


# drop the columns which are not very important

# In[ ]:


y=y.map({'Yes':1
      ,'No':0})


# In[ ]:


y


# In[ ]:


X


# In[ ]:


f,ax=plt.subplots(figsize=(20,12))
sns.heatmap(X[list(X.columns)].corr(),linewidths=0.25,square=True,linecolor='w',cmap='Oranges_r',annot=True)


# In[ ]:


fig=px.box(df,x='Couple',y='Age')
fig.show()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=7)
model.fit(X, y)


# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 7)
model2 = DecisionTreeRegressor()
model2.fit(train_X, train_y)

val_predictions = model2.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# In[ ]:




