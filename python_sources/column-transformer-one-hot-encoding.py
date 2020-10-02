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


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# In[ ]:


df=pd.read_csv("../input/car-price-prediction3-features/carprices.csv")
df


# In[ ]:


el=LabelEncoder()
dlf=df
dlf["Car Model"]=el.fit_transform(dlf["Car Model"])
dlf


# In[ ]:


x=np.array(dlf[["Car Model","Mileage","Age(yrs)"]])
x


# In[ ]:


y=np.array(dlf["Sell Price($)"])
y


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
x_test


# In[ ]:


transformer=ColumnTransformer( transformers=[('OneHot',OneHotEncoder(),[0])],remainder="passthrough")
X=transformer.fit_transform(x_train)
x_test=transformer.fit_transform(x_test)
X


# In[ ]:


X=X[:,1:]
X


# In[ ]:


reg=LinearRegression()
reg.fit(X,y_train)


# In[ ]:


reg.predict([[0,0,70000,6]])


# In[ ]:


reg.score(X,y_train)


# In[ ]:


x_test


# In[ ]:


x_test=x_test[:,1:]
x_test


# In[ ]:


y_test


# In[ ]:


reg.fit(x_test,y_test)


# In[ ]:


reg.predict([[0,0,70000,6]])


# In[ ]:


reg.score(x_test,y_test)


# If you want an explanation for eatch line let me know.
# 
# 
# Do upvote if you like the notebook.

# In[ ]:




