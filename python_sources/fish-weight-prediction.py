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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/fish-market/Fish.csv")


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


data['Species'].value_counts()


# In[ ]:


data =pd.get_dummies(data)


# In[ ]:


data.columns


# In[ ]:


X=data[['Length1', 'Length2', 'Length3', 'Height', 'Width',
       'Species_Bream', 'Species_Parkki', 'Species_Perch', 'Species_Pike',
       'Species_Roach', 'Species_Smelt', 'Species_Whitefish']]


# In[ ]:


y = data['Weight']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


regression = LinearRegression()


# In[ ]:


regression.fit(X_train,y_train)


# In[ ]:


y_predict = regression.predict(X_test)


# In[ ]:


output = pd.DataFrame({"actual":y_test,"predicted":y_predict})


# In[ ]:


output


# In[ ]:


plt.scatter(y_test,y_predict)


# In[ ]:





# In[ ]:





# In[ ]:




