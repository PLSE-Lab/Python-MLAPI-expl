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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


data = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")


# In[ ]:


#As Always the first few rows
data.head()


# In[ ]:


data.describe()


# In[ ]:


data.isnull().sum()


# In[ ]:


#There seems no null values, which make EDA much simpler


# In[ ]:


#Lets see the corelation between the vaues, but the serial no column has no signifiance, 
#so we will drop Serial no
data.drop(['Serial No.'],inplace=True,axis=1)


# In[ ]:


data.head()


# In[ ]:


data.corr()


# In[ ]:


corr = data.corr()
sns.heatmap(corr,cmap="Blues",annot=True)


# In[ ]:


#Chance of Admit is the Value to be predicted
#Hatmap shows that GRE Score,TOEFL Score,CGPA has high impact on Chance of Admit


# In[ ]:


df = data.drop(['Chance of Admit '],axis=1)
y = data['Chance of Admit ']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)


# In[ ]:


from sklearn.linear_model import LinearRegression
log_reg=LinearRegression()
log_reg.fit(X_train,y_train)


# In[ ]:


pred = log_reg.predict(X_test)


# In[ ]:


from sklearn.metrics import r2_score
print (r2_score(y_test, pred))
print (pred)


# In[ ]:




