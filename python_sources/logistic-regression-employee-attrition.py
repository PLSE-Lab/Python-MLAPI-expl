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
import seaborn as sns
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas_profiling


# In[ ]:


df = pd.read_csv("../input/HR-Employee-Attrition.csv")


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


pandas_profiling.ProfileReport(df)


# In[ ]:


num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns


# In[ ]:


print(num_cols)
print(cat_cols)


# In[ ]:


df.Attrition.replace({"Yes":1,"No":0},inplace = True)


# In[ ]:


df.Over18.replace({"Y":1,"No":0},inplace = True)


# In[ ]:


df.OverTime.replace({"Yes":1,"No":0},inplace = True)


# In[ ]:


df_onehot = pd.get_dummies(df[cat_cols.drop(["Attrition","Over18","OverTime"])])


# In[ ]:


df_final = pd.concat([df_onehot,df[num_cols],df["Attrition"],df["Over18"],df["OverTime"]],axis =1)


# In[ ]:


df_final.head()


# In[ ]:


df_final.corr()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[ ]:


X = df_final.drop(columns=['Attrition'])
Y = df_final['Attrition']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3,random_state=0)


# In[ ]:


log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)


# In[ ]:


train_predict = log_reg.predict(X_train)


# In[ ]:


metrics.confusion_matrix(y_train,train_predict)


# In[ ]:


metrics.accuracy_score(y_train,train_predict)


# In[ ]:


test_pred = log_reg.predict(X_test)


# In[ ]:


metrics.accuracy_score(y_test,test_pred)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, test_pred))

