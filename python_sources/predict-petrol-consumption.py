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

df=pd.read_excel("../input/measurements2.xlsx")

print(df.head())

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns
sns.heatmap(df.isnull())


# In[ ]:


null_values=df.isnull().sum().sort_values(ascending=False)
ax=sns.barplot(null_values.index,null_values.values)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
import matplotlib.pyplot as plt
plt.show()


# In[ ]:


df.drop(['refill gas','refill liters','specials'],axis=1,inplace=True)
sns.heatmap(df.isnull())


# In[ ]:


temp_inside_mean=np.mean(df['temp_inside'])


# In[ ]:


print(temp_inside_mean)


# In[ ]:


df['temp_inside'].fillna(temp_inside_mean,inplace=True)


# In[ ]:


sns.heatmap(df.isnull())


# In[ ]:


from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
l=LinearRegression()


# In[ ]:


x=df.drop(['consume','gas_type'],axis=1)


# In[ ]:


y=df['consume']


# In[ ]:


l.fit(x,y)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[ ]:


l.fit(x_train,y_train)


# In[ ]:


y_pred=l.predict(x_test)


# In[ ]:


print(l.coef_,l.intercept_)


# In[ ]:


from sklearn import metrics
print(metrics.mean_squared_error(y_test,y_pred))
print(metrics.mean_absolute_error(y_test,y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[ ]:


""""from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['gas_type'] = le.fit_transform(df['gas_type'])"""


# In[ ]:


dum1=pd.get_dummies(df['gas_type'])
print(dum1)


# In[ ]:


df=pd.concat([df,dum1],axis=1)


# In[ ]:


df.drop('gas_type',axis=1,inplace=True)


# In[ ]:


x1=df.drop('consume',axis=1)


# In[ ]:


y1=df['consume']


# In[ ]:


from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
l=LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=42)


# In[ ]:


l.fit(x_train,y_train)


# In[ ]:


y_pred_1=l.predict(x_test)
print(y_pred_1)


# In[ ]:


from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred_1)))

