#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv('../input/train.csv', sep=',')
df_train.head()


# In[ ]:


df_test = pd.read_csv('../input/test.csv', sep=',')
df_test.head()


# In[ ]:


df_store = pd.read_csv('../input/store.csv', sep=',')
df_store.head()


# In[ ]:


df_train.StateHoliday = df_train.StateHoliday.replace(0,'0')
df_test.StateHoliday = df_test.StateHoliday.replace(0,'0')


# In[ ]:


X_oh_tr = pd.get_dummies(df_train.StateHoliday, prefix='StateHoliday', prefix_sep='=')
X_oh_te = pd.get_dummies(df_test.StateHoliday, prefix='StateHoliday', prefix_sep='=')


# In[ ]:


df_1 = pd.concat([df_train.drop(['Date','Customers','Sales','StateHoliday'], axis=1),X_oh_tr],axis=1)
df_2 = pd.concat([df_test.drop(['Id','Date','StateHoliday'],axis=1), X_oh_te], axis=1)
df_2['StateHoliday=b'] = 0
df_2['StateHoliday=c'] = 0
df_2.Open = df_2.Open.fillna(0)


# In[ ]:


X_train, y_train = df_1.values, df_train.iloc[:,3].values
X_test = df_2.values


# In[ ]:


model = RandomForestRegressor()
y_hat = model.fit(X_train,y_train).predict(X_test)


# In[ ]:


cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=10).mean()


# In[ ]:


df_out = pd.DataFrame({'Sales': y_hat})
df_out.index.name = 'Id'
df_out.index = df_out.index + 1
df_out.head()


# In[ ]:


df_train.head()


# In[ ]:


df_out.to_csv('out.csv')


# In[ ]:




