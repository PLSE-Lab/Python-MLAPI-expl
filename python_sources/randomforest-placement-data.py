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
get_ipython().run_line_magic('matplotlib', 'inline')
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


df = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
df.head()


# In[ ]:


df.columns


# In[ ]:


df.isna().any()


# In[ ]:


df = df.fillna(0)


# In[ ]:


df1 = df.copy()


# In[ ]:


df.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
oh = OneHotEncoder()

df['gender'] = le.fit_transform(df['gender'])
df['status'] = le.fit_transform(df['status']) 
df['workex'] = le.fit_transform(df['workex'])
df['ssc_b'] = le.fit_transform(df['ssc_b'])
df['hsc_b'] = le.fit_transform(df['hsc_b'])
df['hsc_s'] = le.fit_transform(df['hsc_s'])
df['degree_t'] = le.fit_transform(df['degree_t'])
df['hsc_b'] = le.fit_transform(df['hsc_b'])
df['specialisation'] = le.fit_transform(df['specialisation'])


# In[ ]:


#df['ssc_b'] = oh.fit_transform(df['ssc_b'])
#df['hsc_b'] = oh.fit_transform(df['hsc_b'])


# In[ ]:


#for col in df.columns:
 #   le.fit_transform(df.columns)
df.head()


# In[ ]:


X  = df.drop(['sl_no', 'mba_p',], axis = 1)
y= df['mba_p']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 43)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train, y_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#le.fit(X, y)
#le.fit(y)


# In[ ]:


classifier=RandomForestClassifier(n_estimators=100)
model = classifier.fit(X_train, y_train.astype('int'))


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[ ]:


from sklearn import metrics
from sklearn.metrics import r2_score


# In[ ]:



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred).round(3))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred).round(3))  
print('Root Mean Squared:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))
print('r2_score:', r2_score(y_test, y_pred).round(3))
print('Accuracy:', metrics.accuracy_score(y_test.astype('int'), y_pred).round(3))


# ***Random Forest has very poor accuracy score, hence this model cannot be used***

# # Don't forget to vote. Thanks
