#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/HRDataset_v9.csv')


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


x=data.iloc[:,:-1]
y=data.iloc[:,-1:]


# In[ ]:


x.drop(['Employee Name','Employee Number','State','DOB','Sex','MaritalDesc','CitizenDesc','Hispanic/Latino','RaceDesc','Date of Hire','Manager Name','Reason For Term','Employee Source','Date of Termination','Position'],axis=1,inplace=True)


# In[ ]:


x.info()


# In[ ]:


label=x.iloc[:,10:]
label.head()


# In[ ]:


label.shape


# In[ ]:


data['Employment Status'].value_counts


# In[ ]:


data['Employment Status'].unique()


# In[ ]:


data['Department'].unique()


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
label=ohe.fit_transform(label).toarray()
print(label)


# In[ ]:


data2=pd.DataFrame(data=label,index=range(310),columns=['Active','Future Start','Leave of Absence','Terminated for Cause','Voluntarily Terminated','Admin Offices','Executive Office','IT/IS','Production       ','Sales','Software Engineering',])


# In[ ]:


data2.head()


# In[ ]:


df=pd.concat([x.iloc[:,:10],data2],axis=1)


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
y.iloc[:,0]=le.fit_transform(y.iloc[:,0])


# In[ ]:


y.head()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dt_reg=DecisionTreeRegressor(random_state=0)
dt_reg.fit(df,y)


# In[ ]:


y_predict = dt_reg.predict(df)

from sklearn.metrics import accuracy_score

accuracy_score(y, y_predict)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
df, y = make_regression(n_features=4, n_informative=2,
                      random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0,
                             n_estimators=10)
regr.fit(df, y)


# In[ ]:


y.shape


# In[ ]:


y_predict2 = regr.predict(df)
#accuracy_score(y, y_predict2)
regr.score(df,y)
#print(regr.feature_importances_)


# In[ ]:





# In[ ]:




