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


data=pd.read_csv('/kaggle/input/insurance/insurance.csv')
data.head()


# In[ ]:


import seaborn as sns
sns.countplot(data.sex)


# In[ ]:


sns.distplot(data.charges)


# In[ ]:


crossTab=pd.crosstab(data.sex,data.smoker)
crossTab.plot.bar(stacked=True,color=['c','r'])


# In[ ]:


sns.scatterplot(data.bmi,data.charges,hue=data.children)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()
data.sex=la.fit_transform(data.sex)
data.smoker=la.fit_transform(data.smoker)
data.region=la.fit_transform(data.region)


# In[ ]:


data.head()


# In[ ]:


sns.heatmap(data.corr(),annot=True,cmap='rainbow')


# In[ ]:


yval=data.iloc[:,[0,2,6]].values


# In[ ]:


from sklearn.cluster import KMeans
t=[]
for i in range(1,10):
    model=KMeans(n_clusters=i)
    model.fit(yval)
    t.append(model.inertia_)  


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(range(1,10),t,color='y')
plt.plot(range(1,10),t,color='c')


# In[ ]:


model=KMeans(n_clusters=3)
model.fit(yval)
pred=model.predict(yval)
a=model.cluster_centers_


# In[ ]:


plt.scatter(yval[pred==0,0],yval[pred==0,1],label='grp 1')
plt.scatter(yval[pred==1,0],yval[pred==1,1],label='grp 2')
plt.scatter(yval[pred==2,0],yval[pred==2,1],label='grp 3')
plt.legend()
plt.scatter(a[:,0],a[:,1],marker='x',color='r')


# In[ ]:


model.predict([[19,0,0]])


# In[ ]:


x=data.drop('charges',axis=1)
y=data['charges']


# In[ ]:


from sklearn.model_selection import train_test_split
xr,xt,yr,yt=train_test_split(x,y,test_size=0.1)


# In[ ]:


from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
model=LGBMRegressor(n_estimators=30)
model.fit(xr,yr)
yp=model.predict(xt)


# In[ ]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print(r2_score(yt,yp))
print(mean_absolute_error(yt,yp))
print(mean_squared_error(yt,yp))


# In[ ]:




