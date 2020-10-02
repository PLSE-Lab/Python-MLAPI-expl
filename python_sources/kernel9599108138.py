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
import seaborn as sn
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.drop(['id','host_id','host_name','last_review'],axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


data['name'].unique()


# In[ ]:


data['reviews_per_month']=data['reviews_per_month'].fillna((data['reviews_per_month']).mean())


# In[ ]:


data.isnull().any()


# In[ ]:


data.dropna(inplace=True)
data.isnull().sum()


# In[ ]:


data['neighbourhood_group'].unique()


# In[ ]:


data['neighbourhood'].unique()


# In[ ]:


data['room_type'].value_counts().plot(kind='bar')


# In[ ]:


corr=data.corr()
plt.figure(figsize=(10,8))
sn.heatmap(corr,annot=True)


# In[ ]:


plt.figure(figsize = (10,8))
sn.barplot(x='room_type', y='price', data=data, palette="Set2")


# In[ ]:


sn.scatterplot(x='room_type',y='price',data=data)


# In[ ]:


not_renting_out = np.array(data['availability_365']==0).sum()
print(not_renting_out)


# In[ ]:


data=data[data['availability_365']!=0]
data.head()


# In[ ]:


sn.countplot(x='neighbourhood_group',data=data)


# In[ ]:


data['room_type'] = data['room_type'].astype('category').cat.codes
data['neighbourhood_group'] = data['neighbourhood_group'].astype('category').cat.codes


# In[ ]:


sn.countplot(x='room_type',data=data)


# In[ ]:


sn.countplot(x='neighbourhood_group',data=data)


# In[ ]:


data.head()


# In[ ]:


x=data.drop(['name','neighbourhood','latitude','longitude','reviews_per_month','calculated_host_listings_count','price'],axis=1)
y=data['price']


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
slc=StandardScaler()
x_train=slc.fit_transform(x_train)
x_test=slc.transform(x_test)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
model= LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
#print(predictions)
print(r2_score(y_test, predictions))
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(rmse)


# In[ ]:


error_frame = pd.DataFrame({'Actual': np.array(y_test).flatten(), 'Predicted': predictions.flatten()})
error_frame.head(10)


# In[ ]:


sn.boxplot(x=data['price'])


# In[ ]:




