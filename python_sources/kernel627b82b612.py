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


data=pd.read_csv("../input/insurance/insurance.csv")


# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


data.describe()


# In[ ]:


data.head()


# In[ ]:


import matplotlib.pyplot as plt
count1=plt.bar(data['sex'],data['age'])


# In[ ]:


data.groupby(['region']).sex.value_counts()


# In[ ]:


import seaborn as sns
sns.catplot(x='smoker',kind='count',hue='sex',data=data)


# In[ ]:


sns.catplot(x='smoker',y='charges',kind='violin',hue='sex',data=data)


# In[ ]:


sns.boxplot(y='smoker',x='charges',data = data[(data.age == 18)],orient='horizontal')


# In[ ]:


sns.distplot(data['bmi'])


# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
encoder.fit(data['sex'])
data.sex=encoder.transform(data.sex)


# In[ ]:


encoder.fit(data['region'])
data.region=encoder.transform(data.region)


# In[ ]:


encoder.fit(data['smoker'])
data.smoker=encoder.transform(data.smoker)


# In[ ]:


cor=data.corr().charges.sort_values()


# In[ ]:


sns.heatmap(data.corr())


# In[ ]:


#data.hist(column=data['charges'])
 plt.np.histogram(data.charges)


# In[ ]:


sns.distplot(data[(data['smoker']==0)]['charges'])


# In[ ]:


f= plt.figure(figsize=(12,5))
axis=f.add_subplot(121)
sns.distplot(data[(data['smoker']==1)]['charges'],ax=axis)

axis.set_title('Distribution of non smokers')


# In[ ]:





# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


x=data.drop(['charges'],axis=1)
y=data.charges
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=100)


# In[ ]:


lr=LinearRegression().fit(x_train,y_train)


# In[ ]:


y_train_pred=lr.predict(x_train)
y_test_pred=lr.predict(x_test)
print(lr.score(x_test,y_test))


# In[ ]:


from sklearn.metrics import r2_score,mean_squared_error
r2_train=r2_score(y_train,y_train_pred)
r2_test=r2_score(y_test,y_test_pred)
print(r2_train,r2_test)
mse_train=mean_squared_error(y_train,y_train_pred)
mse_test=mean_squared_error(y_test,y_test_pred)
print(mse_train,mse_test)


# In[ ]:




