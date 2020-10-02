#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('/kaggle/input/insurance/insurance.csv')


# In[ ]:


data=df.copy()


# In[ ]:


data.shape


# In[ ]:


data.head(10)


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.sample(50)


# In[ ]:


data.region.value_counts()


# In[ ]:





# In[ ]:


data.columns


# In[ ]:


data.sex.value_counts()


# In[ ]:


data.smoker.value_counts().plot.bar()


# In[ ]:


data.corr().T


# In[ ]:


f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(data.corr(),annot=True,fmt='.2f',ax=ax)
plt.show()


# In[ ]:


data.corr()['charges'].sort_values()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
#sex
le = LabelEncoder()
le.fit(data.sex.drop_duplicates()) 
data.sex = le.transform(data.sex)
# smoker or not
le.fit(data.smoker.drop_duplicates()) 
data.smoker = le.transform(data.smoker)
#region
le.fit(data.region.drop_duplicates()) 
data.region = le.transform(data.region)


# In[ ]:


data.head()


# In[ ]:


f= plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.smoker == 1)]["charges"],color='c',ax=ax)
ax.set_title('Distribution of charges for smokers')

ax=f.add_subplot(122)
sns.distplot(data[(data.smoker == 0)]['charges'],color='b',ax=ax)
ax.set_title('Distribution of charges for non-smokers')
plt.show()


# In[ ]:


sns.catplot(x="smoker", kind="count",hue = 'sex', palette="pink", data=data)


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Box plot for charges of women")
sns.boxplot(y="charges", x="smoker", data =  data[(data.sex == 1)])


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of bmi")
ax = sns.distplot(data["bmi"], color = 'm')


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of charges for patients with BMI greater than 30")
ax = sns.distplot(data[(data.bmi >= 30)]['charges'], color = 'm')


# y=data.iloc[:,[6]]

# x=data.iloc[:,[0]]

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error


# In[ ]:


x


# In[ ]:


x = data.drop(['charges'], axis = 1)
y = data.charges

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state = 42)
lr = LinearRegression().fit(x_train,y_train)

print(lr.score(x_test,y_test))


# In[ ]:


pre=x.iloc[32:33,:]


# In[ ]:


pre


# In[ ]:


lr.predict([[17,1,26,1,0,2]])


# In[ ]:


lr.predict(pre)

