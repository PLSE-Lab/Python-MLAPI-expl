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


# Data is in comma seperated values (C.S.V) format

# In[ ]:


df=pd.read_csv('/kaggle/input/prediction-facebook-comment/Dataset.csv')
df.columns


# In[ ]:


df


# In[ ]:


import numpy as np #used for scientific computation
import pandas as pd #used for data mugging and preprocessing
import matplotlib.pyplot as plt #data visualization library
import seaborn as sns # stastical visualization library
import squarify #used to make square area plots
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df.describe()


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.corr().head()


# correlation map

# In[ ]:



f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


df.isnull().sum()


# Here yellow bars repersent the null values(missing values)

# In[ ]:


plt.figure(figsize=(10,5)) #plt is the object of matplot lib and .figure() is used to show or change properties of graphs
sns.heatmap(df.isnull(),cmap='viridis',yticklabels=False,cbar=False) #heatmaps are matrix plots which can visualize data in 2D
plt.show()


# **Dropping the collums**

# In[ ]:


df=df.drop(['shares'],axis='columns')
df=df.drop(['mon_pub'],axis='columns')
df=df.drop(['thu_pub'],axis='columns')
df=df.drop(['mon_base'],axis='columns')


# In[ ]:


df.columns


# In[ ]:


df.isnull().sum()


# In[ ]:


plt.figure(figsize=(50,100)) #plt is the object of matplot lib and .figure() is used to show or change properties of graphs
sns.heatmap(df.isnull(),cmap='viridis',yticklabels=False,cbar=False)#heatmaps are matrix plots which can visualize data in 2D
plt.show()


# In[ ]:


df['Returns']=df['Returns'].fillna(df['Returns'].mode()[0] )
df['Category']=df['Category'].fillna(df['Category'].mode()[0])
df['commBase']=df['commBase'].fillna(df['commBase'].mode()[0])
df['comm48']=df['comm48'].fillna(df['comm48'].mode()[0])


# In[ ]:


df.isnull().all()


# In[ ]:


plt.scatter(df.sat_base,df.output,marker=".",color="blue")
plt.xlabel('sat_base')
plt.ylabel('output')


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


columns=df.columns.tolist()

#filter the columns to remove data we do not want
columns=[c for c in columns if c not in ['output']]

#store the value we will predicting on
target='output'

x=df[['likes', 'Checkins', 'Returns', 'Category', 'commBase', 'comm24',
       'comm48', 'comm24_1', 'diff2448', 'baseTime', 'length', 'hrs',
       'sun_pub', 'tue_pub', 'wed_pub', 'fri_pub', 'sat_pub', 'sun_base',
       'tue_base', 'wed_base', 'thu_base','fri_base']]
y=df[target]


# In[ ]:


x.shape


# In[ ]:


y.shape


# *** Liner regression ***

# In[ ]:



from sklearn import datasets, linear_model, metrics 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,random_state=1) 
reg = linear_model.LinearRegression() 
reg.fit(x_train, y_train) 


# In[ ]:


print('Coefficients: \n', reg.coef_) 


# In[ ]:


print('Variance score: {}'.format(reg.score(x_test, y_test))) #not the best algo to implement the accuracy is 30%


# In[ ]:


reg.score(x_test,y_test)


# **Random Forest **

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=20)
model.fit(x_train, y_train)


# In[ ]:


model.score(x_test,y_test)


#  **Decision Tree **

# In[ ]:



from sklearn import tree
model = tree.DecisionTreeRegressor()
model.fit(x_train, y_train)
model.score(x_test,y_test)

