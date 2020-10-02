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


df=pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv',index_col='App')
import seaborn as sns
df.head()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df['Installs']=df['Installs'].apply(lambda x:x.replace("+",""))


# In[ ]:


df['Installs']=df['Installs'].apply(lambda x:x.replace(",",""))


# In[ ]:


df=df[df["Installs"]!='Free']
df['Installs']=df['Installs'].astype(int)


# In[ ]:



sns.barplot(y='Installs',x='Type',data=df)


# Well who doesn't like **Free** apps!

# In[ ]:


plt.figure(figsize=(18,4))
sns.barplot(y='Installs',x='Category',data=df)


# In[ ]:


plt.figure(figsize=(20,4))
sns.barplot(x='Android Ver',y='Installs',data=df)


# In[ ]:


df['Reviews']=df['Reviews'].astype(int)


# In[ ]:


plt.figure(figsize=(12,4))
sns.distplot(a=df['Rating'],kde=False)


# **Most of the ratings lie in the 4.0-4.5 region**

# In[ ]:


plt.figure(figsize=(14,4))
sns.barplot(y='Reviews',x='Rating',data=df)


# In[ ]:


def func(x):
    f=list(x)
    if 'k' in f:
        return float(x.replace("k",''))/1000
    elif 'M' in f:
        return float(x.replace('M',''))
    elif x=='Varies with device':
        return 11
df['Size']=df['Size'].apply(lambda x:func(x))


# In[ ]:


plt.figure(figsize=(16,4))
sns.distplot(a=df['Size'])


# ****Most of the apps have a size in range of 0-20 MB****

# In[ ]:


plt.figure(figsize=(20,4))
sns.barplot(x='Content Rating',y='Installs',data=df)


# In[ ]:


df=df.reset_index()


# In[ ]:


android_ver=pd.get_dummies(df['Android Ver'])
df=df.join(android_ver)


# In[ ]:


def func1(x):
    x=x.replace('$','')
    return float(x)


# In[ ]:


df['Price']=df['Price'].apply(lambda x:func1(x))


# In[ ]:


category=pd.get_dummies(df['Category'])
type=pd.get_dummies(df['Type'])
content=pd.get_dummies(df['Content Rating'])
df=df.join(category)
df=df.join(type)
df=df.join(content)


# In[ ]:


df.drop(['Rating','Reviews','Type','Content Rating','Genres','Current Ver','Last Updated','Android Ver','Category'],axis=1,inplace=True)


# In[ ]:


df=df.set_index('App')
df['Installs']=df['Installs'].astype(str)+'+'


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score,precision_score


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(df.drop('Installs',axis=1),df['Installs'],random_state=0)


# **Changing the labels**

# In[ ]:


y_train=y_train.replace({'10000000+':'In Millions','1000000+':'In Millions','500000000+':'In Millions','100000000+':'In Millions','500000+':'In Lakhs','100000+':'In Lakhs',
                 '50000000+':'In Millions','5000000+':'In Millions','1000+':'In Thousands','10000+':'In Thousands','100+':'In Hundreds','5000+':'In Thousands',
                '5+':'In Hundreds','10+':'In Hundreds','50000+':'In Thousands','500+':'In Hundreds','50+':'In Hundreds','1000000000+':'In Millions',
                '1+':'In Hundreds','0+':'In Hundreds'})
y_test=y_test.replace({'10000000+':'In Millions','1000000+':'In Millions','500000000+':'In Millions','100000000+':'In Millions','500000+':'In Lakhs','100000+':'In Lakhs',
                 '50000000+':'In Millions','5000000+':'In Millions','1000+':'In Thousands','10000+':'In Thousands','100+':'In Hundreds','5000+':'In Thousands',
                '5+':'In Hundreds','10+':'In Hundreds','50000+':'In Thousands','500+':'In Hundreds','50+':'In Hundreds','1000000000+':'In Millions',
                '1+':'In Hundreds','0+':'In Hundreds'})


# In[ ]:


for x in range(15,40):
        forest=RandomForestClassifier(max_depth=x,n_estimators=100)
        forest.fit(X_train,y_train)
        y_pred=forest.predict(X_test)
        print('depth= ',x,' estimators= ',100,'accuracy= ',accuracy_score(y_test,y_pred))


# In[ ]:


forest1=RandomForestClassifier(max_depth=19)
forest1.fit(X_train,y_train)
y_pred=forest1.predict(X_test)
final_df=pd.DataFrame({'Pred':y_pred,'Actual':y_test})


# In[ ]:


miss_class=final_df[final_df['Pred']!=final_df['Actual']]


# In[ ]:


h=[]
for x in miss_class.index:
    h.append(df.loc[x]['Installs'])
    


# In[ ]:


final_df_miss_class=pd.DataFrame({'Pred':miss_class['Pred'],'Actual':miss_class['Actual'],'Installs':h})


# **All mis-classified rows from the test set**

# In[ ]:


pd.set_option('display.max_rows',None)
final_df_miss_class


# In[ ]:




