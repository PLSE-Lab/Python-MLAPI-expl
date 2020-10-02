#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
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


#df=pd.read_csv("../input/crude-oil-prices/Crude Oil Prices Daily.xlsx",encoding='utf-8')

from pandas import ExcelWriter
from pandas import ExcelFile

df = pd.read_excel('../input/crude-oil-prices/Crude Oil Prices Daily.xlsx')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


plt.figure(figsize=(15,10))
df['Closing Value'].plot(kind='bar')
plt.title('Variation of oil price over years')
plt.ylabel('Price')
plt.xticks(color = 'w',rotation=90)


# In[ ]:


plt.figure(figsize=(15,10))
df['Closing Value'].plot(kind='line')
plt.title('Variation of oil price over years')
plt.ylabel('Price')
plt.xticks(color = 'w')


# In[ ]:


df['year']=df['Date'].dt.year


# In[ ]:


df.head()


# In[ ]:


x=df['year']
y=df['Closing Value']
plt.bar(x,y)
plt.title('Variation of Price over years')
plt.xlabel('Year')
plt.ylabel('Price in $')


# In[ ]:


df.tail()    # there are some null(NaN) values appearing the data


# In[ ]:


df['year'].isnull().sum()   


# So no need for preprocessing in year fiel

# In[ ]:


df['Closing Value'].isnull().sum()


# We have to preprocess the Closing value field by filling the values in place of Nan

# In[ ]:


df.fillna(df['Closing Value'].mean(),inplace=True)


# In[ ]:


x=df['year'].values.reshape(-1,1)
y=df['Closing Value'].values.reshape(-1,1)


# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)


# In[ ]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(xtrain,ytrain)


# In[ ]:


score=reg.score(xtest,ytest)
print(score)


# In[ ]:


from sklearn.metrics import accuracy_score    #Oil price in 2020
ypredict=reg.predict([[2020]])    
print(ypredict)


# In[ ]:


plt_train=plt.scatter(xtrain,ytrain,color='grey')
plt_test=plt.scatter(xtest,ytest,color='green')
plt.plot(xtrain, reg.predict(xtrain), color='black', linewidth=7)
plt.plot(xtest,reg.predict(xtest),  color='blue', linewidth=2)
plt.title('Regression over distribution of data',fontsize=20)
plt.xlabel("Year")
plt.ylabel("Oil Price")
plt.legend((plt_train, plt_test),("train data", "test data"))
plt.show()


# In[ ]:




