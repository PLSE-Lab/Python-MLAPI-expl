#!/usr/bin/env python
# coding: utf-8

# In[19]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


print(df.Platform.unique())
print(df.Genre.unique())


# In[9]:


wii=df[df.Platform=='Wii']
ps4=df[df.Platform=='PS4']
xone=df[df.Platform=='XOne']

plt.plot(wii.Platform,wii.EU_Sales,color='red',label='wii')
plt.plot(ps4.Platform,ps4.EU_Sales,color='blue',label='ps4')
plt.plot(xone.Platform,xone.EU_Sales,color='green',label='xone')


plt.legend()
plt.grid()
plt.show()


# In[10]:


plt.scatter(wii.NA_Sales,wii.EU_Sales, color='red',label='wii')
plt.scatter(ps4.NA_Sales,ps4.EU_Sales,color='blue',label='ps4')
plt.scatter(xone.NA_Sales,xone.EU_Sales,color='green',label='xone')


plt.grid()
plt.legend()
plt.xlabel('na sales')
plt.ylabel('eu sales')
plt.show()


# In[25]:


plt.hist(wii.Year_of_Release, bins=10,alpha=0.5)
plt.xlabel('Year of Release')
plt.show()

plt.hist(ps4.Year_of_Release, bins=5, alpha=0.5)
plt.xlabel('Year of Release')
plt.show()

plt.hist(xone.Year_of_Release,bins=5,alpha=0.5)
plt.xlabel('Year of Release')
plt.show()


# In[51]:


object=('Wii','Ps4','Xone')
y_pos=np.arange(len(object))
side=(8,3,12)

plt.bar(y_pos,side,align='center',color='green',alpha=0.5)
plt.xticks(y_pos)
plt.ylabel('Usage')
plt.show()


plt.barh(y_pos,side, align='center', color='green',alpha=0.5)
plt.yticks(y_side)
plt.xlabel('Usage')
plt.show()


# In[55]:


plt.subplot(2,1,1)
plt.plot(wii.Platform,wii.EU_Sales,color='red',label='wii')
plt.subplot(2,1,2)
plt.plot(ps4.Platform,ps4.EU_Sales,color='blue',label='ps4')
plt.show()




# In[57]:


data1=df.Year_of_Release.head()
data2=df.Year_of_Release.tail()

vertical_concat=pd.concat([data1,data2],axis=0)
print(vertical_concat)
print('')
horizantal_concat=pd.concat([data1,data2],axis=1)
print(horizantal_concat)


# In[58]:


df.corr()


# In[60]:


import seaborn as sns
sns.heatmap(df.corr())


# In[71]:


f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,ax=ax)


# In[74]:


dictionary={'Turkey':'Galatasaray','Spain':'Barcelona','France':'PSG','Germany':'Bayern Munih'}
print(dictionary.keys())
print(dictionary.values())


# In[78]:


dictionary['Germany']='Dortmund'
print(dictionary)

