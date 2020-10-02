#!/usr/bin/env python
# coding: utf-8

# In this tutorial, we will see python basics for data science. 
# By the end of this tutorial, you will have  a good exposure to basic statistics, data munging, and data visualization.

# In[1]:


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


data=pd.read_csv('../input/pokemon.csv')


# In[3]:


#show top 5 values of the dataset
data.head()


# In[4]:


#all information about data
data.info()


# In[5]:


#summarize the data
data.describe()


# In[6]:


#show the columns of the dataset
data.columns


# **Introduction to Python**
# 
# Matplotlib: - This is a python library allow us to plot data. Different type of plots are as below:
# 
#     1. Line plot: - It is used if X axis is time.
#     2. Scatter plot: - Shows the relationship between two variables.
#     3. Histogram: - It is used to see the distribution of the data.

# In[7]:


import matplotlib.pyplot as plt

#line plot
data.Speed.plot(kind='line',color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Defense.plot(kind='line',color = 'g',label = 'Defense',linewidth=1,alpha = 0.5,grid = True,linestyle = '-')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('line plot')
plt.show()


# In[8]:


#scatter plot
data.plot(kind='scatter',x='Attack',y='Defense',alpha=0.5,color='red')
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.title('Scatter plot')
plt.show()


# In[9]:


#histogram
data.hist(column='Speed',bins=50,figsize=(12,12))


# **Dictionary in Python**
# 
# Each value in a dictionary will have a unique key.

# In[10]:


dict={'India':'Delhi','France':'Paris'}
print(dict.keys())
print(dict.values())


# In[11]:


dict['Australia']='Canberra'     #adds new entry to the dictionary
print(dict)
dict['India']='New Delhi'        #updates existing entry
print(dict)


# **Pandas**
# 
# It is a python library used for data manipulation and data analysis.

# In[12]:


series=data['Speed']
print(type(series))
dataframe=data[['Defense']]
print(type(dataframe))


# In[13]:


x=data['Defense']>200
data[x]


# In[14]:


#filtering data in pandas
data[np.logical_and(data['Defense']>200,data['Attack']<300)]


# In[15]:


#the above code can also be written as
data[(data['Defense']>200) & (data['Attack']<300)]


# **While and for loops**

# In[16]:


i=0
while(i !=5):
    print('i is:',i)
    i+=1
print(i,'is equal to 5')


# In[17]:


#global and local scope
x=2
def f():
    x=3
    return x
print(x)
print(f())


# In[18]:


#nested function
def square():
    def add():
        x=2
        y=4
        z=x+y
        return z
    return add()**2
print(square())


# In[19]:


#default and flexible arguments
def f(a,b=2,c=3):     #default
    y=a+b+c
    return y
print(f(1))
print(f(1,4,3))


# In[20]:


#flexible arguments
def f(*args):
    for i in args:
        print(i)
f(1)
print('---------')
f(1,2,3)


# In[21]:


def f(**kwargs):
    for key,value in kwargs.items():
        print(key,'', value)
f(country='India',capital='New Delhi')


# In[22]:


#lambda function
square= lambda x: x**2
square(5)


# In[23]:


#Iterators
x='Kaggle'
it=iter(x)
print(next(it))
print(*it)


# In[24]:


#zip lists
l1=[1,2,3]
l2=[4,5,6]
l=zip(l1,l2)
print(list(l))


# In[25]:


#list comprehension
x=[1,2,3]
y=[i+1 for i in x]
y


# In[26]:


x=[4,10,20]
y=[i**2 if i==5 else i-5 if i<6 else i+5 for i in x]
print(y)


# In[27]:


#performing list comprehension on our dataset
threshold=sum(data.Speed)/len(data.Speed)
data['Speed_level']=['high' if i>threshold else 'low' for i in data.Speed]
data.loc[:,['Speed_level','Speed']].head()


# In[28]:


data=pd.read_csv('../input/pokemon.csv')
data_new=data.head()
data_new


# In[29]:


# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted=pd.melt(frame=data_new,id_vars='Name',value_vars=['Attack','Defense'])
melted


# In[30]:


melted.pivot(index='Name',columns='variable',values='value')


# In[31]:


data['Type 2'].value_counts(dropna=False)


# In[32]:


#convert datatype
data['Type 1']=data['Type 1'].astype('category')


# In[33]:


data['Speed']=data['Speed'].astype('float')


# In[ ]:




