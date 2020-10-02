#!/usr/bin/env python
# coding: utf-8

# ***1. Introduction to Python***

# ***1.1 Matplotlib***

# In[ ]:


# Import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/creditcard.csv")
df.info()


# In[ ]:


df.head()


# In[ ]:


df.corr()


# In[ ]:


df.columns


# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(21,21))
sns.heatmap(df.corr(), annot=True, linewidths=.6, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


# Line Plot
df.Time.plot(kind = 'line', color = 'red',label = 'Time',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
df.Amount.plot(color = 'blue',label = 'Amount',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()


# In[ ]:


# Scatter Plot 
df.plot(kind='scatter', x='Time', y='Amount',alpha = 0.7,color = 'blue')
plt.xlabel('Time')          
plt.ylabel('Amount')
plt.title('Time Amount Scatter Plot')
plt.show()


# In[ ]:


# Histogram
df.Time.plot(kind = 'hist',bins = 75,figsize = (10,10))
plt.show()


# In[ ]:


# clf() = cleans it up again you can start a fresh
df.Time.plot(kind = 'hist',bins = 50)
plt.clf()


# ***1.2 Dictionary***

# In[ ]:


# Create dictionary and look its keys and values
dict = {"Volkswagen":"Passat", "Renault":"Megane"}
print(dict.keys())
print(dict.values())


# In[ ]:


# Update existing key value
dict["Volkswagen"] = "Golf"
print(dict)

# Add new key-value pair
dict["Nissan"] = "Micra"
print(dict)

# Remove one of entries
del dict["Renault"]
print(dict)

# Check key exist or not
print('Nissan' in dict)

# Remove all entries
dict.clear()
print(dict)

# Deallocate dict object from cache
del dict
print(dict)


# ***1.3 Pandas***

# In[ ]:


df = pd.read_csv("../input/creditcard.csv")
df.head()


# In[ ]:


# Define Series
series = df['Time']
print(type(series))

# Define DataFrames
data_frame = df[['Amount']]
print(type(data_frame))


# In[ ]:


print('A' == 'a')
print(0 == 'zero')
print(5 != 2)
print(False & False)
print(1<2 or 4>3)


# In[ ]:


# Filter data using greater condition
a = df["Amount"] > 12910.93
df[a]


# In[ ]:


df[np.logical_and(df['Time']>48401.0, df['Amount']>=19656.53)]


# In[ ]:


df[(df['Time']>48401.0) & (df['Amount']>=19656.53)]


# ***1.4 While and For Loop***

# In[ ]:


i=-3
while i<5:
    print('value i is:', i)
    i+=1
print('Current value for i is:', i)


# In[ ]:


dict = {"Volkswagen":"Passat", "Renault":"Megane"}

for key, value in enumerate(dict):
    print('key&value of dictionary is:', str(key)+'-'+value)


# In[ ]:


# Get Index&Value of data frame
for index,value in df[['V1']][0:2].iterrows():
    print(index," : ",value)


# ***2. Python Data Science Toolbox***

# ***2.1. User Defined Function***

# In[ ]:


a = df[df.Time>0.117396]
def min_time():
    print(a["Amount"])
min_time()


# In[ ]:


df[df.Time>0.117396]["Amount"]


# ***2.2 Nested Function***

# In[ ]:


def circle_circumference(x):
    pi = 3.14
    return 2*pi*x
print(circle_circumference(3))


# ***2.3  Lambda Function***

# In[ ]:


def square(x):
    print(x**2)
square(6)


# In[ ]:


square = lambda x:x**3+4
print(square(6))


# In[ ]:




