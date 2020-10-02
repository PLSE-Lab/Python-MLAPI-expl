#!/usr/bin/env python
# coding: utf-8

# **1.  Introduction to Python**

# **  1.1. Matplotlib**

# In[ ]:


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/tmdb_5000_movies.csv")
df.info()


# In[ ]:


df.head()


# In[ ]:


df.corr()


# In[ ]:


df.columns


# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


# Line Plot
df.budget.plot(kind = 'line', color = 'yellow',label = 'Budget',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
df.revenue.plot(color = 'blue',label = 'Revenue',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()


# In[ ]:


# Scatter Plot 
df.plot(kind='scatter', x='budget', y='revenue',alpha = 0.7,color = 'orange')
plt.xlabel('Budget')              # label = name of label
plt.ylabel('Revenue')
plt.title('Budget Revenue Scatter Plot')
plt.show()


# In[ ]:


# Histogram
df.vote_average.plot(kind = 'hist',bins = 100,figsize = (12,12))
plt.show()


# In[ ]:


# clf() = cleans it up again you can start a fresh
df.vote_count.plot(kind = 'hist',bins = 50)
plt.clf()


# **  1.2. Dictionary**

# In[ ]:


# Create dictionary and look its keys and values
dict = {"Mobile Phone":"iPhone", "Computer":"Lenovo"}
print(dict.keys())
print(dict.values())


# In[ ]:


# Update existing key value
dict["Computer"] = "Toshiba"
print(dict)

# Add new key-value pair
dict["Smart Watch"] = "Samsung Gear"
print(dict)

# Remove one of entries
del dict["Mobile Phone"]
print(dict)

# Check key exist or not
print('Smart Watch' in dict)

# Remove all entries
dict.clear()
print(dict)

# Deallocate dict object from cache
del dict
print(dict)


# **  1.3. Pandas**

# In[ ]:


df = pd.read_csv("../input/tmdb_5000_movies.csv")
df.head()


# In[ ]:


# Define Series
series = df['budget']
print(type(series))

# Define DataFrames
data_frame = df[['revenue']]
print(type(data_frame))


# In[ ]:


print('a' == 'b')
print('c' == 1)
print(True & True)
print(3>2 or 2<1)


# In[ ]:


# Filter data using greater condition
a = df["revenue"] > 1487965080
df[a]


# In[ ]:


df[np.logical_and(df['revenue']>1487965080, df['budget']>=200000000)]


# In[ ]:


df[(df['revenue']>1487965080) & (df['budget']>=200000000)]


# **  1.4. While and For Loop**

# In[ ]:


i=9
while i>4:
    print('value i is:', i)
    i-=1
print('Current value for i is:', i)


# In[ ]:


df_temp = df[(df['revenue']>1487965080) & (df['budget']>=200000000)]
for i in df_temp['title']:
    print('The most expensive movie is:', i)


# In[ ]:


for index, value in enumerate(df_temp['title']):
    print('Index&value of The most expensive movie is:', str(index)+'-'+value)


# In[ ]:


dict = {"Mobile Phone":"iPhone", "Computer":"Lenovo"}
for key, value in enumerate(dict):
    print('key&value of dictionary is:', str(key)+'-'+value)


# In[ ]:


# Get Index&Value of data frame
for index,value in df_temp[['title']][0:2].iterrows():
    print(index," : ",value)


# 

# **2.  Python Data Science Toolbox**

# **2.1. User Defined Function**

# In[ ]:


a = df[df.budget>270000000]
def highest_budget():
    print(a["title"])
highest_budget()


# In[ ]:


df[df.budget>270000000]["title"]


# **2.2. Nested Function**

# In[ ]:


def square_triangle(x):
    pi = 3.14
    def square():
        return x**2
    return(pi*square())
print(square_triangle(4))


# **2.3. Lambda Function**

# In[ ]:


def square(x):
    print(x**2)
square(4)


# In[ ]:


square = lambda x:x**2+5
print(square(4))


# **2.4. Iterators**

# In[ ]:




