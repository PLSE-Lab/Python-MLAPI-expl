#!/usr/bin/env python
# coding: utf-8

# # **Loading the Libraries**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt


# ** Loading the Data into the DataFrame**

# In[ ]:


data=pd.read_csv('../input/canada1/Canada - Canada by Citizenship.csv')
data.head()


# In[ ]:


print("Data Count :")
pd.DataFrame(data.count())


# **When analyzing a dataset, it's always a good idea to start by getting basic information about your dataframe. We can do this by using the `info()` method.
# **

# In[ ]:


data.info()


# **Let's clean the data set to remove a few unnecessary columns. We can use *pandas* `drop()` method as follows:**

# In[ ]:


# in pandas axis=0 represents rows (default) and axis=1 represents columns.
data.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)
data.head(2)


# **Let's rename the columns so that they make sense. We can use `rename()` method by passing in a dictionary of old and new names as follows:**

# In[ ]:


data.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
data.columns


# **We will also add a 'Total' column that sums up the total immigrants by country over the entire period 1980 - 2013, as follows:**

# In[ ]:


data['Total'] = data.sum(axis=1)
data.head()


# **We can check to see how many null objects we have in the dataset as follows:**

# In[ ]:


data.isnull().sum()


# **Finally, let's view a quick summary of each column in our dataframe using the `describe()` method.**

# In[ ]:


data.describe()


# **Before we proceed, notice that the default index of the dataset is a numeric range from 0 to 194. This makes it very difficult to do a query by a specific country. For example to search for data on Japan, we need to know the corressponding index value.
# This can be fixed very easily by setting the 'Country' column as the index using `set_index()` method.**

# In[ ]:


data.set_index('Country',inplace=True)


# In[ ]:


data.head(3)


# Example: Let's view the number of immigrants from India for the following scenarios:
#     1. The full row data (all columns)
#     2. For years 1980 to 1985

# In[ ]:


# 1. the full row data (all columns)
pd.DataFrame(data.loc['India']).drop(['Continent', 'Region','DevName','Total']).plot()


# 
# 
# Since we converted the years to string, let's declare a variable that will allow us to easily call upon the full range of years:
# 

# In[ ]:


years = list(map(str, range(1980, 2014)))
years


# ### Filtering based on a criteria
# To filter the dataframe based on a condition, we simply pass the condition as a boolean vector. 
# 
# For example, Let's filter the dataframe to show the data on Asian countries (AreaName = Asia).

# In[ ]:


# 1. create the condition boolean series
condition = data['Continent'] == 'Asia'
print (condition)


# In[ ]:


# 2. pass this condition into the dataFrame
data[condition]


# In[ ]:


# we can pass mutliple criteria in the same line. 
# let's filter for AreaNAme = Asia and RegName = Southern Asia

data[(data['Continent']=='Asia') & (data['Region']=='Southern Asia')]


# # Get the data set for China and India, and display dataframe

# In[ ]:


df_CI = data.loc[['India', 'China'], years]
df_CI.head()


# **Plot graph. We will explicitly specify line plot by passing in kind parameter to plot().**

# In[ ]:


df_CI.plot(kind='line')


# **That doesn't look right...
# Recall that pandas plots the indices on the x-axis and the columns as individual lines on the y-axis. Since df_CI is a dataframe with the country as the index and years as the columns, we must first transpose the dataframe using transpose() method to swap the row and columns.**

# In[ ]:


df_CI = df_CI.transpose()
df_CI.head()


# In[ ]:


### type your answer here

df_CI.index = df_CI.index.map(int) # let's change the index values of df_CI to type integer for plotting
df_CI.plot(kind='line')

plt.title('Immigrants from China and India')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()


# ![](http://)# The trend of top 3 countries that contributed the most to immigration to Canada.

# In[ ]:


data.sort_values(by='Total', ascending=False, axis=0, inplace=True)



# get the top 5 entries
df_top3 = data.head(4)


# transpose the dataframe
df_top3 = df_top3[years].transpose() 



print(df_top3)


# Step 2: Plot the dataframe. To make the plot more readeable, we will change the size using the `figsize` parameter.
df_top3.index = df_top3.index.map(int) # let's change the index values of df_top5 to type integer for plotting
df_top3.plot(kind='line', figsize=(14, 8)) # pass a tuple (x, y) size



plt.title('Immigration Trend of Top 3 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')


plt.show()


# In[ ]:




