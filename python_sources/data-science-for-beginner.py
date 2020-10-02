#!/usr/bin/env python
# coding: utf-8

# Hi. I am working data science and want to share my learning.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# * Firstly we need to add the data we want to examine. We can build data frames from csv .

# In[ ]:


#We uses pandas library for add the data.
data = pd.read_csv('../input/DJIA_table.csv')


# * I'm curious about the number of rows and columns.

# In[ ]:


data.shape


# * We need to have information about our data for data analysis.

# In[ ]:


#Lets learn what are the columns names of our data. We call feature 
data.columns


# In[ ]:


#Learn more about our columns
data.info()


# * As you see data have 1989 entries and are non-null all of them and we can see columns types.
# 

# * Date column's type is object but  Datetime would be better.

# In[ ]:


#Lets do Date column's type Datetime  
data.Date= pd.to_datetime(data.Date)  #change of feature type
data.dtypes  #show features type


# * Now let's examine the first and last lines of data so that we can better understand.

# In[ ]:


#Show the top five entries in the database
data.head() 


# In[ ]:


#Gives the last five entries in the database
data.tail()


# * We can reach value we want the line in the column. This is indexing.

# In[ ]:


#We want to reach High column and 0.index
data.High[0]
#We can do that too 
#data["High"][1]
#data.loc[1,["High"]]


# In[ ]:


#We can reach High and Low column's line's all
data[["High","Low"]]


# In[ ]:


#or we can reach the first 5 lines
data.loc[:5,["High","Low"]]


# * We can sort the different columns we want. This is slicing

# In[ ]:


#Gives all columns between Open and Close
data.loc[:3,"Open":"Close"]


# * Now let's examine the data according to the conditions we want. This is filtring

# In[ ]:


filter1 = data.High > 10190
data[filter1]


# In[ ]:


filter2 = data.Low < 10000
data[filter2]


# In[ ]:


## Combining filters
data[filter1 & filter2]


# * We can create new column.

# In[ ]:


#New column is  High and Low column's sum
data["total"] = data.High + data.Low
data.head()


# * We can apply a function on columns of data. This is a apply() function

# In[ ]:


#Lets see how is it.
def operation(x):   #We create a function
    return x/10000
data.Open.apply(operation)  #We use the function with apply() 


# * If create function with lambda, the code is written on single line.

# In[ ]:


#Lets see
data.Open.apply(lambda x: x/10000) #of course we use apply() 


# * Remember we have index. Lets see 

# In[ ]:


#Learn index name
print(data.index.name)


# In[ ]:


#If you want to change
data.index.name = "index_name"
print(data.index.name)


# * Our index start to zero.  We wanted to star one?

# In[ ]:


data.index = range(1,1990,1)
data.head()


# * I want to show something i love 

# In[ ]:


#Create a new column. The column write rise for higher values than average else fall
high_mean = data.High.mean()  #calculate average
data["high_level"] = ["rise" if high_mean < each else "fall" for each in data.High]  #values are scanned in high column
data.head(1000)


# In[ ]:


#Create a new column. The column write rise for higher values than average else fall
low_mean = data.Low.mean()  #calculate average
data["low_level"] = ["rise" if low_mean < each else "fall" for each in data.Low]  #values are scanned in low column
data.head(1000)


# * We can use these two columns(high_level , low_level) as index.

# In[ ]:


data1 = data.set_index(["high_level","low_level"])
data1.head(1000)


# * Now i want to delete column.

# In[ ]:


#Delete the total column
data = data.drop(["total"],axis=1)
data.columns


# * I want to see a column's value.

# In[ ]:


#Values in the high_level column
data.high_level


# In[ ]:


#Different values in the high_level column
data.high_level.unique()


# * We can build dataframe from dictionaries.

# In[ ]:


dictionary = {"Sex":["F","F","M","M"],
              "Size":["S","L","S","L"],
              "Age": [10,48,24,35]}
data_1 = pd.DataFrame(dictionary)
data_1


# * We can change index and columns. This is pivoting.

# In[ ]:


#We new index sex and  columns size 
data_1.pivot(index = "Sex", columns = "Size", values = "Age")


# * We can use two columns(Sex, Size) as index.

# In[ ]:


#We use sex and size column as index
data_2 = data_1.set_index(["Sex","Size"])
data_2


# * We have more than one index. Now we have it down.

# In[ ]:


#Choose size for index
data_2.unstack(level=0)


# In[ ]:


#Choose sex for index
data_2.unstack(level=1)


# * We change column's location.

# In[ ]:


#I chenge size column and sex column location
data_3 = data_2.swaplevel(0,1)
data_3


# * Create list of values of choosed two column of wanted column. This is melt() function.

# In[ ]:


#Create list of values of size and age of each sex  column
pd.melt(data_1, id_vars = "Sex" , value_vars = ["Size","Age"])


# * Let's finally group.

# In[ ]:


data_1


# In[ ]:


#Calculate averange by sex
data_1.groupby("Sex").mean()


# In[ ]:


#Max value by size
data_1.groupby("Size").max()


# In[ ]:


#Min value by size
data_1.groupby("Size").min()

