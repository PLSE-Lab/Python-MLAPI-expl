#!/usr/bin/env python
# coding: utf-8

# # **CLEANING DATA**

# **When you are analyzing any data, there could be some unclean data. So, what is the *unclean data*?**
# 
# **First of all, the column names may cause problems for you: Upper-lower case letters will be irregular, there may be spaces between the words etc.**
# 
# **And then, you may experience some missing data. **
# 
# **As you know there are thousands of languages around the world and some of them uses different alphabets and writing systems. So, it would be a problem in your analyses.**
# 
# **All of these examples correspond to *unclean data*.**
# 
# **In conclution, perhaps we will never encounter a proper/perfect data set throughout our life. Let's learn together, how to deal with it!**

#  **P.S. IN PREPARING THIS DOCUMENT, I MOSTLY USED THE KERNEL CALLED "Data ScienceTutorial for Beginners". https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners**
# 
#  **I AM PREPARING THIS KERNEL TO LEARN, NOT TO TEACH. BUT HOPEFULLY, IT WILL ALSO HELPFUL FOR YOU. ENJOY! **

# In[ ]:


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


# In[ ]:


data = pd.read_csv("../input/athlete_events.csv")


# # **DIAGNOSIS AND DISCOVERY**

# At the begining we should diagnose or explore data. With *info, head, tail etc.* methods

# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


data.tail()


# As you see there is some missing data(NaN). In column names, there is no space between the words but there is upper-lower case letters. With this methods we examine some basic problems in our data set. Also, we become familiar with data. There may be other problems, it is just a foreknowledge. 

# # **EXPLORATORY DATA ANALYSIS**

# **value_counts():** As the name suggests, it counts the values. Let's try it:
# 
# P.S. If you use this method like that *value_counts()* or *value_counts(dropna = True)*, you will not see the number of missing data. With *dropna = False* you can check the number of missing data.

# In[ ]:


print((data.Age).value_counts(dropna = False))


# To get some basic statistical information we can use **describe()** method.

# In[ ]:


data.describe()


# # **VISUAL EXPLORATORY DATA ANALYSIS**

# **boxplot():** it visualize basic statistical information.

# In[ ]:


data.boxplot(column = "Age")


# In this graph:
# * The black line at top is max.
# * The blue line at top is %75.
# * The green line is median of the age column.
# * The blue line at the bottom is %25.
# * The black line at the bottom is min.
# * The circles are outliers.
# 

# # **TIDY DATA**

# We will use **melt()** function for tidy data. At this step, our purpose is not analyzing. We want to standardize our data for analysis. With standardization i mean that, we will reshape data set without changing the data. 

# In[ ]:


data_tidy = data.head()
data_tidy


# In[ ]:


melted = pd.melt(frame = data_tidy, id_vars = "Name", value_vars = ["Weight","Height"])
melted


# # **PIVOTING DATA**

# Reverse of melting with **pivot()**.

# In[ ]:


melted.pivot(index = "Name", columns = "variable", values = "value")


# # **CONCATENATING DATA**

# We can put together two different dataframe with **concat()**.

# **1) Vertical Concatenating Example**

# In[ ]:


data1 = data.head()
data2 = data.tail()
conc_row = pd.concat([data1,data2], axis = 0, ignore_index = True)
conc_row


# **2) Horizontal Concatenating Example**

# In[ ]:


data3 = data.Name.tail()
data4 = data.Age.tail()
conc_col = pd.concat([data3,data4], axis = 1)
conc_col


# # **CHANGING DATA TYPES**

# In[ ]:


data.dtypes


# In[ ]:


data.Name = data.Name.astype('category')
data.Year = data.Year.astype('float')


# In[ ]:


data.dtypes


# # **MISSING DATA**
# 

# In[ ]:


data.info()


# In[ ]:


data.Medal.value_counts(dropna = False)


# As you see, 85% of athletes medal information is NaN. In other words, a large part of the data is missing. Let's try to fix that problem.
# 
# In this situation, dropping is not a good solution. But as you know we are just learning. So let's start with dropping with:

# In[ ]:


data.Medal.dropna(inplace = True)


# In[ ]:


assert data.Medal.notnull().all()


# Assertions are simply boolean expressions that checks if the conditions return true or not. If it is true, the program does nothing and move to the next line of code. However, if it's false, the program stops and throws an error.(Source: https://www.programiz.com/python-programming/assert-statement)

# In[ ]:


data.Medal.value_counts(dropna = False)


# As another solution, let's fill the missing values with mean of that value:
# * The number of missing data in *Age* column is 9474 and the mean of age value is 25.55898. 

# In[ ]:


data.Age.value_counts(dropna = False)


# In[ ]:


data.Age.describe()


# In[ ]:


data.Age.fillna(float(26), inplace = True)


# In[ ]:


assert data.Age.notnull().all()


# In[ ]:


data.Age.value_counts(dropna = False)

