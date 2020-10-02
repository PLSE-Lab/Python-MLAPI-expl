#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv("../input/flavors_of_cacao.csv")
# Any results you write to the current directory are saved as output.


# In[ ]:


from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


data.head()


# Converting percetanges to decimal values

# In[ ]:


data["Cocoa\nPercent"] = data["Cocoa\nPercent"].apply(lambda x: float(float(x.split("%")[0])/100))
data.head()


# Checking for columns missing values

# In[ ]:


data.columns[data.isnull().any()].tolist()


# Finding correlation between features

# In[ ]:


sns.heatmap(data.corr(), annot=True)


# Only **REF** and **Review Date** have significant correlation between them

# In[ ]:


data.columns


# In[ ]:


# data[data.columns[8]].unique().tolist()


# In[ ]:


plt.figure(figsize=(8,6))
sns.distplot(data['Rating'],bins=5,color='red')


# The Rating distribution is skewed with major values between 2-4

# In[ ]:


plt.figure(figsize=(8,6))
sns.distplot(data['Cocoa\nPercent'],bins=5,color='red')


# Most of the chocolate bars have around 70% cocoa 

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x = data['Rating'], y = data['Cocoa\nPercent'], s=15)
plt.ylabel('Cocoa\nPercent', fontsize=13)
plt.xlabel('Rating', fontsize=13)
plt.show()


# Some inferences from the above graph:
# 1. Most chocolate bars having rating between 2-4 have cocoa content also around 70%.
# 2. Some of the chocolate bars with cocoa content around 100% have lower or average ratings. This tells us that there are other factors in play when ratings are given, which seems logical too.
# 3. The chocolate bar having rating 5 has cocoa content around 70%. This confirms our above statement.
# 4. Even the bars having cocoa content less than 50% have average ratings.

# Let's look at the chocolate bars with rating 5.

# In[ ]:


data[data["Rating"] == 5.0]


# Let's try to find out what could be the reason for this 5 rating.

# In[ ]:


data[data[data.columns[0]] == "Amedei"]


# So the bars made by **Amedei** are generally higher rated.

# In[ ]:


data[data[data.columns[1]].isin(["Chuao", "Toscano Black"])]


# So its not only the company but also the origin of the bean([Chuao](https://en.wikipedia.org/wiki/Chuao)) and name, which makes sense.

# In[ ]:


data_bean_type = data[data[data.columns[7]].isin(["Trinitario", "Blend", "Criollo"])]


# In[ ]:


data_bean_type[data_bean_type["Rating"] < 2.50]


# So on the basis of **Bean Type**, there are some bars with lower ratings. The bar with 100% cocoa content also falls in this category, even though it has the same Bean Type as that of the one with Rating 5. 
# 
# Lets see what the company **Hotel Chocolat** tells us.

# In[ ]:


data[data[data.columns[0]] == "Hotel Chocolat"]


# So it has bars in all the categories, interesting! Looks like **too much cocoa spoils the bar**.

# In[ ]:


data[data[data.columns[4]] > 0.9]


# WOW!!
# 
# Looks like I'm right upto some extent.
