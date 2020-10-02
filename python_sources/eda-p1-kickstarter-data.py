#!/usr/bin/env python
# coding: utf-8

# # *Problem description:*
# You've been hired by an angel investor to evaluate kickstarter projects to see what upcoming market trends will be based off of historical kickstarter data. Help them analyze this data set to gather key insights.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## 1. DataFrame attributes
# 
# a. Load the csv file corresponding to the year 2018 as a DataFrame named 'projects_2018' and display the first 5 rows. 

# In[ ]:


projects_2018 = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")
projects_2018.head()


# 
# b. Output the number of rows and columns in the DataFrame

# In[ ]:


print("There are", projects_2018.shape[0], "rows")
print("There are", projects_2018.shape[1], "columns")


# c. Display the names of the all the columns in the DataFrame

# In[ ]:


print(projects_2018.columns.values)


# ## 2. Select and Query

# a.Create a new DataFrame named 'projects_short' with these columns:
# 
# ['category', 'deadline', 'goal', 'launched', 'pledged', 'state', 'backers', 'country']

# In[ ]:


# Selecting a subset of our orginal dataframe to work with
projects_short = pd.DataFrame(data=projects_2018,columns=['name','category', 'deadline', 'goal', 'launched', 'pledged', 'state', 'backers', 'country'])


# b. Select projects that have greater than 100 backers from this new DataFrame
# 

# In[ ]:


more_than_100 = projects_short[projects_short.backers > 100]
print(more_than_100)


# c. Which project raised the most amount of money? 

# In[ ]:


projects_short.sort_values('pledged', ascending=False).iloc[0]


# d. Find the greatest number of backers for a single project, is this the same amount of backers we found for our previous step for the highest funded project? What could have caused this to occur?

# In[ ]:


maximum = projects_short.backers.max()
print(maximum)


# Possible answer: Some projects may utilize high dollar amount rewards more effectively, thereby lowering the amount of backers that they would need to hit a given goal amount.

# e. Which projects had less than 10 backers but more than $1000 in pledged dollars?

# In[ ]:


projects_short.query("pledged > 1000 and backers <=10")


# ## 3. Summary statistics + Groupby + Frequency

# a. Find the category with the highest average pledged amount

# In[ ]:


projects_short.groupby('category').mean().sort_values("pledged",ascending = False).iloc[0]


# ## 4. Sorting and plotting

# a. Count the amount of projects from each country and display it on a pie plot
# 

# In[ ]:


sorted_by_frequency = projects_short.groupby('country').size().sort_values(ascending=False).plot.pie()


# b. Determine which country recieved the highest level of backing on average
# 
# *Hint: groupby country and aggregate the backers with a simple mean() method*

# In[ ]:


projects_short.groupby('country').mean().sort_values("backers",ascending=False)


# c. Plot the relationship between the goal amount and the amount pledged

# In[ ]:


projects_short.plot.scatter(x="goal",y="pledged")

