#!/usr/bin/env python
# coding: utf-8

# # Traffic Collision Data: A Pandas Tutorial
# 
# Welcome Programmers!!
# 
# Pandas is a very effective Python module that allows programmers to clean, manage, analyze, and extract insights from any datasets. In this tutorial, we will be using fundamental, yet effective operations of Pandas by analyzing LA's traffic collision data from 2010 to 2019. Let's see what Pandas can tell us about this data!

# a. Let's first import our module from Python library.

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


# # **1. Display**
# 
# The first step of data analysis is to load the data set we will be working with and assigning it as a DataFrame to a variable. 
# By doing so, we can easily manage and manipulate our data sets. 
# 
# *Format*
# > DataFrame = pd.read_csv(file directory)
# 
# Once we have created a DataFrame, we can display the first 5 rows of the data sets using the .head() function. The default is set to 5 rows, which you can assign to a different number by inserting a number as an argument to the .head() function. You can also display the last 5 rows by using the .tail() function.
# 
# *Format*
# > DataFrame.head()
# 
# 
# 
# a. Load the csv file as a DataFrame named 'collision' and display the first 5 rows.

# In[ ]:


collision = pd.read_csv('../input/traffic-collision-data-from-2010-to-present/traffic-collision-data-from-2010-to-present.csv')

# Renaming the columns to get rid of spaces
cols = collision.columns
cols = cols.map(lambda x: x.replace(' ', '_'))
collision.columns = cols

collision.head()


# # 2. Data Attributes
# 
# Pandas has multiple methods to identify what attributes the data sets have. 
# * .shape: Output the number of rows and columns of the data set.
# * .columns: Output the names of column as a list
# 
# 
# 
# 
# Let's identify how big of a data set we are dealing with.
# 
# **a. How many collisions have occurred in LA during 2010 - 2019?**

# In[ ]:


collision.shape[0]


# **b. Output the column names as a list**

# In[ ]:


collision.columns


# # 3. Select
# 
# Now that we have a clear idea of what kind of data set we are dealing with, let's learn how we can select specific types of data from our data set. There are two major ways of doing this:
# 1. Using index or specific element
# 2. Using conditional statement
# 
# To use index, you can use .iloc function to specify the row numbers that you would like to see. If you would like to search a specific element, you can use the .loc function with the name of the element in first column to identify. 
# 
# *Format*
# > .iloc[#:#]
# 
# > .loc['Element_Name']
# 
# To use conditional statement, you will be inserting the condition you want inside of a square bracket. 
# 
# *Format*
# >DataFrame[Condition about DataFrame]
# 
# 
# Let's see how we can use these to select specific data. 
# 
# **a. Display the traffic collision occurred at Hollywood only**

# In[ ]:


# Hint: Use conditioning to filter data from Hollywood
collision[collision.Area_Name == 'Hollywood']


# We can use all select methods to identify a very specific data element from the extensive amount of data. 
# 
# Here is the situation:
# 
# **b. You are an investigator of a collision case that occurred in Hollywood at 14:50. Find the report number of this collision given that the victim is a 29 years old female.**

# In[ ]:


# Hint: Use the conditioning to sort out the victim profile and the time occurred. 
# Hint: Then set 'Area_Name' as the index, and access the first element (the report number) in the row indexed as 'Hollywood'
collision[(collision.Victim_Age == 29.0) & 
          (collision.Victim_Sex == 'F') & 
          (collision.Time_Occurred == 1450)]\
    .set_index('Area_Name')\
    .loc['Hollywood']\
    .iloc[0]


# # 4. Summarize
# 
# As part of data analysis, you would want to summarize what you have found and Pandas has various tools to help you with that. 
# * sum(): Output the sum of the rows
# * mean(): Output the average of the rows
# * median(): Output the median of the rows
# * std(): Output the standard deviation of the rows
# 
# Let's find out what is the average time of collision occurred and victim's age.
# 
# **a. Compute the average of collision time occurred and victim's age**

# In[ ]:


collision[['Time_Occurred','Victim_Age']].mean()


# Data can be even more effective when it is sorted and summarized using a visual representation. 
# Pandas allows you to sort your data via a specific filter in a descending or ascending order. 
# 
# *Format
# > DataFrame.sort_values('Filter', ascending = True/False)
# 
# To represent your findings with a visual representation, Pandas provide numerous tools to do that:
# * DataFrame.plot(): Output a line graph
# * DataFrame.plot.bar(): Output a bar graph
# * DataFrame.plot.barh(): Output a horizontal bar graph
# * DataFrame.plot.pie(): Output a pie chart
# 
# Let's find out if there is any time of the day that collisions occur more frequently. For policy makers, these findings might allow them to put more precautions or measures to prevent collisions in these times. 
# 
# 
# b. Create a bar chart that summarizes the top 20 frequent time of collision occurred

# In[ ]:


# Hint: First sort the values by 'Time_Occurred' in descending order and count the frequencies of each time. 
# Hint: Select only the top 20 time and plot as a bar graph.

collision.sort_values('Time_Occurred', ascending = False).Time_Occurred.value_counts().iloc[:20].plot.bar()


# It seems that collisions occur frequently ranging from 15:00~21:00, corresponds with when people get off from work. 

# # 5. Split-Apply-Combine, Frequency Counts

# Another great benefit of using Pandas is that it allows us to easily group data based on specific criteria and summarize those data. It can do so by splitting the data by specific columns, apply summary statistics, and combines these data. Programmers can perform this operation by using the .groupby() function.
# 
# *Format
# > .groupby('Column_Name')
# 
# 
# Let's see what areas in LA have the most number of collisions occurred. For a policy maker, this information could be useful to allocate police and transportation workforce more efficiently to prevent traffic collisions.
# 
# 
# **a. Create a horizontal bar graph that summarizes the area of collision occurred**

# In[ ]:


# Hint: First group the data set by 'Area_Name' and count the frequencies using .size() function.
# Hint: Sort values in ascending order and plot as a horizontal bar graph.

collision.groupby('Area_Name').size().sort_values(ascending = True).plot.barh()


# Let's see how number of collisions have been on a yearly basis. Since the data does not provide the year per se, we will be creating a new column called 'Year', which only takes account of the year. 
# 
# * Format:
# > DataFrame['New_Column'] = Whatever the New column is about
# 
# b. Create a new column called 'Year' that only takes the year of collisions occurred. 
#    Then plot a bar chart the summarizes yearly trend of number of collisions occurred.

# In[ ]:


# Hint: Parse the first 4 letters of elements in the 'Date_Occurred' column using .str[] function.

collision['Year'] = collision['Date_Occurred'].str[:4]

collision.groupby('Year').size().plot.bar()


# The number of collisions seems to have increased especially over 2016-2018. Let's take a closer look at the data by looking at monthly trend.
# 
# c. Create a new column called 'Year_Month' that takes account of the year and the month of collisions occurred. Then draw a line graph that shows the monthly trend of collisions.
# 

# In[ ]:


# Adjusting the dimensions of the graph for a better visual view
import matplotlib.pyplot as plt
plt.figure(figsize=(15,6))


collision['Year_Month'] = collision['Date_Occurred'].str[:7]
collision.groupby('Year_Month').size().plot()


# # Conclusion

# From our data analysis using Pandas, we can conclude that there has been monthly and yearly increase in the number of collisions in LA during the period 2010-2019. Collisions occurred more frequently in areas such as 77th street, Southwest, Wilshire, and West LA. Collisions occurred more frequently at the time frame of 15:00~21:00. These findings could be very useful especially for policy makers or civil engineers who would be interesting in putting preventative measures to reduce the number of collisions in LA.
# 
# I hope you enjoyed my tutorial on Pandas and hopefully you were able to learn something from me. Pandas is a very effective tool that has so much potential when used properly and creatively. Why don't you find a data set that interests you and try to do an analysis yourself?!
# 

# In[ ]:




