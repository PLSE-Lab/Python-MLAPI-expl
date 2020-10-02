#!/usr/bin/env python
# coding: utf-8

# # Game of Thrones: Pandas Tutorial

# # What are Pandas?
# Pandas is a fast, powerful, and easy to use open source data analysis and manipulation tool, built in python. Today we will be demonstrating a multitude of methods in Pandas through the analysis of a dataset showcasing the battles in the popular series, Game of Thrones. 

# # Import
# First and foremost, we begin by loading and importing the Pandas method and the dataset csv file(s) and saving them as DataFrames, otherwise known as objects which represent data in tables.
# 
# In this tutorial, we save them under the names 'battles' and 'deaths.'

# In[ ]:


import pandas as pd
battles = pd.read_csv('/kaggle/input/game-of-thrones/battles.csv')
deaths = pd.read_csv('/kaggle/input/game-of-thrones/character-deaths.csv')


# # Display
# Oftentimes when working with datasets, it can be overwhelming. There can be what seems like an abundance of information to work with and you may want to selectively view bits and pieces of the dataset at a time - rather than the entire thing at once.
# 
# You can use the .head() function to display the top number of rows and the .tail() function for the bottom rows in a dataset. If you would like to specify how many rows, use the parenthesis () to fill in number of your choice. Otherwise, a default of 5 rows will be displayed. 
# 
# 1. Examine the datasets and display the top 7 rows and the bottom 5 rows.

# In[ ]:


battles.head(7)


# In[ ]:


# Since there are no specifications for rows in () - default 5 rows will be displayed
deaths.tail()


# # Data Attributes
# Data attributes are the characteristics of a dataset and we will observe two attributes in this step. 
# * .column - Finding the column names listed throughout the dataset
# * .shape - With shape we find the size of the dataset (how many rows and columns exist)
# 
# **Note:** .shape[0] will only display the output of rows and .shape[1] will only display the column output

# 2. What are the names of the columns as listed in the DataFrame? (Hint: Use the .column method)

# In[ ]:


battles.columns


# 3. How many rows and columns are in the datasets? (Hint: Use the .shape method)
# 
# 
# You can use this method to compare the dimensions of both datasets and assess what sizes you are working with. 

# In[ ]:


battles.shape


# In[ ]:


deaths.shape[1]


# In[ ]:


deaths.shape[0]


# # Select
# Since we've taken a closer look at our Dataset now, we can select and locate specific data within the larger set. The methods for select are:
# * .iloc[] - selecting by row numbers
# * .loc[] - selecting by labels
# * conditional - specifying conditions within the selection []
# 

# 4. Display the list of deaths ranging from rows seven through eleven (Hint: Use iloc[])

# In[ ]:


deaths.iloc[7:11]


# 5. How did battles in the year 299 affect the number of major_deaths? (Hint: Use conditional selecting)

# In[ ]:


battles[battles.year == 299]


# # Summarize
# Summary methods in Pandas include:
# * .sum() - row sum
# * .mean() - row average
# * .median() - row median
# * .std() - row standard deviation 
# 

# 6. What is the average number of deaths for all the battles in the dataset? (Hint: Use .mean())

# In[ ]:


round(battles.major_death.mean(),2)


# 7. Now let's say I want to combine Select and Summarize and find the average number of deaths for battles in only the year 299. How can I specify this? (Hint: Use conditional select and .mean() together)

# In[ ]:


battles[ battles.year == 299 ].major_death.mean()


# # Sort
# Now that we are able to Select and Summarize, we can add Sort as our step 3. After examining average deaths and deaths per year, let's say we want to find the year with the highest number of battles and the year with the lowest.
# 
# Sort methods in Pandas include:
# * .sort_index() and sort_values(). - Displays values in the table in an order from highest to lowest or vice versa
# * .max() and .min() - returning the maximum/minimum value
# * .idxmax() and .idxmin() - returning the index of the maximum/minimum value
# 

# 8. List the highest number of battles that took place in a year using two different methods. (Hint: Use .max() and .sort_values())

# In[ ]:


battles.year.max()


# In[ ]:


battles.year.sort_values(ascending=False).iloc[0]


# # Split-Apply-Combine
# Using this we can easily group data and display a summary of a specific statistic we select in the parenthesis () of our groupby. 
# First, it *splits* the data according to column, then *applies* the summary statistic, and lastly it *combines* the dataset into a new one.
# 
# Split-apply-combine methods include:
# * .groupby()

# 9. Use .groupby() to find the average major capture per battle.

# In[ ]:


battles.groupby('name').major_capture.mean()


# # Frequency Counts
# This is yet another way to find the size of a particular group. For instance, if we wanted to count the number of attacker kings throughout all the years, we can do so through the frequency count method:
# * .groupby().size()

# 10. Find the number of battles fought by each attacker king, respectively. 

# In[ ]:


battles.groupby('attacker_king').size()


# # Plots 
# This tool is great for visualizing the data we have gathered. There are a multitude of plots to pick from including: bar, line, pie, and scatter. The method for plots is:
# * .plots()
# When picking a type of plot, you simply add it before the parenthesis like this: .plot.bar()
# * you can also label your y and x values accordingly!

# 11. Display a pie chart to distribute the number of deaths.
# 
# From the graph, it is clear that there are more battles in which there are no deaths than there are a maximum of one.

# In[ ]:


battles.groupby('major_death').size().plot.pie()


# Display a bar chart that showcases the ranges of individuals who died in relation to each allegience.
# 
# From the graph, it seems that most individuals who died do not have an allegiance to any particular house. However, out of those that do it seems that the Night's Watch has suffered the most deaths and House Arryn has suffered the least.

# In[ ]:


deaths.groupby('Allegiances').size().plot.bar(x='Allegiance', y='Name')


# # The End
# The world of data has much to offer and countless possibilities in allowing us to predict, analyze, and visualize. Hope you enjoyed my tutorial and learned something new from it!
