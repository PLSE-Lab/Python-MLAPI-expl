#!/usr/bin/env python
# coding: utf-8

# # Do We Drink More Coffee During the Week?
# 
# Here's a fun little analysis of  data from a bakery called "The Bread Basket", which is located in the historic center of Edinburgh. Of course, this data is from one local in Scotland, so clearly cannot truly be generalized, but still fun to look at nonetheless. My assumption looking at this data is that we tend to drink more coffee on weekdays to power us through the work day.
# 
# This is a work in progress! Please upvote and leave a comment with what you'd like to see next, or questions about what I have done so far.

# ## Import and Clean Data
# 
# Will use pandas to import the data, retain only coffee transactions, group by date, and add a "day of week" variable.

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('../input/BreadBasket_DMS.csv')


# In[ ]:


coffee_only  = df[df['Item'] == 'Coffee']


# In[ ]:


coffee_only = coffee_only.groupby(['Date']).size().reset_index(name='counts')
coffee_only['day_of_week'] = pd.to_datetime(coffee_only['Date']).dt.day_name()


# ## Visualizations
# 
# Importing the seaborn library to work on some visualizations.

# In[ ]:


import seaborn as sns


# Using seaborn to generate a violin plot to explore the distribution of number of coffees sold by day of week.

# In[ ]:


sns.mpl.rc('figure', figsize=(9,6))
ax = sns.violinplot(x='day_of_week', y = 'counts', data = coffee_only, width = 0.8)
ax.set(xlabel='Day of Week', ylabel='Coffees Sold')
ax.set_title('Distribution of Coffees Sold by Day of Week')


# ## Initial Thoughts
# 
# The initial violin plot is quite interesting... on the face of it, it seems like more coffee is sold on the weekends (Saturdays, to be specific). Furthermore, it looks like there is quite a lot of variance in the number of coffees sold on Saturday, whereas days like Tuesday have a much more compact distribution.
# 
# More exploration is needed; there are obvious potentially-confounding variables still at play. For example, Edinburgh is quite tourism-heavy, so there could simply be a much larger number of patrons on the weekend. Viewing the data normalizing by total number of transactions may be the logical next step. 
# 
# ## Next Steps
# 
# More visualizations to come! Please leave any thoughts you have thus far!
