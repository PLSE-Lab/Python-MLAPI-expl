#!/usr/bin/env python
# coding: utf-8

# Setting up the environment

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
print("Setup Complete")


# Example of the data:

# In[ ]:


crimes_filepath = "../input/crimes-in-boston/crime.csv"

crimes_data = pd.read_csv(crimes_filepath, index_col=0, encoding="latin-1")

crimes_data.head()


# There were two tricky parts to importing this data.
# 1. index_col="id" threw an error. I used index_col = 0 instead.
# 2. I was being told that certain bytes could not be read with utf-8 encoding. So I hardcoded encoding="latin-1".

# Questions:
# 1. What day of what month generally has the highest amount of crime?
# 2. Is there a relationship between type of crime committed and the time of day?
#    - The best way to do this depends on the total number of crime types. 
# 
# Jobs to be done:
# 1. Create a heatmap with days of the week and month of the year on the X and Y axis, respectively
# 2. Create a swarmplot with type of crime committed on the x axis and time of day on the y axis.
# 

# Let's start with the heatmap. In Excel, I would first create a pivot table summing the number of crimes committed on each day of the week with month of the year as an index. Then I would build a visualization. So first I need to figure out how to make that pivot table. Let's just start by firing up the engine to see if I can generate ANY pivot table:

# In[ ]:


pd.pivot_table(crimes_data, index=["DISTRICT", "MONTH"])

Seems to work! Let's drop all the data we don't need for the heatmap and put it into a new dataframe:
# In[ ]:


heatmap_data = crimes_data.filter(['MONTH','DAY_OF_WEEK'],axis=1)
heatmap_data.head()


# Now for the pivot table:

# In[ ]:


heatmap_pivot_table = pd.pivot_table(heatmap_data, index=["MONTH"], columns=["DAY_OF_WEEK"], aggfunc=[len])
heatmap_pivot_table.head()


# And finally, the heatmap:

# In[ ]:


plt.figure(figsize=(14,7))
plt.title("Total amount of crime by days of the week")
sns.heatmap(data=heatmap_pivot_table)


# What do you see? What might be the reason behind these trends? How might the Boston Police Department change their operations to correct for this?

# In[ ]:


swarmplot_data = crimes_data.filter(['OFFENSE_CODE_GROUP','OCCURRED_ON_DATE'], axis=1)
swarmplot_data.head()


# We have to remove the dates, leaving just the time. This is a little tricky. 
# On the first line, we convert the the column to the datetime format, so we can then call the time object.

# In[ ]:


swarmplot_data['OCCURRED_ON_DATE'] = swarmplot_data['OCCURRED_ON_DATE'].str.slice(10, 13, 1)
swarmplot_data['OCCURRED_ON_DATE'].head() 


# Now for the swarmplot:

# In[ ]:


swarmplot_data['OCCURRED_ON_DATE'] = swarmplot_data['OCCURRED_ON_DATE'].astype(int)
swarmplot_data.head() 


# And now for the swarmplot:

# In[ ]:


plt.figure(figsize=(75,100))
plt.title("Crimes in Boston")
plt.yticks(rotation=45)
sns.set(font_scale=5) 
sns.swarmplot(y=swarmplot_data['OFFENSE_CODE_GROUP'].head(2000), x=swarmplot_data['OCCURRED_ON_DATE'].head(2000), size=10)


# Looking at this swarmplot, I'm starting to rethink that it's the right visualization to communicate this data. 
# 
# -For one, there are too many categories for type of crime. The labels' font is incredibly small, and I have to scroll on and on to see the whole graph. 
# 
# -In addition, the graphical demands of creating this visualization is too damn high. It takes a couple of minutes just to plot the first ten thousand rows, and there are over three hundred thousand. 
# 
# I'm thinking a heatmap is actually the way to go for this visualization as well. 
# 
# Let's start fresh from the original data set:

# In[ ]:


crime_type_data = crimes_data.filter(['OFFENSE_CODE_GROUP','OCCURRED_ON_DATE'], axis=1)
crime_type_data.head()


# In[ ]:


crime_type_data['OCCURRED_ON_DATE'] = crime_type_data['OCCURRED_ON_DATE'].str.slice(10, 13, 1)
crime_type_data.head()


# In[ ]:


crime_type_data['OCCURRED_ON_DATE'] = crime_type_data['OCCURRED_ON_DATE'].astype(int)
crime_type_data.head()


# In[ ]:


crime_type_pivot_table = pd.pivot_table(crime_type_data, index=["OFFENSE_CODE_GROUP"], columns=["OCCURRED_ON_DATE"], aggfunc=[len])
crime_type_pivot_table.head()


# In[ ]:


plt.figure(figsize=(30,30))
plt.title("Crime Time")
sns.heatmap(data=crime_type_pivot_table)


# And it seems to work! It makes sense that crime reports would spike during the middle of the afternoon, and fall flat in the middle of the night. 
# 
# Things I would like to do:
# 1. Create a minimum threshold for number of crimes committed, and remove the rows that do not satisfy the criteria. For example, the prostitution and manslaughter data isn't the most riveting.
# 2. Change/remove the titles of the axes. Do they have to be the same as the column names in the table?
# 3. Normalize data across each crime type. Right now, the eye is drawn to crimes that are reported in the highest overall volume, while the goal of the visualization is to emphasize the crimes that experience the highest variation depending on time of day. 
