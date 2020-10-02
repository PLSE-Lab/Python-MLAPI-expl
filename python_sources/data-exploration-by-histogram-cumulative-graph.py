#!/usr/bin/env python
# coding: utf-8

# # Data Exploration by Histogram & Cumulative graph

# ## Problem Statement

# Lets assume that the allowable maintenance time of machines in a specific manufacturing plant is 180 minutes. Based on previous data, we want to analyze whether allowing 180 minutes for maintenance is reasonable. Histogram and Cumulative Graph of the maintenance time of machine breakdown will be leveraged in this project.

# ![](https://imgur.com/KMQIQKE.png)

# ## Import the dataset & necessary libraries

# In[ ]:


# import the libraries
import pandas # Data import
import pylab as pl # Data visualization

# import the dataset
maintenance_df = pandas.read_excel("../input/maintenance-time-of-machines/maintenance time of machines.xlsx") # Line 1
maintenance_df.head()


# ## Draw the Histogram

# In[ ]:


# Draw the Histogram
maintenance_time = maintenance_df.iloc[:,1]
N, bins, patches = pl.hist(maintenance_time, bins=16)
# the selection of the number of bins is ofter craft. Since our range of our data is 160 (180-20), I choose 16 so that the BARs are evenly spaced by 10.
pl.title("Histogram")
pl.xlabel("Time of Maintenance")
pl.ylabel("Frequency")
jet = pl.get_cmap('jet', len(patches))
for i in range(len(patches)):
    patches[i].set_facecolor(jet(i))


# > ***Interpreting the histogram:***
# 
# 
# As observed in the above **histogram**, there are 40 machines with the maintenance time raning from 20 minutes to 30 minutes. Similar situation arises for another 40 machines. So, basically, 80 machines out of 100 machines in our dataset have a maintenance time of 40 minutes or lower. Great observation! Lets justify with a cumulative graph. 
# 

# ## Draw a Cumulative Graph

# In[ ]:


# Develop a Cumulative graph
N, bins, patches = pl.hist(maintenance_time, cumulative = True, bins=16)
pl.xlabel("Time of Maintenance")
pl.ylabel("Cumulative Probability Distribution")
jet = pl.get_cmap('jet', len(patches))
for i in range(len(patches)):
    patches[i].set_facecolor(jet(i))


# > ***Interpreting the Cumulative graph:***
# 
# 
# From the **Cumulative Graph** above, it is reasonable to conclude that the maintenance time of  80% machines are approximately 40 minutes or less.
# Therefore, we may allow 40 minutes as maintenance time and take care of the remaining ~20% machines differently, such as instally surrogated machines.

# ### Business Perspective

# Ultimately, just by simple Data Exploration, we can recommend the manufacturing floor manager to reduce the allocated maintenance time from **180** minutes to **40** minutes. Awesome!  :) 
