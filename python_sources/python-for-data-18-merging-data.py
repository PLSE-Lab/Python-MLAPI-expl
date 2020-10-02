#!/usr/bin/env python
# coding: utf-8

# # Python for Data 18: Merging Data
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)

# Data you use for your projects won't always be confined to a single table in a CSV or excel file. Data is often split across several tables that you need to combine in some way. DataFrames can be joined together if they have columns in common. Joining tables in various ways is a common operation when working with databases but you can also join data frames in Python using functions included with pandas.
# 
# First, let's import some libraries and create some dummy medical data tables to use as examples for this lesson.

# In[15]:


import numpy as np
import pandas as pd
import os


# In[16]:


table1 = pd.DataFrame({"P_ID" : (1,2,3,4,5,6,7,8),
                     "gender" : ("male", "male", "female","female",
                                "female", "male", "female", "male"),
                     "height" : (71,73,64,64,66,69,62,72),
                     "weight" : (175,225,130,125,165,160,115,250)})

table1


# In[17]:


table2 = pd.DataFrame({"P_ID" : (1, 2, 4, 5, 7, 8, 9, 10),
                     "sex" : ("male", "male", "female","female",
                            "female", "male", "male", "female"),
                     "visits" : (1,2,4,12,2,2,1,1),
                     "checkup" : (1,1,1,1,1,1,0,0),
                     "follow_up" : (0,0,1,2,0,0,0,0),
                     "illness" : (0,0,2,7,1,1,0,0),
                     "surgery" : (0,0,0,2,0,0,0,0),
                     "ER" : ( 0,1,0,0,0,0,1,1) } ) 

table2


# Both data frames contain the column "P_ID" but the other columns are different. A unique identifier like an ID is usually a good key for joining two data frames together. You can combine two data frames by a common column with merge():

# In[18]:


combined1 = pd.merge(table1,       # First table
                    table2,        # Second table
                    how="inner",   # Merge method
                    on="P_ID")     # Column(s) to join on

combined1


# Inspecting the new combined data frame, we can see that the number of records dropped from 8 in the original tables to 6 in the combined table. If we inspect the P_ID column closely, we see that the original data tables contain some different values for P_ID. Note that inside the merge function we set the argument "how" to "inner". An inner join only merges records that appear in both columns used for the join. Since patients 3 and 6 only appear in table1 and patients 9 and 10 only appear in table2, those four patients were dropped when we merged the tables together.
# 
# Inner joins ensure that we don't end up introducing missing values in our data. For instance, if we kept patients 3 and 6 in the combined data frame, those patients would end up with a lot of missing values because they aren't present in the table2. If you want to keep more of your data and don't mind introducing some missing values, you can use merge to perform other types of joins, such as left joins, right joins and outer joins:

# In[19]:


# A left join keeps all key values in the first(left) data frame

left_join = pd.merge(table1,       # First table
                    table2,        # Second table
                    how="left",   # Merge method
                    on="P_ID")     # Column(s) to join on

left_join


# In[20]:


# A right join keeps all key values in the second(right) data frame

right_join = pd.merge(table1,       # First table
                    table2,        # Second table
                    how="right",   # Merge method
                    on="P_ID")     # Column(s) to join on

right_join


# In[21]:


# An outer join keeps all key values in both data frames

outer_join = pd.merge(table1,      # First table
                    table2,        # Second table
                    how="outer",   # Merge method
                    on="P_ID")     # Column(s) to join on

outer_join


# By this point, you may have noticed that the two data frames contain a second column in common. The first table contains the column "gender" while the second contains the column "sex", both of which record the same information. We can solve this issue by first renaming one of the two columns so that their names are the same and then supplying that column's name as a second column to merge upon:

# In[22]:


table1.rename(columns={"gender":"sex"}, inplace=True) # Rename "gender" column

combined2 = pd.merge(table1,               # First data frame
                  table2,                  # Second data frame
                  how="outer",             # Merge method
                  on=["P_ID","sex"])    # Column(s) to join on

combined2


# By renaming and merging on the sex column, we've managed to eliminate some NA values in the outer join. Although an outer joins can introduce NA values, they can also be helpful for discovering patterns in the data. For example, in our combined data, notice that the two patients who did not have values listed for height and weight only made visits to the ER. It could be that the hospital did not have patients 9 and 10 on record previously and that it does not take height and weight measurements for ER visits. Using the same type of intuition, it could be that patients 3 and 6 have height and weight measurements on file from visits in the past, but perhaps they did not visit the hospital during the time period for which the visit data was collected.

# ## Wrap Up

# The pandas function merge() can perform common joins to combine data frames with matching columns. For some projects, you may have to merge several tables into one to get the most out of your data.
# 
# Now that we know how to prepare and merge data, we're ready to learn more about two of the most common tools for exploring data sets: frequency tables and plots.

# ## Next Lesson: [Python for Data 19: Frequency Tables](https://www.kaggle.com/hamelg/python-for-data-19-frequency-tables)
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)
