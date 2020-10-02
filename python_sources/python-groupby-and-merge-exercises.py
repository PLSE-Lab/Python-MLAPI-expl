#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

#Squelch SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Load dataframe
bill_of_materials = pd.read_csv('../input/bill_of_materials.csv')
bill_of_materials.head()

# Function to get component data
def pull_out_component(raw, component_num):
    return pd.DataFrame({'tube_assembly_id': raw.tube_assembly_id,
                         'component_id': raw['component_id_' + str(component_num)],
                         'component_count': raw['quantity_' + str(component_num)]})

### Component counts ###
component_counts = pd.concat((pull_out_component(bill_of_materials, i) for i in range(1, 9)), axis=0)
component_counts.dropna(axis=0, inplace=True)

# List of component files
files = ['comp_adaptor.csv', 'comp_boss.csv', 'comp_elbow.csv', 'comp_float.csv',
         'comp_hfl.csv', 'comp_nut.csv', 'comp_other.csv', 'comp_sleeve.csv',
         'comp_straight.csv', 'comp_tee.csv', 'comp_threaded.csv']

# Combine all component data into one dataframe
all_component_data = pd.concat([pd.read_csv('../input/'+f) for f in files], axis=0)

### Component weights ###
component_weights = all_component_data[['component_id', 'weight']]
component_weights.fillna(0, inplace=True)


# # Merge and Groupby Exercises
# ---
# 
# Manufacturers of large machines routinely analyze the materials they use. This dataset contains information about various tubes used in machines. You can imagine that big machines require different kinds of tubes; cars have tubes, airplanes have tubes, and bulldozers have tubes too. Let's talk a little about what a tube looks like in this data set.
# 
# The smallest part of a tube assembly is a *component*. Components might be things like clamps, gaskets, or some other kind of fittings. In general, tube assemblies consist of one or more component and can be different combinations of components as well.  In this exercise, you'll analyze the data and answer some questions about weights. Before you can analyze the data though, you have to get it into a manageable format. You'll need to use what you've learned about merging and grouping. 
# 
# ---
# 
# 
# **You have these two dataframes to answer the questions. Take a look and get familiar with them**
# - **component_counts**: contains the quantity of components in each tube assembly 
# - **component_weights**: contains the weights of individual components
# <br><br>
# 
# ---
# 
# Resources<br>
# *Links to the tutorials*<br>
# [Merge tutorial][2]<br>
# [Groupby Tutorial][3] 
# 
# 
# *Pandas documentation*<br>
# [Merge Docs][4]<br>
# [Groupby Docs][5]
# 
# ---
# 
# Hint: For these exercises, you may also find the [sort_values][1] method useful<br>
# 
# 
#   [1]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
#   [2]: https://www.kaggle.com/crawford/python-merge-tutorial
#   [3]: https://www.kaggle.com/crawford/python-groupby-tutorial
#   [4]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html
#   [5]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html

# # Exercises: 
# ---
# 
# ### What are the five heaviest tube assemblies and their weights?
# 
# Expected output:
# 
# <table>
# <tr><td>TA-01619</td><td>19.261</td></tr>
# <tr><td>TA-04072</td><td>19.200</td></tr>
# <tr><td>TA-18779</td><td>16.844</td></tr>
# <tr><td>TA-18226</td><td>16.077</td></tr>
# <tr><td>TA-14536</td><td>15.648</td></tr>
# </table>

# In[ ]:


# Find the five heaviest tube assembly id's and their weights


# <br><br>
# ### How many tube_assembly_id's have more than five components (sum of component_count)?
# 
# Expected output: 120
# 

# In[ ]:


# How many tube_assembly_id's have more than five components? (sum of component_count)


# <br><br>
# ### How many component_id's are used in more than 50 tube assemblies?
# 
# Expected output: 69
# 

# In[ ]:


# How many component_id's are used in more than 50 tube assemblies?


# <br><br>
# ### What is the average weight of the five heavist component_id's?
# 
# Expected output: 95.4894

# In[ ]:


# What is the average weight of the five heaviest component_id's?

