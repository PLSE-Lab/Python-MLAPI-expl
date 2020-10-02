#!/usr/bin/env python
# coding: utf-8

# # Word Income Database - Starter Kernel

# Here's a little starter kernel to get you going with the WID database.

# **Import Libraries**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from subprocess import check_output

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 100
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')


# **Define some constants**

# In[2]:


FOLDER_ROOT = './..'
FOLDER_INPUT = FOLDER_ROOT + '/input'
FOLDER_OUTPUT = FOLDER_ROOT + '/output'


# **Check available data sources**

# In[3]:


print(check_output(["ls", FOLDER_INPUT]).decode("utf8"))


# **Countries**

# In[4]:


countries_df = pd.read_csv(FOLDER_INPUT + '/countries.csv')
countries_df.head()


# **WID - World Income Database**

# In[5]:


wid_df = pd.read_csv(FOLDER_INPUT + '/wid.csv')
wid_df.head()


# **Variables**

# In[7]:


variables_df = pd.read_csv(FOLDER_INPUT + '/wid_variables.csv', encoding='ISO-8859-1')
variables_df.head()


# ## Finding variables

# In[8]:


# Variable name contains `population`
population = variables_df['Variable Name'].str.contains("population", case=False)

# Variable name contains `all ages`
all_ages = variables_df['Variable Name'].str.contains('all ages', case=False)

# Variable name contains `individuals`
individuals = variables_df['Variable Name'].str.contains('individuals', case=False)


# In[9]:


pd.set_option('display.max_colwidth', -1)

# Filter the available variables
result_df = variables_df[population & all_ages & individuals]

# Let's see the important columns
result_df[['Variable Code', 'Variable Name', 'Variable Description']]


# So, let's assume I'd like to show some population data. I will use the `npopul999i` variable.

# In[10]:


my_var = 'npopul999i'
variables_df[variables_df['Variable Code'] == my_var].T


# ## Extracting data from WID dataframe

# In[11]:


fr_population_df = wid_df[wid_df.country == 'FR'][['country', 'year', 'perc', my_var]]
fr_population_df.head()


# **Plot population changes if France**

# In[12]:


plt.plot(fr_population_df['year'], fr_population_df[my_var] / 1000000)
plt.title("France population till 2016")
plt.xlabel("Years")
plt.ylabel("Population in million")
plt.show()


# ## Resources

# Check out the WID's official papers [here](http://wid.world/news/) for more inspiration. 

# **If you found this notebook helpful or you just liked it, an upvote would be very much appreciated! Thanks :)**
