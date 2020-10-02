#!/usr/bin/env python
# coding: utf-8

# <h2>NFL Punt Competition
# <h3>A new file (player_punt_data.analysis.csv) was created and uploaded to show the positions that have the most interaction on a punt. The file has 3259 entries. 
# 

# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
import pandas
import seaborn


# In[ ]:


data = pandas.read_csv('../input/player_punt_data_analysis.csv')


# 
# 

# 

# In[ ]:


### import new data
data


# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
df1 = pandas.read_csv('../input/player_punt_data_analysis.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'player_punt_data_analysis.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df1.head(5)


# 

# ## Conclusion
# The file shows that the defensive positions cb, olb, de, ss, fs, ilb, dt account for 70% of all plays. 
