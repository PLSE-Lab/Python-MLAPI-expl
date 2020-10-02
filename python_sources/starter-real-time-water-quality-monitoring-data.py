#!/usr/bin/env python
# coding: utf-8

# ## Check water quality data.

# ## Exploratory Analysis
# To begin this exploratory analysis, first use `matplotlib` to import libraries and define functions for plotting the data. Depending on the data, not all plots will be made. (Hey, I'm just a kerneling bot, not a Kaggle Competitions Grandmaster!)

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# There are 11 csv files in the current version of the dataset:
# 

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# The next hidden code cells define functions for plotting data. Click on the "Code" button in the published kernel to reveal the hidden code.

# Now you're ready to read in the data and use the plotting functions to visualize the data.

# ### Let's check 1st file: /kaggle/input/Johnstone_river_coquette_point_joined.csv

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# Johnstone_river_coquette_point_joined.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/Johnstone_river_coquette_point_joined.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'Johnstone_river_coquette_point_joined.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df1.head(5)


# Distribution graphs (histogram/bar graph) of sampled columns:

# ### Let's check 2nd file: /kaggle/input/Johnstone_river_innisfail_joined.csv

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# Johnstone_river_innisfail_joined.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df2 = pd.read_csv('/kaggle/input/Johnstone_river_innisfail_joined.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'Johnstone_river_innisfail_joined.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df2.head(5)


# Distribution graphs (histogram/bar graph) of sampled columns:

# In[ ]:


plotScatterMatrix(df2, 12, 10)


# ### Let's check 3rd file: /kaggle/input/Mulgrave_river_deeral_joined.csv

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# Mulgrave_river_deeral_joined.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df3 = pd.read_csv('/kaggle/input/Mulgrave_river_deeral_joined.csv', delimiter=',', nrows = nRowsRead)
df3.dataframeName = 'Mulgrave_river_deeral_joined.csv'
nRow, nCol = df3.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df3.head(5)


# Missing data visualization

# In[ ]:


import missingno as msno
import seaborn as sns; sns.set(style="whitegrid", font_scale=2)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



# missing data plot
# pick up one station and check the missing data status
# missing value happens at different timestamps for different water quality variables


file = '/kaggle/input/Tully_river_euramo_joined.csv'

df = pd.read_csv(file)
df.set_index('Timestamp', inplace=True)
df.drop(columns=['Dayofweek', 'Month'],inplace=True)
df = df.loc['2019-02-01T00:00:00':'2019-03-31T00:00:00']
df.replace(0, np.nan, inplace=True)


msno.matrix(df.set_index(pd.period_range(start='2019-02-01', periods=1393, freq='H')) , freq='10D', fontsize=20)


# ## Conclusion
# The water quality data are collected from real world monitoring system. A lot of data are missing for the water quality variables.

# In[ ]:




