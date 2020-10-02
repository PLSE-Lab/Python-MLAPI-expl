#!/usr/bin/env python
# coding: utf-8

# ### A few small examples of cleaning data with pandas

# In[ ]:


# Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Import data and convert time columns from string to timestamp
df = pd.read_csv('../input/startup-investments-crunchbase/investments_VC.csv', encoding = "ISO-8859-1", parse_dates=['founded_month', 'founded_quarter', 'founded_year'])


# In[ ]:


# Strip empty spacing from column headers for easier column calling
df.rename(columns=lambda x: x.strip(), inplace=True)


# ### Top and Tail of data

# In[ ]:


df.head()


# In[ ]:


df.tail()


# #### Seems to be a lot of NaN values towards the end of the data set. The next line sums up all NaN values within each column using the function 'isnull()'. Check it out:

# ### Calculating the sum of all NaN values within each column.

# In[ ]:


df.isnull().sum()


# #### The most reocurring value is shown to be 4856. This gives the impression that there are 4856 rows altogether that have NaN value in all columns. Let's test this out.

# In[ ]:


# Shape of data before removing NaN rows
df.shape


# In[ ]:


# Remove any row with NaN as ALL column values
# (Calling the data base affected by this command by a new name just in case there is an error)
data = df.dropna(how='all')


# In[ ]:


# Shape of data after removing NaN rows
data.dropna(how='all').shape


# In[ ]:


54294 - 49438


# For the sake of simplicity and easier for analysis. dropna will be used for all records with NaN anywhere.

# In[ ]:


data = data.dropna(how='any')


# In[ ]:


data.shape


# ### Correcting data types for columns to maintain consistency throughout data

# In[ ]:


pd.set_option('display.max_columns', 40)
data.head()


# In[ ]:


# Some columns are strings, some numeric and some time valued. Timestamp was already done on importing so now to make
# sure that other columns are correct data types. Numeric columns will be converted into float data types.
data.dtypes


# #### data.dtypes shows the data types of all columns. If there are a mixed data type in a column it will display 'object'. A previous run showed a ValueError that highlighted commas, dashes, no values as a problem for the funding_total_usd column.

# In[ ]:


# Remove all commas in funding_total_usd column
data['funding_total_usd']= data['funding_total_usd'].str.replace(',', '')
data['funding_total_usd']= data['funding_total_usd'].str.replace('-', '0')


# Commas, dashes and no values removed. Now can convert column to float dtype

# In[ ]:


data.head()


# In[ ]:


# convert whole column funding_total_usd to float data type
data['funding_total_usd'] = data['funding_total_usd'].astype(float)


# In[ ]:


data.head()


# In[ ]:


data[['state_code', 'region', 'city']]= data[['state_code', 'region', 'city']].astype(str)


# In[ ]:


state_count = data['state_code'].value_counts()


# In[ ]:


print(f"Minimum = {np.min(state_count)}")
print(f"Maximum = {np.max(state_count)}")
print(f"Median = {np.median(state_count)}")
print(f"Mean = {round(np.mean(state_count), 2)}")
print(f"Standard deviation = {round(np.std(state_count), 2)}")
print(f"Variance = {round(np.var(state_count), 2)}")


# In[ ]:


state_count1 = state_count[:29,]
state_count2 = state_count[30:,]

plt.figure(figsize=(20, 10))
plt.subplot(211)
plt.plot(state_count1, color='tab:blue', marker='o')
plt.plot(state_count1, color='black')

plt.subplot(212)
plt.plot(state_count2, color='tab:blue', marker='o')
plt.plot(state_count2, color='black')

plt.show()


# In[ ]:




