#!/usr/bin/env python
# coding: utf-8

# # Asian Games Medal - Data Analysis and Graphs
# 
# ### Below are few Steps for Analysis
# 
# * Import Packages
# * Import CSV file
# * Total Number of Records
# * Data Type
# * Top 10 Records
# * Rename Field Name
# * Top 10 Records after Rename Field Name
# * Top 15 Year Wise Medals Count
# * Top 15 Counrty Wise Medals Count
# * Unique Counrty Name
# * Max, Min and Mean for Total Medals
# * Max, Min and Mean for Total Medals  ( Top Country - China (CHN) )
# 
# # Import Packages

# In[ ]:


import pandas as pd
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn
import os
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 9999
pd.options.display.float_format = '{:20,.2f}'.format


# # Import CSV file

# In[ ]:


DataSet = pd.read_csv('../input/asiangamestop10.csv').fillna(0)


# # Total Number of Records

# In[ ]:


TotalRowCount = len(DataSet)
print("Total Number of Data Count :", TotalRowCount)


# # Data Type

# In[ ]:


DataSet.dtypes


# # Top 10 Records

# In[ ]:


DataSet.head(10)


# # Rename Field Name

# In[ ]:


DataSet.rename(columns={'NOC' : 'Country',}, inplace=True)


# # Top 10 Records after Rename Field Name

# In[ ]:


DataSet.head(10)


# # Top 15 Year Wise Medals Count

# In[ ]:


TotalPrice = DataSet.groupby(['Year'])['Total'].sum().nlargest(15)
print("Top 15 Year Wise Medals Count\n")
print(TotalPrice)
plt.figure(figsize=(22,7))
GraphData=DataSet.groupby(['Year'])['Total'].sum().nlargest(15)
GraphData.plot(kind='bar')
plt.ylabel('Medals Count')
plt.xlabel('Year')


# # Top 15 Counrty Wise Medals Count

# In[ ]:


TotalPrice = DataSet.groupby(['Country'])['Total'].sum().nlargest(15)
print("Top 15 Country Medals Count \n")
print(TotalPrice)
plt.figure(figsize=(22,7))
GraphData=DataSet.groupby(['Country'])['Total'].sum().nlargest(15)
GraphData.plot(kind='bar')
plt.ylabel('Medals Count')
plt.xlabel('Country Name')


# # Unique Counrty Name

# In[ ]:


UniqueNOC = DataSet['Country'].unique()
print("All Unique Country Name \n")
print(UniqueNOC)


# # Max, Min and Mean for Total Medals

# In[ ]:


print ("Max Medal Mode is :",DataSet['Total'].max())
print ("Min Medal Mode is :",DataSet['Total'].min())
ItemTypeMean = DataSet['Total'].mean()
print ("Mean Medal Mode is :", round(ItemTypeMean))


# # Max, Min and Mean for Total Medals  ( Top Country - China (CHN) )

# In[ ]:


ItemData=DataSet[DataSet['Country']=='China (CHN)']
print ("China (CHN) - Max Medal Mode is :",ItemData['Total'].max())
print ("China (CHN) - Min Medal Mode is :",ItemData['Total'].min())
ItemTypeMean = ItemData['Total'].mean()
print ("China (CHN) - Mean Medal Mode is :", round(ItemTypeMean))

