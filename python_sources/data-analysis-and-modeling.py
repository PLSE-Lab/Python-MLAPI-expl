#!/usr/bin/env python
# coding: utf-8

# # Following are steps for Data Analysis and Modeling
# 
# 1. Import Packages
# 2. Import CSV file
# 3. Check Total Records in CSV file
# 4. Check DataType of CSV file
# 5. Print first 10 and last 10 recods from DataSet
# 6. Total 10 Country wise count with graph
# 7. Total Pack Price for Top 15 Countries with graph
# 8. First Line Designation Wise Count with graph
# 10. Shipment Mode percentage wise Pie Chart
# 11. Unquie Manufacturing Site Names
# 12. Shipment Mode, Min and Mean value for Air
# 13. Top 10 Manufacturing Site for all Shipment Mode with Graph
# 14. Top 10 Manufacturing Site for Air Shipment Mode with Graph
# 15. Pack Price analysis using Distributions and Plot Graph
# 16. Shipment Mode and Pack Price in Bar Plot Graph
# 
# Now, lets begin with Data!!!

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


DataSet = pd.read_csv('../input/SCMS_Delivery_History_Dataset.csv').fillna(0)


# # Check Total Records in CSV file

# In[ ]:


TotalRowCount = len(DataSet)
print("Total Number of Data Count :", TotalRowCount)


# # Check DataType of CSV file

# In[ ]:


DataSet.dtypes


# # Print first 10 and last 10 recods from DataSet

# In[ ]:


DataSet.head(10)


# In[ ]:


DataSet.tail(10)


# # Total 10 Country wise count with graph

# In[ ]:


DataSet = DataSet.dropna()
ItemCount = DataSet["Country"].value_counts().nlargest(10)
print("Top 10 Countries Wise Count \n")
print(ItemCount)
sn.set_context("talk",font_scale=1)
plt.figure(figsize=(22,6))
sn.countplot(DataSet['Country'],order = DataSet['Country'].value_counts().nlargest(10).index)
plt.title('Top 10 Countries Wise Count \n')
plt.ylabel('Total Count')
plt.xlabel('Country Name')


# # Total Pack Price for Top 15 Countries with graph

# In[ ]:


TotalPrice = DataSet.groupby(['Country'])['Pack Price'].sum().nlargest(15)
print("Total Pack Price for Top 15 Countries\n")
print(TotalPrice)
plt.figure(figsize=(22,6))
GraphData=DataSet.groupby(['Country'])['Pack Price'].sum().nlargest(15)
GraphData.plot(kind='bar')
plt.ylabel('Total Pack Price')
plt.xlabel('Country Name')


# # First Line Designation Wise Count

# In[ ]:


sn.set_context("talk",font_scale=1)
plt.figure(figsize=(5,6))
sn.countplot(DataSet['First Line Designation'],order = DataSet['First Line Designation'].value_counts().nlargest(10).index)
plt.title('First Line Designation Wise Count \n')
plt.ylabel('Total Count')
plt.xlabel('First Line Designation')


# # Shipment Mode percentage wise Pie Chart

# In[ ]:


ShippingMode = DataSet["Shipment Mode"].value_counts()
labels = (np.array(ShippingMode.index))
sizes = (np.array((ShippingMode / ShippingMode.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(title="Shipment Mode")
dat = [trace]
fig = go.Figure(data=dat, layout=layout)
py.iplot(fig, filename="Shipment Mode")


# # Unquie Manufacturing Site Names

# In[ ]:


UniqueItem = DataSet['Manufacturing Site'].unique()
print("All Unique Manufacturing Site \n")
print(UniqueItem)


# # Shipment Mode, Min and Mean value for Air

# In[ ]:


ItemData=DataSet[DataSet['Shipment Mode']=='Air']
print ("The Max Air Shipment Mode is :",ItemData['Unit of Measure (Per Pack)'].max())
print ("The Min Air Shipment is :",ItemData['Unit of Measure (Per Pack)'].min())
ItemTypeMean = ItemData['Unit of Measure (Per Pack)'].mean()
print ("The Mean Air Shipment is :", round(ItemTypeMean,2))


# # Top 10 Manufacturing Site for all Shipment Mode with Graph

# In[ ]:


plt.figure(figsize=(22,6))
TopFiveManufacturingSite=DataSet.groupby('Manufacturing Site').size().nlargest(10)
print(TopFiveManufacturingSite)
TopFiveManufacturingSite.plot(kind='bar')
plt.title('Top 10 Manufacturing Site \n')
plt.ylabel('Total Count')
plt.xlabel('Manufacturing Site Name')


# # Top 10 Manufacturing Site for Air Shipment Mode with Graph

# In[ ]:


# Top 10 Air Shipment Mode in Bar Chart
ItemData=DataSet[DataSet['Shipment Mode']=='Air']
DataSet[DataSet["Shipment Mode"]=='Air']['Manufacturing Site'].value_counts()[0:10].to_frame().plot.bar(figsize=(22,6))
ItemSupplier = DataSet[DataSet["Shipment Mode"]=='Air']['Manufacturing Site'].value_counts()[0:10]
print("Top 10 Air Manufacturing Site \n")
print(ItemSupplier)
plt.title('Top 10 Air Manufacturing Site\n')
plt.ylabel('Air Count')
plt.xlabel('Manufacturing Site')


# # Shipment Mode and Pack Price in Bar Plot Graph

# In[ ]:


plt.subplots(figsize = (18,6))
plt.xticks(rotation = 90)
sn.barplot('Shipment Mode','Pack Price', data = DataSet)
plt.show()


# # Conclusion
# 
# * Top Country for Pack Price : Nigeria - 25,620.72
# * Top Shipping Mode : Air
# * The Max Air Shipment Mode is : 1000
# * The Min Air Shipment is : 1
# * The Mean Air Shipment is : 82.35
# * Top Manufacturing Site : Aurobindo Unit III, India - 3172
# * Top Air Manufacturing Site : Aurobindo Unit III, India - 1694

# # This is my first kaggle notebook.Hope it helped. Please leave your comments / suggestions below. Please upvote if you liked it.
