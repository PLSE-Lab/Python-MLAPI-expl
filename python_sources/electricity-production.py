#!/usr/bin/env python
# coding: utf-8

# # Visualizing Africa's Total production of electricity 2000 - 2014
# This notebook visualizes Data retrieved from www.data.world. We tried to intrepret the information using Pivot Tables and a wide range of Graphs. 
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

name_sheet=pd.read_csv('../input/african_energy.csv',encoding="ISO-8859-1")
name_sheet


# In[ ]:


name_sheet.pop('Indicator')

#handle missing data on imported data
name_sheet=name_sheet.fillna(0)

#view summary of how data looks
name_sheet.describe()


# #### Create a Pivot Table
# We will create a pivot table to show us the amount of Electricity Produced by each African country between 2000 -2014. 

# In[ ]:


african_sheet=name_sheet.pivot_table(index='Date',aggfunc={'Value':sum},columns='RegionName')
african_sheet=african_sheet.fillna(0)
african_sheet


# #### Plot Cluster Map

# In[ ]:


sns.clustermap(african_sheet['Value'],standard_scale=1)


# #### Plot Joint KDE for  Algeria vs Nigeria

# In[ ]:


sns.jointplot(african_sheet['Value','Nigeria'],african_sheet['Value','Algeria'],african_sheet,kind='kde')


# #### Histogram & KDE Plot for Nigeria

# In[ ]:


sns.distplot(african_sheet['Value','Nigeria'],bins=15).set_title('Histogram & KDE Distribution for Nigeria')


# #### Heat Map showing Electricity production in Africa

# In[ ]:


sns.heatmap(african_sheet['Value']).set_title('Heatmap Distribution for Africa')


# #### Nigeria's Electricity Production  Between 2000 - 2014

# In[ ]:


sns.violinplot(african_sheet['Value','Nigeria'],inner='stick').set_title('Violin Distribution for Nigeria')


# In[ ]:


sns.pairplot(data=name_sheet,hue='RegionName')


# In[ ]:


sns.set(style="darkgrid")
mapping=name_sheet[name_sheet.RegionName=='Nigeria'].plot(kind='scatter',x='Date',y='Value',color='red',label='Nigeria')
name_sheet[name_sheet.RegionName=='South Africa'].plot(kind='scatter',x='Date',y='Value',color='green',label='South Africa',ax=mapping)
name_sheet[name_sheet.RegionName=='Sudan'].plot(kind='scatter',x='Date',y='Value',color='blue',label='Sudan',ax=mapping)
name_sheet[name_sheet.RegionName=='Rwanda'].plot(kind='scatter',x='Date',y='Value',color='orange', label='Rwanda', ax=mapping)
name_sheet[name_sheet.RegionName=='Egypt'].plot(kind='scatter',x='Date',y='Value',color='purple', label='Egypt', ax=mapping)
mapping.set_xlabel('Date')
mapping.set_ylabel('Energy in GWh')
mapping.set_title('Total production of electricity 2000 - 2014')
mapping=plt.gcf()
mapping.set_size_inches(10,6)

