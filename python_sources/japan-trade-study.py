#!/usr/bin/env python
# coding: utf-8

# In[81]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
plt.style.use('bmh')
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
print(plt.style.available)


# Trade statistics from 1988 to 2015

# In[82]:


# Trade from 1988 to 2015
trade_df = pd.read_csv('../input/year_1988_2015.csv')

trade_df.head()


# In[83]:


# Country names and areas
country_df = pd.read_csv('../input/country_eng.csv')

country_df.head()


# In[84]:


# Merge the two tables
df_main = pd.merge(country_df, trade_df, on=['Country'])
df_main.head()


# ## Global Trade ##
# First we will examine the different regions in the world that Japan Trades with and see which ones are the highest.

# In[69]:


region  = df_main.groupby(['Year', 'Area'])['VY'].agg(sum).to_frame().unstack()
# Drop intergrated area and special area
region = region.drop(region.columns[[3, 8]], axis=1)
region 


# In[70]:


region['VY'].plot(figsize=(12,12))
plt.ylabel('Trade')
plt.title('Trade for Each Region against Year')


# The first thing that we can see from this graph is that the amount of trade with Asia increases dramatically from the 1990s onwards. However trade with other regions in the world maintain a steady level. There is a spike in trade from Middle East between 2005-2010, this could be due to higher demand for fossil fuels. We see another spike from the Middle East in 2011 after the Fukushima incident. This caused the closure of many Nuclear power plants, as a result Japan imported more natural gas to meet energy needs.
# Another thing that is striking about the graph in the dramatic drop in 2008 for trade with all regions. This could be due to the Financial Crisis that affected global trade and caused recession throughout the world.

# Exports and Imports by Region
# =======

# In[71]:


# Export table, exports = 1
region_export = df_main[df_main.exp_imp==1].groupby(['Year', 'Area'])['VY'].agg(sum).unstack()
# Drop sepcial area
region_export = region_export.drop(region_export.columns[7], axis=1)
region_export.head()


# In[72]:


region_export.plot()
plt.ylabel('Volume')
plt.title('Exports by Region')


# In[73]:


region_import = df_main[df_main.exp_imp==2].groupby(['Year', 'Area'])['VY'].agg(sum).unstack()
# Drop integrated hozei ar and sepcial area
region_import = region_import.drop(region_import.columns[3], axis=1)
region_import = region_import.drop(region_import.columns[7], axis=1)
region_import.head()


# In[74]:


region_import.plot()
plt.ylabel('Volume')
plt.title('Imports by Region')


# In[75]:


# Comparing the import and exports
compare_exp_imp = pd.pivot_table(df_main, 'VY', 'Year', 'exp_imp', aggfunc=sum).rename(columns={1:'Exports', 2:'Imports'})
compare_exp_imp.head()


# In[76]:


compare_exp_imp.plot(kind='bar')
plt.title('Comparing Exports and Imports from 1988 to 2015')
plt.ylabel('Value')
plt.legend(title='')


# ## Trade by Region ##

# Asia
# =======

# In[77]:


# Finding the top trading Countries in Asia
df_main_asia = df_main[(df_main.exp_imp==2) & (df_main.Area=='Asia')].groupby(['Country_name'])['VY'].agg(sum).sort_values(ascending=False)
df_main_asia.head()


# In[78]:


df_main_asia = df_main[(df_main.exp_imp==2) & (df_main.Area=='Asia')].groupby(['Country_name'])['VY'].agg(sum).sort_values(ascending=True)
df_main_asia.plot(kind='barh')
plt.ylabel('Country')
plt.xlabel('Trade Value')
plt.title('Top Trading Countries in Asia')


# In[79]:


hs2_df = pd.read_csv('../input/hs2_eng.csv')
#df_product_asia = df_main.drop.where(df_main.Area=='Special_Area' & df_main.Area=='Integrated_Hozei_Ar_Special_Area')
#df_product_asia.where(df_main.Area == 'Asia')
#df_product_asia = df_main.drop(df_main['Country_name'=='Special_Area'])
df_product_asia = df_main.drop[df_main.Area=='Special_Area']

    


# In[80]:




