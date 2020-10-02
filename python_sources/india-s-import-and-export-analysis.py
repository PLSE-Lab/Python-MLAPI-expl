#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[42]:


uncleaned_data = pd.read_csv('../input/commodity_trade_statistics_data.csv')
#cleaned data
global_trade_df = uncleaned_data.dropna(how='all')


# In[3]:


global_trade_df.shape


# In[4]:


global_trade_df.head(10)


# In[5]:


global_trade_df.count()


# In[25]:


# the array of unique categories
global_trade_df['category'].unique()


# # A lot of category, let's pick the most imp one

# In[18]:


#get the top 10 commodity which contribute maximum in the trade 
df=global_trade_df.groupby('commodity').trade_usd.mean().reset_index(name='trade_usd')
df = df.nlargest(10,'trade_usd').reset_index()
df


# In[27]:


#checking data type of the column 'year'
global_trade_df['year'].unique()


# In[19]:


trade_by_country = global_trade_df[['country_or_area','year','flow', 'category' ,'trade_usd']]

#using groupby function and building a multiIndex to make analysis easier
trade_by_country = trade_by_country.groupby(['country_or_area','year','flow', 'category'])[['trade_usd']].sum()
trade_by_country.head(30)


# # Analysis of Trade in India

# In[43]:


India_df = global_trade_df[global_trade_df['country_or_area'] == 'India']
India_years = India_df['year'].unique()
India_years.sort()

exports_br = trade_by_country['trade_usd'].loc['India', : ,'Export', 'all_commodities']
imports_br = trade_by_country['trade_usd'].loc['India', : ,'Import', 'all_commodities']


fig=plt.figure(figsize=(10, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size':15})


p2 = plt.bar(India_years, imports_br)
p1 = plt.bar(India_years, exports_br)

plt.title("Exports vs Imports")
plt.ylabel('Trade worth - in 100 billion US dollars')
plt.xlabel('year')
plt.legend((p1, p2), ('Exports', 'Imports'))


# In[24]:


# Most Imp commodity of Imp India
India_df[India_df['flow']=='Import'].groupby('commodity').trade_usd.mean()


# In[25]:


del(global_trade_df)


# In[44]:


India_Import=India_df[India_df['flow']=='Import']
df=India_Import.groupby('commodity').trade_usd.mean().head(10).reset_index()


# In[45]:


df


# In[47]:


#df = df.to_frame().reset_index()


# In[39]:


#drop 'All commodities' to get the barplot properly 
df=df.drop(0).reset_index(drop=True)  
#df.loc[0]==""


# In[40]:


sn.barplot(x='trade_usd',y='commodity',data=df)
#sn.set_xticklabels(rotation=30)
plt.xticks(rotation=20)
plt.title("India's Top 10 Import Commodity")
plt.xlabel("Trade worth - in 100 billion US dollars")
plt.ylabel("Commodity")


# In[70]:


# Global_India=Global_India[Global_India['flow']=='Export']
# df=Global_India.groupby('commodity').trade_usd.agg(['count','min','max','mean']).head(10)


# In[46]:


India_Export=India_df[India_df['flow']=='Export']
#df=India_Export.groupby('commodity').trade_usd.mean().head(10).reset_index()
df


# In[47]:


df=df.drop(0).reset_index(drop=True)

sn.barplot(x='trade_usd',y='commodity',data=df)
#sn.set_xticklabels(rotation=30)
plt.xticks(rotation=20)
plt.title("India's Top 10 Export Commodity")
plt.xlabel("Trade worth - in 100 billion US dollars")
plt.ylabel("Commodity")


# In[ ]:




