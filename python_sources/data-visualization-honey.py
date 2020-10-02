#!/usr/bin/env python
# coding: utf-8

# # Honey Production In The USA (1998-2012)
# 
# 
# #### Data Fields
# 
#    * state
#    * numcol : number of honey producing colonies
#    * yieldpercol : yield per colony (lbs)
#    * totalprod : total production (lbs: numcol * yieldpercol)
#    * stocks : stocks held by producers (lbs)
#    * priceperlb : avg price per lb 
#    * prodvalue : totalprod * priceperlb
#    * year 
#    
# #### My natural fascination with bees led me to explore this dataset. SAVE THE BEES!

# #### Import Data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


honey = pd.read_csv('../input/honeyproduction.csv')


# In[3]:


honey.head()


# In[4]:


print(honey['state'].unique())
print('\n')
print("Number of States in dataset: " + str(len(honey['state'].unique())))


# #### Visualize Change in Honey Pricings

# In[95]:


sns.set_style("darkgrid")
honey.groupby('year')['priceperlb'].mean().plot(figsize=(15,4)).set_title('$ Price per lb.')


# #### Visualize the Supply & Demand of Honey

# In[96]:


honey.groupby('year')[['totalprod', 'prodvalue']].sum().plot(figsize=(18,4)).set_title("Total Production (lbs) vs. Production Value ($)")


# #### Correlation between  Production of Honey and Price of Honey

# In[100]:


honey_cor = honey[['totalprod','priceperlb']]
honey_cor.corr()


# ### Total Production per State (years: 1989 - 2012)

# In[94]:


group= honey.groupby('state')
first_year = honey['year'].min()
last_year= honey['year'].max()

ordered_names = sorted(group['totalprod'].sum().sort_values().index)


fig, axes = plt.subplots(nrows=11, ncols=4, sharex=True, sharey=True, figsize=(18,25))
axes_list = [item for sublist in axes for item in sublist] 

for state in ordered_names:
    selection= group.get_group(state) 
    ax = axes_list.pop(0)
    selection.plot(x='year', y='totalprod', label=state, ax=ax, legend=False)
    ax.set_title(state, fontsize=17)
    ax.tick_params(
        which='both',
        bottom='off',
        left='off',
        right='off',
        top='off',
        labelsize=16 
    )
    ax.grid(linewidth=0.25)
    ax.set_xlim((first_year, last_year))
    ax.set_xlabel("")
    ax.set_xticks((first_year, last_year))
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
for ax in axes_list:
    ax.remove()
    
plt.subplots_adjust(hspace=1)
plt.tight_layout()


# #### Total Production missing for States: 
# * MD
# * NV
# * OK
# * SC
# 

# In[101]:


pd.pivot_table(honey, index='state', columns='year', values='totalprod', aggfunc=np.sum)


# #### Visualizing Average Yield per Colony 
# 
# * Yield per Colony is down (21.4%) in 2012 compared to 1998

# In[110]:


honey.groupby('year')['yieldpercol'].mean().plot(figsize=(15,5)).set_title("Yield Per Colony (lbs)")

