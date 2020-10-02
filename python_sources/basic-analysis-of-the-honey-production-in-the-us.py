#!/usr/bin/env python
# coding: utf-8

# # <center>Honeybees and Neonic Pesticides</center>
# ## <center>Are Neonic pesticides connected to the decline of bees colonies?</center>

# <img src='https://images.newscientist.com/wp-content/uploads/2017/07/25204611/p8401619.jpg'>

# The data come from Kevin Zmith on [Kaggle](https://www.kaggle.com/kevinzmith/honey-with-neonic-pesticide), inspired by the dataset Honey Production in the USA, extended to the period 1998-2017. Additionnaly, the data from USGS's Pesticide National Synthesis Project has been agregated, allowing evaluation of the statistical connections between Honey Production and the use of Neonicotinoid (neonic) pesticides.

# ### Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as py
import cufflinks as cf
cf.go_offline()
pd.set_option('display.max_columns', 30)


# ## Data cleaning

# ### Importing dataset

# In[ ]:


data = pd.read_csv("../input/honey-with-neonic-pesticide/vHoneyNeonic_v03.csv")


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.columns


# ### I convert the columns using pounds to kg 

# In[ ]:


data.insert(loc=3, column='yieldpercol_kg', value=data["yieldpercol"]*0.45359237)
data.insert(loc=5, column='totalprod_kg', value=data["totalprod"]*0.45359237)
data.insert(loc=6, column='totalprod_to', value=data["totalprod"]*0.00045359237)
data.insert(loc=8, column='stocks_to', value=data["stocks"]*0.00045359237)
data.insert(loc=10, column='priceperkg', value=data["priceperlb"]/0.45359237)
data.insert(loc=11, column='pricepertonne', value=data["priceperlb"]/0.00045359237)
data.head()


# In[ ]:


data = data.rename(columns={"nCLOTHIANIDIN": "CLOTHIANIDIN", "nIMIDACLOPRID": "IMIDACLOPRID",
                     "nTHIAMETHOXAM": "THIAMETHOXAM", "nACETAMIPRID": "ACETAMIPRID",
                    "nTHIACLOPRID": "THIACLOPRID","nAllNeonic":"AllNeonic"})
data.to_csv('vHoneyNeonic_v04.csv')


# In[ ]:


data.isnull().sum()


# ### Deleting rows with missing values because they concern Neonic pesticides features and I want to analyze their effects on the honey production. We already know that 237 honey producers that were'nt analyze or did'nt want to display this information

# In[ ]:


data = data.dropna()


# In[ ]:


data.shape


# ## Data Analyse

# ### Top 10 of States producing honey

# In[ ]:


data.groupby("StateName")['totalprod_to'].sum().sort_values(ascending=False)[:10]


# In[ ]:


data.groupby("Region")['totalprod_kg'].sum().sort_values(ascending=False)


# ### Evolution of the price of honey

# In[ ]:


evo_price = data.groupby("year", as_index=False).agg({'priceperkg':'mean'})
evo_price.iplot(kind='line', x='year', xTitle='Year', color='orange',
           yTitle='Price of honey (dollars)', title='Evolution of the price of honey')


# The price of honey has seen a five-fold increase in 12 years !

# ### Production by year 

# In[ ]:


prod_by_year = data.groupby("year", as_index=False).agg({'totalprod_to':'mean'})
prod_by_year.iplot(kind='bar', x='year', xTitle='Year', color='red',
           yTitle='Production of honey (Tonne)', title='Evolution of the production of honey')


# ### Is there a correlation between the price and the production ? 

# In[ ]:


data['priceperkg'].corr(data['totalprod_kg'])


# The production only has a 23% impact on the price of honey ! Other features should enter into account... The market ?

# ### Use of Neonic by state

# In[ ]:


data.groupby("StateName")['AllNeonic'].sum().sort_values(ascending=False)


# ### Evolution of the use of Neonic pesticides 

# In[ ]:


evo_neonic = data.groupby("year", as_index=False).agg({'AllNeonic':'mean'})
evo_neonic.iplot(kind='bar', x='year', xTitle='Year', color='green',
           yTitle='Use of Neonic pesticides (kg)', title='Evolution of the use of Neonic pesticides')


# There is an 4460% increase of use of Neonic between 1995 and 2014 !

# ### Is there a correlation between the production and the use of Neonic ?

# In[ ]:


data['totalprod_kg'].corr(data['AllNeonic'])


# The correlation is low between the production of honey and the use of Neonic pesticides (11%)

# In[ ]:


evo_col = data.groupby("year", as_index=False).agg({'numcol':'count'})
evo_col.iplot(x='year', xTitle='Year', color='purple',
           yTitle='Number of colonies', title='Evolution of the number of colonies')


# ### Is there a correlation between the number of colonies and the use of Neonic pesticides ?

# In[ ]:


data['numcol'].corr(data['AllNeonic'])


# The correlation is low between the number of colonies and the use of Neonic pesticides (19%)

# In[ ]:




