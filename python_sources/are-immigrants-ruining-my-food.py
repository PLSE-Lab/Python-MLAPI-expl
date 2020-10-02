#!/usr/bin/env python
# coding: utf-8

# # Are immigrants ruining my food?
# ### a rough look at the link between ethnic populations and food in US cities
# 
# In this analysis, I will look at how ethnic population size is linked to the quantity and quality of restaurants in a few major US cities.  I combine data on ethnic populations in US cities (from the American Community Survey) and restaurant quantity / quality data from Yelp (https://www.kaggle.com/yelp-dataset/yelp-dataset). As the Yelp data is limited to a few geographic areas, we'll focus on 6 metropolitan areas that are clearly marked in both datasets.
# 
# Let's begin by **importing some packages to work with**.

# In[ ]:


# data manipulation packages
import numpy as np
import pandas as pd
import re

# plotting packages
import matplotlib.pyplot as plt
import seaborn as sns


# Next, let's **read in the data**.  The datasets are pretty large, so this can take up to 30 seconds.

# In[2]:


# Import Yelp and American Community Survey (ACS) datasets
yelp = pd.read_csv('../input/yelp-dataset/yelp_business.csv')
acs = pd.read_csv('../input/ipumsancestryextract/2012-16_ipums_ancestry_extract.csv')


# ### 1. Data preparation
# #### 1a. Mapping Yelp geography to ACS
# First, I mapped the 6 biggest geographic areas present in both the Yelp and ACS data to each other. I map city names in the Yelp data ('city' column below) to 2013 Metropolitan areas in the ACS data by name, in an external 'mapping' file:

# In[18]:


#Import Yelp to ACS mapping
geo_map_yelp = pd.read_csv('../input/ipumsancestryextract/geo_map_yelp.csv')
geo_map_yelp


# We can then apply the mapping to the Yelp data, and filter out other cities not in our identified 6 (by using the 'inner' option while merging). Once done, let's print out the states left in the dataset, to check if this is done correctly:

# In[4]:


# Merge Yelp dataset with Yelp to ACS Mapping, dropping entries not in our 6 cities
yelp_geo = pd.merge(yelp, geo_map_yelp, on='city', how='inner', sort=False)
yelp_geo['state'].value_counts()


# We still have a tail of states who are not supposed to be in our analysis. This is because there are multiple cities named after our target cities in the US and overseas (e.g. Las Vegas, Florida).  We'll have to drop these out:

# In[5]:


# Drop out duplicate cities in other states
yelp_geo = yelp_geo.loc[(yelp_geo['state']=='AZ') |
                        (yelp_geo['state']=='NV') |
                        (yelp_geo['state']=='NC') |
                        (yelp_geo['state']=='OH') |
                        (yelp_geo['state']=='PA') |
                        (yelp_geo['state']=='IL')
                        ].reset_index(drop=True)

yelp_geo['state'].value_counts()


# #### 1b. Mapping Yelp restaurant ethnicities to a standardised list
# Beyond geography, we'll also have to match the ethnicity of restaurants to the ethnicity of populations.  As there are hundreds of restaurant ethnic tags in Yelp and hundreds of ethnic ancestry tags in the ACS data, I put together a simplified ethnic mapping with 9 categories (based on the ACS groupings).  I then label all Yelp businesses according to this mapping. The Yelp dataset also contains American restaurants and other business types, which will be skipped by this mapping.

# In[6]:


# prepare ethnicity in yelp data (by restaurant category)
# mapping source: https://www.yelp.com/developers/documentation/v3/category_list
eth_map_yelp = pd.read_csv('../input/ipumsancestryextract/eth_map_yelp.csv', index_col='Yelp_clean')

eth_map_yelp.head(10)


# In[7]:


# create new dataframe for Yelp data with labelled ethnic data
yelp_geo_eth = yelp_geo.copy()
yelp_geo_eth['ethnicity']=""
    
# convert the mapping to a dictionary to label the dataset
eth_dict_yelp = eth_map_yelp.to_dict()['ethnicity']

# Label all Yelp businesses (including restaurants) by ethnicity.  
# Note: this is very slow, using two for loops, which can take up to 1 minute. This can be substantially improved.
for k, v in eth_dict_yelp.items():
    for index, element in yelp_geo_eth.loc[:,'categories'].iteritems():
        if k in element:
            yelp_geo_eth.loc[yelp_geo_eth.index[index],'ethnicity']=v


# I can now **group the Yelp data together at the geography and ethnicity level:**

# In[8]:


yelp_grouped=yelp_geo_eth.groupby(['MET2013','ethnicity']).agg({'stars':['mean','median'], 'business_id':'count', 'review_count':'sum', 'Metropolitan area, 2013 OMB delineations':'first'})
yelp_grouped.columns=['mean_stars','median_stars','restaurant_count','review_count','area_name']
yelp_grouped.head(10)


# #### 1c. Prepare the ACS data
# We now have to apply the same ethnic mapping to the ACS data.

# In[9]:


# Import ACS ethnicity mapping
eth_map_acs = pd.read_csv('../input/ipumsancestryextract/eth_map_acs.csv')
eth_map_acs.iloc[[0,1,2,3,-4,-3,-2,-1]]


# In[10]:


# Merge ethnicity mapping with the ACS dataset, and drop irrelevant populations (e.g. American)
acs_eth = pd.merge(acs, eth_map_acs, on='ANCESTR1', how='inner', sort=False)
acs_eth = acs_eth[(acs_eth['ethnicity']!='american') & (acs_eth['ethnicity']!='na')]
acs_eth.head(5)


# We now can **group the ACS data together at the geography and ethnicity level.**
# (Note: we sum the 'PERWT' column, which is the weighted value of each individual in the dataset, to arrive at the estimated population size.  The ACS data provides these weights.)

# In[11]:


acs_grouped = acs_eth.groupby(['MET2013','ethnicity']).agg({'PERWT':'sum'})
acs_grouped.head(15)


# And **finally create the combined ACS & Yelp dataset.** Ideally we would have a larger sample size (and drop out rows with only e.g. 1 restaurant generating the rating), but this will have to do for now.

# In[12]:


comb_data=pd.merge(acs_grouped,yelp_grouped,left_index=True,right_index=True,how='inner').reset_index()
comb_data


# ### 2. Data analysis
# 
# We can use regression to try and understand the relationships in this dataset.  Due to the limited size of the combined dataset (~50 observations), we're unlikely to find any statistically significant results.  However, using Seaborn plots, we can get a rough intuition about relationships from the data, to be confirmed by further analysis with an expanded dataset.
# 
# Let's first take a look at the relationship between **ethnic population and number of restaurants of that cuisine:**

# In[13]:


# plot restaurant count by number of people (by ethnicity)
plt.figure(figsize=(20,10))

sns.regplot('PERWT','restaurant_count',data=comb_data)
plt.title("Ethnic restaurant count by population", size=22)
plt.xlabel("Ethnic population")
plt.ylabel("Ethnic restaurant count")
plt.show()


# While there does seem to be a slight positive relationship (roughly +200 restaurants for every +400K people?), the residuals seem to increase with the independent variable. Using the log of the independent and dependent variables improves this somewhat:

# In[ ]:


# Create log of variables
comb_data['log_pop']=comb_data['PERWT'].apply(np.log1p)
comb_data['log_rest_cnt']=comb_data['restaurant_count'].apply(np.log1p)

#Plot figure with log of variables
plt.figure(figsize=(20,10))
sns.regplot('log_pop','log_rest_cnt', data=comb_data)
plt.title("Ethnic restaurant count by population", size=22)
plt.xlabel("Log of ethnic population")
plt.ylabel("Log of ethnic restaurant count")
plt.show()


# So we could roughly say that a 1% increase in ethnic population increases # of restaurants by 1%. A slight positive relationship is completely unsurprising.
# 
# Let's next take a look at the **relationship between population and quality of restaurants** (we'll look at the mean star rating for restaurants corresponding to each ethnicity).

# In[150]:


# plot restaurant rating by population
sns.lmplot('PERWT','mean_stars', data=comb_data, 
            fit_reg=True, size=6, aspect=3, legend=False)
plt.title("Ethnic restaurant rating by population", size=22)
plt.xlabel("Ethnic population")
plt.ylabel("Mean ethnic restaurant stars")
plt.show()


# The residuals seem to decrease with the independent variable here, so let's try using the log of the independent variable:

# In[151]:


# plot restaurant rating by log population
sns.lmplot('log_pop','mean_stars', data=comb_data, 
            fit_reg=True, size=6, aspect=3, legend=False)
plt.title("Ethnic restaurant rating by population", size=22)
plt.xlabel("Log ethnic population")
plt.ylabel("Mean ethnic restaurant stars")
plt.show()


# Again somewhat improved.  Interestingly, there appears to be **a very slight negative relationship (if any)**, suggesting that % increase in ethnic population decreases mean ratings of ethnic restaurants.
# 
# This is an interesting topic, so it's worth disecting a bit how population can affect restaurant quality, in two broad categories:
# -  **Supply side:** growing supply of people of one ethnicity may mean more cooks etc. that could in theory allow more restaurants to open and improve food quality
# -  **Demand side:** growing population of one ethnicity should increase demand for their cuisine - resulting in more restaurants.  However, they may also be demanding more authentic flavours, which may not be to Americans' taste, and result in drop in star rating.  Alternatively, perhaps an increasing population demanding authentic flavour rates existing restaurants that cater to American tastes lower. 
# 
# So supply and demand side factors can help explain the increase in restaurants, but it appears demand side factors may  contribute to decline in ratings.
# 
# Given how diverse the cuisines included are (especially in terms of fit to American tastes), it's worth looking at **very rough trends by ethnicity.** We only have ~5 data points per ethnicity so this is a very rough, initial look. 

# In[153]:


# exclude Brazil and Pacific ethnicities
comb_data_trimmed = comb_data.loc[(comb_data['ethnicity']!='brazil')&(comb_data['ethnicity']!='pacific')]

# Plot facet chart for each ethnicity
ax = sns.lmplot('log_pop','mean_stars', data=comb_data_trimmed, 
               col='ethnicity', hue='ethnicity', col_wrap=3, scatter_kws={'s':100},
               fit_reg=True, size=6, aspect=2, legend=False)
ax.set(ylim=(2.5,5))
ax.set_titles(size=24)

plt.show()


# There seem to a few groupings here:
# -  **Positive relationship:** Subsaharan African, Middle East/North African and East Asian roughly increase with population
# -  **Negative relationship:** Eastern European, West Indies and South-Asian roughly decrease with population
# -  **Flat:** Western European and Latin American are roughly flat with population.
# -  **Excluded:** I excluded Brazil and Pacific from this chart due to limited data points.
# 
# It's a bit **hard to discern what differentiates ethnicities with Positive vs. Negative relationships** (topic for further investigation with more data!). However, the **Flat ethnicities** are the ones with largest populations, and most common cuisine presence in the US. Perhaps they are so integrated into American cuisine, that they are not connected to the ethnic population anymore? E.g. American style Western European cuisine exists regardless of the Western European population, and is entirely prepared by Americans. 
# 
# That's all I have time for on this for now - please let me know your feedback and thoughts!

# In[ ]:




