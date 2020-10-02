#!/usr/bin/env python
# coding: utf-8

# **INTRODUCTION**
# 
# We will analyse the World Happiness Report for the years 2015, 2016 & 2017. In particular we are interested in finding those countries whose overall 'Happiness Rank' has decreased the most between 2015 and 2017 and the main reason for such a move.
# 
# First we will read in the necessary dataframes and make sure that the data is consistent with regard to the country names (rows) and column headings. We can then create a new dataframe (happy_final) listing countries with their 'Happiness Rank' for each of the 3 years and a new column 'Change in Rank' showing the change between 2015 and 2017.
# 
# We then visualize these changes in rank by plotting two bar charts for the countries in the regions 'Latin America & Caribbean' and 'Sub-Saharan Africa'.
# 
# Finally, we look at the countries with the largest decrease in rank and calculate which attribute from our original dataframes contributes the most to this change.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt #plotting charts
import seaborn as sns #plotting charts


# In[ ]:


# Read in the data from the 3 reports
happy_2015 = pd.read_csv('../input/2015.csv')
happy_2016 = pd.read_csv('../input/2016.csv')
happy_2017 = pd.read_csv('../input/2017.csv')


# In[ ]:


happy_2015.head()


# In[ ]:


happy_2016.head()


# In[ ]:


happy_2017.head()


# In[ ]:


happy_2015.info()


# In[ ]:


happy_2016.info()


# In[ ]:


happy_2017.info()


# The 2017 dataframe has different heading names from the 2015/2016 data and there seems to be a mismatch in number of countries between the 3 sets of data, which we can explore and clean up now. Otherwise the data types seem good.

# In[ ]:


# Let's find the missing countries between 2015 & 2016 by merging the two tables and 
# creating a new _merge column

merge1516 = pd.merge(happy_2015,happy_2016,on='Country',how='outer',indicator=True)

# List the countries included in 2015 but not 2016. 
not16 = merge1516[merge1516['_merge'] == 'left_only']['Country'].tolist()
not16


# In[ ]:


# List the countries included in 2016 but not 2015
not15 = merge1516[merge1516['_merge'] == 'right_only']['Country'].tolist()
not15


# In[ ]:


# Fix "Somaliland Region" in 2015 table
# Remove null value countries from both dataframes

happy_2015['Country'].replace(to_replace = 'Somaliland region',value='Somaliland Region',inplace=True)
not15.remove('Somaliland Region')
not16.remove('Somaliland region')
happy_2015_clean = happy_2015[~happy_2015['Country'].isin(not16)]
happy_2016_clean = happy_2016[~happy_2016['Country'].isin(not15)]


# In[ ]:


# Use the same method to compare the 'clean' 2016 table to 2017 table

merge1617 = pd.merge(happy_2016_clean,happy_2017,on='Country',how='outer',indicator=True)

# Countries in 2016_clean but not in 2017
not17 = merge1617[merge1617['_merge'] == 'left_only']['Country'].tolist()
not17


# In[ ]:


# Countries in 2017 but not in 2016_clean
not16_new = merge1617[merge1617['_merge'] == 'right_only']['Country'].tolist()
not16_new


# In[ ]:


# Make the names for 'Taiwan' and 'Hong Kong' consistent across both happy_2016_clean and 
# happy_2017 tables

happy_2017['Country'].replace(to_replace = 'Hong Kong S.A.R., China',value = 'Hong Kong',inplace=True)
happy_2017['Country'].replace(to_replace = 'Taiwan Province of China',value = 'Taiwan',inplace=True)
not17.remove('Taiwan')
not17.remove('Hong Kong')
not16_new.remove('Taiwan Province of China')
not16_new.remove('Hong Kong S.A.R., China')

# Finally, clean up null values from all 3 dataframes

happy_2015_clean = happy_2015_clean[~happy_2015_clean['Country'].isin(not17)]
happy_2016_clean = happy_2016_clean[~happy_2016_clean['Country'].isin(not17)]
happy_2017_clean = happy_2017[~happy_2017['Country'].isin(not16_new)]


# In[ ]:


happy_2017_clean.head()


# In[ ]:


# rename column headings
happy_2017_clean = happy_2017_clean.rename(columns={'Happiness.Rank': 'Happiness Rank 2017'})
happy_2015_clean = happy_2015_clean.rename(columns={'Happiness Rank': 'Happiness Rank 2015'})
happy_2016_clean = happy_2016_clean.rename(columns={'Happiness Rank': 'Happiness Rank 2016'})


# In[ ]:


# Create a new dataframe of Country, Region, Happiness Rank from all 3 clean
# dataframes

column_list_2015 = ['Country','Region','Happiness Rank 2015']
happy_2015_clean_final = happy_2015_clean[column_list_2015]

column_list_2016 = ['Country','Region','Happiness Rank 2016']
happy_2016_clean_final = happy_2016_clean[column_list_2016]

column_list_2017 = ['Country','Happiness Rank 2017']
happy_2017_clean_final = happy_2017_clean[column_list_2017]

happy_final = happy_2015_clean_final.merge(
    happy_2016_clean_final,on='Country').merge(
    happy_2017_clean_final,on='Country')  # merge all 3 data frames on 'Country' column

happy_final


# In[ ]:


# Remove 'Region_y' column and rename 'Region_x' to 'Region'

happy_final.drop('Region_y',axis=1,inplace = True)
happy_final.rename(columns={'Region_x':'Region'},inplace = True)


# In[ ]:


happy_final.head(10)


# In[ ]:


# Let's see which countries have increased/decreased their Happiness Rank the most between 2015
# and 2017

# Add a new column showing the change in rank between 2017 and 2015
happy_final['Change in Rank'] = happy_final['Happiness Rank 2015'] - happy_final['Happiness Rank 2017']


# In[ ]:


happy_final.head(10)


# In[ ]:


# The top 10 countries that have seen their happiness rank decrease the most between 2015 and 2017
happy_final.sort_values(by='Change in Rank',axis=0,ascending=True).head(10)


# In[ ]:


# Look at the countries in Latin America and Caribbean only
happy_latin = happy_final[happy_final['Region'] == 'Latin America and Caribbean']
happy_latin.head()


# In[ ]:


# Plot a bar chart of countries in Latin America & Caribbean versus their change in rank
import seaborn as sns
plt.figure(figsize=(14,8))
sns.set_style("whitegrid")
sns.barplot(x='Country',y='Change in Rank',
            data=happy_latin.sort_values(by='Change in Rank',ascending=False),palette='muted')
plt.xticks(rotation=90)
plt.grid(b=True,which='major')


# In[ ]:


# Plot a similar bar chart for those countries in Sub-Saharan Africa
happy_sub_saharan = happy_final[happy_final['Region'] == 'Sub-Saharan Africa']
happy_sub_saharan_sort = happy_sub_saharan.sort_values(by='Change in Rank', ascending=False)
plt.figure(figsize=(14,8))
sns.set_style("whitegrid")
sns.barplot(x='Country',y='Change in Rank',data=happy_sub_saharan_sort,palette = 'muted')
plt.xticks(rotation=90)
plt.grid(b=True,which='major')


# In[ ]:


# Find the countries that have a decrease in rank that is more than 2 times the standard 
# deviation
happy_final[happy_final['Change in Rank'] < - happy_final['Change in Rank'].std() *2]


# In[ ]:


# Let's look at the attributes that contributes the most to the decrease in happiness 
# in the countries Venezuela, Zambia, Liberia and Haiti

countries = ['Venezuela','Zambia','Liberia','Haiti']
happy_2015_copy = happy_2015.copy()
happy_2017_copy = happy_2017.copy()
happy_2015_copy = happy_2015_copy.set_index('Country')
happy_2017_copy = happy_2017_copy.set_index('Country')

# List the attributes (column headings) we are looking at
attrib_2015 = ['Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom'
              , 'Trust (Government Corruption)', 'Generosity']
attrib_2017 = ['Economy..GDP.per.Capita.','Family','Health..Life.Expectancy.','Freedom'
              ,'Trust..Government.Corruption.','Generosity']

# Rename 2017 headings to match those of 2015
happy_2017_copy = happy_2017_copy.rename(columns=dict(zip(attrib_2017,attrib_2015)))

# Loop over the 4 countries, list the percentage changes, find the index of the minimum
# change and print out the relevant attritbute.
for country in countries:
    attrib_pct = []
    print(country + ':')
    for i in range(6):
       my_attrib = (happy_2017_copy.loc[country,attrib_2015[i]] - 
                    happy_2015_copy.loc[country,attrib_2015[i]]) *100 / happy_2015_copy.loc[country,attrib_2015[i]]
       attrib_pct.append(my_attrib)
    min_index_1 = attrib_pct.index(min(attrib_pct))
    
    print(attrib_2015[min_index_1] + "\n" )



# **Conclusion:**
# 
# The countries that have seen their 'Happiness Rank' decrease the most between 2015 and 2017 seem to be concentrated
# in the Latin America & Caribbean and Sub-Saharan Africa regions with the attributes that contribute the most to this 'change' in happiness being 'Freedom' and 'Trust (Government Corruption).

# In[ ]:




