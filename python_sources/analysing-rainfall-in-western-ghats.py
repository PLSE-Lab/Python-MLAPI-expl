#!/usr/bin/env python
# coding: utf-8

# # **Analysing Rainfall in Western Ghats**
# 
# The Western Ghats of India are a very important resource in India. It is home to a whopping 1/3rd of all animal species that exist in the country, thick rainforests and massive mountains (with some parts rising above 2500 meters). The region is also known for its rich sources of fresh water, with several natural springs and waterfalls that embellish the forests. However, the global climate is changing and we must study its impact of the rainfall pattern in the Western Ghats.
# 
# We have monthly and annual rainfall measurements of 115 years (1901 to 2015).
# 
# Our objectives would be:
# * Extract the data of subdivisions that lie in the region of the Western Ghats.
# * Analyse the season rainfall
# * Analyse the trend of the annual rainfall in Western Ghats
# * Compare with rest of the country

# In[ ]:


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Read the dataset for district wise rainfall of India
india_df = pd.read_csv('/kaggle/input/rainfall-in-india/rainfall in india 1901-2015.csv') 
india_df.info()


# In[ ]:


# Let's take care of the null values if any
# We will front fill the null values to maintain the trend

india_df['JAN'].fillna(method = 'ffill', inplace = True)
india_df['FEB'].fillna(method = 'ffill', inplace = True)
india_df['MAR'].fillna(method = 'ffill', inplace = True)
india_df['APR'].fillna(method = 'ffill', inplace = True)
india_df['MAY'].fillna(method = 'ffill', inplace = True)
india_df['JUN'].fillna(method = 'ffill', inplace = True)
india_df['JUL'].fillna(method = 'ffill', inplace = True)
india_df['AUG'].fillna(method = 'ffill', inplace = True)
india_df['SEP'].fillna(method = 'ffill', inplace = True)
india_df['OCT'].fillna(method = 'ffill', inplace = True)
india_df['NOV'].fillna(method = 'ffill', inplace = True)
india_df['DEC'].fillna(method = 'ffill', inplace = True)
india_df['ANNUAL'].fillna(method = 'ffill', inplace = True)
india_df['Jan-Feb'].fillna(method = 'ffill', inplace = True)
india_df['Mar-May'].fillna(method = 'ffill', inplace = True)
india_df['Jun-Sep'].fillna(method = 'ffill', inplace = True)
india_df['Oct-Dec'].fillna(method = 'ffill', inplace = True)


# This data set contains data of the whole country. We need the rainfall measurements of only those subdivisions which lie within our region of interest.
# 
# ![image.png](attachment:image.png)
# Image source: Trends in the rainfall pattern over India - Pulak Guhathakurta, M. Rajeevan
# 
# We will consider the following metereological subdivsions which approximately cover the Western Ghats.
# 
# 1. MADHYA MAHARASHTRA
# 2. KONKAN & GOA
# 3. COASTAL KARNATAKA
# 4. SOUTH INTERIOR KARNATAKA
# 5. KERALA

# In[ ]:


# Subdivisions
subdivisions_list = ['MADHYA MAHARASHTRA','KONKAN & GOA','COASTAL KARNATAKA','SOUTH INTERIOR KARNATAKA', 'KERALA']

# Now let's carve out the region of interest
western_ghats_df = india_df.loc[india_df['SUBDIVISION'].isin(subdivisions_list)]
rest_of_india_df = india_df.loc[~india_df['SUBDIVISION'].isin(subdivisions_list)]
display(western_ghats_df)

# Madhya Maharashtra subdivision
madhya_maharashtra_df = western_ghats_df.loc[western_ghats_df['SUBDIVISION'] == 'MADHYA MAHARASHTRA']

# Konkan & Goa subdivision
konkan_goa_df = western_ghats_df.loc[western_ghats_df['SUBDIVISION'] == 'KONKAN & GOA']

# Costal Karnataka subdivision
coastal_karnataka_df = western_ghats_df.loc[western_ghats_df['SUBDIVISION'] == 'COASTAL KARNATAKA']

# South Interior Karnataka subdivision
south_interior_karnataka_df = western_ghats_df.loc[western_ghats_df['SUBDIVISION'] == 'SOUTH INTERIOR KARNATAKA']

# Kerala subdivision
kerala_df = western_ghats_df.loc[western_ghats_df['SUBDIVISION'] == 'KERALA']


# In[ ]:


# Annual subdivision wise rainfall pattern in Western Ghats

plt.figure(figsize=(10,10))
plt.plot(madhya_maharashtra_df['YEAR'], madhya_maharashtra_df['ANNUAL'].ewm(span=10, adjust=False).mean())
plt.plot(konkan_goa_df['YEAR'], konkan_goa_df['ANNUAL'].ewm(span=10, adjust=False).mean())
plt.plot(coastal_karnataka_df['YEAR'], coastal_karnataka_df['ANNUAL'].ewm(span=10, adjust=False).mean())
plt.plot(south_interior_karnataka_df['YEAR'], south_interior_karnataka_df['ANNUAL'].ewm(span=10, adjust=False).mean())
plt.plot(kerala_df['YEAR'], kerala_df['ANNUAL'].ewm(span=10, adjust=False).mean())
plt.ylabel("Annual Rainfall (mm)")
plt.xlabel("Year")
plt.legend(['Madhya Maharashtra', 'Konkan & Goa', 'Coastal Karnataka', 'South Interior Karnataka', 'Kerala'], loc='upper left')
plt.show()


# * It is the coastal portion of the ghats that receive the most rainfall with Coastal Karnataka leading the way.
# * There is a gradual increasing trend in annual rainfall in Coastal Karnataka and Konkan Goa in the last 115 years.
# * There is a very gentle declining trend in annual rainfall in Kerala.
# * Although Madhya Maharashtra and South Interior Karnataka have a much lower annual rainfall, the annual amounts are increasing.

# In[ ]:


# Rainfall pattern in June-September in Western Ghats

plt.figure(figsize=(10,10))
plt.plot(madhya_maharashtra_df['YEAR'], madhya_maharashtra_df['Jun-Sep'].ewm(span=10, adjust=False).mean())
plt.plot(konkan_goa_df['YEAR'], konkan_goa_df['Jun-Sep'].ewm(span=10, adjust=False).mean())
plt.plot(coastal_karnataka_df['YEAR'], coastal_karnataka_df['Jun-Sep'].ewm(span=10, adjust=False).mean())
plt.plot(south_interior_karnataka_df['YEAR'], south_interior_karnataka_df['Jun-Sep'].ewm(span=10, adjust=False).mean())
plt.plot(kerala_df['YEAR'], kerala_df['Jun-Sep'].ewm(span=10, adjust=False).mean())
plt.ylabel("Rainfall in June-Sep (mm)")
plt.xlabel("Year")
plt.legend(['Madhya Maharashtra', 'Konkan & Goa', 'Coastal Karnataka', 'South Interior Karnataka', 'Kerala'], loc='upper left')
plt.show()


# * The majority of rainfall in the Western Ghats is received in the June-September period.
# * Once again Coastal Karnataka kicks off with the highest amounts.
# * Kerala has a surprisingly lower rainfall in this period as compared to Coastal Karnataka and Konkan Goa.

# In[ ]:


# Rainfall pattern in October-December in Western Ghats

plt.figure(figsize=(10,10))
plt.plot(madhya_maharashtra_df['YEAR'], madhya_maharashtra_df['Oct-Dec'].ewm(span=10, adjust=False).mean())
plt.plot(konkan_goa_df['YEAR'], konkan_goa_df['Oct-Dec'].ewm(span=10, adjust=False).mean())
plt.plot(coastal_karnataka_df['YEAR'], coastal_karnataka_df['Oct-Dec'].ewm(span=10, adjust=False).mean())
plt.plot(south_interior_karnataka_df['YEAR'], south_interior_karnataka_df['Oct-Dec'].ewm(span=10, adjust=False).mean())
plt.plot(kerala_df['YEAR'], kerala_df['Oct-Dec'].ewm(span=10, adjust=False).mean())
plt.ylabel("Rainfall in Oct-Dec (mm)")
plt.xlabel("Year")
plt.legend(['Madhya Maharashtra', 'Konkan & Goa', 'Coastal Karnataka', 'South Interior Karnataka', 'Kerala'], loc='upper left')
plt.show()


# * After September, the monsoon recedes in most parts of the ghats. 
# * Kerala maintains a significantly higher rainfall in Oct-Dec period.

# In[ ]:


# Rainfall pattern in March-May in Western Ghats

plt.figure(figsize=(10,10))
plt.plot(madhya_maharashtra_df['YEAR'], madhya_maharashtra_df['Mar-May'].ewm(span=10, adjust=False).mean())
plt.plot(konkan_goa_df['YEAR'], konkan_goa_df['Mar-May'].ewm(span=10, adjust=False).mean())
plt.plot(coastal_karnataka_df['YEAR'], coastal_karnataka_df['Mar-May'].ewm(span=10, adjust=False).mean())
plt.plot(south_interior_karnataka_df['YEAR'], south_interior_karnataka_df['Mar-May'].ewm(span=10, adjust=False).mean())
plt.plot(kerala_df['YEAR'], kerala_df['Mar-May'].ewm(span=10, adjust=False).mean())
plt.ylabel("Rainfall in March-May (mm)")
plt.xlabel("Year")
plt.legend(['Madhya Maharashtra', 'Konkan & Goa', 'Coastal Karnataka', 'South Interior Karnataka', 'Kerala'], loc='upper left')
plt.show()


# * We see that the rainfall in pre-monsoon season is also significantly higher in Kerala.
# * Probably, this early onset and delayed offset of monsoon in Kerala is what contributes to the annual totals.

# In[ ]:


# Analyse the seasonal variation in annual rainfall for the subdivsions
sns.set_style('whitegrid')
plt.figure(figsize=(10, 10))
plt.xticks(rotation='vertical')
sns.boxplot(x='SUBDIVISION', y='ANNUAL', data=western_ghats_df)
plt.ylabel("Annual Rainfall in Western Ghats (mm)")
plt.xlabel("Subdivision")
plt.show()


# In[ ]:


# Annual rainfall pattern in Western Ghats

western_ghats_yearly_df = western_ghats_df.groupby('YEAR').mean().reset_index()

plt.figure(figsize=(10,10))
plt.plot(western_ghats_yearly_df['YEAR'], western_ghats_yearly_df['ANNUAL'].ewm(span=20, adjust=False).mean())
plt.ylabel("Annual Rainfall in Western Ghats (mm)")
plt.xlabel("Year")
plt.show()


# The average annual rainfall in the Western Ghats region has been gradually increasing since 1901.

# In[ ]:


# Seasonality in rainfall in Western Ghats
western_ghats_df.groupby('YEAR').sum().drop(['ANNUAL','Jan-Feb','Mar-May','Jun-Sep','Oct-Dec'], axis=1).T.plot(figsize=(10,10), legend=False)
plt.ylabel("Rainfall (mm)")
plt.xlabel("Year")
plt.show()


# The rainfall in the Western Ghats hits the peak in July.  

# In[ ]:


# Compare rainfall in western ghats with rest of india
rest_of_india_yearly_df = rest_of_india_df.groupby('YEAR').mean().reset_index()

plt.figure(figsize=(10,10))
plt.plot(western_ghats_yearly_df['YEAR'], western_ghats_yearly_df['ANNUAL'].ewm(span=20, adjust=False).mean())
plt.plot(rest_of_india_yearly_df['YEAR'], rest_of_india_yearly_df['ANNUAL'].ewm(span=20, adjust=False).mean())
plt.ylabel("Annual Rainfall (mm)")
plt.xlabel("Year")
plt.legend(['Western Ghats', 'Rest of India'], loc='upper left')
plt.show()


# The average annual rainfall in the western ghats is significantly higher than the rest of the country. But this is expected. What we are more interested in is the trend, which is clearly increasing in the region while it is decreasing in rest of the country.
# 
# Although climate change is severely affecting the indian subcontinent since the last 100 years, the Western Ghats have been more robust than the other regions.
# 
# PS: I do not wish to encourage continuing of our old destructive lifestyles. The forests have a way of looking after themselves, they can absorb a certain amount of pressure from civilization, but there is a limit.
