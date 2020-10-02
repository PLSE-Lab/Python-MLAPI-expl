#!/usr/bin/env python
# coding: utf-8

# # Import the libraries and the dataset

# In[ ]:


import pandas as pd


# In[ ]:


import_data = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_import.csv')


# In[ ]:


import_data.head()


# In[ ]:


import_data.shape


# ## Get the different Commodities, Countries name and Year

# In[ ]:


unique_commodities = import_data['Commodity'].unique()


# In[ ]:


len(unique_commodities)


# In[ ]:


unique_countries = import_data['country'].unique()


# In[ ]:


len(unique_countries)


# In[ ]:


unique_year = import_data['year'].unique()


# In[ ]:


len(unique_year)


# ## Visualize the value count for each year

# In[ ]:


yearly_data = import_data.groupby('year')['value'].sum()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.set()


# In[ ]:


font_dict = {
    'size': 20,
    'weight': 'bold'
}


# In[ ]:


plt.figure(figsize=(10, 4))
plt.subplot(211)
plt.title("Import Quantity-vs-Year", fontdict=font_dict)
plt.bar(x=yearly_data.index, height=yearly_data.values)
plt.ylabel('Import Quantity')
plt.subplot(212)
plt.plot(yearly_data)
plt.xlabel('Year')
plt.ylabel('Import Quantity')
plt.show()


# ## Visualize the import count by each country, on yearly basis

# In[ ]:


country_import = import_data.groupby(['country', 'year'])['value'].sum()


# In[ ]:


country_import = country_import.sort_index(axis=0, ascending=True)


# In[ ]:


country_import


# In[ ]:


country_import.loc['AFGHANISTAN TIS']


# In[ ]:


plt.figure(figsize=(10, 8))
plt.pie(country_import.loc['AFGHANISTAN TIS'].values, labels=country_import.loc['AFGHANISTAN TIS'].index,
       autopct="%.2f")
plt.title("Trade Analysis ", fontdict=font_dict)


# In[ ]:


import_data['Commodity'].unique()[:20]


# ## Some of the questions that need to be answered?
# 
# 1. What did India imported the most in any 2016, 2018 year? 
# 2. Which commodity forms a major chunk of trade?
# 3. How has the trade between India and any given country grown over time? (Countries - USA, China, IRAN, Japan, Russia, Afghanistan, Pakistan)

# ### 1. What did India import the most in any 2016, 2018 year? 

# In[ ]:


commodities_2016_sum = import_data[import_data['year'] == 2016].groupby('Commodity')['value'].sum()


# In[ ]:


commodities_2016_sum.sort_values(ascending=False).head(1).index


# In[ ]:


commodities_2018_sum = import_data[import_data['year'] == 2018].groupby('Commodity')['value'].sum()


# In[ ]:


commodities_2018_sum.sort_values(ascending=False).head(1).index


# ### Answer - Most Imported Item
# 
# 1. 2016 - NATURAL OR CULTURED PEARLS.. (Jewellery Item)
# 2. 2018 - MINERAL FUELS.. (Natural Resource)

# ### 2. Which commodity forms a major chunk of trade?

# In[ ]:


commodities_sum = import_data.groupby('Commodity')['value'].sum().sort_values(ascending=False)


# In[ ]:


commodities_sum = commodities_sum[:10]


# In[ ]:


## reduce the name of the commodities so that it's easier to visualize
commodities_sum.index = commodities_sum.index.map(lambda x: x[:15])


# In[ ]:


plt.figure(figsize=(10, 8))
plt.pie(commodities_sum, labels=commodities_sum.index, autopct="%.2f")
plt.title('Resource-vs-Import Share Percent', fontdict=font_dict)


# ### Answer - Most Imported Item
# 
# 1. Mineral Fuels.
# 2. Natural or Cultured Pearls.

# ### 3. How has the trade between India and any given country grown over time? (Countries - USA, China, IRAN, Japan, Russia, United Arab Emirates, UK, Afghanistan, Pakistan)

# In[ ]:


import_data['country'][import_data['country'].str.startswith('U')].unique()


# In[ ]:


import_data['country'][import_data['country'].str.startswith('C')].unique()


# In[ ]:


import_data['country'][import_data['country'].str.startswith('I')].unique()


# In[ ]:


import_data['country'][import_data['country'].str.startswith('J')].unique()


# In[ ]:


import_data['country'][import_data['country'].str.startswith('R')].unique()


# In[ ]:


import_data['country'][import_data['country'].str.startswith('A')].unique()


# In[ ]:


import_data['country'][import_data['country'].str.startswith('P')].unique()


# In[ ]:


countries_list = ['U S A', 'U ARAB EMTS', 'U K', 'CHINA P RP', 'IRAN', 'JAPAN', 'RUSSIA', 'AFGHANISTAN TIS', 'PAKISTAN IR']


# In[ ]:


specific_countries_data = import_data[import_data['country'].isin(countries_list)].groupby(['country', 'year'])['value'].sum()


# In[ ]:


specific_countries_data


# #### Which country accounts for maximum trade in the list of specified countries

# In[ ]:


countries_share = import_data[import_data['country'].isin(countries_list)].groupby('country')['value'].sum().sort_values(ascending=False)


# In[ ]:


countries_share = countries_share.apply(lambda x: round(100 * x/countries_share.sum()))


# In[ ]:


countries_share


# In[ ]:


plt.figure(figsize=(12, 8))
plt.pie(countries_share, labels=countries_share.index, autopct="%.2f")
plt.title('Countries Percentage Share', fontdict=font_dict)


# ### How the trade has grown over time for China?

# In[ ]:


plt.figure(figsize=(10, 8))
plt.subplot(211)
plt.bar(x=specific_countries_data.loc['CHINA P RP'].index, height=specific_countries_data.loc['CHINA P RP'].values)
plt.title('Import-vs-Year for China', fontdict=font_dict)
plt.subplot(212)
plt.plot(specific_countries_data.loc['CHINA P RP'])
plt.xlabel('Year')
plt.ylabel('Import in US($)')


# ### Visualizing trade statistics regarding specified countries

# In[ ]:


for country in countries_list:
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.bar(x=specific_countries_data.loc[country].index, height=specific_countries_data.loc[country].values)
    title_name = 'Import-vs-Year for ' + country
    plt.title(title_name, fontdict=font_dict)
    plt.subplot(212)
    plt.plot(specific_countries_data.loc[country])
    plt.xlabel('Year')
    plt.ylabel('Import in US($)')

