#!/usr/bin/env python
# coding: utf-8

# # Information about the data - 
# 
# The dataset consists of trade values for import of commodities in million US '$'. The dataset is tidy and each row consists of a single observation.<br><br>
# <b>Importing</b> means buying foreign goods and services by citizens, businesses and government of a country. No matter, how they are sent to the country. They can be shipped, sent it by e-mail, or even hand carried in personal luggage on a plane.<br><br>
# 
# <br>
# <p>In this kernel, we are going to 2 important visualization - </p>
# <br>
# 1. Name of the top 10 countries from which, India import the maximum number of commodities, with it's percentage compared to whole import data. However, we are also going to visualize the top 5 commodities in those countries and show their percentage as well.
# <br>
# 2. Name of the top 10 commodities, which India import the maximum. However, not only that we are also going to see the name of those top 5 countries from which we have imported the particular commodity.
# <br><br><br>
# <p>Visualization used - <b>Sunburst<b></p>

# ### Let's start exploring our dataset

# ### Importing the pandas libraries for data analysis

# In[ ]:


import pandas as pd


# ### Import the data set

# In[ ]:


data = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_import.csv')


# In[ ]:


data.head()


# ### Data Description - 
# 
# 1. HSCode - Commodity Code.
# 2. Commodity - Name of the commodity which is being imported from other country to India.
# 3. Value - Value of commodity in million US '$'.
# 4. Country - Name of the country, from which the commodity is being imported.
# 5. Year - Year number in which the particular commodity is being imported.

# ### Cut short the name of commodities to 15 characters only

# In[ ]:


data['Commodity'] = data['Commodity'].map(lambda x: x[:15])


# In[ ]:


data.head(2)


# ### Done

# ### Get the name of top 10 countries with percentage that has the highest share in import percentage

# In[ ]:


highest_imported_country = data.groupby(['country'])['value'].sum()


# In[ ]:


highest_imported_country.sort_values(ascending=False)


# In[ ]:


## get the percentage share
highest_imported_country = highest_imported_country.apply(lambda x:round(100 * x/highest_imported_country.sum())).sort_values(ascending=False)


# In[ ]:


## cut short the data to only top 10 countries
highest_imported_country = highest_imported_country[:10]


# In[ ]:


highest_imported_country


# ### Name of the countries, from which India imports the maximum amount of goods are - 
# 1. CHINA P RP
# 2. U S A
# 3. U ARAB EMTS
# 4. SAUDI ARAB
# 5. SWITZERLAND
# 6. IRAQ
# 7. AUSTRALIA
# 8. INDONESIA
# 9. GERMANY
# 10. KOREA RP

# #### Done

# ### Get the name of top 10 commodities, which India imports and get their percentage share as well

# In[ ]:


highest_imported_commodity = data.groupby('Commodity')['value'].sum()


# In[ ]:


highest_imported_commodity.sort_values(ascending=False)


# In[ ]:


## getting the percentage share
highest_imported_commodity = highest_imported_commodity.apply(lambda x:round(100 * x/highest_imported_commodity.sum())).sort_values(ascending=False)


# In[ ]:


## get the top 10 commodities only
highest_imported_commodity = highest_imported_commodity[:10]


# In[ ]:


highest_imported_commodity


# ### Name of the top 10 commodities, that India imports are - 
# 
# 1. Mineral Fuels - 33%
# 2. Natural or Cult (Jewellery) - 16.0
# 3. Electrical Machinery - 9.0
# 4. Nuclear Reactor - 8.0
# 5. Organic Chemicals - 4.0
# 6. Iron and Steel - 3.0
# 7. Plastic and Art - 3.0
# 8. Animal or Vegetable - 2.0
# 9. Optical, or Photog - 2.0
# 10. Inorganic Chemicals - 1.0

# #### Done

# ## Now, get the name and percentage share of top 5 commodities that India gets from the list of top 10 countries

# In[ ]:


top_10_countries = highest_imported_country.index.tolist()


# In[ ]:


countries_commodity_data = data.groupby(['country', 'Commodity'])['value'].sum()


# In[ ]:


countries_commodity_data


# In[ ]:


countries_commodity_data.groupby(level=1).sum()


# In[ ]:


sorted_country_commodity_data = countries_commodity_data.groupby(level=0).apply(lambda x: round(100 * x/x.sum())).sort_values(ascending=False)


# In[ ]:


sorted_country_commodity_data.loc['CHINA P RP']


# ### Conclusion = Our data has been sorted in such a manner that firstly, we will get the name of the country and then the commodity percentage share from that particular country

# ### Now, we will reduce our data to top 5 values only from each country

# In[ ]:


sorted_country_commodity_data = sorted_country_commodity_data.groupby(level=0).head()


# In[ ]:


sorted_country_commodity_data.loc['CHINA P RP']


# ### Now, selecting the data only for the required countries only

# In[ ]:


sorted_country_commodity_data.head(2)


# In[ ]:


top_10_countries


# In[ ]:


## This command is working.. need to check to get the multiindex for all the top countries
# sorted_country_commodity_data.loc[sorted_country_commodity_data.index.get_level_values('country') == 'SAUDI ARAB']

sorted_country_commodity_data.loc[sorted_country_commodity_data.index.get_level_values('country') == 'SAUDI ARAB']


# In[ ]:


sorted_country_commodity_data.loc['SAUDI ARAB']


# In[ ]:


for country in top_10_countries:
    print(country)
    print(sorted_country_commodity_data.loc[country])


# ### As of now, we have the name of the country and top 5 commodities, India imports from them, with their percentage share with respect to that particular country only

# ### Now, creating our master SunBurst diagram that shows the name of the country and then the name of commodities with the percentage values as well

# In[ ]:


labels_list = top_10_countries.copy()


# In[ ]:


highest_imported_country.loc['U S A']


# In[ ]:


labels_list = []
values_list = []
for country in top_10_countries:
    ## 1. add the percentage value of country
    values_list.append(highest_imported_country.loc[country])
    ## 2. add the percentage value of commodities
    minerals = sorted_country_commodity_data.loc[country].index.tolist()
    minerals_values = sorted_country_commodity_data.loc[country].values.tolist()
    for i in range(0, len(minerals)):
        minerals[i] = country + "-" + minerals[i]
    labels_list.append(country)
    labels_list.extend(minerals)
    values_list.extend(minerals_values)


# In[ ]:


print("Labels count:", len(labels_list))
print("Values count:", len(values_list))


# In[ ]:


parent_list = []
current_country = ""
i = 0
for country in labels_list:
    if (country in top_10_countries):
        parent_list.append("Commodities")
        current_country = country
#         values_list.append(i)
    else:
        parent_list.append(current_country)
    i = i+1


# In[ ]:


len(parent_list)


# In[ ]:


import matplotlib.pyplot as plt
import plotly.graph_objects as go


# <h2><center>Name of the top 10 countries and top 5 commodities, imported from those country</center></h2>

# In[ ]:


trace = go.Sunburst(
    labels = labels_list,
    parents = parent_list,
    values = values_list,
#     branchvalues="total",
    outsidetextfont = {"size": 20, "color": "#377eb8"},
    marker = {"line": {"width": 2}},
)

layout = go.Layout(
    margin = go.layout.Margin(t=0, l=0, r=0, b=0)
)

go.Figure([trace], layout).show()


# <br><br>

# ## Get the name of the countries from which top 10 commodities are being imported

# In[ ]:


top_10_commodities = highest_imported_commodity.index.tolist()


# In[ ]:


top_10_commodities


# ## Group countries by commodities and their percentage share

# In[ ]:


commodity_countries_data = data.groupby(['Commodity', 'country'])['value'].sum()


# In[ ]:


commodity_countries_data


# In[ ]:


commodity_countries_data.groupby(level=1).sum()


# In[ ]:


sorted_commodity_countries_data = commodity_countries_data.groupby(level=0).apply(lambda x: round(100 * x/x.sum())).sort_values(ascending=False)


# In[ ]:


sorted_commodity_countries_data.loc['MINERAL FUELS, ']


# ### Conclusion = Our data has been sorted in such a manner that firstly, we will get the name of the commodity and then the countries percentage share for that particular commodity

# ### Now, we will reduce our data to top 5 countries name only from each commodity

# In[ ]:


sorted_commodity_countries_data = sorted_commodity_countries_data.groupby(level=0).head()


# In[ ]:


sorted_commodity_countries_data.loc['MINERAL FUELS, ']


# ### Now, selecting the data only for the required countries only

# In[ ]:


sorted_commodity_countries_data.head(2)


# In[ ]:


top_10_commodities


# In[ ]:


## This command is working.. need to check to get the multiindex for all the top countries
# sorted_commodity_countries_data.loc[sorted_commodity_countries_data.index.get_level_values('Commodity') == 'MINERAL FUELS, ']

sorted_commodity_countries_data.loc[sorted_commodity_countries_data.index.get_level_values('Commodity') == 'MINERAL FUELS, ']


# In[ ]:


sorted_commodity_countries_data.loc['MINERAL FUELS, ']


# In[ ]:


for commodity in top_10_commodities:
    print(commodity)
    print(sorted_commodity_countries_data.loc[commodity])


# ### As of now, we have the name of the commodity and top 5 countries, India imports from them, with their percentage share with respect to that particular commodity

# ### Now, creating our master SunBurst diagram that shows the name of the commodity and then the name of countries with the percentage values as well

# In[ ]:


labels_list_commodities = top_10_commodities.copy()


# In[ ]:


highest_imported_commodity.loc['ANIMAL OR VEGET']


# In[ ]:


labels_list_commodities = []
values_list_commodities = []
for commodity in top_10_commodities:
    ## 1. add the percentage value of country
    values_list_commodities.append(highest_imported_commodity.loc[commodity])
    ## 2. add the percentage value of commodities
    countries = sorted_commodity_countries_data.loc[commodity].index.tolist()
    countries_values = sorted_commodity_countries_data.loc[commodity].values.tolist()
    for i in range(0, len(countries)):
        countries[i] = countries[i] + "-" + commodity
    labels_list_commodities.append(commodity)
    labels_list_commodities.extend(countries)
    values_list_commodities.extend(countries_values)


# In[ ]:


print("Labels count:", len(labels_list_commodities))
print("Values count:", len(values_list_commodities))


# In[ ]:


parent_list_commodities = []
current_commodity = ""
i = 0
for commodity in labels_list_commodities:
    if (commodity in top_10_commodities):
        parent_list_commodities.append("Commodities")
        current_commodity = commodity
#         values_list.append(i)
    else:
        parent_list_commodities.append(current_commodity)
    i = i+1


# In[ ]:


len(parent_list_commodities)


# In[ ]:


import matplotlib.pyplot as plt
import plotly.graph_objects as go


# <h2><center>Top 10 commodities and top 5 countries from which they are being imported</center></h2>

# In[ ]:


trace = go.Sunburst(
    labels = labels_list_commodities,
    parents = parent_list_commodities,
    values = values_list_commodities,
#     branchvalues="total",
    outsidetextfont = {"size": 20, "color": "#377eb8"},
    marker = {"line": {"width": 2}},
)

layout = go.Layout(
    margin = go.layout.Margin(t=0, l=0, r=0, b=0)
)

go.Figure([trace], layout).show()


# <br><br>

# In[ ]:


top_10_commodities


# # Conclusion - 
# 
# <b> Top 10 countries accounting for maximum chunk of import trade are - </b>
# 1. China
# 2. Saudi Arab
# 3. United Arab Emirates
# 4. Unites States of America
# 5. Switzerland
# 6. Iraq
# 7. Korea
# 8. Germany
# 9. Indonesia
# 10. Australia
# <br><br><br>
# 
# <b>Top 10 commodities accounting for maximum chunk of import trade are - </b>
# 1. Mineral Fuels.
# 2. Natural or Cultured Pearls.
# 3. Electrical Machinery.
# 4. Nuclear Reactor.
# 5. Organic Chemicals.
# 6. Iron and Steel.
# 7. Plastic.
# 8. Animal or Vegetables.
# 9. Opticals.
# 10. Inorganic Chemicals.
