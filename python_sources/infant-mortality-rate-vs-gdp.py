#!/usr/bin/env python
# coding: utf-8

# # In this analysis we will explore the World Development Indicators Dataset 
# ## 1) Track how GDP and infant mortality relate in USA over years
# ## 2) Find the relation between Infant mortality and GDP of all the countries in 2010
# ## 3) Map out the infant mortality over the world map and find heated areas where Infant mortality is higher

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import pandas as pd 
import folium
get_ipython().run_line_magic('matplotlib', 'inline')


# Load the data set in the notebook

# In[ ]:


Indicators = pd.read_csv('/kaggle/input/world-development-indicators/Indicators.csv')


# In[ ]:


Indicators.head(5)


# Using masking to filter infant mortality rate and GDP of USA

# In[ ]:


hist_country = 'USA'
mortality_stage = []
mask3 = Indicators['IndicatorCode'].str.contains('SP.DYN.IMRT.IN') 
mask4 = Indicators['CountryCode'].str.contains(hist_country)

mortality_stage = Indicators[mask3 & mask4]


# In[ ]:


hist_indicator1 = 'GDP per capita \(constant 2005'
hist_country1 = 'USA'

mask1 = Indicators['IndicatorName'].str.contains(hist_indicator1) 
mask2 = Indicators['CountryCode'].str.contains(hist_country1)

gdp_stage1 = Indicators[mask1 & mask2]


# In[ ]:


mortality_stage.head(5)


# In[ ]:


gdp_stage1.head(5)


# Checking to see number of years of data available to find correlation between GDP and infant mortality rate

# In[ ]:


print("GDP Min Year = ", gdp_stage1['Year'].min(), "max: ", gdp_stage1['Year'].max())
print("Mortality min Year = ", mortality_stage['Year'].min(), "max: ", mortality_stage['Year'].max())


# Trucating Mortality rate years to match available GDP data

# In[ ]:


mortality_stage_trunc = mortality_stage[mortality_stage['Year'] < 2015]


# In[ ]:


print("Mortality min Year = ", mortality_stage_trunc['Year'].min(), "max: ", mortality_stage_trunc['Year'].max())


# In[ ]:


fig, axis = plt.subplots()
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.set_title('Infant mortality rate vs. GDP \(per capita\)',fontsize=10)
axis.set_xlabel(gdp_stage1['IndicatorName'].iloc[0],fontsize=10)
axis.set_ylabel(mortality_stage_trunc['IndicatorName'].iloc[0],fontsize=10)

X = gdp_stage1['Value']
Y = mortality_stage_trunc['Value']

axis.scatter(X, Y)
plt.show()


# In[ ]:


np.corrcoef(gdp_stage1['Value'],mortality_stage_trunc['Value'])


# ### As we can see from the above scatter plot and correlation values that the infant mortality rate is almost inversly proportional to GDP

# # Relation between GDP and infant mortality of all the countries for 2010 year

# In[ ]:


hist_year = 2010
mask5 = Indicators['IndicatorCode'].str.contains('SP.DYN.IMRT.IN') 
mask6 = Indicators['Year'].isin([hist_year])

mortality_stage1 = Indicators[mask5 & mask6]


# In[ ]:


mortality_stage1.head()


# In[ ]:


len(mortality_stage1)


# In[ ]:


hist_indicator2 = 'GDP per capita \(constant 2005'
hist_year = 2010

mask7 = Indicators['IndicatorName'].str.contains(hist_indicator2) 
mask8 = Indicators['Year'].isin([hist_year])
gdp_stage2 = Indicators[mask7 & mask8]


# In[ ]:


gdp_stage2.head(5)


# In[ ]:


len(gdp_stage2)


# In[ ]:


gdp_mortality= mortality_stage1.merge(gdp_stage2, on='CountryCode', how='inner')


# In[ ]:


gdp_mortality.head()


# In[ ]:


len(gdp_mortality)


# In[ ]:


fig, axis = plt.subplots()

axis.yaxis.grid(True)
axis.set_title('Infant mortality vs. GDP \(per capita\)',fontsize=10)
axis.set_xlabel(gdp_mortality['IndicatorName_y'].iloc[0],fontsize=10)
axis.set_ylabel(gdp_mortality['IndicatorName_x'].iloc[0],fontsize=10)

X = gdp_mortality['Value_y']
Y = gdp_mortality['Value_x']

axis.scatter(X, Y)
plt.show()


# ### Here in this plot we can see that countries with higher GDP has low infant mortality. This confirms our earlier analysis of USA data that GDP and Infant mortality are inversly related

# # Now Lets map the infant moratlity rates for all the countries as per latest data available and see the hot areas where there is more infant deaths

# In[ ]:


plot_data = mortality_stage1[['CountryCode','Value']]
plot_data.head()


# In[ ]:


hist_indicator = mortality_stage1.iloc[0]['IndicatorName']


# In[ ]:


country_geo = 'https://raw.githubusercontent.com/python-visualization/folium/588670cf1e9518f159b0eee02f75185301327342/examples/data/world-countries.json'


# In[ ]:


map = folium.Map(location=[100, 0], zoom_start=1.5)


# In[ ]:


folium.Choropleth(geo_data=country_geo, data=plot_data,
             columns=['CountryCode', 'Value'],
             key_on='feature.id',
             fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2,
             legend_name=hist_indicator).add_to(map)


# In[ ]:


map.save('plot_data.html')


# In[ ]:


from IPython.display import HTML
HTML('<iframe src=plot_data.html width=700 height=450></iframe>')


# ### From the map we can see that high Infant mortality rate are concentrated in third world countries in Africa where GDP is Low. And also Infant mortality is low in  even with central asia high GDP this is because of terrorism 

# In[ ]:




