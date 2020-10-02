#!/usr/bin/env python
# coding: utf-8

# # Finally Python has its own One Liner graph creation library
# 
# I Know, I know. Pandas has the iplot function already provided by Plotly. 
# 
# Then we have seaborn also. And both of them are pretty good.
# 
# I distinctly remember the time when Seaborn came. I was really so fed up with Matplotlib. To create even simple graphs I had to run through so many StackOverflow threads. The time I could have spent in thinking good ideas for presenting my data, was being spent in handling Matplotlib. And it was frustrating. 
# 
# Seaborn is much better than Matplotlib, yet it also demands a lot of code for a simple **"good looking"** graph. 
# 
# When Plotly came it tried to solve that problem. And when conjuncted with Pandas, plotly is a great tool. 
# 
# Just using the `iplot` function you can do so much with Plotly. 
# 
# But still it is not very intuitive. At least for me. 
# 
# I still didn't switched to Plotly just because I had spent enough time with Seaborn to do things "quickly" enough and I didn't want to spend any more time learning a new visualization library. I had created my own functions in Seaborn to create the visualizations I mostly needed. Yet it was still a workaround. I had given up hope of having anything better. 
# 
# Comes ***Plotly Express***  in picture. And is it awesome. 
# 
# According to the crestors of Plotly Express (who also created Plotly obviously), Plotly Express is to Plotly what Seaborn is to Matplotlib. 
# > a terse, consistent, high-level wrapper around Plotly.py for rapid data exploration and figure generation.
# 
# I just had to try it out. 
# 
# And have the crestors made it easy to start experimenting with it? I am doing a disservice to my blog, but you can actually look up at the notebook provided by the authors to check out pretty much the power of this damn library. 
# 
# One liners to do everything you want? Check.
# 
# Standardized functions? Learn to create a scatterplot and you have pretty much learned this tool. Check.

# # An Interesting Dataset albeit a little depressing
# 
# In this post I will be using the suicide dataset from 1985-2015
# 

# In[ ]:


import pandas as pd
import numpy as np
import plotly_express as px

# Suicide Data
suicides = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")
del suicides['HDI for year']
del suicides['country-year']

# Country ISO Codes
iso_country_map = pd.read_csv("../input/countries-iso-codes/wikipedia-iso-country-codes.csv")
iso_country_map = iso_country_map.rename(columns = {'English short name lower case':"country"})

# Country Continents
concap =pd.read_csv("../input/country-to-continent/countryContinent.csv", encoding='iso-8859-1')[['code_3', 'continent', 'sub_region']]
concap = concap.rename(columns = {'code_3':"Alpha-3 code"})

correct_names = {'Cabo Verde': 'Cape Verde', 'Macau': 'Macao', 'Republic of Korea': "Korea, Democratic People's Republic of" , 
 'Russian Federation': 'Russia',
 'Saint Vincent and Grenadines':'Saint Vincent and the Grenadines' 
 , 'United States': 'United States Of America'}

def correct_country(x):
    if x in correct_names:
        return correct_names[x]
    else:
        return x

suicides['country'] = suicides['country'].apply(lambda x : correct_country(x))

suicides = pd.merge(suicides,iso_country_map,on='country',how='left')
suicides = pd.merge(suicides,concap,on='Alpha-3 code',how='left')
suicides['gdp'] = suicides['gdp_per_capita ($)']*suicides['population']


# In[ ]:


suicides_gby_Continent = suicides.groupby(['continent','sex','year']).aggregate(np.sum).reset_index()
suicides_gby_Continent['gdp_per_capita ($)'] = suicides_gby_Continent['gdp']/suicides_gby_Continent['population']
suicides_gby_Continent['suicides/100k pop'] = suicides_gby_Continent['suicides_no']*1000/suicides_gby_Continent['population']
# 2016 data is not full
suicides_gby_Continent=suicides_gby_Continent[suicides_gby_Continent['year']!=2016]


# In[ ]:


suicides_gby_Continent.head()


# In[ ]:


suicides_gby_Continent_2007 = suicides_gby_Continent[suicides_gby_Continent['year']==2007]


# In[ ]:


px.scatter(suicides_gby_Continent_2007,x = 'suicides/100k pop', y = 'gdp_per_capita ($)')


# In[ ]:


px.scatter(suicides_gby_Continent_2007,x = 'suicides/100k pop', y = 'gdp_per_capita ($)',color='continent')


# In[ ]:


px.scatter(suicides_gby_Continent_2007,x = 'suicides/100k pop', y = 'gdp_per_capita ($)',color='continent',size ='suicides/100k pop')


# In[ ]:


px.scatter(suicides_gby_Continent_2007,x = 'suicides/100k pop', y = 'gdp_per_capita ($)', size = 'suicides/100k pop', color='continent',symbol='sex')


# In[ ]:


px.scatter(suicides_gby_Continent_2007,x = 'suicides/100k pop', y = 'gdp_per_capita ($)', size = 'suicides/100k pop', color='continent',facet_col='sex')


# In[ ]:


px.scatter(suicides_gby_Continent,x = 'suicides/100k pop', y = 'gdp_per_capita ($)',color='continent',
           size='suicides/100k pop',symbol='sex',animation_frame='year', animation_group='continent',range_x = [0,0.6],
          range_y = [0,70000],text='continent')


# In[ ]:


european_suicide_data = suicides[suicides['continent'] =='Europe']
european_suicide_data_gby = european_suicide_data.groupby(['age','sex','year']).aggregate(np.sum).reset_index()
european_suicide_data_gby['suicides/100k pop'] = european_suicide_data_gby['suicides_no']*1000/european_suicide_data_gby['population']

# A single line to create an animated Bar chart too.
px.bar(european_suicide_data_gby,x='age',y='suicides/100k pop',facet_col='sex',animation_frame='year', 
       animation_group='age', 
       category_orders={'age':['5-14 years', '15-24 years', '25-34 years', '35-54 years', 
       '55-74 years', '75+ years']},range_y=[0,1])


# In[ ]:


suicides_map = suicides.groupby(['year','country','Alpha-3 code']).aggregate(np.sum).reset_index()[['country','Alpha-3 code','suicides_no','population','year']]
suicides_map["suicides/100k pop"]=suicides_map["suicides_no"]*1000/suicides_map["population"]
px.choropleth(suicides_map, locations="Alpha-3 code", color="suicides/100k pop", hover_name="country", animation_frame="year",
             color_continuous_scale=px.colors.sequential.Plasma)

