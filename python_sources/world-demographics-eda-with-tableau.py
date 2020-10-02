#!/usr/bin/env python
# coding: utf-8

# # World Demographic Analysis 1960 - 2016
# 
# We will perform some Exploratory Data Analysis (EDA) with Tableau using the three datasets. The three datasets comprise of: (1) Population, (2) Fertility Rate, (3) Life Expectancies for countries around the globe from years 1960 to 2016. The datasets do not have information on the continent, therefore we will create a Continent variable so that we can also look at the changes/trends between the continents in the 30 year periods.

# In[ ]:


# Importing the libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import pycountry
#!pip install pycountry-convert
import pycountry_convert as pc

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Population Dataset

# In[ ]:


# Import Population Dataset
df_population = pd.read_csv('../input/country_population.csv')
df_population.drop(labels=['Indicator Name','Indicator Code'],axis=1, inplace=True)


# In[ ]:


df_population.sample(n=5)


# The dataset is wide, where each year has it's own column. To make it easier to deal with the data, we need to create a Year column so that the dataset goes from wide to long.

# In[ ]:


df_population = df_population.melt(id_vars=['Country Name','Country Code'],value_vars=df_population.columns[2:],var_name='Year',value_name='Population')


# In[ ]:


df_population = df_population.sort_values(by=['Country Name','Year'],axis=0)


# In[ ]:


df_population['Year'] = pd.to_numeric(df_population['Year'])


# ## Fertility Rate Dataset

# In[ ]:


df_fertility = pd.read_csv('../input/fertility_rate.csv')
df_fertility.drop(['Indicator Name','Indicator Code'],axis=1,inplace=True)


# In[ ]:


df_fertility.sample(n=5)


# In[ ]:


# Changing the dataframe from wide to long
df_fertility = df_fertility.melt(id_vars=['Country Name','Country Code'],value_vars=df_fertility.columns[2:],var_name='Year',value_name='Fertility')


# In[ ]:


df_fertility.sort_values(['Country Name','Year'],inplace=True)


# In[ ]:


df_fertility['Year'] = pd.to_numeric(df_fertility['Year'])


# ## Life Expectancy Dataset

# In[ ]:


df_life_expectancy = pd.read_csv('../input/life_expectancy.csv')
df_life_expectancy.drop(['Indicator Name','Indicator Code'],axis=1,inplace=True)


# In[ ]:


df_life_expectancy.sample(n=5)


# In[ ]:


df_life_expectancy = df_life_expectancy.melt(id_vars=['Country Name','Country Code'], value_vars=df_life_expectancy.columns[2:],var_name='Year', value_name='Life Expectancy')


# In[ ]:


df_life_expectancy.sort_values(by = ['Country Name','Year'],inplace=True)


# In[ ]:


df_life_expectancy['Year'] = pd.to_numeric(df_life_expectancy['Year'])


# ## Merging the Population, Fertility Rate, and Life Expectancy dataframes

# In[ ]:


df = df_population.merge(df_life_expectancy,on=['Country Name','Country Code','Year']).merge(df_fertility,on=['Country Name','Country Code','Year'])


# In[ ]:


df.sample(n=5)


# In[ ]:


sns.heatmap(df.isnull(),cbar=False)


# Some missing values, however not too many.

# ## Adding in the Continents variable

# In[ ]:


# This is our function for using the country code to identify the country 
# and then assigning a continent code to a new column

def match_to_continent(country_code):
        
    try:    
        alpha2_code = pc.country_alpha3_to_country_alpha2(country_code)
        continent_code = pc.country_alpha2_to_continent_code(alpha2_code)
        return continent_code
    
    except:
        return None


# In[ ]:


df['Continent Code'] = df['Country Code'].apply(lambda x: match_to_continent(x))


# In[ ]:


df.sample(n=5)


# In[ ]:


# Check for any observations that do not have a Continent Code

df[df['Continent Code'].isnull()].iloc[:,0].unique()


# In[ ]:


# Remove the observations that do not have a Continent Code

without_continents = list(df[df['Continent Code'].isnull()].iloc[:,0].unique())
df = df[~df['Country Name'].isin(without_continents)]


# In[ ]:


# Mapping the Continent Codes to the Continent

continents = {
    'NA': 'North America',
    'SA': 'South America', 
    'AS': 'Asia',
    'OC': 'Oceania',
    'AF': 'Africa',
    'EU': 'Europe',
    'AN': 'Antarctica'
}

df['Continent'] = df['Continent Code'].map(continents)
df.drop(['Continent Code'],axis=1,inplace=True)
#df.to_csv(path + '\\world_data.csv',index=False)


# In[ ]:


df.sample(n=5)


# # EDA with Tableau
# Below is an interactive plot which looks at population, fertility rate and life expectancy for years 1960 to 2016, between countries and continents.

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1563850256650' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Wo&#47;World_Bank_Data_15636119038020&#47;World_DemAnalysis_smallWidth&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='World_Bank_Data_15636119038020&#47;World_DemAnalysis_smallWidth' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Wo&#47;World_Bank_Data_15636119038020&#47;World_DemAnalysis_smallWidth&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1563850256650');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height='2527px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# The above plot has five dimensions: (1) Fertility Rate, (2) Life Expectancy Rate, (3) Population shown by size of the circle, (4) Continent shown by color, (5) Time by scrolling between years 1960 to 2016.
# 
# If you move the scroll from 2016 to 1960, you'll see the relationships between Life Expectancy and Fertility Rate between the continents with respect to time. From the analysis, some of the life expectancies in African and Asian countries between the 1960s to 1990s were around 28 to 50 years of age. In the 1960s to about mid 1970s, Africa and Asia were mostly clustered together. We can see the trend gradually moving towards a lower fertility rate and a higher life expectancy rate over the years.
