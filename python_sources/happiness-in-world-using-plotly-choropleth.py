#!/usr/bin/env python
# coding: utf-8

# # World Happiness on Interactive Maps.

# ## Loading Required Libraries.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go


# In[ ]:


init_notebook_mode(connected=True)


# ## Reading Data and Assessing it.

# In[ ]:


df_2015 = pd.read_csv('../input/world-happiness/2015.csv')
df_2015.head()


# In[ ]:


df_2015.info()


# In[ ]:


df_2016 = pd.read_csv('../input/world-happiness/2016.csv')
df_2016.head()


# In[ ]:


df_2016.info()


# In[ ]:


df_2017 = pd.read_csv('../input/world-happiness/2017.csv')
df_2017.head()


# In[ ]:


df_2017.info()


# In[ ]:


df_2015 = df_2015.replace(['United States', 'Russia'], ['United States of America', 'Russian Federation'])
df_2016 = df_2016.replace(['United States', 'Russia'], ['United States of America', 'Russian Federation'])
df_2017 = df_2017.replace(['United States', 'Russia'], ['United States of America', 'Russian Federation'])


# In[ ]:


df_code = pd.read_csv('../input/country-code/country_code.csv')
df_code.head()


# In[ ]:


df_code = df_code.drop(['Unnamed: 0','code_2digit'], axis=1)
df_code.head()


# In[ ]:


df_code = df_code.rename(index=str, columns={'Country_name':'Country'})


# In[ ]:


df_2015 = df_2015.merge(df_code, how='inner', on='Country')
df_2016 = df_2016.merge(df_code, how='inner', on='Country')
df_2017 = df_2017.merge(df_code, how='inner', on='Country')


# In[ ]:


df_2015.info()


# # Exploratory Data Analysis
# ### For each year we are going to project different factors of Hapiness and Happiness Rank on world map.

# In[ ]:


def plotmap(df, plot_series, color_title, scl, title ):
    data = dict(type = 'choropleth',
               locations = df['code_3digit'],
               z = df[plot_series],
               text = df['Country'],
               colorbar = {'title':color_title},
               colorscale = scl)

    layout = dict(title=title,
                geo=dict(showframe=False, projection={'type':'equirectangular'}))

    fig = go.Figure(data = [data], layout = layout)
    iplot(fig)


# ## 1. Happiness Rank

# In[ ]:


plotmap(df_2015, 'Happiness Rank', 'Happiness Rank 2015', 'RdBu', '2015 Happiness Ranking')


# In[ ]:


plotmap(df_2016, 'Happiness Rank', 'Happiness Rank 2016', 'RdBu', '2016 Happiness Ranking')


# In[ ]:


plotmap(df_2017, 'Happiness.Rank', 'Happiness Rank 2017', 'RdBu', '2017 Happiness Ranking')


# ## 2. GDP per Capita : The extent to which GDP contributes to the calculation of the Happiness Score.

# In[ ]:


plotmap(df_2015, 'Economy (GDP per Capita)', 'GDP per capita', 'Greens', '2015 GDP per Capita')


# In[ ]:


plotmap(df_2016, 'Economy (GDP per Capita)', 'GDP per capita', 'Greens', '2016 GDP per Capita')


# In[ ]:


plotmap(df_2017, 'Economy..GDP.per.Capita.', 'GDP per capita', 'Greens', '2017 GDP per Capita')


# ## 3. Family : The extent to which Family contributes to the calculation of the Happiness Score

# In[ ]:


plotmap(df_2015, 'Family', 'Family score', 'Blues', '2015 Family Score')


# In[ ]:


plotmap(df_2016, 'Family', 'Family score', 'Blues', '2016 Family Score')


# In[ ]:


plotmap(df_2017, 'Family', 'Family score', 'Blues', '2017 Family Score')


# ## 4. Health (Life Expectancy) : The extent to which Health (Life Expectancy) contributes to the calculation of the Happiness Score.

# In[ ]:


plotmap(df_2015, 'Health (Life Expectancy)', 'Life Expectancy score', 'YlGnBu', '2015 Life Expectancy Score')


# In[ ]:


plotmap(df_2016, 'Health (Life Expectancy)', 'Life Expectancy score', 'YlGnBu', '2016 Life Expectancy Score')


# In[ ]:


plotmap(df_2017, 'Health..Life.Expectancy.', 'Life Expectancy score', 'YlGnBu', '2017 Life Expectancy Score')


# ## 5. Freedom : The extent to which Freedom contributes to the calculation of the Happiness Score

# In[ ]:


plotmap(df_2015, 'Freedom', 'Freedom score', 'Greys', '2015 Freedom Score')


# In[ ]:


plotmap(df_2016, 'Freedom', 'Freedom score', 'Greys', '2016 Freedom Score')


# In[ ]:


plotmap(df_2017, 'Freedom', 'Freedom score', 'Greys', '2017 Freedom Score')


# ## 6. Trust (Government Corruption) : The extent to which Trust (Government Corruption) contributes to the calculation of the Happiness Score

# In[ ]:


plotmap(df_2015, 'Trust (Government Corruption)', 'Government Corruption score', 'Hot', '2015 Government Corruption Score')


# In[ ]:


plotmap(df_2016, 'Trust (Government Corruption)', 'Government Corruption score', 'Hot', '2016 Government Corruption Score')


# In[ ]:


plotmap(df_2017, 'Trust..Government.Corruption.', 'Government Corruption score', 'Hot', '2017 Government Corruption Score')


# ## 7. Generosity : The extent to which Generosity contributes to the calculation of the Happiness Score

# In[ ]:


plotmap(df_2015, 'Generosity', 'Generocity score', 'Reds', '2015 Generocity Score')


# In[ ]:


plotmap(df_2016, 'Generosity', 'Generocity score', 'Reds', '2016 Generocity Score')


# In[ ]:


plotmap(df_2017, 'Generosity', 'Generocity score', 'Reds', '2017 Generocity Score')


# ## In here we can clearly see that the European, American and Australian regions have better scores in each of the field and African regions have the worst. But Asians regions are on better end in some cases but on worse end in some i.e. in mixed or mid range.
