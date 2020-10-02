#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.go_offline()
init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')
p = sns.color_palette(["mediumseagreen", "sandybrown", "royalblue", "orangered", "saddlebrown", "darkorchid"])
sns.set()


# In[ ]:


df = pd.read_csv('../input/world-happiness-report-2019.csv')


# In[ ]:


df.columns = ['country', 'ladder', 'ladder_sd', 'positive_affect', 'negative_affect', 'social_support', 'freedom', 'corruption', 'generosity', 'gdp_per_capita', 'healthy_life_expectancy']


# In[ ]:


df.sample(5)


# In[ ]:


df.info()


# In[ ]:


pd.isnull(df).sum()


# ## Add continent column

# In[ ]:


asia = ["Israel", "United Arab Emirates", "Singapore", "Thailand", "Taiwan Province of China",
                 "Qatar", "Saudi Arabia", "Kuwait", "Bahrain", "Malaysia", "Uzbekistan", "Japan",
                 "South Korea", "Turkmenistan", "Kazakhstan", "Turkey", "Hong Kong S.A.R., China", "Philippines",
                 "Jordan", "China", "Pakistan", "Indonesia", "Azerbaijan", "Lebanon", "Vietnam",
                 "Tajikistan", "Bhutan", "Kyrgyzstan", "Nepal", "Mongolia", "Palestinian Territories",
                 "Iran", "Bangladesh", "Myanmar", "Iraq", "Sri Lanka", "Armenia", "India", "Georgia",
                 "Cambodia", "Afghanistan", "Yemen", "Syria"]
europe = ["Norway", "Denmark", "Iceland", "Switzerland", "Finland",
                 "Netherlands", "Sweden", "Austria", "Ireland", "Germany",
                 "Belgium", "Luxembourg", "United Kingdom", "Czech Republic",
                 "Malta", "France", "Spain", "Slovakia", "Poland", "Italy",
                 "Russia", "Lithuania", "Latvia", "Moldova", "Romania",
                 "Slovenia", "North Cyprus", "Cyprus", "Estonia", "Belarus",
                 "Serbia", "Hungary", "Croatia", "Kosovo", "Montenegro",
                 "Greece", "Portugal", "Bosnia and Herzegovina", "Macedonia",
                 "Bulgaria", "Albania", "Ukraine"]
north_america = ["Canada", "Costa Rica", "United States", "Mexico",  
                 "Panama","Trinidad and Tobago", "El Salvador", "Belize", "Guatemala",
                 "Jamaica", "Nicaragua", "Dominican Republic", "Honduras",
                 "Haiti"]
south_america = ["Chile", "Brazil", "Argentina", "Uruguay",
                 "Colombia", "Ecuador", "Bolivia", "Peru",
                 "Paraguay", "Venezuela"]
australia = ["New Zealand", "Australia"]
d_asia = dict.fromkeys(asia, 'Asia')
d_europe = dict.fromkeys(europe, 'Europe')
d_north_america = dict.fromkeys(north_america, 'North America')
d_south_america = dict.fromkeys(south_america, 'South America')
d_australia = dict.fromkeys(australia, 'Australia')
continent_dict = {**d_asia, **d_europe, **d_north_america, **d_south_america, **d_australia}
df["continent"] = df["country"].map(continent_dict)
df.continent.fillna("Africa", inplace=True)


# In[ ]:


df.loc[df['country'] == 'Uzbekistan', 'freedom'] = np.nan


# In[ ]:


df[df.isnull().any(axis=1)]


#  ## Life satisfaction across the globe
# > - Important: All data in this dataset are rankings. In this case, the lower the value (deeper the color), the higher the life satisfaction

# In[ ]:


data = dict(type = 'choropleth', 
           locations = df['country'],
           locationmode = 'country names',
           z = df['ladder'], 
           text = df['country'],
           colorbar = {'title':'Ladder'},
           colorscale = "Blues")
layout = dict(title = 'Life satisfaction ladder 2019', 
             geo = dict(showframe = False, 
                       projection = {'type': 'mercator'}))
choromap = go.Figure(data = [data], layout=layout)
iplot(choromap)


# ## Overall Correlation Matrix

# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="RdYlBu")


# ## Overall boxplot

# In[ ]:


sns.boxplot(x="ladder", y="continent", data=df, palette=p)


# ## Freedom ranking per continent

# In[ ]:


sns.FacetGrid(df, hue="continent", height=7, palette=p).map(sns.kdeplot, "freedom")
plt.legend()


# In[ ]:


north_america = df[df.continent=="North America"]
europe = df[df.continent=="Europe"]
asia = df[df.continent=="Asia"]


# ## Ladder distribution in North America

# In[ ]:


sns.distplot(north_america.ladder, color='royalblue')


# ## Correlation between freedom and happiness in North America

# In[ ]:


sns.jointplot("freedom", "ladder", data=north_america, kind='reg', color='royalblue')


# ## Correlation between generosity and happiness in North America

# In[ ]:


sns.jointplot("generosity", "ladder", data=north_america, kind='reg', color='royalblue')


# In[ ]:


np.corrcoef(x=north_america.freedom, y=north_america.ladder)


# ## How is healthy life expectancy and life satisfaction correlated in North America

# In[ ]:


sns.distplot(north_america.healthy_life_expectancy, color='royalblue')


# In[ ]:


north_america.iplot(kind='scatter', mode='markers', x='healthy_life_expectancy', y='ladder', text='country', color='royalblue', xTitle='Healthy Life Expectancy', yTitle='Ladder')


# ## How gdp per capita plays a role in all of these factors in North America

# In[ ]:


sns.pairplot(north_america[['gdp_per_capita', 'freedom', 'healthy_life_expectancy', 'ladder', 'continent']], hue='continent', palette="Blues")


# ## How corruption is correlated to life satisfaction in Europe

# In[ ]:


sns.distplot(europe.corruption, color='mediumseagreen')


# > - We can see there is little middle ground in Europe

# In[ ]:


sns.jointplot("corruption", "ladder", data=europe, kind='reg', color='mediumseagreen')


# ## Correlation between generosity and life satisfaction in Europe

# In[ ]:


sns.jointplot("generosity", "ladder", data=europe, kind='reg', color='mediumseagreen')


# ## Correlation between gpd per capita and life satisfaction in Europe

# In[ ]:


sns.jointplot("gdp_per_capita", "ladder", data=europe, kind='reg', color='mediumseagreen')


# ## Correlation between freedom and life satisfaction in Asia

# In[ ]:


asia.iplot(kind='scatter', mode='markers', x='freedom', y='ladder', text='country', color='orangered', xTitle='Freedom', yTitle='Ladder')


# ## Correlation between social support and life satisfaction

# In[ ]:


sns.jointplot("social_support", "ladder", data=asia, kind='reg', color='orangered')


# ## Correlation between healthy life expectancy and life satisfaction

# In[ ]:


sns.jointplot("healthy_life_expectancy", "ladder", data=asia, kind='reg', color='orangered')


# ## How gdp per capita plays an important role in Asia

# In[ ]:


sns.pairplot(north_america[['gdp_per_capita', 'social_support', 'healthy_life_expectancy', 'ladder', 'continent']], hue="continent", palette="Reds")


# In[ ]:


np.corrcoef(x=asia.freedom, y=asia.ladder)


# ## Summary: Correlation Matrix of three continents

# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(north_america.corr(), cmap="Blues", annot=True)


# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(europe.corr(), cmap="YlGn", annot=True)


# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(asia.corr(), cmap="Reds", annot=True)

