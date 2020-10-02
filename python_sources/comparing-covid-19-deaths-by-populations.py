#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("/kaggle/input/corona-virus-report/covid_19_clean_complete.csv")
df.head()


# In[ ]:


df['Country/Region'] = df['Country/Region'].replace({'US': 'United States',
                                                     'Taiwan*':'Taiwan'})


# In[ ]:


get_ipython().system('pip install countryinfo')


# Add Population to DataFrame and remove countries we couldn't find a number for

# In[ ]:


from countryinfo import CountryInfo

Country_Populations = {}
for i in sorted(list(df['Country/Region'].unique())):
    try:
        Country_Populations[i] = CountryInfo(i).population()
    except (KeyError):
        Country_Populations[i] = np.nan
        
df['Populations'] = df['Country/Region'].map(Country_Populations)
df = df[~df['Populations'].isna()] # remove countries we couldn't get the population for


# Most charts I've come across comparing countries start when each country hits ten cases. This number seems reasonable so we'll filter out all deaths below 10 while the df has deaths as cumulative,

# In[ ]:


df = df[df['Deaths'] >= 10]


# Remove Province/Sates with Nan values. This leaves only the rows which list the deaths for the entire country.

# In[ ]:


df = df[df['Province/State'].isna()]
df['Date'] = df['Date'].astype('datetime64[ns]')


# In[ ]:


top_deaths = list((df[df['Populations'] >= 1000000].groupby(['Country/Region']).Deaths.max() /df[df['Populations'] >= 1000000].groupby(['Country/Region']).Populations.max()).sort_values().index[-10:])


# Undo cumulative figure for deaths to get daily death rate.

# In[ ]:


df.sort_values(['Country/Region','Date'], inplace=True)
df['Daily_Deaths'] = df['Deaths'].diff()

mask = df['Country/Region'] != df['Country/Region'].shift(1)
df['Daily_Deaths'][mask] = np.nan
df.head()


# Change the number of deaths per day to a rolling 7 day average. This will help with smoothing out the graph.

# In[ ]:


df['Deaths_7_Days_Rolling'] = df.groupby(['Country/Region'])['Daily_Deaths'].rolling(7).mean().values


# Calculate the fraction of rolling deaths by the population.

# In[ ]:


df['%_Deaths_per_Pop'] = df['Deaths_7_Days_Rolling']/df['Populations']


# Rank the days since 10 deaths for each country so we can plot them on the same x-axis

# In[ ]:


df['Rank'] = df.groupby('Country/Region').Date.rank()
df.head()


# In[ ]:





# We just want to plot the top ten countries that have the highest number of deaths per population, and only include countries with over 1 million inhabitants.

# In[ ]:


filtered_df = df[(df["Country/Region"].isin(top_deaths)) & (df['Populations'] >= 1000000)]


# In[ ]:


filtered_df = filtered_df.sort_values(by='Date')
filtered_df['Days Since 10 Deaths'] =filtered_df['Rank']


# In[ ]:


sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(filtered_df, row="Country/Region", hue="Country/Region", aspect=11, height=1, palette = pal)

# Draw the densities in a few steps
g.map(sns.lineplot, 'Days Since 10 Deaths', "%_Deaths_per_Pop", clip_on=False, color="w", lw=2,estimator=None)
g.map(plt.axhline, y=0, lw=2, clip_on=False)
g.map(plt.fill_between, 'Days Since 10 Deaths','%_Deaths_per_Pop')

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)
    
g.map(label, 'Days Since 10 Deaths')

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-0.4)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)


# In[ ]:




