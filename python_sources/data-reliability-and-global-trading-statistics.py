#!/usr/bin/env python
# coding: utf-8

# # Exploring Global Trading Statistics
# I tried to create a global flow-chart for all goods and an Export-Import as well.
# That's when I stumbled upon a curiosity. No country with an initial letter after 'm' seems to export or import oil in any kind.
# 
# Investigating this a little bit further, e. g. the dataset doesn't contain oil trading for the USA in any year at all.
# 
# Therefore the dataset has considerable gaps in it's data. Analyses regarding the trade-value or the overall import and export will be highly skewed.
# Please consider this for your analyses.
# 
# Have a good day.
# 
# ## Import Libraries

# In[1]:


# To store the data
import pandas as pd

# To do linear algebra
import numpy as np

# To create plot
import matplotlib.pyplot as plt

# To create nicer plots
import seaborn as sns

# To create interactve plots
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)


# ## Load Data

# In[2]:


df = pd.read_csv('../input/commodity_trade_statistics_data.csv', low_memory=False)
print('Entries: {}\tFeatures: {}'.format(df.shape[0], df.shape[1]))
df.head(3)


# In[3]:


def plotCount(col, n=40):
    plot_df = df[col].value_counts()[:n]

    data = go.Bar(x = plot_df.index,
                  marker = dict(color = '#551a8b'),
                  y = plot_df)

    layout = go.Layout(title = 'Trades Per {}'.format(col.capitalize()),
                       xaxis = dict(title = col.capitalize()),
                       yaxis = dict(title = 'Count'))

    fig = go.Figure(data=[data], layout=layout)
    iplot(fig)


# # How many trades are there over the years?

# In[4]:


plotCount('year')


# # Are there more Exports or Imports worldwide?

# In[5]:


plotCount('flow')


# # Which country has the most trades overall?

# In[6]:


plotCount('country_or_area')


# # What is the entity for measuring?

# In[7]:


plotCount('quantity_name')


# # Which are the best and worst countries for the Export-Import difference?

# In[8]:


country_df = df.groupby(['country_or_area', 'year', 'flow'])['trade_usd'].sum()

im_export_df = country_df.loc[:, :, 'Export'].rename('Export').to_frame().join(country_df.loc[:, :, 'Import'].rename('Import'))
diff = (im_export_df['Export'] - im_export_df['Import']).sort_values(ascending=False).rename('Bilanz').reset_index()

n = 3
flop = diff[diff['year']==2016].sort_values('Bilanz')['country_or_area'].values[:n].tolist()
top = diff[diff['year']==2016].sort_values('Bilanz', ascending=False)['country_or_area'].values[:n].tolist()


data = []

for country in top+flop:
    plot_df = diff[diff['country_or_area']==country].sort_values('year')
    data.append(go.Scatter(x = plot_df['year'],
                           y = plot_df['Bilanz'],
                           name = country))

layout = go.Layout(title = 'Best And Worst {} Countries: Import-Export Difference'.format(n),
                   xaxis = dict(title = 'Year'),
                   yaxis = dict(title = 'Export-Import Difference in USD'))

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# # How big is the Export-Import difference for the countries?

# In[9]:


year = 2016
plot_df = diff[diff['year']==year]
maximum = plot_df['Bilanz'].max()
minimum = plot_df['Bilanz'].min()

scl = [[0.0, '#ff0000'], [abs(minimum)/(abs(minimum)+maximum), '#ffffff'],[1.0, '#00ff00']]

# Data for the map
data = [dict(type='choropleth',
             colorscale = scl,
             autocolorscale = False,
             locations = plot_df['country_or_area'],
             z = plot_df['Bilanz'],
             locationmode = 'country names',
             marker = dict(line = dict (color = '#000000',
                                        width = 1)),
             colorbar = dict(title = 'Differenz in USD'))]

# Layout for the map
layout = dict(title = 'Export-Import Difference {}'.format(year))
    
fig = dict(data=data, layout=layout)
iplot(fig)


# # Which is the most exported commodity of each country?

# In[10]:


products = df[(df['flow']=='Export') & (df['year']==2016)].groupby(['country_or_area', 'commodity'])['trade_usd'].sum().reset_index()
plot_df = products[~products['commodity'].isin(['ALL COMMODITIES', 'Commodities not specified according to kind'])].sort_values(['country_or_area', 'trade_usd'], ascending=False).groupby('country_or_area').first().reset_index()

scl = [[0.0, '#ff0000'],[1.0, '#00ff00']]

# Data for the map
data = [dict(type='choropleth',
             colorscale = scl,
             autocolorscale = False,
             locations = plot_df['country_or_area'],
             z = plot_df['trade_usd'],
             text = plot_df['commodity'],
             locationmode = 'country names',
             marker = dict(line = dict (color = '#000000',
                                        width = 1)),
             colorbar = dict(title = 'Export in USD'))]

# Layout for the map
layout = dict(title = 'Export of the most valuable commodity {}'.format(year))
    
fig = dict(data=data, layout=layout)
iplot(fig)


# # No country with an inital letter after 'm' is exporting oil?
# # The dataset sees unreliable and won't be investigated further.

# In[22]:


commodity = 'Oils petroleum, bituminous, distillates, except crude'
year = 2016

plot_df = df[(df['commodity']==commodity) & (df['year']==year)][['country_or_area', 'flow', 'trade_usd']].replace({'flow':{'Re-Export':'Export', 'Re-Import':'Import'}}).groupby(['country_or_area', 'flow']).sum().reset_index()
plot_df = plot_df[plot_df['country_or_area'].isin(plot_df['country_or_area'].value_counts()[plot_df['country_or_area'].value_counts()>1].index)]
n = int(np.ceil(np.sqrt(plot_df.shape[0]/2)))

fig, axarr = plt.subplots(n, n, figsize=(16, 16))

s = plot_df['trade_usd'].max() / 1000
for i, country in enumerate(plot_df['country_or_area'].unique()): 
    exp, imp = plot_df[plot_df['country_or_area']==country].sort_values('flow')['trade_usd'].values
    
    axarr[i//n][i%n].scatter(x=[0, 1], y=[0, 0], s=[imp/s, exp/s])
    axarr[i//n][i%n].set_title(country)
    
    plt.sca(axarr[i//n][i%n])
    plt.xticks([0, 1], ['Import', 'Export'])
    
    axarr[i//n][i%n].set_yticks([], [])
plt.tight_layout()
plt.show()


# # The USA aren't importing or exporting oil in any year! This implies a highly skewed exploration.

# In[23]:


df[(df['country_or_area']=='USA') & (df['commodity']=='Oils petroleum, bituminous, distillates, except crude')]


# In[ ]:




