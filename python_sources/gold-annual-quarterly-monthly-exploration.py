#!/usr/bin/env python
# coding: utf-8

# # Exploring Gold Reserves by Different Time Periods
# 
# A quick overview of the IMF gold reserves dataset, which is available for all countries since 1950, on a monthly, quarterly, and annual basis.  
# Some regions are also included like "world" and "Europe", and several others exist.  
# 
# I have an interactive [dashboard for analyzing gold reserves and trends](https://www.dashboardom.com/gold-reserves) based on this dataset if you are interested.  
# 

# In[ ]:


import pandas as pd
import plotly
import plotly.graph_objects as go


# In[ ]:


gold = pd.read_csv('../input/gold-reserves-by-country-quarterly/gold_reserves_annual_quarterly_monthly.csv')
gold


# In[ ]:


russia_monthly = gold[(gold['Country Name']=='Russian Federation') & (gold['period'] == 'month')]
russia_monthly


# In[ ]:


turkey_quarterly = gold[(gold['Country Name']=='Turkey') & (gold['period'] == 'quarter')]
turkey_quarterly


# ### Get the exact name of a country
# Many countries aren't listed with the typical name they are known for like "China", which is listed as "China, P.R.: Mainland". A simple way of find this out is to run a quick search for part of that string and get the available option(s): 

# In[ ]:


gold['Country Name'][gold['Country Name'].str.contains('china', case=False)].unique()


# In[ ]:


china_annual = gold[(gold['Country Name']=='China, P.R.: Mainland') & (gold['period'] == 'year')]
china_annual.head()


# ### Compare the reserves in Q2 of 2019 vs Q2 of 2009 `q209_q219`

# In[ ]:


q209_q219 = gold[gold['Time Period'].isin(['2009Q2', '2019Q2'])]
q209_q219


# In[ ]:


print('Top 10 Countries that changed gold reserves 2019Q2 vs. 2009Q1')
pivoted = (q209_q219
           .pivot_table(index='Country Name', columns='Time Period', values='tonnes')
           .assign(diff=lambda df: df['2019Q2']-df['2009Q2'])
           .dropna()
           .sort_values('diff', ascending=False))
q2_top10 = pivoted.head(10).append(pivoted.tail(10))

q2_top10.style.format("{:,.0f}")


# In[ ]:


fig = go.Figure()
fig.add_bar(y=q2_top10.index[::-1], x=q2_top10['diff'][::-1], 
            orientation='h', marker={'color': (['red'] * 10) + (['green']*10)})
fig.layout.height = 600
fig.layout.xaxis.title = 'Tonnes'
fig.layout.title = 'Top Gainers and Losers of Gold Reserves 2019-Q2 vs 2009-Q2'

fig


# ## Make it a function to visualize the top `n` gainers and losers in any two selected quarters

# In[ ]:


def top_gainers_losers(df=gold, first_quarter='2009Q2', second_quarter='2019Q2', top_n=10):
    df = df[df['Time Period'].isin([first_quarter, second_quarter])]
    pivoted = (df
               .pivot_table(index='Country Name', columns='Time Period', values='tonnes')
               .assign(diff=lambda df: df.iloc[:, 1] - df.iloc[:, 0])
               .dropna()
               .sort_values('diff', ascending=False))
    top10 = pivoted.head(top_n).append(pivoted.tail(top_n))
    fig = go.Figure()
    fig.add_bar(y=top10.index[::-1], x=top10['diff'][::-1], 
                orientation='h', marker={'color': (['red'] * 10) + (['green']*10)})
    fig.layout.height = 600
    fig.layout.xaxis.title = 'Tonnes'
    fig.layout.title = 'Top Gainers and Losers of Gold Reserves ' + second_quarter + ' vs. ' + first_quarter

    return fig


# Let's try how the function works with "2005Q2" and "2007Q1"

# In[ ]:


top_gainers_losers(df=gold, first_quarter='2005Q2', second_quarter='2007Q1')


# It seems it works fine, but many "countries" are regions and groups of countries. Let's remove those.  
# We first create a list of those regions, and remove them from our `df` in the function. 

# In[ ]:


to_remove = ['Europe', 'CIS', 'Middle East, North Africa, Afghanistan, and Pakistan', 'Sub-Saharan Africa',
             'Emerging and Developing Asia', 'Euro Area', 'Advanced Economies', 'World']


# In[ ]:


top_gainers_losers(df=gold[~gold['Country Name'].isin(to_remove)], first_quarter='2005Q2', second_quarter='2007Q1')


# In[ ]:


top_gainers_losers(df=gold[~gold['Country Name'].isin(to_remove)], first_quarter='2000Q1', second_quarter='2010Q1')


# In[ ]:


top_gainers_losers(df=gold[~gold['Country Name'].isin(to_remove)], first_quarter='1990Q1', second_quarter='2000Q1')

