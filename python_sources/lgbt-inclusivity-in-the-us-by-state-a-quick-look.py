#!/usr/bin/env python
# coding: utf-8

# As a part of the LGBT+ community, I am predictably interested in finding out which states are more supportive of the LGBT+ community compared to the others. Here I try to look state based demographic data and the policy tally helpfully collected by the Movement Advancement Project to find the predicting factors that contribute to LGBT+ inclusivity in the United States.
# 
# # Data:
# 
# I have used four datasets:
# 1. ACS Demographic Data from [here](https://www.kaggle.com/muonneutrino/us-census-demographic-data/kernels?sortBy=hotness&group=everyone&pageSize=20&datasetId=7001&kernelType=all&language=Python)
# 2. Movement Advancement Project Policy Tally Data from [their wbsite](http://www.lgbtmap.org/)
# 3. State Ideology data collected by [Richard C. Fording](https://rcfording.wordpress.com/state-ideology-data/)
# 4. U.S. Religious Demographic Data collected by [ARDA](http://www.thearda.com/Archive/Files/Descriptions/RCMSCY10.asp)

# ## Importing Modules and Data

# In[1]:


import numpy as np 
import pandas as pd 
import os
#print(os.listdir("../input"))

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode, plot, iplot
#init_notebook_mode(connected=True)

sns.set()


# In[2]:


df_2017 = pd.read_csv("../input/us-census-demographic-data/acs2015_census_tract_data.csv")
df_2015 = pd.read_csv("../input/us-census-demographic-data/acs2015_census_tract_data.csv")
lgbt = pd.read_csv('../input/map-lgbt-policy-tally/MAP.csv')
ideo = pd.read_csv('../input/state-ideology/stateideology_v2018.csv',header=None)
arda = pd.read_csv('../input/arda-state/ARDA_State.csv')


# # Initial Exploration and Cleaning
# 
# ### Movement Advancement Project Data
# 
# This one is pretty clean, since I had to plug and chug data from their website into a csv file myself.

# In[3]:


lgbt.info()


# Here SO_Tally is the pertains to sexual orientation policies, and GI_Tally pertains to gender identity policies. Tot_Tally is the sum of the two.

# In[4]:


lgbt.head()


# In[5]:


lgbt.describe()


# Clearly, there is a huge variance in the policy tallies by state. It's clear from looking at the standard deviation, the quantiles, and the range. It can also be seen just by looking at the tallies for Alabama and California in the first few rows of the dataset. On an average, states are more accepting (in terms of their policy) when it comes to sexual orientation than gender identity. 

# ### State Ideology Data
# The state ideology data does not come with column names. I've looked at the data description on the website to learn what the columns are, and I'll rename the columns accordingly.

# In[6]:


id_col = ['State', 'State_id','Year','Citi_ideo','Govt_ideo']

ideo.columns = id_col


# In[7]:


ideo.isna().sum()


# In[8]:


print(ideo.loc[ideo.State.isna()])


ideo.loc[ideo.Citi_ideo.isna()]


# According to the exploration, the missing data for State and Year can easily be filled in witht he forward fill method. I'll use forward fill for the others as well, because it seems reasonable that not too much will change in one year. But we must ensure that the data is sorted by state and year, otherwise forward fill will yield disastrous results.

# In[9]:


ideo = ideo.sort_values(['State_id','Year']).fillna(method='ffill')


# Clearly, to predict support for the LGBT community in 2019, ideology data from the 1970s isn't particularly relevant. So I will create a dataframe with ideology data from the recent years.

# In[10]:


ideo_recent = ideo.loc[ideo['Year']>=2015].groupby('State',as_index=False)['State_id','Citi_ideo','Govt_ideo'].mean()

ideo_recent.head()


# ### ARDA (Religious Demographic) Data
# 
# Let's first select relevant columns.

# In[11]:


arda_col = ['STNAME']
for col in arda.columns:
    if 'RATE' in col:
        arda_col.append(col)
arda_col


# In[12]:


arda_rel = arda[arda_col[:9]]

arda_rel['STID'] = lgbt.loc[lgbt['State']==arda['STNAME'],['State_id']]
arda_rel.fillna(0,inplace=True)
arda_rel.head()


# In[13]:


for col in arda_rel.columns:
    if col in ['STNAME','STID','TOTRATE']:
        continue
    else:
        arda_rel[col] = arda_rel[col]/arda_rel['TOTRATE']

arda_rel.head()


# ### US ACS Census Tract Data

# In[14]:


full = [df_2017,df_2015]


# In[15]:


for df in full:
    df['Other Race'] = 100 - df['White']-df['Black']-df['Hispanic']-df['Asian']-df['Native']-df['Pacific']


# In[16]:


df_2017.columns


# For this analysis, we don't really care about the transportation or employment type. We, however, are interested to see if sex, race or poverty has any correlation at all to political alignment and therefore possibly on LGBT+ inclusivity.

# In[17]:


rel_col = ['State','CensusTract','TotalPop','Men','Women','White', 'Black', 'Hispanic', 'Asian', 'Pacific', 'Native', 'Other Race',
           'Income','IncomePerCap','Poverty','Unemployment']


# In[18]:


df_2017 = df_2017[rel_col]
df_2015 = df_2015[rel_col]


# Now let's define some functions and aggregate the data by state. By simply taking the mean, we would be making a mistake since the total population in each census tract is varied. We must therefore take the weighted mean of the rates.

# In[19]:


def by_state(df):
    
    df1 = lgbt.copy()[['State', 'State_id']]
        
    for col in df.columns:
        if col in ['CensusTract','State']:
            continue
        if 'TotalPop' in col:
            df1[col] = df.groupby(['State'],as_index=False)[col].sum()[col]
        else:
            df1[col+'_rate'] = df.groupby(['State'],as_index=False)[col].sum()[col].divide(df1['TotalPop'])
        
    return df1


def clean(df):
    df2 = df.copy()    
    for col in df.columns:
        
        if col in ['CensusTract','State','TotalPop','Men','Women']:
            continue
        
        df2[col]=df2[col].mul(df2['TotalPop'],fill_value=1)
        
        dfnew = by_state(df2)
    return dfnew


# In[20]:


df17_state=clean(df_2017)

df17_state.head()


# # Visual Exploration
# 
# Now let's make some plots and observe some trends.

# In[21]:


init_notebook_mode(connected=True)

so_scale = [
    [0.0, 'rgb(242,240,247)'],
    [0.2, 'rgb(218,218,235)'],
    [0.4, 'rgb(188,189,220)'],
    [0.6, 'rgb(158,154,200)'],
    [0.8, 'rgb(117,107,177)'],
    [1.0, 'rgb(84,39,143)']
]

gi_scale = [
    [0.0, 'rgb(247,240,242)'],
    [0.2, 'rgb(235,218,218)'],
    [0.4, 'rgb(220,189,188)'],
    [0.6, 'rgb(200,154,158)'],
    [0.8, 'rgb(177,107,117)'],
    [1.0, 'rgb(143,39,84)']]
    
so_data = [dict(type='choropleth',
    colorscale = so_scale,
    autocolorscale = False,
    locationmode = 'USA-states',
    locations = lgbt['State_id'],
    z = lgbt['SO_Tally'],
    colorbar = go.choropleth.ColorBar()
)]

so_layout = dict(
    title = 'Sexual Orientation Tally',
    geo = dict(
        scope = 'usa',
        projection = {'type':'albers usa'},
        showlakes = True,
        lakecolor = 'rgb(255, 255, 255)')
)

fig1 = go.Figure(data = so_data, layout = so_layout)
iplot(fig1)

gi_data = [dict(type='choropleth',
    colorscale = gi_scale,
    autocolorscale = False,
    locationmode = 'USA-states',
    locations = lgbt['State_id'],
    z = lgbt['GI_Tally'],
    colorbar = go.choropleth.ColorBar()
)]

gi_layout = dict(
    title = 'Gender Identity Tally',
    geo = dict(
        scope = 'usa',
        projection = {'type':'albers usa'},
        showlakes = True,
        lakecolor = 'rgb(255, 255, 255)')
)

fig2 = go.Figure(data = gi_data, layout = gi_layout)
iplot(fig2)


# In[22]:


data = [dict(type='choropleth',
    colorscale = 'Viridis',
    autocolorscale = False,
    reversescale=True,
    locationmode = 'USA-states',
    locations = lgbt['State_id'],
    z = lgbt['Tot_Tally'],
    text = lgbt['State_id'].astype(str),
    colorbar = go.choropleth.ColorBar()
)]

layout = dict(
    title = 'Total Policy Tally',
    geo = dict(
        scope = 'usa',
        projection = {'type':'albers usa'},
        showlakes = True,
        lakecolor = 'rgb(255, 255, 255)')
)

fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[23]:


ideo_recent.head()


# In[45]:


_ = ideo_recent.plot(kind='bar',x='State',y=['Citi_ideo','Govt_ideo'],stacked=True,color=['darkblue','skyblue'],figsize = (20,8),
                     legend=None)
_ = plt.xticks(rotation=90)
_ = plt.xlabel('State')
_ = plt.ylabel('Ideology')
_ = plt.legend(('Citizen Idelogy','Govt. Ideology'))


# Let's look at the highest and lowest scoring states in terms of ideology.

# In[48]:


ideo_recent['Tot_ideo'] = ideo_recent['Citi_ideo']+ideo_recent['Govt_ideo'] 

ideo_recent.sort_values(['Tot_ideo'],ascending=False).head(10).plot(kind='bar',x='State',y=['Citi_ideo','Govt_ideo'],
                                                                    figsize=(12,6),stacked=True,color=['darkblue','lightblue'],legend=None)
plt.xticks(rotation=60)
plt.xlabel('State')
plt.ylabel('Ideology')
plt.legend(('Citizen Idelogy','Govt. Ideology'))
plt.title('States With Highest Ideology Score')
ideo_recent.sort_values(['Tot_ideo','Citi_ideo','Govt_ideo'],ascending=False).tail(10).plot(kind='bar',x='State',y=['Citi_ideo','Govt_ideo'],
                                                                                            figsize=(12,6),stacked=True,color=['darkblue','lightblue'],legend=None)
plt.xticks(rotation=60)
plt.xlabel('State')
plt.ylabel('Ideology')
plt.legend(('Citizen Idelogy','Govt. Ideology'))
plt.title('States With Lowest Ideology Score')


# In[29]:


_ = lgbt.plot(kind='bar',x='State_id',y=['SO_Tally','GI_Tally'],stacked=True,color=['navy','lightseagreen'],figsize = (20,8),
                     legend=None)
_ = plt.xticks(rotation=60)
_ = plt.xlabel('State')
_ = plt.ylabel('Policy Tally')
_ = plt.legend(('Sexual Orientation Policy','Gender Identity Policy'),loc ='upper right')


# In[34]:


lgbt.sort_values(['Tot_Tally'],ascending=False).head(10).plot(kind='bar',x='State',y=['SO_Tally','GI_Tally'],
                                                              figsize=(12,6),stacked=True,color=['navy','lightseagreen'],legend=None)
plt.xticks(rotation=60)
plt.xlabel('State')
plt.ylabel('Policy Tally')
plt.legend(('Sexual Orientation Tally','Gender Identity Tally'))
plt.title('States With Highest Policy Tally')
lgbt.sort_values(['Tot_Tally','GI_Tally','SO_Tally'],ascending=True).head(10).plot(kind='bar',x='State',y=['SO_Tally','GI_Tally'],
                                                                                            figsize=(12,6),stacked=True,color=['navy','lightseagreen'],legend=None)
plt.xticks(rotation=60)
plt.xlabel('State')
plt.ylabel('Policy Tally')
plt.legend(('Sexual Orientation Tally','Gender Identity Tally'))
plt.title('States With Lowest Policy Tally')


# # Correlation Heatmap
# 
# I will start by merging the datasets into one so I can plot correlation matrices and so on.

# In[ ]:


full = df17_state.merge(lgbt,on=['State','State_id'],how='inner').merge(arda_rel,left_on=['State','State_id'],right_on=['STNAME','STID'],how='inner').drop(
    ['STNAME','STID','TOTRATE'],axis=1)
full.head()


# I'm going to drop some columns as they will most likely be irrelevant or redundant. For example, income and income per capita are redundant, so I'll only keep income per capita.

# In[ ]:


full.drop(['Other Race_rate','Income_rate','Men_rate','Women_rate'],axis=1,inplace=True)
data = full.merge(ideo_recent.drop('State_id',axis=1),on=['State'])


# In[ ]:


corr = data.corr()
cmap = sns.cubehelix_palette(8,light=1, as_cmap=True)
_ , ax = plt.subplots(figsize =(14, 10))
hm = sns.heatmap(corr, ax= ax, annot= True,linewidths=0.2,cmap=cmap)


# Conclusions:
# 
# 1. Citizen and government ideology directly affects LGBT+ policy tally, which is not sirprising, given that policies are very closely related to government ideology.
# 2. Religious demographics do affect LGBT+ inclusivity. the Evangelist demographic being the most important negative factors.
# 3. There seems to be a significant correlation between hispanic and asian populations with the policy tally. However, that does not necessarily mean those communities are more inclusive. It is much more likely that people from certain racial/ethnic demographics end up living in states which are more inclusive towards the LGBT+ community, as they tend to also be more inclusive towards a diverse racial demographics.
# 4. One can use county level demographics data to make predictions on citizen ideology by county for each state.
