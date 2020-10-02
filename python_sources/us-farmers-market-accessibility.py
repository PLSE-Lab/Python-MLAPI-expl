#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns


# ### TASK - How accessible are farmers markets?
# 
# - One might assume farmers markets would be more common in wealthier areas of the US, but is this the case?
# - There are many ways to define accessibility. Here are a few ideas, spanning economic, geographical and the seasonal definitions:
# 
#  #### 1. Farmers Markets per capita
#  #### 2. Average distance to a farmer market in a locality/Density of farmers markets per unit land area
#  #### 3. How frequently are markets available during the year/seasons?
#  #### 4. Are the markets advertised on the web/social media? i.e how do people find out about them
#  #### 5. Are markets clustered in particular geographical subregions of the US?
#  #### 6. Average prices for commodities? (How accessible is their pricing?)
#  
# 
# - Based on the available data, we can work with metrics 1, 3, 4 and 5.

# In[ ]:


markets = pd.read_csv('../input/farmers-markets-in-the-united-states/farmers_markets_from_usda.csv')
counties = pd.read_csv('../input/farmers-markets-in-the-united-states/wiki_county_info.csv')


# In[ ]:


pd.set_option('display.max_columns', None)
markets.head(5)


# In[ ]:


markets.columns


# In[ ]:


counties.head(5)


# ### APROACH
# 
# - The columns "county" and "state" can be used to join the two dataframes and doing a groupby on these and counting the "MarketName" column will give a markets per county/state column
# - The "per capita income" column in counties should be sufficient to answer the more wealth == more farmers markets question
# - You could use the "population" and "number of households" columns, to get a farmers markets per capita/household figure
# - To visualise the geographical relationship, I'm thinking of several maps of the US, with heatmaps to visualise income, population and farmers market density, per county or state
# - To test the hypothesis, "are markets more common in wealthier areas?" implied by the task, I can assess the correlation between wealth indicators and farmers market/per capita data
# 
# #### Additional points
# - The produce columns ("Beans" Y/N etc) could be one hot encoded and used as categorical variables or summed to give a produce diversity metric for a given market, if we wanted to use them
# - The "Season Date" columns could be converted and summed to give an overall "time frame available" metric but this data appears to be incomplete
# - The social media links could be used to get a sense of how well advertised the markets are
# - The x and y columns are latitude and longitude values allowing granular plotting of geographical distributions if desired
# - The 'Credit	WIC	WICcash	SFMNP	SNAP' columns indicate available payment methods, the more payment methods the more accessible the market I suppose?
# - We also have "median household income"	"median family income" columns, which should track with "per capita income" but would allow you to be more granular in a sociological sense (differentiating "families" and "households")

# # State Level - per capita income vs markets per capita

# In[ ]:


# groupby county and groupby state to get markets/county, markets/state
market_per_county = markets.groupby('County')
market_per_state = markets.groupby('State')


# In[ ]:


# count the number of markets per state
market_per_state = market_per_state['MarketName'].count()
market_per_state.sort_values(inplace = True)


# In[ ]:


ax = market_per_state.plot.bar(figsize = (10, 6))
ax.set_ylabel('Market Count')
ax.set_title('Market Count (States)');


# ### Looks like CA and NY have the most farmers markets. This could be because of their relative wealth and/or high populations
# 

# In[ ]:


counties.head()


# In[ ]:


counties.columns


# In[ ]:


counties.dropna(inplace = True)


# In[ ]:


# lets format the numerical wealth metrics so they can be used
wealth_metrics = ['per capita income','median household income', 'median family income']
for i in wealth_metrics:
    counties[i] = counties[i].str.strip('$,')
    counties = counties.replace(',','', regex=True)
    counties[i] = counties[i].astype('int32')


# In[ ]:


counties.head()


# In[ ]:


# we can now groupby state and average over the wealth metrics
counties_state_gb = counties.groupby(['State'])[wealth_metrics].mean().reset_index()
counties_state_gb.sort_values(by = 'per capita income', inplace = True)


# In[ ]:


counties_state_gb.head()


# In[ ]:


# In the interests of brevity lets focus on per capita income for now
ax = counties_state_gb[['State', 'per capita income']].plot.bar(x = 'State', figsize = (10, 6))
ax.set_ylabel('per capita income')
ax.set_title('Per Capita Income (States)');


# In[ ]:


# combine the markets and counties tables and plot on one chart
combined_state_df = pd.merge(market_per_state, counties_state_gb, how = 'left', on = 'State')
combined_state_df.sort_values(by = 'per capita income', inplace = True)


# In[ ]:


combined_state_df.head()


# In[ ]:


# sum the number of househodls per state and use this to calculate the markets per capita for each state
counties['population'] = counties['population'].astype('int32')
counties_sum = counties.groupby('State').sum().reset_index()


# In[ ]:


counties_sum.head()


# In[ ]:


# we don't need the sum of the other columns so isolate the state and population columns
counties_sum = counties_sum[['State', 'population']]


# In[ ]:


combined_state_df.head()


# In[ ]:


counties_sum.head()


# In[ ]:


combined_state_pop_df = pd.merge(combined_state_df, counties_sum, how = 'right', on = 'State')
combined_state_pop_df.head(5)


# In[ ]:


combined_state_pop_df['market_per_capita'] = combined_state_pop_df['MarketName'] / combined_state_pop_df['population']
combined_state_pop_df.head(5)


# In[ ]:


# min max scaling to get the no. of markets and per capita income on a comparable scale for plotting
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x = combined_state_pop_df[['market_per_capita']].values.astype(float)
x_scaled = min_max_scaler.fit_transform(x)
combined_state_pop_df['norm_mpc'] = x_scaled

min_max_scaler = preprocessing.MinMaxScaler()
x = combined_state_pop_df[['per capita income']].values.astype(float)
x_scaled = min_max_scaler.fit_transform(x)
combined_state_pop_df['norm_pci'] = x_scaled


# In[ ]:


combined_state_pop_df.head()


# In[ ]:


# By eye, is there any correlation between markets per capita and income per capita, at the state level? Answer: maybe
f, ax = plt.subplots(figsize=(10, 5))
plt.xticks(rotation=90, fontsize=10)
plt.ylabel('Normalised Value')
plt.bar(height="norm_pci", x="State", data=combined_state_pop_df, label="Per Capita Income", color="lightgreen", alpha = 0.5);
plt.bar(height="norm_mpc", x="State", data=combined_state_pop_df, label="Per Capita Markets", color="black", alpha = 0.5);
plt.title('Per Capita Income vs Per Capita Markets (States)')
plt.legend();


# In[ ]:


# checking for correlation between markets per capita and income per capita at a state level
import scipy.stats as stats
g = sns.jointplot(data = combined_state_pop_df, x = "per capita income", y = "market_per_capita", kind = 'reg')
g.annotate(stats.pearsonr);


# ### From the above charts and stats...
# 
# - There is moderate positive correlation between per capita income and per capita farmers markets at a state level (Pearsons r of 0.43 with a two-tailed p-value of 0.0015)
# - Perhaps if we look at the county level, per capita income will correlate differently with markets per capita?

# # County Level - per capita income vs markets per capita

# In[ ]:


# we've got two county columns in the markets and counties dataframe, lets use sets and an intersection operation to find those county values common to both
unique_counties_markets = set(markets['County'].unique())
unique_counties_counties = set(counties['county'].unique())
shared_counties = unique_counties_markets.intersection(unique_counties_counties)


# In[ ]:


len(counties)


# In[ ]:


len(shared_counties)


# In[ ]:


markets['county'] = markets['County']


# In[ ]:


# use the isn() operator to subset each dataframe based on shared_counties set membership and then merge them
markets_shared = markets.loc[markets['county'].isin(shared_counties)]
counties_shared = counties.loc[counties['county'].isin(shared_counties)]
combined_county = pd.merge(markets_shared, counties_shared, how = 'left', on = 'county')


# In[ ]:


combined_county['county'].value_counts()


# In[ ]:


combined_county.columns


# In[ ]:


# state to state2letter. Thanks to https://gist.github.com/rogerallen/1583593
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}


# In[ ]:


State_2_letter = []
for i in combined_county['State_x']:
    State_2_letter.append(us_state_abbrev.get(i))
combined_county['State_2_letter'] = State_2_letter


# In[ ]:


combined_county['State_2L_county'] = combined_county['State_2_letter'] + ', ' + combined_county['county']


# In[ ]:


combined_county['State_2L_county'].head()


# In[ ]:


combined_county['State_2L_county'].unique()


# In[ ]:


# calculate markets per capita (county)
market_per_county = combined_county.groupby('State_2L_county', as_index=False)
market_per_county = market_per_county['MarketName'].count()
market_per_county = pd.DataFrame(market_per_county)


# In[ ]:


market_per_county.columns


# In[ ]:


market_per_county.rename(columns = {'State_2L_county':'county'}, inplace=True)
combined_county.drop(['county'], axis=1, inplace=True)
combined_county.rename(columns = {'State_2L_county':'county'}, inplace=True)


# In[ ]:


combined_county.shape


# In[ ]:


combined_county.columns


# In[ ]:


# sum the populations over the counties and get a mean of each wealth metric over the counties in separate dataframes
combined_county_population = combined_county.groupby(['county'])['population'].sum().reset_index()
combined_county_pci = combined_county.groupby(['county'])[wealth_metrics].mean().reset_index()


# In[ ]:


combined_county_population.head()


# In[ ]:


combined_county_pci.head()


# In[ ]:


market_per_county.columns


# In[ ]:


#combine the dataframes and calculate the markets per capita
county_markets_pop = pd.merge(market_per_county, combined_county_population, how = 'left', on = 'county')
county_markets_pop_pci = pd.merge(county_markets_pop, combined_county_pci, how = 'left', on = 'county')
county_markets_pop_pci['markets_pc'] = county_markets_pop_pci['MarketName'] / county_markets_pop_pci['population']


# In[ ]:


county_markets_pop_pci


# In[ ]:


# scaling again for our simple barplot comparison of counties sorted by per capita income 
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x = county_markets_pop_pci[['markets_pc']].values.astype(float)
x_scaled = min_max_scaler.fit_transform(x)
county_markets_pop_pci['norm_mpc'] = x_scaled

min_max_scaler = preprocessing.MinMaxScaler()
x = county_markets_pop_pci[['per capita income']].values.astype(float)
x_scaled = min_max_scaler.fit_transform(x)
county_markets_pop_pci['norm_pci'] = x_scaled

county_markets_pop_pci.sort_values(by = 'per capita income', inplace = True)


# In[ ]:


# many counties in puerto rico are topping the list ;)
county_markets_pop_pci.head(10)


# In[ ]:


# Too many counties to show on the x axis conveniently on a non-interactive plot but we can visualise the relationship between per capita income and per capita markets at the county level
f, ax = plt.subplots(figsize=(10, 5))
plt.xticks(rotation=90, fontsize=10)
plt.ylabel('Normalised Value')
plt.xlabel('Counties')
plt.bar(height="norm_pci", x="county", data=county_markets_pop_pci, label="Per Capita Income", color="lightgreen", alpha = 0.5);
plt.bar(height="norm_mpc", x="county", data=county_markets_pop_pci, label="Per Capita Markets", color="black", alpha = 0.5);
plt.title('Per Capita Income vs Per Capita Markets (Counties)')
plt.legend();


# In[ ]:


g = sns.jointplot(data = county_markets_pop_pci, x = "per capita income", y = "markets_pc", kind = 'reg')
g.annotate(stats.pearsonr);


# - The plots and stats above suggests no correlation between per capita income and farmers markets per capita at a county level

# ## Other possibilities
# - Make a correlation matrix. Is there any ( linear or almost linear?) correlation between markets per capita and the other numerical variables (particularly the wealth and sociological metrics) in the dataset at all
# 
# - A map of the US with a heatmap of per capita markets/per capita income, may reveal some geographical correlations
# 
# ****** I'd like to try making some maps so I'm going to do the last option ******
# 

# # Correlation Matrix - numerical variables vs markets per capita
# 
# - NOTE - I'm going to generate correlation matrices below based on Pearson's r, a metric of linear correlation. Some correlations are non-linear and for these you can use Spearman's or Kendall's correlation metrics

# In[ ]:


combined_state_pop_df.head()


# In[ ]:


combined_state_pop_df.rename(columns={'MarketName':'Market_Count'}, inplace=True)


# In[ ]:


# Add some extra columns to the wealth_metrics list to use for specify data for the correlation matrices
numericals = ['Market_Count', 'per capita income',
       'median household income', 'median family income', 'population', 'market_per_capita']


# In[ ]:


from string import ascii_letters

def plot_corr_matrix(df, columns, plot_title):
    sns.set(style="white")
    correlations = df[columns].corr()
    mask = np.triu(np.ones_like(correlations, dtype=np.bool))
    f, ax = plt.subplots(figsize=(10, 6))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    ax = sns.heatmap(correlations, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5, "label": "Pearsons r"})
    ax.set_title(plot_title)
    plt.show()

plot_corr_matrix(combined_state_pop_df, numericals, 'State Level - Numerical Features vs Markets')


# - The correlation matrix above indicates that at a state level, there is moderate linear correlation between population and both the Market_Count and the markets per capita
# - Interestingly, population appears to be a better predictor of markets per capita and total markets per state than the wealth metrics
# - The various wealth metrics are also correlated with market count and markets per capita at a state level as expected

# In[ ]:


county_markets_pop_pci.rename(columns={'markets_pc':'market_per_capita'}, inplace=True)
county_markets_pop_pci.rename(columns={'MarketName':'Market_Count'}, inplace=True)


# In[ ]:


# in the absence of consistent column names....we must specify what things are in the counties df
# numericals_counties = ['Market_Count', 'per capita income',
#        'median household income', 'median family income', 'population', 'markets_pc']


# In[ ]:


# my dataframe names could be clearer...
county_markets_pop_pci.head()


# In[ ]:


plot_corr_matrix(county_markets_pop_pci, numericals, 'County Level - Numerical Features vs Markets')


# - Looking at the county level, it seems population correlates strongly with the count of markets in a county (Market_Count), but much less with the wealth metrics
# - Markets per capita (market_per_capita) on the other hand, shows little correlation with population or wealth metrics

# # Maps of the US with the per capita income and density of farmers markets indicated

# In[ ]:


# plotly chloropleth tutorial here: https://plotly.com/python/mapbox-county-choropleth/
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties_json = json.load(response)  # load the geographical data for US counties


# In[ ]:


# load the FIPS county codes 
get_ipython().system('curl -O https://coastwatch.pfeg.noaa.gov/erddap/convert/fipscounty.csv')


# In[ ]:


fips_county = pd.read_csv('./fipscounty.csv')


# In[ ]:


fips_county.head()


# In[ ]:


fips_county.shape


# In[ ]:


# Get rid of the state only rows with a lambda function
fips_county_names = fips_county[fips_county['Name'].apply(lambda x: len(x) > 2)]


# In[ ]:


fips_county_names.shape


# In[ ]:


fips_county_names.head()


# In[ ]:


county_markets_pop_pci.head()


# In[ ]:


fips_county_names.rename(columns={'Name':'county'}, inplace=True)


# In[ ]:


fips_county_names['county'] = fips_county_names['county'].astype('str')
fips_county_names['FIPS'] = fips_county_names['FIPS'].astype('str');


# In[ ]:


county_markets_pop_pci['county'] = county_markets_pop_pci['county'].astype('str')
county_markets_pop_pci['county'] = county_markets_pop_pci['county'].str.strip()


# In[ ]:


# make a single dataframe with the fips codes and markets per capita
combo_FIPS_markets = pd.merge(fips_county_names, county_markets_pop_pci, how = 'inner', on = 'county')


# In[ ]:


# FIPS are 5 digit codes but the 0 are missing from the first 10000 in the data I've found here. So lets add some zeros
def add_zero(df):
    for i,j in enumerate(df['FIPS']):
        if len(j) == 4:
            df['FIPS'][i] = '0'+ j
        else:
            continue
    return df


# In[ ]:


add_zero(combo_FIPS_markets)


# In[ ]:


#combo_FIPS_markets['FIPS'].loc[combo_FIPS_markets['county'].str.contains('CA,')]


# In[ ]:


counties_copy = counties
State_2_letter = []
for i in counties_copy['State']:
    State_2_letter.append(us_state_abbrev.get(i))
counties_copy['State_2_letter'] = State_2_letter


# In[ ]:


counties_copy['State_2L_county'] = counties_copy['State_2_letter'] + ', ' + counties_copy['county']


# In[ ]:


counties_copy.drop(['county'], axis=1, inplace=True)
counties_copy.rename(columns = {'State_2L_county':'county'}, inplace=True)


# In[ ]:


# Now we have a dataframe with almost complete per capita income data for each US county
county_pc_FIPS = pd.merge(fips_county_names, counties_copy, how = 'inner', on = 'county')


# In[ ]:


add_zero(county_pc_FIPS)


# In[ ]:


import plotly.express as px

fig = px.choropleth_mapbox(county_pc_FIPS, geojson=counties_json, locations='FIPS', color='per capita income',
                           color_continuous_scale="Viridis",
                           range_color=(0, max(county_pc_FIPS['per capita income'])),
                           mapbox_style="carto-positron",
                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                           opacity=0.5,
                           labels={'per capita income':'per capita income'}
                          )

#fig.add_trace(px.scatter_mapbox(markets, lat=markets['x'], lon=markets['y']))

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, title = 'Per capita income in US counties')
# fig.update_layout()
fig.show()


# - Per capita income at a county level is higher around cities, where there are more jobs, more big companies and higher salaries
# - There are a few missing counties in the income data here, I'll have to check if this is due to anything I have done or whether data for these counties are missing

# In[ ]:


import plotly.graph_objects as go
fig = go.Figure(data=go.Scattergeo(
        lon = markets['x'],
        lat = markets['y'],
        mode = 'markers',
        marker_color = 'blue',
        geojson = counties_json
        ))
fig.update_layout(geo_scope='usa', title = 'Farmers Market Locations')
fig.show()


# - Above you can see that farmers markets appear to be clustered in the most populated areas...which makes good business sense ;)

# In[ ]:


fig = px.choropleth_mapbox(combo_FIPS_markets, geojson=counties_json, locations='FIPS', color='market_per_capita',
                           color_continuous_scale="picnic",
                           range_color=(0, max(combo_FIPS_markets['market_per_capita'])),
                           mapbox_style="carto-positron",
                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                           opacity=0.5,
                           labels={'market_per_capita':'markets per capita'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# - Interestingly at the county level, a few obscure locations have the highest farmers markets per capita, like the red county near Billings Montana

# ## These maps would be more useful if:
# - They were consolidated into one figure
# - You could toggle between state and county groupbys, the location of markets, per capita income, farmers markets per capita and population
# - Perhaps there are easier libraries to work with than plotly for this purpose?

# # Conclusions
# 
# - We established that per capita income was moderately linearly correlated (Pearsons r 0.43 p=0.0015) with per capita farmers markets at the state level but not at the county level
# - Population showed more correlation with market counts and markets per capita at a state level, than the various wealth metrics such as per capita income. This makes sense as farmers markets will do more business and therefore be more attactive for people to set up, in areas where there are more people and generally more wealth
# - At a county level, farmers market density is far more strongly correlated with population, than the wealth metrics. Again this make sense, as more markets will pop up where there are more people and therefore more demand for food in general
# - From the maps above we can see that farmers markets are widespread across the counties and states of the US and commonly located near population centres
# - An overlay of population and agricultural production on the some of the maps above, might indicate if farmers markets are more accessible (by the metric of farmers markets per capita) in "rural" areas of the US, where there tends to be more agriculture and fewer people
# - Farmers market per capita is probably not the best metric of accessibility in many ways, as it doesn't really take into account how close people are to the market. A vast, less populated state will have more farmers markets per capita but they may not be easily accessible if the average journey time to a market is prohibitive
# - It would be nice to do some multivariate clustering or correlation to see if there are any other more complex higher dimensional correlations, between the total or per capita farmers markets in each of the US states or counties and combinations of the available features in the data
# 

# # Things to improve/reflect upon
# - The naming of dataframes and groupby objects needs to be clear to avoid confusion
# - Do you need to use sets to find common column values in multiple dataframes or can you do it within Pandas?
# - Pearson r measures linear correlations, Spearmans and Kendall correlation test might tell you something about non-linear relationships
# - The normalised overlayed barplot of "per capita income" vs "markets per capita" is redundant, if you have a pairplot with a linear model, confidence interval and pearsonsr. Though I suppose it gives you a sense of if the linear relationship between per capita income and per capita markets is strong enough to see by eye.
# - Working with maps inevitably requires external data, the veracity/completeness of which you have to be careful with. I think the map figures would benefit from consolidation and the addition of data selection widgets as discussed
# - Using more functions to organise and reuse code blocks for plotting, groupby and variable formatting operations on data frames, would make things quicker to develop and easier to read back over. I will have try to remember to do this from the beginning of my next analysis
# 
# 
