#!/usr/bin/env python
# coding: utf-8

# ## Farmers Markets in the United States
# We are given 2 data sets
# 1. A data set about the farmers markets
# 2. A data set about the counties in the United States
# 
# Through these 2 data sets, we hope to answer the following question:
# 
# > One criticism of farmers markets is that they are largely inaccessible to many Americans, especially those of low socio-economic status. 
# > Does the data reflect this criticism?
# 
# In this notebook, I will be using heatmaps to visualise the data presented in the data sets and attempt to address the question.
# 
# ## Short answer
# Tl;dr: No, the data does not reflect that lower income regions have less access to farmers markets, in fact the income level is quite constant throughout the entire region. Rather, the markets are present in wherever the population is larger, regardless of income level. However, the data set doesn't reflect the cost of items sold at those markets, I feel that such a variable can better answer the question posed above.

# In[ ]:


# install the necessary libraries
get_ipython().system('pip install us')
get_ipython().system('pip install gmaps')


# In[ ]:


# import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # for plotting graphs
import math

import us # for converting state names to numeric geolocation
import plotly.graph_objects as go # to plot heatmaps
import gmaps # to plot heatmaps using Google Maps

# libraries for folium heatmaps
import folium
from folium import plugins

# libraries for displaying image files in a notebook
from IPython.display import display
from PIL import Image


# # Farmers Market Data set
# Let's have a look at the variables present in this data set and the number of missing values.

# In[ ]:


farmers_mkt = pd.read_csv('/kaggle/input/farmers-markets-in-the-united-states/farmers_markets_from_usda.csv')

print(farmers_mkt.shape)
print('-'*40)
farmers_mkt.isna().sum()


# ## Some information about Farmers Markets data set
# * FMID: Farmers Market Identitification Number.
# * Website: Website for the market.
# * Facebook/Twitter/Youtube/Other Media: Handles or websites to the market's social media page.
# * zip: zipcode
# * Season dates: There are many dates here, Season1Dates etc. Refers to the dates that belongs to this Season.
# * Season times: There are many season duration and dates here, Season1Time etc. Refers to the operating times in this Season.
# * x: Longitude
# * y: Latitude
# * Location: Generic location of market, some values include 'Local government building grounds'.
# * Credit/WIC/WICcash/SFMNP/SNAP: Payment mode. Y indicating the payment mode is accepted.
#     * Credit: Credit Card
#     * WIC: WIC Farmers Market Nutrition Program (WIC-FMNP)
#     * WICcash: WIC Cash Value Vouchers
#     * SFMNP: Senior Farmers Markets Nutrition Program (SFMNP)
#     * SNAP: Supplemental Nutrition Assistance Program (SNAP)
# * Organic/Bakedgoods/Cheese/.../WildHarvested: Products sold. Y indicating the market sells this category of food
# * updateTime: Time where the data is updated

# In[ ]:


farmers_mkt.head()


# # Data Wrangling
# 
# Let's simplify the farmers market data set for visualisation purposes.
# 
# ## Dealing with variables with too many missing values
# * Information about seasons is largely missing, I will drop these variables

# In[ ]:


farmers_mkt = farmers_mkt.drop(columns=['Season1Date',
       'Season1Time', 'Season2Date', 'Season2Time', 'Season3Date',
       'Season3Time', 'Season4Date', 'Season4Time'])


# There appears to be 2823 rows in the data set that does not have information about the products sold, let's create a variable that reflects if products are being shown at a market.

# I am going to include a column of 1's in the data set for heatmap visualisation purposes

# In[ ]:


# if products are not shown, we assign a value 1, otherwise assign 0
farmers_mkt['No products'] = farmers_mkt['Bakedgoods'].isna().astype(int)

farmers_mkt.loc[:,['MarketName','No products']].head()


# In[ ]:


farmers_mkt['Is market'] = [1]*farmers_mkt.shape[0]


# ## Social media platforms
# 
# * The social media websites are not very useful, you can possibly build a web-scraper to scrape data off the websites but that's too much effort for me
# * First I will replace the columns of the social media platforms with value 1 if the cell is non-empty, 0 otherwise
# * Then I will create a new variable called 'Media' by summing up the values for ['Website', 'Facebook', 'Twitter', 'Youtube', 'OtherMedia']
# * 'Media' will represent how tech-savvy the market is

# In[ ]:


# function that formats the variables corresponding to each social media platform
# if a social media platform is present, we assign it a value 1, otherwise value 0 is assigned
def format_social_media(df):
    vars = ['Website', 'Facebook', 'Twitter', 'Youtube', 'OtherMedia']
    
    # if a social media site is present, we assign 1, otherwise 0
    for var in vars:
        df[var] = (df[var].notnull()).astype('int')
    
    return df

# apply the function to the data set
farmers_mkt = format_social_media(farmers_mkt)

# add a media score, a sum of the values of the 5 media types
farmers_mkt['Media'] = farmers_mkt.loc[:, 'Website':'OtherMedia'].sum(1)

# drop the columns of each media channel
farmers_mkt = farmers_mkt.drop(columns=['Website', 'Facebook', 'Twitter', 'Youtube',
       'OtherMedia', 'updateTime'])


# ## Grouping of Products sold
# There are too many categories of products in the data set, let's do some feature engineering.
# 
# 1. If a product is available is in a market, I will assign it a value 1 instead of 'Y', similarly I will assign a value of 0 for 'N'.
# 2. Next I will group the products into the following new categories:
#     * Plants: 'Trees', 'Plants', 'Nursery', 'Flowers'
#     * Meat: 'Meat', 'Poultry', 'Seafood'
#     * Dairy: 'Cheese', 'Eggs', 'Tofu'
#     * Fresh produce: 'Organic', 'Herbs', 'Vegetables', 'Mushrooms', 'WildHarvested', 'Beans', 'Fruits', 'Grains', 'Nuts'
#     * Confectionery: 'Bakedgoods', 'Honey', 'Jams', 'Maple', 'Coffee', 'Juices', 'Wine'
#     * Others: 'Crafts', 'Prepared', 'Soap', 'PetFood'
# 
# 3. The new categories will have values equal to the sum of its corresponding constituent categories.
#     * E.g. If a market sells Cheese and Eggs only, then it has a value of 2 for the variable 'Dairy'.
# 4. I will create a variable called 'Number of products' by summing the values of the above categories

# In[ ]:


# products that need feature engineering
products = ['Organic', 'Bakedgoods', 'Cheese', 'Crafts', 'Flowers', 'Eggs',
       'Seafood', 'Herbs', 'Vegetables', 'Honey', 'Jams', 'Maple', 'Meat',
       'Nursery', 'Nuts', 'Plants', 'Poultry', 'Prepared', 'Soap', 'Trees',
       'Wine', 'Coffee', 'Beans', 'Fruits', 'Grains', 'Juices', 'Mushrooms',
       'PetFood', 'Tofu', 'WildHarvested']

# changing all the 'Y' to 1 and 'N' to 0
for product in products:
    try:
        farmers_mkt[product] = farmers_mkt[product].replace(to_replace=['Y', 'N'], value=[1,0])
    except:
        continue


# In[ ]:


# temporary dataframe to store our new variables
products_df = pd.DataFrame()

# dictionary of new variables
new_products = {'Plants':['Trees', 'Plants', 'Nursery', 'Flowers'],                'Meat':['Meat', 'Poultry', 'Seafood'],                'Dairy':['Cheese', 'Eggs', 'Tofu'],                'Fresh produce':['Organic', 'Herbs', 'Vegetables', 'Mushrooms', 'WildHarvested', 'Beans', 'Fruits', 'Grains', 'Nuts'],                'Confectionery':['Bakedgoods', 'Honey', 'Jams', 'Maple', 'Coffee', 'Juices', 'Wine'],                'Others':['Crafts', 'Prepared', 'Soap', 'PetFood']}

# creating new product categories
for product in new_products.keys():
    try:
        products_df[product] = farmers_mkt.loc[:,new_products[product]].sum(1)
    except:
        print(product)

# simply sum the columns up to obtain number of categories present        
products_df['Number of products'] = products_df.sum(1)

# drop the product categories from the main dataset and add new product categories
farmers_mkt = farmers_mkt.drop(columns=products)
farmers_mkt = pd.concat([farmers_mkt, products_df], axis=1)


# ## Use of vouchers at the markets
# 
# * Similarly, I will encode 'Y' as 1 and 'N' as 0 for the following variables: 'Credit', 'WIC', 'WICcash', 'SFMNP', 'SNAP'
# * Note that the variables 'WIC', 'WICcash', 'SFMNP', 'SNAP' correspond to payment modes offered to the lower income population
# * I will create a new variable 'Low income friendly' as the sum of the values in ['WIC', 'WICcash', 'SFMNP', 'SNAP']. 

# In[ ]:


payment_modes = ['Credit', 'WIC', 'WICcash', 'SFMNP', 'SNAP']
for payment_mode in payment_modes:
    try:
        farmers_mkt[payment_mode] = farmers_mkt[payment_mode].replace(to_replace=['Y', 'N'], value=[1,0])
    except:
        continue
farmers_mkt['Low income friendly'] = farmers_mkt.loc[:, 'WIC':'SNAP'].sum(1)


# ## Final data set
# Let's have a look at the data set after some data wrangling :)

# In[ ]:


farmers_mkt.head()


# # County info data set
# Let's have a look at the variables present in this data set and the number of missing values.

# In[ ]:


county_info = pd.read_csv('/kaggle/input/farmers-markets-in-the-united-states/wiki_county_info.csv')

print(county_info.shape)
print('-'*20)
county_info.isna().sum()


# In[ ]:


county_info.head()


# # Data Wrangling
# Let's modify the data set further for ease of visualisation

# ## Dropping redundant variables, missing data
# * The variable 'number' is clearly redundant, let's drop it.
# * There are only at most 5 rows with missing values out of the entire data set with 3233 rows, let's drop them too

# In[ ]:


# the variable 'number' is redundant, so we drop them
county_info = county_info.drop(columns = 'number')

# drop rows with missing values
county_info = county_info.dropna()


# ## Change currency to numeric values
# First note that variables ['per capita income', 'median household income', 'median family income', 'population',	'number of households'] are all of string type and contains characters '$' and ','
# 
# Let's change these variables to float type

# In[ ]:


# the figures are strings, have a dollar sign and commas, the following changes them to integers
def remove_char(df):
    bad_var = ['per capita income', 'median household income', 'median family income', 'population', 'number of households']
    bad_tokens = ['$', r',']
    
    for var in bad_var:
        df[var] = df[var].replace('[\$,]', '', regex=True).astype(int)
    return df


# In[ ]:


county_info = remove_char(county_info)


# ## Obtain geolocation of each county/state
# * It is generally difficult to pinpoint the location of a market from its county name unless we use the [Google Geocoding API](https://developers.google.com/maps/documentation/geocoding/intro). However the Google Geocoding API isn't free of charge.
# * I have tried using other APIs to map county names to latitude and longitude numbers but some problems I encountered were:
#     * processing time is too slow
#     * the naming conventions of the county in the data set doesn't follow other APIs and doesn't follow the farmers market data set's naming convention.
#     * there aren't many APIs that support county names to geolocation conversion
# 
# If you have a better solution, please let me know :)
#     
# A solution would be to do visualisation at the state level using this Python library: [us](https://pypi.org/project/us/).
# 
# The us Python library has its own naming convention for the US states, let's check which state in the county_info data set doesn't follow the naming convention.

# In[ ]:


# list of all the unique states in the data set
states = list(county_info['State'].unique())

# check which U.S. state isn't named correctly
for state in states:
    if (us.states.lookup(state) == None):
        print(state)


# 
# 'U.S. Virgin Islands' is not the correct name, let's rename 'U.S. Virgin Islands' to 'Virgin Islands'.

# In[ ]:


county_info.loc[county_info['State'] == 'U.S. Virgin Islands', 'State'] = 'Virgin Islands'


# ## Create state level data set from county level data set
# Next, let's create a state-level data set from the county_info data set
# * 'Per capita income', 'Population', 'Number of households', 'Number of markets' can possibly be combined to a state-level variable.
# * However, that is not the case for variables 'Median household income' and 'Median family income'.

# In[ ]:


# list of all the unique states in the data set
states = list(county_info['State'].unique())

states_coded = []

# obtains the FIPS code from state name
for state in states:
    states_coded.append(us.states.lookup(state).abbr)


# In[ ]:


# ready to create state-level data set
state_info = pd.DataFrame()

# retain state names in state-level data set for reference
state_info['State'] = states

# FIPS code of each state
state_info['State code'] = states_coded

# variables to be included in new data set
cols = ['Per capita income', 'Population', 'Number of households', 'Number of markets']

# initialisation
for var in cols:
    state_info[var] = ''

temp = []

# computation for state-level variables
for i in range(len(states)):
    num_household = 0
    
    # dataframe of all counties in state state[i]
    state_df = pd.DataFrame(county_info.loc[county_info['State'] == states[i], :]).reset_index()
    
    total_popn = sum(state_df['population'])
    state_info.loc[i, 'Population'] = total_popn
    state_info.loc[i, 'Number of households'] = state_df['number of households'].sum()
    state_info.loc[i, 'Number of markets'] = farmers_mkt[farmers_mkt['State'] == states[i]].shape[0]
    temp += [round(state_df['per capita income'].dot(state_df['population'] / total_popn))]

state_info['Per capita income'] = temp
state_info['Per capita income'] = state_info['Per capita income'].astype(int)

state_info.head()


# # Data Visualisation
# Now that we have our data sets ready, let's visualise them!
# 
# I will generally use heatmaps to visualise the data. The heatmaps can be plotted in 2 different ways:
# 1. [gmaps](https://pypi.org/project/gmaps/) library 
# 2. [plotly](https://plotly.com/python/maps/) library 

# ## Visualisations using gmaps heatmap
# We can visualise our data on Google Maps, but we require an API key which can be obtained for free I believe.
# 
# You can retrieve an API key by following the instructions [here](https://developers.google.com/maps/documentation/javascript/get-api-key).
# 
# Also, you may need to enable an [extension](https://jupyter-gmaps.readthedocs.io/en/latest/install.html) for the gmaps to display on jupyter notebook.
# 
# Note:
# * Kaggle has not enabled the gmaps notebook extension so the heatmap cannot be displayed here, I have included images of what they should look like if you run this on a jupyter notebook.
# * The heatmap is supposed to be interactive, you can zoom in and out for a more detailed view, but you can't see it here as I have displayed them as images.

# In[ ]:


# many of the steps required to plot a gmaps heatmap are repetitive
# the following function eases this process

# input: a dataframe containing longitude and latitude values and the variable you wish to visualise
# output: a corresponding heatmap
def plot_gmaps(df, var):
    # obtain your own API key with the link above
    API_KEY = YOUR_API_KEY

    gmaps.configure(api_key=API_KEY)
    
    # a dataframe of longitude and latitudes, this dataframe cannot have missing values
    valid_df = df.loc[~df['x'].isnull(), ['x', 'y', var]]
    
    m = gmaps.Map()
    
    # adding a heatmap layer on top on Google Maps
    heatmap_layer = gmaps.heatmap_layer(
        valid_df[['y','x']], 
        
        # we divide the variable by its max value to ensure all variable have a scale of [0,1]
        # this prevents the heatmap from looking more saturated for a variables with larger scale
        weights=valid_df[var] / valid_df[var].max(),
        max_intensity=100, 
        point_radius=20.0
    )
    m.add_layer(heatmap_layer)
    
    return m


# ## Distribution of markets
# Observe how the middle section of this continent does not have as much farmers market as the rest.

# In[ ]:


# uncomment out the following line if you're using a jupyter notebook
# plot_gmaps(farmers_mkt, 'Is market')

# comment out the following line if you're using a jupyter notebook
display(Image.open("/kaggle/input/figures/gmaps_is_market.png"))


# ## Distribution of low-income friendly markets
# * The data here is weighted by the number of low-income friendly payment modes, namely the 'WIC', 'WICcash', 'SFMNP', 'SNAP'.
# * If a market supports all 4 payment modes, they get a weight of 4.

# In[ ]:


# uncomment out the following line if you're using a jupyter notebook
# plot_gmaps(farmers_mkt, 'Low income friendly')

# comment out the following line if you're using a jupyter notebook
display(Image.open("/kaggle/input/figures/gmaps_low_income_friendly.png"))


# ## Number of markets without products data

# In[ ]:


# uncomment out the following line if you're using a jupyter notebook
# plot_gmaps(farmers_mkt, 'No products')

# comment out the following line if you're using a jupyter notebook
display(Image.open("/kaggle/input/figures/gmaps_no_products.png"))


# ## Distribution of markets that uses social media
# * The data here is weighted by the number of social media platforms ('Website', 'Facebook', 'Twitter', 'Youtube', 'OtherMedia') a market uses.
# * If a market supports all 5 payment modes, they get a weight of 5.

# In[ ]:


# uncomment out the following line if you're using a jupyter notebook
# plot_gmaps(farmers_mkt, 'Media')

# comment out the following line if you're using a jupyter notebook
display(Image.open("/kaggle/input/figures/gmaps_media.png"))


# ## Distribution of markets that sell plants

# In[ ]:


## Distribution of markets that sell plants

# uncomment out the following line if you're using a jupyter notebook
# plot_gmaps(farmers_mkt, 'Plants')

# comment out the following line if you're using a jupyter notebook
display(Image.open("/kaggle/input/figures/gmaps_plants.png"))


# ## Distribution of markets that sell meat

# In[ ]:


## Distribution of markets that sell meat

# uncomment out the following line if you're using a jupyter notebook
# plot_gmaps(farmers_mkt, 'Meat')

# comment out the following line if you're using a jupyter notebook
display(Image.open("/kaggle/input/figures/gmaps_meat.png"))


# ## Distribution of markets that sell fresh produce

# In[ ]:


## Distribution of markets that sell fresh produce

# uncomment out the following line if you're using a jupyter notebook
# plot_gmaps(farmers_mkt, 'Fresh produce')

# comment out the following line if you're using a jupyter notebook
display(Image.open("/kaggle/input/figures/gmaps_fresh_produce.png"))


# ## Distribution of markets that sell dairy products

# In[ ]:


## Distribution of markets that sell dairy products

# uncomment out the following line if you're using a jupyter notebook
# plot_gmaps(farmers_mkt, 'Dairy')

# comment out the following line if you're using a jupyter notebook
display(Image.open("/kaggle/input/figures/gmaps_dairy.png"))


# ## Distribution of markets that sell confectionery products

# In[ ]:


## Distribution of markets that sell confectionery products

# uncomment out the following line if you're using a jupyter notebook
# plot_gmaps(farmers_mkt, 'Confectionery')

# comment out the following line if you're using a jupyter notebook
display(Image.open("/kaggle/input/figures/gmaps_confectionery.png"))


# ## Distribution of markets that sell other miscellaneous products

# In[ ]:


## Distribution of markets that sell other miscellaneous products

# uncomment out the following line if you're using a jupyter notebook
# plot_gmaps(farmers_mkt, 'Others')

# comment out the following line if you're using a jupyter notebook
display(Image.open("/kaggle/input/figures/gmaps_others.png"))


# ## Number of products sold in the markets

# In[ ]:


## Distribution of number of products sold in the markets

# uncomment out the following line if you're using a jupyter notebook
# plot_gmaps(farmers_mkt, 'Number of products')

# comment out the following line if you're using a jupyter notebook
display(Image.open("/kaggle/input/figures/gmaps_num_products.png"))


# ## Use of social media in the farmers markets
# Let's see which farmers markets are more tech-savvy and uses social media.

# In[ ]:


## Distribution of markets that use social media

# uncomment out the following line if you're using a jupyter notebook
# plot_gmaps(farmers_mkt, 'Media')

# comment out the following line if you're using a jupyter notebook
display(Image.open("/kaggle/input/figures/gmaps_media.png"))


# ## County information
# Now let's visualise some statistics about the counties!

# In[ ]:


state_info['Number of markets per capita'] = state_info['Number of markets'] / state_info['Population']

# many of the steps for plotting heatmaps are repetitive, this function takes care of that
# input: a dataframe df and a variable in df that you wish to plot
# output: None. A heatmap will be plotted directly.
def plot_heatmap(df, var): 
    # plotting the heatmap by states
    fig = go.Figure(data=go.Choropleth(
        locations=df['State code'], # Spatial coordinates
        z = df[var].astype(float), # Data to be color-coded
        locationmode = 'USA-states', # set of locations match entries in `locations`
        colorscale = 'Reds',
        colorbar_title = var,
        text = df['State']
    ))

    fig.update_layout(
        title_text = var + ' by state<br>(Hover over the states for details)',
        geo_scope='usa', # limit map scope to USA
    )

    fig.show()


# In[ ]:


plot_heatmap(state_info, 'Population')


# In[ ]:


plot_heatmap(state_info, 'Per capita income')


# The income per capita seems similar throughout all states, let's generate more visualisations to derive more details.

# In[ ]:


# sort the dataframe by the 'per capita income' variable, ascending order
state_info = state_info.sort_values(by=['Per capita income'])

fig = plt.figure()
ax = fig.add_axes([0,0,2,1])
ax.bar(state_info['State'], state_info['Per capita income'])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


state_info['Per capita income'].describe()


# In[ ]:


plot_heatmap(state_info, 'Number of households')


# In[ ]:


plot_heatmap(state_info, 'Number of markets')


# # Analysis
# * Interestingly the per capita income level of different states are largely similar, around USD25,000.
# * However, note that the states with the lowest per capita income are generally small islands like 'Samoa', 'Mariana Islands', 'Puerto Rico' and 'Guam' are not really visible in the heatmap, so the heatmap isn't representative of smaller regions.
# * There are a few states earning much more than the mean like 'Columbia' and 'Massachusetts'.
# * Generally, states with a larger population have a higher number of households and higher concentration of markets, so it is not generally true that lower income states do not have access to farmers markets.
# * The variety of products sold in the markets and use of social media is also largely correlated with the size of population the markets are in.
