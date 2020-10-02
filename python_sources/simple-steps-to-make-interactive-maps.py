#!/usr/bin/env python
# coding: utf-8

# ## Simple Steps to Make An Interactive Map

# In[112]:


import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import gc
import warnings
warnings.filterwarnings("ignore")


# This notebook shows how to make a simple interactive map using plotly.
# 
# First of all, we will load the data we need. There are six csv files in the input file, but here we only focus on two of them: the donations dataset and the donors dataset. They can be merged based on the unique Donor ID. 

# In[116]:


# Load and merge datasets. It may take a little while as the datasets are large.
donations = pd.read_csv('../input/Donations.csv')
donors = pd.read_csv('../input/Donors.csv')
df = donations.merge(donors, on="Donor ID", how="left")
# Delete and collect garbage
del donations, donors
gc.collect()
# A quick look at the variable names and data types
df.dtypes


# Before betting into the map, we will first of all get the aggregated data by state. Here is a short list of the data that we are going to plot:
# 
# 1. Number of unique donors in each state;
# 2. Number of donations in each state;
# 3. Number of total tonation amount in each state;
# 4. Average amount per donation in each state. 
# 
# We can accomplish this by simply using groupby() and agg()
# 

# In[104]:


# Get aggregated data at the state level
state = df.groupby('Donor State', as_index=False).agg({'Donor ID': 'nunique',
                                                       'Donation ID': 'count',
                                                       'Donation Amount':'sum'})    
# rename the columns
state.columns = ["State", "Donor_num", "Donation_num", "Donation_sum"]
# Get average donation amount
state["Donation_ave"]=state["Donation_sum"]/state["Donation_num"]
# Clean garbage
del df
gc.collect()
# A quick look at the dataframe we got
state.head()


# Now we are ready to make the plot.  Let's import the libraries we need and set up for using plotly. 

# In[117]:


import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()


# We will create a new column showing the text we would like to see when the mouse hovers on each state. The target is something like:
# 
# **Alabama** <br>
# Number of donors:   <br>
# Number of donations:  <br>
# Average amount per donation:  <br>
# Total donation amount:  <br>
# 

# In[118]:


# Convert numerical variables into strings first
for col in state.columns:
    state[col] = state[col].astype(str)

state['text'] = state['State'] + '<br>' +    'Number of donors: $' + state['Donor_num']+ '<br>' +    'Number of donations: $'+ state['Donation_num']+ '<br>'+    'Average amount per donation: $' + state['Donation_ave']+ '<br>' +    'Total donation amount:  $' + state['Donation_sum']


# Plotly only takes two letter state code, but we only have full state name in our dataset.  We'll use a dictionary to map the state names into state codes. 

# In[119]:


state_codes = {
    'District of Columbia' : 'DC','Mississippi': 'MS', 'Oklahoma': 'OK', 
    'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 
    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 
    'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 
    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 
    'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 
    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 
    'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',
    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 
    'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 
    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 
    'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 
    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 
    'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 
    'Nevada': 'NV', 'Maine': 'ME', 'other': ''}

state['code'] = state['State'].apply(lambda x : state_codes[x])


# Now we are ready to set up for plotting. 
# 
# There are basically two parts to set up before sending the data to plotly: the data and the layout.  Below is a simple set up but there are many adjustments that you can make. 

# In[109]:


data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = state['code'], # The variable identifying state
        z = state['Donation_sum'].astype(float), # The variable used to adjust map colors
        locationmode = 'USA-states', 
        text = state['text'], # Text to show when mouse hovers on each state
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(  
            title = "USD")  # Colorbar to show beside the map
        ) ]

layout = dict(
        title = 'Donation by State<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )


# You are two lines away from your interactive map!

# In[110]:


fig = dict(data=data, layout=layout)
iplot(fig)

