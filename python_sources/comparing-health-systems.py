#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import math

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


# What is the best way to run a national health system? Does spending more always lead to better results, or does it matter who spends the money and where it goes? This is the basic question in this notebook. That means the roadmap will be as follows: 
# 
# First I'll explore the world bank data to see which countries spend the highest amounts of their money on their healthcare. Ofcourse there is more than one way to spend money on healthcare. Governments often spend most of the money, but people also spend out of pocket money. And ofcourse there is, in some countries, an influx of foreign aid also adds to the amount of money spend.
# 
# When that is done I'll start to look at how all this increases or decreases the effectiveness of the overall health system. There is an obvious problem of corrolation vs causation here. The richer countries are more likely to be able to spend more money, but they are also the key beneficiaries of new technologies and often have healthier environments. This means that even if we'd find an effect of health spending on healthiness, it may be an effect of wealth, the environment or technology. So, we should be carefull with the claims we make and take into consideration many more factors than just healthcare spending and effectiveness.
# 
# The thirds step is to look at how general effectiveness translates into the ability of a country to keep epidemics of communicable diseases under control. Ofcourse these analyses have the same caveats as the earlier steps. This will lead towards the last part, where I'll take a look at Covid-19 and how succesfull the responses have been.

# # Table of Contents
# 1. [Exploring Health Systems](#Exploring-Health-Systems)
# 2. [Health System Effectiveness](#Health-System-Effectiveness)
# 3. [Infectious Diseases](#Infectious-Diseases)
# 4. [Covid-19 Readiness](#Covid-19-Readiness)

# # Exploring Health Systems
# 
# Let's take a look at the data from the world bank. The full dataset can be found [here](https://www.kaggle.com/danevans/world-bank-wdi-212-health-systems). The figures below show the ten biggest and lowest spenders. These figures include government spending, out of pocket spending and foreign aid. It is immediately visible that looking at total spending won't be enough, as there are many different things going on. Broadly speaking we can discern two groups. First there are the smaller nations, which have a very small GDP. The Marshall Islands, Micronesia, Tuvalu, Sierra Leone, Kiribati & Palau can all be found at or near the bottom of [list](https://www.worldometers.info/gdp/gdp-by-country/) of GDP numbers. This means that relatively small increases of spending in absolute terms would lead to big rises in the relative terms shown in the graph.
# 
# The United States, Brazil & Switzerland clearly belong to a second group of countries with a big GDP. They spend, both in absolute and in relative terms, a lot of money on their healthcare. It may be interesting to note in the current crises that both the United States and Brazil spend so much on healthcare. Both countries have been hit hard by Covid-19 and have struggled to find an appropriate response.
# 
# Cuba stands more or less alone, for political and socio-economic reasons. It does not have a large GDP, but it is well known for it's healthcare system. The government spends a lot on healthcare and it would be interesting to look how effective the system is in the next parts.

# In[ ]:


healthsysdf = pd.read_csv('../input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv') #load first dataset

healthsysdf = healthsysdf.drop(columns = 'Province_State') #drop useless columns
healthsysdf = healthsysdf.drop(columns = 'Country_Region')
healthsysdf['Total_Gov_Spend'] = healthsysdf.apply(lambda row: (row.Health_exp_pct_GDP_2016 / 100) * row.Health_exp_public_pct_2016, axis = 1) #calculate total government spending
healthsysdf['Outofpocket_Spend'] = healthsysdf.apply(lambda row: (row.Health_exp_pct_GDP_2016 / 100) * row.Health_exp_out_of_pocket_pct_2016, axis = 1) #calculate total out of pocket spending
healthsysdf['Other_Spend'] = healthsysdf.apply(lambda row: row.Health_exp_pct_GDP_2016 - row.Total_Gov_Spend - row.Outofpocket_Spend, axis = 1)


countrycodes = ['AFG', 'ALB', 'DZA', 'AND', 'AGO', 'ATG', 'ARG', 'ARM', 'AUS', 'AUT', 'AZE', 'BHS', 'BHR', 'BGD', 'BRB', 'BLR', 'BEL', 'BLZ', 'BEN', 'BTN', 'BOL', 'BIH', 'BWA', 'BRA',
                'BRN', 'BGR', 'BFA', 'BDI', 'CPV', 'KHM', 'CMR', 'CAN', '', 'CAF', 'TCD', '', 'CHL', 'CHN', '', '', 'COL', 'COM', 'COD', 'COG', 'CRI', 'CIV', 'HRV', 'CUB', 'CYP',
                'CZE', 'DNK', 'DJI', 'DMA', 'DOM', 'ECU', 'EGY', 'SLV', 'GNQ', 'ERI', 'EST', 'SWZ', 'ETH', '', 'FJI', 'FIN', 'FRA', '', 'GAB', 'GMB', 'GEO', 'DEU', 'GHA', 'GRC', '',
                'GRD', '', 'GTM', 'GIN', 'GNB', 'GUY', 'HTI', 'HND', 'HUN', 'ISL', 'IND', 'IDN', 'IRN', 'IRQ', 'IRL', '', 'ISR', 'ITA', 'JAM', 'JPN', 'JOR', 'KAZ', 'KEN', 'KIR', '', 'KOR', '', 'KWT',
                'KGZ', 'LAO', 'LVA', 'LBN', 'LSO', 'LBR', '', '', 'LTU', 'LUX', 'MDG', 'MWI', 'MYS', 'MDV', 'MLI', 'MLT', 'MHL', 'MRT', 'MUS', 'MEX', 'FSM', 'MDA', 'MCO', 'MNG', 'MNE', 'MAR',
                'MOZ', 'MMR', 'NAM', 'NPL', 'NLD', '', 'NZL', 'NGA', 'NER', 'NGA', 'MKD', '', 'NOR', 'OMN', 'PAK', 'PLW', 'PAN', 'PNG', 'PRY', 'PER', 'PHL', 'POL', 'PRT', '', 'QAT', 'ROU', 'RUS',
                'RWA', 'WSM', 'SMR', 'STP', 'SAU', 'SEN', 'SRB', 'SYC', 'SLE', 'SGP', '', 'SVK', 'SVN', 'SLB', '', 'ZAF', '', 'ESP', 'LKA', 'KNA', 'LCA', '', 'VCT', 'SDN', 'SUR', 'SWE', 'CHE', '', 'TJK',
                'TZA', 'THA', 'TLS', 'TGO', 'TON', 'TTO', 'TUN', 'TUR', 'TKM', '', 'TUV', 'UGA', 'UKR', 'ARE', 'GBR', 'USA', 'URY', 'UZB', 'VUT', 'VEN', 'VNM', '', '', 'YEM', 'ZMB', 'ZWE']

healthsysdf['Country_Codes'] = countrycodes #add country codes for use in map


# In[ ]:


bginfo = pd.read_csv('../input/undata-country-profiles/country_profile_variables.csv') #load second dataset
bginfo.rename(columns = {'country':'World_Bank_Name'}, inplace=True) #rename dataset to make combining easy

bginfo = bginfo.replace({'United States of America':'United States', 'Viet Nam': 'Vietnam'})

healthsysdf = healthsysdf.replace({'Yemen, Rep.': 'Yemen'})

healthsysdf = pd.merge(healthsysdf, bginfo, on='World_Bank_Name', how='outer') #combining datasets

healthsysdf = healthsysdf.dropna(thresh=3) #drop countries with little data

# Get the countries with GDP set below 0 and drop them from dataset
badgdp = healthsysdf[ healthsysdf['GDP: Gross domestic product (million current US$)'] < 0 ].index
healthsysdf.drop(badgdp , inplace=True)

# Create smaller regional groupings
healthsysdf.replace({'SouthernAsia':'Asia', 'WesternAsia':'Asia', 'EasternAsia':'Asia','CentralAsia':'Asia', 'South-easternAsia':'Asia',
                     'WesternEurope':'Europe', 'SouthernEurope':'Europe', 'EasternEurope':'Europe', 'NorthernEurope':'Europe',
                     'NorthernAfrica':'Africa', 'MiddleAfrica':'Africa', 'WesternAfrica':'Africa', 'EasternAfrica':'Africa', 'SouthernAfrica':'Africa',
                     'SouthAmerica':'Americas', 'Caribbean':'Americas', 'CentralAmerica':'Americas', 'NorthernAmerica':'Americas',
                     'Polynesia':'Oceania', 'Melanesia':'Oceania', 'Micronesia':'Oceania'}, inplace=True )


# In[ ]:


total_exp = healthsysdf.sort_values('Health_exp_pct_GDP_2016', ascending = False)
top_ten_exp = total_exp.head(10)
total_exp = total_exp.sort_values('Health_exp_pct_GDP_2016')
low_ten_exp = total_exp.head(10)

fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

fig.add_trace(
    go.Bar(x=top_ten_exp['World_Bank_Name'], y=top_ten_exp['Health_exp_pct_GDP_2016']),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=low_ten_exp['World_Bank_Name'], y=low_ten_exp['Health_exp_pct_GDP_2016']),
    row=1, col=2
)


fig.update_layout(
    title={
        'text': "Ten highest and lowest spenders",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    plot_bgcolor= 'white',
    paper_bgcolor= 'white',
    yaxis_title="% of GDP spent on healthcare",
    showlegend=False,
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    )
)
fig.show()


# Looking at the lowest spenders also gives us two groups. One is a set of small countries with high GDP. Monaco and oil rich countries such as Brunei and Qater belong to this group. My guess would be that the combination of a small, densely packed population and relatively high GDP means they fall at the tail end of this graph.
# 
# The other group are developing countries that don't spend a lot of money on healthcare relative to their GDP. In order to answer the question of how healthcare spending influences it's effectiveness it will be interesting to compare those developing nations that spend relatively a lot and those that don't.

# In[ ]:


import plotly.graph_objects as go
import pandas as pd

fig = go.Figure(data=go.Choropleth(
    locations = healthsysdf['Country_Codes'],
    z = healthsysdf['Health_exp_pct_GDP_2016'],
    text = healthsysdf['World_Bank_Name'],
    colorscale = 'blues',
    autocolorscale=False,
    colorbar_tickprefix = '% ',
    marker_line_color='darkgray',
    marker_line_width=0.5,
))

fig.update_layout(
    title_text='Percentage of GDP spent on Healthcare',
    font=dict(
        family="Courier New, monospace",
        size=14),
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    )
)

fig.show()


# The Map above shows the level of healthcare spending for all the countries in the World Bank data. You can see a pattern where the United States spends the most followed by most of western and northern Europe, Brazil, Japan, Australia & New Zealand. Interestingly the levels of spending in Sierra Leone, Afghanistan, Zimbabwe & Namibia are also very high. This is most likely due to an influx of foreign aid money.
# 
# This last theory is somewhat supported by the map below. It shows the amount of money the government spends on healthcare as a percentage of GDP. As opposed to out of pocket spending & foreign aid. The pattern looks mostly the same, except that Sierra Leone and Afghanistan have dropped of to similar levels of spending as the countries around them. Namibia & Zimbabwe are still some of the highest spending countries in Africa however.

# In[ ]:


fig = go.Figure(data=go.Choropleth(
    locations = healthsysdf['Country_Codes'],
    z = healthsysdf['Total_Gov_Spend'],
    text = healthsysdf['World_Bank_Name'],
    colorscale = 'blues',
    autocolorscale=False,
    colorbar_tickprefix = '% ',
    marker_line_color='darkgray',
    marker_line_width=0.5,
))

fig.update_layout(
    title_text='Government Spending on Healthcare',
    font=dict(
        family="Courier New, monospace",
        size=14),
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    )
)

fig.show()


# There is a point to be made for measure spending per capita rather than a percentage of GDP. There are multiple countries with high GDP (China & Brazil stand out) that also have very large populations. That means that even if they spend a high percentage of their GDP on healthcare, they still spend less per person than other smaller countries. The map below shows the levels of spending per capita. Now the pattern is clearer than ever. The highest spenders are the United States, Canada, western & northern Eurpe, Japan, Australia & New Zealand.

# In[ ]:


fig = go.Figure(data=go.Choropleth(
    locations = healthsysdf['Country_Codes'],
    z = healthsysdf['per_capita_exp_PPP_2016'],
    text = healthsysdf['World_Bank_Name'],
    colorscale = 'blues',
    autocolorscale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
))

fig.update_layout(
    title_text='Healthcare Spending per Capita',
    font=dict(
        family="Courier New, monospace",
        size=14),
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    )
)

fig.show()


# It is not just a question of spending the most money. It may also matter who spends the money. Looking at my own country (the Netherlands) there are three main ways healthcare is paid for. Direct government spending, insurance & out of pocket. We may add foreign aid as well. Either the government spends the money directly. Most services of General Practicioners are free in the Netherlands, paid for by the state. Another way of paying is by being insured or by paying for healthcare directly out of pocket.
# 
# In the world bank data that we are currently looking at insurance and state spending both fall under the government spending category. This may be misleading since people still need to pay for insurance. The same can be said about government. You pay for it through taxes. But there are often different levels of insurance, the most expensive of which may not be accesible for poorer elements of the country.
# 
# It would be interesting to compare the different models of government vs insurance spending. This is however beyond the scope of the current dataset. We can look at the levels of out of pocket spending however. This is the money directly spend by individuals for their healthcare. The graph below shows these levels for the G8 countries. We can see that the mixture between government and out of pocket spending are mostly the same for the United States, the United Kingdom, France, Germany, Canda & Japan. The rate of out of pocket spending is higher however in China & Russia.

# In[ ]:


g8_list = ['Canada', 'United Kingdom', 'United States', 'Russian Federation', 'Germany', 'France', 'Japan', 'China']

g8_sub = healthsysdf.loc[healthsysdf['World_Bank_Name'].isin(g8_list)]
g8_sub = g8_sub.sort_values('Health_exp_pct_GDP_2016', ascending=False)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=g8_sub['World_Bank_Name'],
    y=g8_sub['Health_exp_pct_GDP_2016'],
    name='Total Spending',
    marker_color='darkblue'
))
fig.add_trace(go.Bar(
    x=g8_sub['World_Bank_Name'],
    y=g8_sub['Total_Gov_Spend'],
    name='Government Spending',
    marker_color='mediumaquamarine'
))
fig.add_trace(go.Bar(
    x=g8_sub['World_Bank_Name'],
    y=g8_sub['Outofpocket_Spend'],
    name='Private (out of pocket) Spending',
    marker_color='lightsteelblue'
))
fig.add_trace(go.Bar(
    x=g8_sub['World_Bank_Name'],
    y=g8_sub['Other_Spend'],
    name='Other',
    marker_color='grey'
))

fig.update_layout(
    barmode='group',
    title={
        'text': "G8 Healthcare spending",
        'y':0.9,
        'x':0.4,
        'xanchor': 'center',
        'yanchor': 'top'},
    plot_bgcolor= 'white',
    paper_bgcolor= 'white',
    yaxis_title="% of GDP spent on healthcare",
    showlegend=True,
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    )
)
fig.show()


# So what are the levels in other parts of the world? Brazil stands out in the graph below as one of the only countries were more is spend out of pocket than by the government. Switzerland also has, for a country with such high spending per capita, a relatively high level of out of pocket spending.

# In[ ]:


interest_sub_list = ['Norway', 'Ireland', 'Netherlands', 'Switzerland', 'Brazil', 'Argentina', 'Mexico', 'Algeria', 'Namibia', 'Rwanda', 'South Africa', 'Indonesia', 'India', 'Australia']

interest_sub = healthsysdf.loc[healthsysdf['World_Bank_Name'].isin(interest_sub_list)]
interest_sub = interest_sub.sort_values('Health_exp_pct_GDP_2016', ascending=False)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=interest_sub['World_Bank_Name'],
    y=interest_sub['Health_exp_pct_GDP_2016'],
    name='Total Spending',
    marker_color='darkblue'
))
fig.add_trace(go.Bar(
    x=interest_sub['World_Bank_Name'],
    y=interest_sub['Total_Gov_Spend'],
    name='Government Spending',
    marker_color='mediumaquamarine'
))
fig.add_trace(go.Bar(
    x=interest_sub['World_Bank_Name'],
    y=interest_sub['Outofpocket_Spend'],
    name='Private (out of pocket) Spending',
    marker_color='lightsteelblue'
))
fig.add_trace(go.Bar(
    x=interest_sub['World_Bank_Name'],
    y=interest_sub['Other_Spend'],
    name='Other',
    marker_color='grey'
))

fig.update_layout(
    barmode='group',
    title={
        'text': "Healthcare spending",
        'y':0.9,
        'x':0.4,
        'xanchor': 'center',
        'yanchor': 'top'},
    plot_bgcolor= 'white',
    paper_bgcolor= 'white',
    yaxis_title="% of GDP spent on healthcare",
    showlegend=True,
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    )
)
fig.show()


# # Health System Effectiveness (WIP)

# So what does the amount a society spends on healthcare mean for the quality of the healthcare? There are many ways to look at healthcare succes, with different levels of granularity. This dataset doesn't really support a deep dive. So for now we're looking at two things. First does more spending increase the capacity in healthcare systems? Does it increase the amount of beds and doctors?
# 
# After that we'll build a model that looks at how big of an effect healthcare spending has on outcomes such as life expectancy & child mortality.

# In[ ]:


healthsysdf = healthsysdf.dropna(subset=['GDP: Gross domestic product (million current US$)'])
size=healthsysdf['GDP: Gross domestic product (million current US$)']
sizeref = 2.*max(healthsysdf['GDP: Gross domestic product (million current US$)'])/(100**2)

fig = px.scatter(x=healthsysdf['per_capita_exp_PPP_2016'], y=healthsysdf['Physicians_per_1000_2009-18'],                 
                 size=size, color=healthsysdf['Region'],
                 hover_name=healthsysdf['World_Bank_Name'])

# Tune marker appearance and layout
fig.update_traces(mode='markers', marker=dict(sizemode='area',
                                              sizeref=sizeref, line_width=2))

fig.update_layout(
    title={
        'text': "Spending vs Physicians per 1000 people",
        'y':0.9,
        'x':0.4,
        'xanchor': 'center',
        'yanchor': 'top'},
    plot_bgcolor= 'white',
    paper_bgcolor= 'white',
    xaxis_title="Healthcare spending per capita",
    yaxis_title="Physicians per 1000 people",
    showlegend=True,
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    )
)

fig.show()


# The graph above shows how many doctors a country has per 1000 inhabitants plotted against the money they spend. Immediately we can see some complicating matters. Looking completely at the left of the graph we can see a couple of countries that do form a upward gradient. That means that if you spend a little of your GDP on healthcare, spending more does mean more doctors. After a while however the effect tapers off. The US stands out again, spending a lot whilst not having that many doctors. Cuba stands out for the exact opposite reasons.
# 
# There are a couple of reasons why the curve might look like this. The first thing any country does when it increases healthcare spending from 0 is investing in hospital beds and staff. This is the quickest way to improve healthcare. At some point however it is no longer efficient to spend money on doctors, but rather on public health programs, vaccinations, digital systems and the like. Another reason may be that as health systems become larger and larger they become harder to manage. So a larger portion of the healthcare money goes to organizational and adiminstrative elements of the system and not towards doctors.

# In[ ]:





# ## Infectious Diseases

# ## Covid-19 Readiness
