#!/usr/bin/env python
# coding: utf-8

# # Japan Trade Statistics
# In this notebook, I will focus on the exploration of data concerning Japan trade.
# I will compare the trend of trade volume in years and the difference among countries.

# ## Modules

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import os
import sys
import warnings
warnings.filterwarnings("ignore")


# ## Data

# In[ ]:


sorted(os.listdir("../input"))


# Country_eng.csv

# In[ ]:


country = pd.read_csv("../input/country_eng.csv")


# In[ ]:


country.head(2)


# It is a dictionary to associate the country code with the English name and its area

# hs2_eng.csv

# In[ ]:


hs2_eng = pd.read_csv("../input/hs2_eng.csv")


# In[ ]:


hs2_eng.tail(2)


# The hs2 has too many categories, we will use another simplified version based on hs2

# 01-05  Animal & Animal Products<br>
# 06-15  Vegetable Products<br>
# 16-24  Foodstuffs<br>
# 25-27  Mineral Products<br>
# 28-38  Chemicals & Allied Industries<br>
# 39-40  Plastics / Rubbers<br>
# 41-43  Raw Hides, Skins, Leather, & Furs<br>
# 44-49  Wood & Wood Products<br>
# 50-63  Textiles<br>
# 64-67  Footwear / Headgear<br>
# 68-71  Stone / Glass<br>
# 72-83  Metals<br>
# 84-85  Machinery / Electrical<br>
# 86-89  Transportation<br>
# 90-97  Miscellaneous<br>
# 
# http://www.foreign-trade.com/reference/hscode.htm

# ### Some real trade data

# I will  focus on year_1988_2015.csv

# In[ ]:


year_1988_2015 = pd.read_csv("../input/year_1988_2015.csv")


# Size, shape, type, features, missing...

# In[ ]:


year_1988_2015.head(2)


# In[ ]:


year_1988_2015.info()


# Year<br>
# Country<br>
# VY is a value of goods which was traded<br>

# Filter features 

# In[ ]:


columns = ["Year", "Country", "VY", "hs2"]
year_data = year_1988_2015[columns]


# In[ ]:


year_data.head(2)


# Then we can start to explore based on the year_data

# ## Exploration

# ### Quetions
# I will try to answer the following questions through the exploration

# 1.Trend of total amount of trade <br>
# 2.Does Japan trade more with Asia or other areas<br>
# 3.What is the dominant country in the dominant area<br>
# 4.What is the structure of trading with the dominant country? Which goods dominate? Was the structure changing?<br>
# 5.Difference when trade with different continents<br>
# 
# I will try to visualize and answer the four questions above 

# ### 1.Trend of total # of trade 

# In[ ]:


year_sum = year_data.groupby(["Year"])["VY"].sum()
year_sum = year_sum.reset_index()


# In[ ]:


# Line plot for yearly trading volue
fig, ax = plt.subplots(figsize = [10,6])

ax.plot(year_sum["Year"], year_sum["VY"], zorder=10);
ax.grid(True, zorder=5)
plt.xlim([year_sum.ix[0,"Year"], year_sum.ix[year_sum.shape[0]-1,"Year"]]);
plt.title("Trade trend from 1998 to 2016");


# From this graph, we see a upper trend of trade of Japan. However, an obvious downturn happens at 2008 because the worldwide financial crisis. Japan start to revive from the crisis quickly in about an year and spend about 5 years to reach the previous peak of trading value 

# ### 2.Does Japan trade more with Asia or other continents

# Then, we start to have a look at the trading preference of Japan to different continents

# We need to build a dictionary first to associate the country with the corresponding area

# In[ ]:


country.head(2)


# In[ ]:


# create country map
area_map = pd.Series(country["Area"]) 
area_map.index=country["Country"]

# create area data
year_data["Area"] = area_map[year_data["Country"]].tolist()
targetarea = [x not in ["Special_Area", "Integrated_Hozei_Ar_Special_Area"] for x in year_data["Area"]]
area_data = year_data.loc[targetarea,:].groupby(["Year", "Area"])["VY"].sum().reset_index()


# In[ ]:


area_data.head(2)


# Then, we can compare the different trading value of each area in each year with Japan

# In[ ]:


area_sum = area_data.groupby("Area")["VY"].sum().reset_index().sort_values(by="VY", ascending=True)


# In[ ]:


# barplot
fig, ax = plt.subplots(figsize = (10,6))
x_pos = np.arange(len(area_sum));
ax.barh(x_pos, area_sum["VY"], align='center',color='green');
ax.set_yticks(x_pos);
ax.set_yticklabels(area_sum["Area"]);
ax.set_xlabel("Trade Value");
ax.set_title("Total amount of trade from 1988 - 2015 with all areas");


# We can see that the Asia is the most important trading partner of Japan over 1988 - 2015. North America and Western_Europe follows. However, did this rank stay constant over the years? We have to look at the structure of each year.

# In[ ]:


# parameters for plotting next graph
areas = np.unique(area_data["Area"])

# offset for annotation
area_offset = {'Africa':-0.05, 'Asia':-0.05, 'Central_and_East_Europe_Russia':-0.05, 'Middle_East':-0.05,
       'Middle_and_South_America':-0.1, 'North_America':-0.05, 'Oceania':0,
       'Western_Europe':-0.05}

# Color Palettes
color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']


# In[ ]:


# Plotting the graph for yearly trading value of each area with Japan
fig, ax = plt.subplots(1, 1, figsize=(12, 14));
for rank, area in enumerate(areas):
    areaplot = area_data.ix[area_data.Area==area,:]
    plt.plot(areaplot.Year,
                    areaplot.VY,
                    lw=2.5,
                    color=color_sequence[rank])
    y_pos = areaplot["VY"].tail(1) + area_offset[area]*10**10
    plt.text(2015.5, y_pos, area, fontsize=14, color=color_sequence[rank])
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.tick_params(axis='y', labelsize=20)
plt.xlim([1988,2015]);
plt.xticks(range(1988, 2015, 2), fontsize=14);
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3);
plt.title("Japanese yearly trading value to areas over 1988 - 2015", size=20);
fig.savefig('jap_area_trade.png')


# This graph shows that back to 1988, The North America is the biggest trading partner of Japan, 
# Howver, after 1990, Asia start to replace the position of North America and lead with a very hugh gap compared to other areas. <br><br>
# A following question is: ***what is the driving force (the dominant country) of the sharp increase since 1990?***<br>
# We can draw another plot for the Asian countries from 1988 to 2014 but only focus on top 12 countries in Asia that traded with Japan most frequetly.

# ### 3.What is the dominant country in Asia

# In[ ]:


# create a map associate country code with the country's name
country_map = pd.Series(country["Country_name"]) 
country_map.index=country["Country"]
year_data["Country_eng"] = country_map[year_data["Country"]].tolist()

# trading countries in Asia
print(np.unique(year_data.ix[year_data["Area"]=="Asia","Country_eng"]))


# In[ ]:


# create asian countries' data and filter the top 12 countries
asia_data = year_data.ix[year_data["Area"]=="Asia",:]             .groupby(["Year", "Country_eng"])["VY"].sum().reset_index()
asia_country_sum = asia_data.groupby("Country_eng")["VY"].sum()                     .reset_index().sort_values(by="VY",ascending=False)
topcountry = asia_country_sum["Country_eng"][0:12]
print(topcountry.values)


# We can create a graph similar to the above one

# In[ ]:


# parameters for plotting next graph
countries = asia_data.Country_eng

# offset for annotation
country_offset = {"People's_Republic_of_China":0, 'Republic_of_Korea':0, 'Taiwan':0,
       'Thailand':0, 'Hong_Kong':0.035, 'Indonesia':0, 'Malaysia':0.02, 'Singapore':-0.005,
       'Philippines':-0.02, 'Viet_Nam':-0.05, 'India':-0.02, 'Brunei':0}


# In[ ]:


# These are the colors that will be used in the plot
fig, ax = plt.subplots(1, 1, figsize=(12, 14));
for rank, country in enumerate(topcountry):
    countryplot = asia_data.ix[asia_data.Country_eng==country,:]
    plt.plot(countryplot.Year,
                    countryplot.VY,
                    lw=2.5,
                    color=color_sequence[rank])
    y_pos = countryplot["VY"].tail(1) + country_offset[country]*10**10
    plt.text(2015.5, y_pos, country, fontsize=13, color=color_sequence[rank])
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.tick_params(axis='y', labelsize=15)
plt.xlim([1988,2015]);
plt.xticks(range(1988, 2015, 2), fontsize=14);
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3);
plt.title("Japanese yearly trading value to Asian countries over 1988 - 2015", size=20);
fig.savefig('jap_asia_trade.png') 


# Clearly, the dominant driving force of the increase of Japanese trade to the Asian Area is its trade with China.

# Another following question of this is that what kinds of products contirbute most to this trading change?<br>

# ### 4.What is the structure of trading with China?

# We first have a look at the general structure of commodities in the Japanese Trade and translate the hs2 code to the simplified categories of commodity

# In[ ]:


def rule(x):
    if x >= 1 and x <= 5:
        return "Animal & Animal Products"
    elif x >= 6 and x <= 15:
        return "Vegetable Products"
    elif x >= 16 and x <= 24:
        return "Foodstuffs"
    elif x >= 25 and x <= 27:
        return "Mineral Products"
    elif x >= 28 and x <= 38:
        return "Chemicals & Allied Industries"
    elif x >= 39 and x <= 40:
        return "Plastics / Rubbers"
    elif x >= 41 and x <= 43:
        return "Raw Hides, Skins, Leather, & Furs"
    elif x >= 44 and x <= 49:
        return "Wood & Wood Products"
    elif x >= 50 and x <= 63:
        return "Textiles"
    elif x >= 64 and x <= 67:
        return "Footwear / Headgear"
    elif x >= 68 and x <= 71:
        return "Stone / Glass"
    elif x >= 72 and x <= 83:
        return "Metals"
    elif x >= 84 and x <= 85:
        return "Machinery / Electrical"
    elif x >= 86 and x <= 89:
        return "Transportation"
    else:
        return "Miscellaneous"


# In[ ]:


# hs2 code translation
year_data["goods"] = year_data["hs2"].apply(rule)


# Then, let's visualize the structure with the pie plot

# In[ ]:


# wrangle the data to get ratio of trading value of goods
goods_sum = year_data.groupby("goods")["VY"].sum().reset_index().sort_values(by="VY", ascending=False)
goods_sum["ratio"] = goods_sum["VY"]/goods_sum["VY"].sum()


# In[ ]:


# plotly pie plot
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

labels = goods_sum["goods"]
values = goods_sum["ratio"]*100

layout = go.Layout(
    title='Value of trading goods with all countries'
)

data = go.Pie(labels=labels, values=values)

fig = go.Figure(data=[data], layout=layout)
iplot(fig, filename='ratioofcom.html')


# The top three kinds of trading Machienery / Ellectircal, Transporation and Mineral Products. For both import and export

# Let's see what Japan trade with China

# In[ ]:


# create china data
china_data = year_data.ix[year_data.Country_eng=="People's_Republic_of_China",:]
china_sum = china_data.groupby("goods")["VY"].sum().reset_index().sort_values(by="VY", ascending=False)
china_sum["ratio"] = china_sum["VY"]/china_sum["VY"].sum()


# In[ ]:


# plotly pie plot
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

labels = china_sum["goods"]
values = china_sum["ratio"]*100

layout = go.Layout(
    title='Ratio of Commodities with China'
)

data = go.Pie(labels=labels, values=values)

fig = go.Figure(data=[data], layout=layout)
iplot(fig, filename='ratioofcomchina.html')


# We can see that the Michinery / Electrical is the dominant part in the trading between China and Japan, followed by Textiles and Miscellaneous. We do not see the Transportation and Mineral Products appearing at the top.

# How did the trading structure with China change from 1988 to 2015?

# In[ ]:


china_year= china_data.groupby(["Year","goods"])["VY"].sum().reset_index()
goods_list = np.unique(china_year["goods"])

# offset for annotation
goods_offset = {'Animal & Animal Products':-0.08, 'Chemicals & Allied Industries':-0.01,
       'Foodstuffs':0.03, 'Footwear / Headgear':-0.005, 'Machinery / Electrical':0,
       'Metals':0.01, 'Mineral Products':-0.03, 'Miscellaneous':0, 'Plastics / Rubbers':-0.005,
       'Raw Hides, Skins, Leather, & Furs':-0.075, 'Stone / Glass':0.013, 'Textiles':0,
       'Transportation':0.01, 'Vegetable Products':-0.05, 'Wood & Wood Products':-0.03}


# In[ ]:


# These are the colors that will be used in the plot
fig, ax = plt.subplots(1, 1, figsize=(12, 14));
for rank, good in enumerate(goods_list):
    goodsplot = china_year.ix[china_year.goods==good,:]
    plt.plot(goodsplot.Year,
                    goodsplot.VY,
                    lw=2.5,
                    color=color_sequence[rank])
    y_pos = goodsplot["VY"].tail(1) + goods_offset[good]*10**10
    plt.text(2015.5, y_pos, good, fontsize=13, color=color_sequence[rank])
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.tick_params(axis='y', labelsize=15)
plt.xlim([1988,2015]);
plt.xticks(range(1988, 2015, 2), fontsize=14);
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3);
plt.title("Japanese yearly trading structure with China over 1988 - 2015", size=20);
fig.savefig('jap_china_trade.png') 


# Since 1990, we start to see the increase of the total amount of trade between China and Japan and the driving force behind this is the increase of Machinary / Electrical. Between 2000 and 2008, we witness the golden period for Machinary / Electrical trading where the speed of increase is extremely higher than before. <br>
# We also notice a sudden increase of Mineral Products at 2008. What happened then?<br>
# The high rank of Miscellaneous and Textiles products also remind me of the "Made in China" phenomenon after 2000 in the world.

# ### 5.Difference when trade with different continents

# We have seen that Japan will trade various structures of goods with different countries or areas. So we want to know how did it do that. Does Japan trade with Middle East countires more Mineral products?

# In[ ]:


goods_area_data = (year_data.groupby(["Area", "goods"])["VY"].sum().unstack().fillna(0)/10**10).round(2)


# In[ ]:


trace = go.Heatmap(z= goods_area_data.values,
                   y= goods_area_data.index.values,
                   x= goods_area_data.columns)
layout = go.Layout(
    autosize=False,
    width=900,
    height=600,
    margin=go.Margin(
        l=200,
        r=50,
        b=200,
        t=100,
        pad=4
    ),
    title="Japanese trade with different areas",
    annotations=[
        dict(
            x=14,
            y=10,
            xref='x',
            yref='y',
            text='Unit - billion',
            showarrow=False,
            arrowhead=7,
            ax=0,
            ay=-40
        )
        ]
)

fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename='heatmap')


# We can get 3 conclusions from this heat map:<br>
# 
#     1.Japan have most of its Machinery / Electrical products sold at Asia, 
#         North America and Western Europe. Asia is the main market.
#     2. Japan relies heavily on the Middle East countries for the Mineral Products
#     3. Japan has little trade in farm produce but much trade in commodities with 
#          high value-added products
