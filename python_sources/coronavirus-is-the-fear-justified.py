#!/usr/bin/env python
# coding: utf-8

# # Coronavirus: is the fear justified?
# 
# This beginning of 2020 has been marked by the outbreak of a coronavirus epidemic in the industrial city of Wuhan, China. The public reaction to this news has inflated worldwide, with most flights to China cancelled, significant falls in stock markets and massive shutdowns of companies in China.
# 
# I am not an epidemiologist and it is hard for me to judge whether these measures are appropiate or not. The point of this notebook is to have a naive look at the actual numbers and use them to raise new questions about the current situation.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.dates import DateFormatter
import seaborn as sns; sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# importing data
data = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")
# other DB 
# data = pd.read_csv('../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200206.csv')


# round up data at the day level
data['Day'] = pd.to_datetime(data['Date']).dt.strftime('%Y/%m/%d')
data.drop(columns=['Last Update','Date'], inplace=True)

# If no data for the provice, replace by the whole country
rows = data[data['Province/State'].isnull()]['Province/State'].index 
data.loc[rows,'Province/State'] = data.loc[rows,'Country']

# get list of days in data set
days_list = list(data.sort_values('Day')['Day'].unique())
provinces_list = list(data['Province/State'].unique())


data.sort_values('Day').tail()


# In[ ]:


# data cleaning
# group by day and Provice; take the max of a given day (sometimes the day is updated several times on the same day)
new_data = data.groupby(['Day','Province/State']).agg('max')
# put 'Province/State' back in feature columns
new_data.reset_index(level=['Day','Province/State'], col_level=1, inplace=True)


# In[ ]:


# visualize grographical differences today

last_day = new_data['Day'].max()

# total cases today
total_confirmed = new_data[new_data['Day']==last_day]['Confirmed'].sum()

# get regional differences
region_data = new_data[new_data['Day']==last_day].groupby(['Province/State']).agg('sum')

region_data['Fraction of infections'] = region_data['Confirmed']/total_confirmed
region_data['Mortality rate'] = region_data['Deaths']/region_data['Confirmed']


region_data_top15 = region_data.sort_values('Confirmed', ascending=False).head(15)


# In[ ]:


import folium
import json
import branca.colormap as cm
latitude = 30.86166
longitude = 114.195397

china_provinces = '/kaggle/input/china-regions-map/china-provinces.json'

china_confirmed_colorscale = cm.linear.YlOrRd_09.scale(0, total_confirmed/20)
china_confirmed_series = region_data_top15['Confirmed']

def confirmed_style_function(feature):
    china_show = china_confirmed_series.get(str(feature['properties']['NAME_1']), None)
    return {
        'fillOpacity': 0.6,
        'weight': 0,
        'fillColor': 'white' if china_show is None else china_confirmed_colorscale(china_show)
    }

china_confirmed_map = folium.Map(location=[35.86, 104.19], zoom_start=4)

folium.TopoJson(
    json.load(open(china_provinces)),
    'objects.CHN_adm1',
    style_function=confirmed_style_function
).add_to(china_confirmed_map)

# localize Wuhan
folium.Marker(
    location=[30.5928, 114.3055],
    popup = folium.Popup('Wuhan.', show=True),
).add_to(china_confirmed_map)

#put color scale
china_confirmed_colorscale.caption = 'Confirmed cases'
china_confirmed_colorscale.add_to(china_confirmed_map)


china_confirmed_map


# The map shows the regions impacted. The source of the outbreak happened in Wuhan in the Hubei province. We will see that this region has by far the highest number of cases (the color or Hubei above is saturated in the scale chosen).

# ## TOC:
# * [1. Global Trends](#global_trends)
# * [2. Situation in China](#china)
# * [3. Situation outside China](#outside_china)
# * [4. Risk for new outbreaks](#new_sources)
# * [Conclusions](#conclusion)

# # 1. Global trends <a class="anchor" id="global_trends"></a>
# 
# Let's have a look at the trend on a worldwide scale. As can be seen below, there has been a pretty steep increase in the cases since the outbreak of the epidemy in the end of January. However, the situation seems to get in control from the lower graph, as the daily number of new cases has been decreasing in relative numbers (the spike on February 12 is related to a change in the methodology of defining when a patient is considered a positive case). The new daily infections has even decreased in absolute numbers in recent days.

# In[ ]:


# check global cases (all regions together) 
global_data = new_data.groupby(['Day']).agg('sum')

# build the figure
fig = plt.figure("", figsize=(12, 12))

ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

sns.barplot(x=global_data.index, y=global_data['Confirmed'], ci=None, ax=ax1)
ax1.set_title('Total confirmed cases')

#new_data['Confirmed'].plot.bar(stacked=True)
sns.barplot(x=global_data.index, y=global_data['Confirmed'].pct_change()*100, ci=None, ax=ax2)
# assign locator and formatter for the xaxis ticks.


# tilt the labels since they tend to be too long
fig.autofmt_xdate()
ax2.set_title('Relative increase [%]')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_ylabel('Increase relative to previous day [%]')
plt.show()


# # 2. Situation in China <a class="anchor" id="china"></a>
# 
# Geographically, the epidemic remains pretty localized to the Hubei province, which accounts for more than 80% of the total cases. The Hubei province being an industrial powerhouse of China, it is not surprising to see the region of Guangdong, China's main export hub, coming second. Notice that, at this point, we are talking of very small numbers of around a thousand infected cases (out of a polulation of more than 100 million). The neighbouring provinces of Hubei have equivalent numbers of cases to the Guangdong province. This can also be visualized in the map at the beginning of this notebook.
# 
# The mortality rate related to the coronavirus seems to have stabilized below 3% in Hubei and is much lower in any other region, suggesting that the actual rate will probably be revised to a lower value by the end of the epidemics. Such a mortality rate is rather low compared to other epidemics as SRAS or Ebola. Notice that some provinces as Zhejiang have reported no fatality so far out of 1000 cases, which is surprising. Either the cases in this province are all recent infections and we might expect some deaths in the coming days, either the data have not been transmitted yet.

# In[ ]:


region_data_top15.style.set_properties(**{'text-align': 'left'})
region_data_top15.drop(['Sno'], axis=1).style.format({'Mortality rate': '{:,.2%}'.format, 'Fraction of infections': '{:,.2%}'.format})


# We can visualize the cases per province to highlight how the epidemic is so far a problem in Hubei only.

# In[ ]:




fig = plt.figure("", figsize=(14, 14))

sns.set(font_scale=1.1)

ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

palette1 = sns.color_palette("Paired", 8)
sns.barplot(y=region_data_top15.index, x='Confirmed', 
            data=region_data_top15, ax=ax1,
            palette = palette1)
ax1.set_title('First 15 region impacted by the coronavirus',size=16)


# exclude Hube2 = palette1
palette2 = palette1[1:] + palette1[:1] # move color representation by 1
sns.barplot(y=region_data_top15.drop(index=['Hubei']).index, x='Confirmed', 
            data=region_data_top15.drop(index=['Hubei']), ax=ax2,
           palette = palette2)
ax2.set_title('Zoom without Hubei province',size=16);


# # 3. Situation outside China <a class="anchor" id="outside_china"></a>
# 
# The virus has not spread outside China so far, except for a few travellers, and numbers are totally under control, including neigbouring countries like Taiwan, Japan or Thailand. It is striking to see that there are far more infected people on the Diamond Prince cruise ship than in any other region worldwide, which means that current measures have been very effective this far. 

# In[ ]:


# all data with 'Others' label for the country turn out to be the cruise ship, replacing them
rows = data[data['Country']=='Others'].index
data.loc[rows,'Country'] = 'Diamond Princess cruise ship'

# remove China
data_other_countries = data[(data['Country'] != 'Mainland China') & (data['Country'] != 'China')]

# Get all country
data_other_countries = data_other_countries[data_other_countries['Day']==last_day].groupby(['Country']).agg('sum')
data_other_countries.drop(['Sno'], axis=1).sort_values('Confirmed', ascending=False).head(15)


# # 4. What is the risk for new outbreaks? <a class="anchor" id="new_sources"></a>
# 
# Besides the situation in Hunan which seems to be improving, the main fear of the global community is that new outbreaks elsewhere, which could complicate the fight against the virus as some medical devices like masks are now in short supply globally. 
# 
# To assess this on a data point of view, let us look at the speed at which the virus is spreading compared to the first outbreak in Hubei. To be rigorous, numbers should be normalized by the population of the regions and density should be taken into account. We neglect this for the time being and will compare the order of magnitudes only.

# In[ ]:


# get the list of the first 15 provinces
top_15_provinces = region_data_top15.index.to_list()

# convert Day back to date format
new_data['Day'] = pd.to_datetime(new_data['Day'], infer_datetime_format=True) 


fig = plt.figure("", figsize=(14, 16))

sns.set(font_scale=1.1)

ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

for province in top_15_provinces:
    
    # get days after which the number of cases got over 400
    if max(new_data[(new_data['Province/State'] == province)]['Confirmed']) > 400 :
        date_start = new_data[(new_data['Province/State'] == province) & (new_data['Confirmed'] > 400)]['Day'].iloc[0]
        realtive_days = (new_data[new_data['Province/State'] == province][['Day']]  - date_start).values.astype('timedelta64[D]')
        ax1.plot(realtive_days, new_data[new_data['Province/State'] == province][['Confirmed']])
        
        if province != 'Hubei' and province != 'Diamond Princess cruise ship':
            ax2.plot(realtive_days, new_data[new_data['Province/State'] == province][['Confirmed']].pct_change()*100, linestyle='--', marker = '*')
        else:
            ax2.plot(realtive_days, new_data[new_data['Province/State'] == province][['Confirmed']].pct_change()*100, linestyle='--', marker = '*', linewidth=3)
        
        
        
ax1.set_xlim([-5,15])    
ax1.set_ylim([0,3000])
ax1.legend(top_15_provinces)
ax1.set_xlabel('Days after more than 400 cases have been confirmed')
ax1.set_ylabel('confirmed cases')
ax1.set_title('Absolute numbers',size=16)

ax2.set_xlim([-5,15])    
ax2.set_ylim([0,50])
ax2.legend(top_15_provinces)
ax2.set_xlabel('Days after more than 400 cases have been confirmed')
ax2.set_ylabel('confirmed cases')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_ylabel('Increase relative to previous day [%]')
ax2.set_title('Daily increase',size=16);


# As can be seen from the graphs above, the spread of the virus in the other regions is nowhere near that in Hubei. The daily rate of new infections has fallen safely below 10% in all regions. One exception is the cruise boat (visible in bold on the left side of the graph since the total number is still lower than 400), which is a very specific case that is not relevant for comparing with entire regions (it is probably very interesting for physicians to better understand the propagation mechanisms). It remains a warning that the virus can spread quickly in closed environments where people interact much with each other, and some places could be at risk (hospitals, prisons, schools, some businesses).

# # Conclusions <a class="anchor" id="conclusion"></a>
# 
# At this point (February 16, 2020), numbers seems to show that most of the coronavirus epidemic has remained confined to the Hubei province. This means that measures to contain the virus have been efficient or (and) that the virus does not spread as fast as initially feared.
# 
# Actually, we could ask ourselves whether these measures were not exaggerated. On a Chinese level, millions of people have been confined in Wuhan and its surroundings and most travel is limited within a significant part of the country. This has had a big impact on people's lives and the economy, with most businesses running in slow mode. Seen from the outside, we might wonder if the tremendous effort to fight the epidemic is a reasonable allocation of resources in a country where basic healthcare is patchy and air pollution is responsible of around 1 million deaths every year ([see here](https://www.scmp.com/news/china/science/article/2166542/air-pollution-killing-1-million-people-and-costing-chinese)). It is possible that Chinese leaders have felt a weakening in their authority over the last year, among riots in Hong Kong and trade fights with USA, and are willing to pick a visible topic as the virus to show their effectiveness to the public (if we forget the first weeks of the epidemic, when doctors have been arrested under "fake news" accusations). It is much easier to gain credit from stopping an epidemic than from a general upscale of health and environmental conditions. This is not to minimize the current efforts though, and we should also be very grateful to doctors and leaders for measures that have prevented a global pandemic. At this point, they seem to be winning a difficult battle.
# 
# On an international level, there has clearly been some overreaction too. Many countries have stopped most of their flights even to Chinese regions that are mostly free from the virus as Beijing or Shanghai and have even repatriated their citizens. It is hard to ignore the political situation in these decisions that seem to stem from a lack of trust towards China, which has probably grown in recent years. On a positive side, scientists have collaborated globally, with Chinese groups sharing the genome of the virus from the start in order to boost research towards a vaccine. Let us hope that such collaborations show the world that humanity gains when people collaborate beyond borders and narrow-minded nationalism.
# 
# Coming back to the data, the next step would be to enrich the analysis by comparing with other data sources. It would be interesting to put the current epidemic in perspective with other ones (seasonal flu, SRAS, ebola). Also, the major slowdown of the Chinese activity during the last few weeks is a good opportunity to study its impact on other parameters (air pollution, travel patterns, etc.), as discussed in this interesting [article](https://www.carbonbrief.org/analysis-coronavirus-has-temporarily-reduced-chinas-co2-emissions-by-a-quarter). 
