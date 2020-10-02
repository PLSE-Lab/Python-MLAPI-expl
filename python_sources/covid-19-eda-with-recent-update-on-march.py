#!/usr/bin/env python
# coding: utf-8

# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F518134%2F485aa04e87e4e45c91815101784c6d95%2Fcorona-4930541_1280.jpg?generation=1585438527494582&alt=media)
# 
# # COVID-19: EDA with recent update on March
# 
# The kernel is inspired by the great EDA kernel [COVID-19: Digging a Bit Deeper](https://www.kaggle.com/abhinand05/covid-19-digging-a-bit-deeper) by @abhinand05 in week1.<br/>
# Please upvote his kernel as well :)<br/>
# I will write an EDA including the recent updates.
# 
# [Note] It seems JHU has changed the data format, and stopped providing recovered cases. So I could not analyze recovered cases.
# 
# 
# ### Version History
# 
# The data is updated and hence the figure is updated everyday. Below are the version history to see the information until specified date.
# 
#  - Version 15: Added **Daily NEW confirmed cases** analysis & **Asia** region EDA.
#  - [Version 18](https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-march?scriptVersionId=31151381): Shows figure as of 2020/3/28.
#  - Version 19: Added <span style="color:red">**sigmoid fitting to estimate when the coronavirus converge in each country**</span>, jump to [When will it converge? - Estimation by sigmoid fitting](#id_converge).

# ## Table of Contents
# 
# 
# **[Load Data](#id_load)**<br/>
# **[Worldwide trend](#id_ww)**<br/>
# **[Country-wise growth](#id_country)**<br/>
# **[Going into province](#id_province)**<br/>
# **[Zoom up to US: what is happening in US now??](#id_province)**<br/>
# **[Europe](#id_europe)**<br/>
# **[Asia](#id_asia)**<br/>
# **[Which country is recovering now?](#id_recover)**<br/>
# **[When will it converge? - Estimation by sigmoid fitting](#id_converge)**<br/>
# **[Further reading](#id_ref)**<br/>

# In[ ]:


import gc
import os
from pathlib import Path
import random
import sys

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import scipy as sp


import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.templates.default = "plotly_dark"

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# --- setup ---
pd.set_option('max_columns', 50)


# <a id="id_load"></a>
# # Load Data
# 
# Load data and convert Japanese columns into English so that others can understand :).

# In[ ]:


# Input data files are available in the "../input/" directory.
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    filenames.sort()
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().run_cell_magic('time', '', "datadir = Path('/kaggle/input/covid19-global-forecasting-week-2')\n\n# Read in the data CSV files\ntrain = pd.read_csv(datadir/'train.csv')\ntest = pd.read_csv(datadir/'test.csv')\nsubmission = pd.read_csv(datadir/'submission.csv')")


# In[ ]:


train


# In[ ]:


test


# In[ ]:


submission


# In[ ]:


train.rename({'Country_Region': 'country', 'Province_State': 'province', 'Id': 'id', 'Date': 'date', 'ConfirmedCases': 'confirmed', 'Fatalities': 'fatalities'}, axis=1, inplace=True)
test.rename({'Country_Region': 'country', 'Province_State': 'province', 'Id': 'id', 'Date': 'date', 'ConfirmedCases': 'confirmed', 'Fatalities': 'fatalities'}, axis=1, inplace=True)
train['country_province'] = train['country'].fillna('') + '/' + train['province'].fillna('')
test['country_province'] = test['country'].fillna('') + '/' + test['province'].fillna('')


# <a id="id_ww"></a>
# # Worldwide trend

# In[ ]:


ww_df = train.groupby('date')[['confirmed', 'fatalities']].sum().reset_index()
ww_df['new_case'] = ww_df['confirmed'] - ww_df['confirmed'].shift(1)
ww_df.tail()


# In[ ]:


ww_melt_df = pd.melt(ww_df, id_vars=['date'], value_vars=['confirmed', 'fatalities', 'new_case'])
ww_melt_df


# When we see the confirmed cases in world wide, it just look like exponential growth curve. The number is increasing very rapidly especially recently. **the number almost doubled in last 1 week**...
# Confirmed cases reached 593K people, and **27K people already died at March 27**.

# In[ ]:


fig = px.line(ww_melt_df, x="date", y="value", color='variable', 
              title="Worldwide Confirmed/Death Cases Over Time")
fig.show()


# Moreover, when we check the growth in log-scale below figure, we can see that the speed of confirmed cases growth rate **slightly increases** when compared with the beginning of March and end of March.<br/>
# In spite of the Lockdown policy in Europe or US, the number is still increasing rapidly.

# In[ ]:


fig = px.line(ww_melt_df, x="date", y="value", color='variable',
              title="Worldwide Confirmed/Death Cases Over Time (Log scale)",
             log_y=True)
fig.show()


# It looks like `fatalities` curve is just shifted the `confirmed` curve to below in log-scale, which means mortality rate is almost constant.
# 
# Is it true? Let's see mortality rate in detail.<br/>
# We see that mortality rate is kept almost 3%, however it is slightly **increasing recently to go over 4%** at the end of March.
# 
# Why? I will show you later that Europe & US has more seriously infected by Coronavirus recently, and mortality rate is high in these regions.

# In[ ]:


ww_df['mortality'] = ww_df['fatalities'] / ww_df['confirmed']

fig = px.line(ww_df, x="date", y="mortality", 
              title="Worldwide Mortality Rate Over Time")
fig.show()


# <a id="id_country"></a>
# # Country-wise growth

# In[ ]:


country_df = train.groupby(['date', 'country'])[['confirmed', 'fatalities']].sum().reset_index()
country_df.tail()


# What kind of country is in the dataset? How's the distribution of number of confirmed cases by country?

# In[ ]:


countries = country_df['country'].unique()
print(f'{len(countries)} countries are in dataset:\n{countries}')


# In[ ]:


target_date = country_df['date'].max()

print('Date: ', target_date)
for i in [1, 10, 100, 1000, 10000]:
    n_countries = len(country_df.query('(date == @target_date) & confirmed > @i'))
    print(f'{n_countries} countries have more than {i} confirmed cases')


# In[ ]:


ax = sns.distplot(np.log10(country_df.query('date == "2020-03-27"')['confirmed'] + 1))
ax.set_xlim([0, 6])
ax.set_xticks(np.arange(7))
_ = ax.set_xticklabels(['0', '10', '100', '1k', '10k', '100k'])


# It is difficult to see all countries so let's check top countries.

# In[ ]:


top_country_df = country_df.query('(date == @target_date) & (confirmed > 1000)').sort_values('confirmed', ascending=False)
top_country_melt_df = pd.melt(top_country_df, id_vars='country', value_vars=['confirmed', 'fatalities'])


# Now US and Italy has more confirmed cases than China, and we can see many Europe countries in the top.
# 
# Korea also appears in relatively top despite of its population, this is because Korea execcutes inspection check aggressively.

# In[ ]:


fig = px.bar(top_country_melt_df.iloc[::-1],
             x='value', y='country', color='variable', barmode='group',
             title=f'Confirmed Cases/Deaths on {target_date}', text='value', height=1500, orientation='h')
fig.show()


# Let's check these major country's growth by date.
# 
# As we can see, Coronavirus hit China at first but its trend is slowing down in March which is good news.<br/>
# Bad news is 2nd wave comes to Europe (Italy, Spain, Germany, France, UK) at March.<br/>
# But more sadly 3rd wave now comes to **US, whose growth rate is much much faster than China, or even Europe**. Its main spread starts from middle of March and its speed is faster than Italy. Now US seems to be in the most serious situation in terms of both total number and spread speed.<br/>
# 

# In[ ]:


top30_countries = top_country_df.sort_values('confirmed', ascending=False).iloc[:30]['country'].unique()
top30_countries_df = country_df[country_df['country'].isin(top30_countries)]
fig = px.line(top30_countries_df,
              x='date', y='confirmed', color='country',
              title=f'Confirmed Cases for top 30 country as of {target_date}')
fig.show()


# In terms of number of fatalities, Europe is more serious than US now.

# In[ ]:


top30_countries = top_country_df.sort_values('fatalities', ascending=False).iloc[:30]['country'].unique()
top30_countries_df = country_df[country_df['country'].isin(top30_countries)]
fig = px.line(top30_countries_df,
              x='date', y='fatalities', color='country',
              title=f'Fatalities for top 30 country as of {target_date}')
fig.show()


# Now let's see mortality rate by country

# In[ ]:


top_country_df = country_df.query('(date == @target_date) & (confirmed > 100)')
top_country_df['mortality_rate'] = top_country_df['fatalities'] / top_country_df['confirmed']
top_country_df = top_country_df.sort_values('mortality_rate', ascending=False)


# Italy is the most serious situation, whose mortality rate is over 10% as of 2020/3/28.<br/>
# We can also find countries from all over the world when we see top mortality rate countries.<br/>
# Iran/Iraq from Middle East, Phillipines & Indonesia from tropical areas.<br/>
# Spain, Netherlands, France, and UK form Europe etc. It shows this coronavirus is really world wide pandemic.
# 
# [UPDATE]: According to the comment by @elettra84, 10% of Italy is due to extreme outlier of Lombardy cluster. Except that mortality rate in Italy is comparable to other country. Refer [Lombardy cluster in wikipedia](https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_Italy#Lombardy_cluster).

# In[ ]:


fig = px.bar(top_country_df[:30].iloc[::-1],
             x='mortality_rate', y='country',
             title=f'Mortality rate HIGH: top 30 countries on {target_date}', text='mortality_rate', height=800, orientation='h')
fig.show()


# How about the countries whose mortality rate is low?
# 
# By investigating the difference between above & below countries, we might be able to figure out what is the cause which leads death.<br/>
# Be careful that there may be a case that these country's mortality rate is low due to these country does not report/measure fatality cases properly.

# In[ ]:


fig = px.bar(top_country_df[-30:],
             x='mortality_rate', y='country',
             title=f'Mortality rate LOW: top 30 countries on {target_date}', text='mortality_rate', height=800, orientation='h')
fig.show()


# Let's see number of confirmed cases on map. Again we can see Europe, US, MiddleEast (Turkey, Iran) and Asia (China, Korea) are red.

# In[ ]:


all_country_df = country_df.query('date == @target_date')


# In[ ]:


all_country_df['confirmed_log1p'] = np.log10(all_country_df['confirmed'] + 1)
fig = px.choropleth(all_country_df, locations="country", 
                    locationmode='country names', color="confirmed_log1p", 
                    hover_name="country", hover_data=["confirmed"],
                    range_color=[all_country_df['confirmed_log1p'].min(), all_country_df['confirmed_log1p'].max()], 
                    color_continuous_scale="peach", 
                    title='Countries with Confirmed Cases')

# I'd like to update colorbar to show raw values, but this does not work somehow...
# Please let me know if you know how to do this!!
trace1 = list(fig.select_traces())[0]
trace1.colorbar = go.choropleth.ColorBar(
    tickvals=[0, 1, 2, 3, 4, 5],
    ticktext=['1', '10', '100', '1000','10000', '10000'])
fig.show()


# When we see mortality rate on map, we see Europe (especaiily Italy) is high. Also we notice MiddleEast (Iran, Iraq) is high.
# 
# When we see tropical area, I wonder why Phillipines and Indonesia are high while other countries (Malaysia, Thai, Vietnam, as well as Australia) are low.
# 
# For Asian region, Korea's mortality rate is lower than China or Japan, I guess this is due to the fact that number of inspection is quite many in Korea.
# Please refer these blogs for detail:
# 
#  - [South Korea launches 'drive-thru' coronavirus testing facilities as demand soars](https://www.japantimes.co.jp/news/2020/03/01/asia-pacific/science-health-asia-pacific/south-korea-drive-thru-coronavirus/#.XoAmw4j7RPY)
#  - [Coronavirus: Why Japan tested so few people](https://asia.nikkei.com/Spotlight/Coronavirus/Coronavirus-Why-Japan-tested-so-few-people)

# In[ ]:


all_country_df['fatalities_log1p'] = np.log10(all_country_df['fatalities'] + 1)

fig = px.choropleth(all_country_df, locations="country", 
                    locationmode='country names', color="fatalities_log1p", 
                    hover_name="country", range_color=[0, 4], 
                    color_continuous_scale="peach", 
                    title='Countries with fatalities')
fig.show()


# In[ ]:


all_country_df['mortality_rate'] = all_country_df['fatalities'] / all_country_df['confirmed']
fig = px.choropleth(all_country_df, locations="country", 
                    locationmode='country names', color="mortality_rate", 
                    hover_name="country", range_color=[0, 0.12], 
                    color_continuous_scale="peach", 
                    title='Countries with mortality rate')
fig.show()


# Why mortality rate is different among country? What kind of hint is hidden in this map? Especially mortality rate is high in Europe and US, is there some reasons?
# 
# There is one interesting hypothesis that BCG vaccination<br/>
# The below figure shows BCG vaccination policy by country: Advanced countries like Europe & US, especially Italy and US does not take BCG vaccination. We can notice this map is indeed similar to mortality rate map above. Is it just accidental?
# 
# ![](https://www.researchgate.net/profile/Alice_Zwerling/publication/50892386/figure/fig2/AS:277209752326147@1443103363144/Map-displaying-BCG-vaccination-policy-by-country-A-The-country-currently-has-universal.png)
# 
# Reference: [If I were North American/West European/Australian, I will take BCG vaccination now against the novel coronavirus pandemic.](https://www.jsatonotes.com/2020/03/if-i-were-north-americaneuropeanaustral.html)
# 
#  - [Australia's Trialing a TB Vaccine Against COVID-19, And Health Workers Get It First](https://www.sciencealert.com/australia-is-trialling-a-tb-vaccine-for-coronavirus-and-health-workers-get-it-first)
# 
# Of course this is just one hypothesis but we can notice/find some hints to tackle Coronavirus like this by carefully analyzing/comparing the data.

# The figure showing fatality growth since 10 deaths.
#  - Ref: [COVID-19 Deaths Per Capita](https://covid19dashboards.com/covid-compare-permillion/)

# In[ ]:


n_countries = 20
n_start_death = 10
fatality_top_countires = top_country_df.sort_values('fatalities', ascending=False).iloc[:n_countries]['country'].values
country_df['date'] = pd.to_datetime(country_df['date'])


df_list = []
for country in fatality_top_countires:
    this_country_df = country_df.query('country == @country')
    start_date = this_country_df.query('fatalities > @n_start_death')['date'].min()
    this_country_df = this_country_df.query('date >= @start_date')
    this_country_df['date_since'] = this_country_df['date'] - start_date
    this_country_df['fatalities_log1p'] = np.log10(this_country_df['fatalities'] + 1)
    this_country_df['fatalities_log1p'] -= this_country_df['fatalities_log1p'].values[0]
    df_list.append(this_country_df)

tmpdf = pd.concat(df_list)
tmpdf['date_since_days'] = tmpdf['date_since'] / pd.Timedelta('1 days')


# In[ ]:


fig = px.line(tmpdf,
              x='date_since_days', y='fatalities_log1p', color='country',
              title=f'Fatalities by country since 10 deaths, as of {target_date}')
fig.add_trace(go.Scatter(x=[0, 21], y=[0, 3], name='Double by 7 days', line=dict(dash='dash', color=('rgb(200, 200, 200)'))))
fig.add_trace(go.Scatter(x=[0, 42], y=[0, 3], name='Double by 14 days', line=dict(dash='dash', color=('rgb(200, 200, 200)'))))
fig.add_trace(go.Scatter(x=[0, 63], y=[0, 3], name='Double by 21 days', line=dict(dash='dash', color=('rgb(200, 200, 200)'))))
fig.show()


# ## Daily NEW confirmed cases trend
# 
# How about **DAILY new cases** trend?<br/>
# We find from below figure:
#  - China has finished its peak at Feb 14, new confirmed cases are surpressed now.
#  - Europe&US spread starts on mid of March, after China slows down.
#  - Current US new confirmed cases are the worst speed, recording worst speed at 20k people/day.

# In[ ]:


country_df['prev_confirmed'] = country_df.groupby('country')['confirmed'].shift(1)
country_df['new_case'] = country_df['confirmed'] - country_df['prev_confirmed']
country_df['new_case'].fillna(0, inplace=True)
top30_country_df = country_df[country_df['country'].isin(top30_countries)]

fig = px.line(top30_country_df,
              x='date', y='new_case', color='country',
              title=f'DAILY NEW Confirmed cases world wide')
fig.show()


# In[ ]:


country_df['date'] = country_df['date'].apply(str)
country_df['confirmed_log1p'] = np.log1p(country_df['confirmed'])
country_df['fatalities_log1p'] = np.log1p(country_df['fatalities'])

fig = px.scatter_geo(country_df, locations="country", locationmode='country names', 
                     color="confirmed", size='confirmed', hover_name="country", 
                     hover_data=['confirmed', 'fatalities'],
                     range_color= [0, country_df['confirmed'].max()], 
                     projection="natural earth", animation_frame="date", 
                     title='COVID-19: Confirmed cases spread Over Time', color_continuous_scale="portland")
# fig.update(layout_coloraxis_showscale=False)
fig.show()


# In[ ]:


fig = px.scatter_geo(country_df, locations="country", locationmode='country names', 
                     color="fatalities", size='fatalities', hover_name="country", 
                     hover_data=['confirmed', 'fatalities'],
                     range_color= [0, country_df['confirmed'].max()], 
                     projection="natural earth", animation_frame="date", 
                     title='COVID-19: Fatalities growth Over Time', color_continuous_scale="portland")
fig.show()


# In[ ]:


country_df.loc[country_df['new_case'] < 0, 'new_case'] = 0.
fig = px.scatter_geo(country_df, locations="country", locationmode='country names', 
                     color="new_case", size='new_case', hover_name="country", 
                     hover_data=['confirmed', 'fatalities'],
                     range_color= [0, country_df['new_case'].max()], 
                     projection="natural earth", animation_frame="date", 
                     title='COVID-19: Daily NEW cases over Time', color_continuous_scale="portland")
fig.show()


# <a id="id_province"></a>
# # Going into province

# How many country has precise province information?<br/>
# It seems it's 8 countries: Australia, Canada, China, Denmark, France, Netherlands, US, and UK.

# In[ ]:


for country in countries:
    province = train.query('country == @country')['province'].unique()
    if len(province) > 1:       
        print(f'Country {country} has {len(province)} provinces: {province}')


# <a id="id_us"></a>
# # Zoom up to US: what is happening in US now??
# 
# As we can see, the spread is fastest in US now, at the end of March. Let's see in detail what is going on in US.

# In[ ]:


usa_state_code_df = pd.read_csv('/kaggle/input/usa-state-code/usa_states2.csv')


# In[ ]:


# Prepare data frame only for US. 

train_us = train.query('country == "US"')
train_us['mortality_rate'] = train_us['fatalities'] / train_us['confirmed']

# Convert province column to its 2-char code name,
state_name_to_code = dict(zip(usa_state_code_df['state_name'], usa_state_code_df['state_code']))
train_us['province_code'] = train_us['province'].map(state_name_to_code)

# Only show latest days.
train_us_latest = train_us.query('date == @target_date')


# When we see inside of the US, we can see **only New York, and its neighbor New Jersey** dominates its spread and are in serious situation.
# 
# New York confirmed cases is over 50k, while other states are less than about 5k confirmed cases.

# In[ ]:


fig = px.choropleth(train_us_latest, locations='province_code', locationmode="USA-states",
                    color='confirmed', scope="usa", hover_data=['province', 'fatalities', 'mortality_rate'],
                    title=f'Confirmed cases in US on {target_date}')
fig.show()


# Mortality rate in New York seems not high, around 1% for now.

# In[ ]:


train_us_latest.sort_values('confirmed', ascending=False)


# In[ ]:


fig = px.choropleth(train_us_latest, locations='province_code', locationmode="USA-states",
                    color='mortality_rate', scope="usa", hover_data=['province', 'fatalities', 'mortality_rate'],
                    title=f'Mortality rate in US on {target_date}')
fig.show()


# **Daily growth**: All state is US got affected from middle of March, and now **growing exponentially**.
# In New York, less than 1k people are confirmed on March 16, but more than 50k people are confirmed on March 30. **50 times explosion in 2 weeks!**

# In[ ]:


train_us_march = train_us.query('date > "2020-03-01"')
fig = px.line(train_us_march,
              x='date', y='confirmed', color='province',
              title=f'Confirmed cases by state in US, as of {target_date}')
fig.show()


# <a id="id_europe"></a>
# # Europe

# In[ ]:


# Ref: https://www.kaggle.com/abhinand05/covid-19-digging-a-bit-deeper
europe_country_list =list([
    'Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland',
    'Italy', 'Latvia','Luxembourg','Lithuania','Malta','Norway','Netherlands','Poland','Portugal','Romania','Slovakia','Slovenia',
    'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus',
    'Albania', 'Bosnia and Herzegovina', 'Kosovo', 'Moldova', 'Montenegro', 'North Macedonia'])

country_df['date'] = pd.to_datetime(country_df['date'])
train_europe = country_df[country_df['country'].isin(europe_country_list)]
#train_europe['date_str'] = pd.to_datetime(train_europe['date'])
train_europe_latest = train_europe.query('date == @target_date')


# When we look into the Europe, its Northern & Eastern areas are relatively better situation compared to Eastern & Southern areas.

# In[ ]:


fig = px.choropleth(train_europe_latest, locations="country", 
                    locationmode='country names', color="confirmed", 
                    hover_name="country", range_color=[1, 50000], 
                    color_continuous_scale='portland', 
                    title=f'European Countries with Confirmed Cases as of {target_date}', scope='europe', height=800)
fig.show()


# Especially **Italy, Spain, German, France** are in more serious situation.

# In[ ]:


train_europe_march = train_europe.query('date > "2020-03-01"')
fig = px.line(train_europe_march,
              x='date', y='confirmed', color='country',
              title=f'Confirmed cases by country in Europe, as of {target_date}')
fig.show()


# In[ ]:


fig = px.line(train_europe_march,
              x='date', y='fatalities', color='country',
              title=f'Fatalities by country in Europe, as of {target_date}')
fig.show()


# When we check daily new cases in Europe, we notice:
# 
#  - Spain & Germany daily growth are more than Italy now, these 2 countries are potentially more dangerous.
#  - Italy new cases are not increasing since March 21, I guess due to lock-down policy is started working. That is not a bad news.

# In[ ]:


train_europe_march['prev_confirmed'] = train_europe_march.groupby('country')['confirmed'].shift(1)
train_europe_march['new_case'] = train_europe_march['confirmed'] - train_europe_march['prev_confirmed']
fig = px.line(train_europe_march,
              x='date', y='new_case', color='country',
              title=f'DAILY NEW Confirmed cases by country in Europe')
fig.show()


# <a id="id_asia"></a>
# # Asia

# In Asia, China & Iran have many confirmed cases, followed by South Korea & Turkey. 

# In[ ]:


country_latest = country_df.query('date == @target_date')

fig = px.choropleth(country_latest, locations="country", 
                    locationmode='country names', color="confirmed", 
                    hover_name="country", range_color=[1, 50000], 
                    color_continuous_scale='portland', 
                    title=f'Asian Countries with Confirmed Cases as of {target_date}', scope='asia', height=800)
fig.show()


# The coronavirus hit Asia in early phase, how is the situation now?<br/>
# China & Korea is already in decreasing phase.<br/>
# However other countries' daily new confirmed cases are still in increasing, especially Iran.

# In[ ]:


top_asian_country_df = top30_country_df[top30_country_df['country'].isin(['China', 'Indonesia', 'Iran', 'Japan', 'Korea, South', 'Malaysia', 'Philippines'])]

fig = px.line(top_asian_country_df,
              x='date', y='new_case', color='country',
              title=f'DAILY NEW Confirmed cases world wide')
fig.show()


# <a id="id_recover"></a>
# # Which country is recovering now?

# We saw that Coronavirus now hits Europe & US, in serious situation. How does it converge?
# 
# We can refer other country where confirmed cases is already decreasing.<br/>
# Here I defined `new_case_peak_to_now_ratio`, as a ratio of current new case and the max new case for each country.<br/>
# If new confirmed case is biggest now, its ratio is 1.
# Its ratio is expected to be low value for the countries where the peak has already finished.  

# In[ ]:


max_confirmed = country_df.groupby('country')['new_case'].max().reset_index()
country_latest = pd.merge(country_latest, max_confirmed.rename({'new_case': 'max_new_case'}, axis=1))
country_latest['new_case_peak_to_now_ratio'] = country_latest['new_case'] / country_latest['max_new_case']


# In[ ]:


recovering_country = country_latest.query('new_case_peak_to_now_ratio < 0.5')
major_recovering_country = recovering_country.query('confirmed > 100')


# The ratio is 0 for the country with very few confirmed cases are reported.<br/>
# I choosed the countries with its confirmed cases more than 100, to see only major countries with the ratio is low.
# 
# We can see:
#  - Middle East coutnries.
#  - South Africa countries.
#  - China & Korea from Asia.

# In[ ]:


fig = px.bar(major_recovering_country.sort_values('new_case_peak_to_now_ratio', ascending=False),
             x='new_case_peak_to_now_ratio', y='country',
             title=f'Mortality rate LOW: top 30 countries on {target_date}', text='new_case_peak_to_now_ratio', height=1000, orientation='h')
fig.show()


# Let's see by map. Yellow countries have high ratio, currently increasing countries. **Blue & purple countries** have low ratio, already decreasing countries from its peak.

# In[ ]:


fig = px.choropleth(country_latest, locations="country", 
                    locationmode='country names', color="new_case_peak_to_now_ratio", 
                    hover_name="country", range_color=[0, 1], 
                    # color_continuous_scale="peach", 
                    hover_data=['confirmed', 'fatalities', 'new_case', 'max_new_case'],
                    title='Countries with new_case_peak_to_now_ratio')
fig.show()


# Let's see some recovering countries.
# 
# ## China
# 
# When we check each state stats, we can see Hubei, the starting place, is extremely large number of confirmed cases.<br/>
# Other states records actually few confirmed cases compared to Hubei.

# In[ ]:


china_df = train.query('country == "China"')
china_df['prev_confirmed'] = china_df.groupby('country')['confirmed'].shift(1)
china_df['new_case'] = china_df['confirmed'] - china_df['prev_confirmed']
china_df.loc[china_df['new_case'] < 0, 'new_case'] = 0.


# In[ ]:


fig = px.line(china_df,
              x='date', y='new_case', color='province',
              title=f'DAILY NEW Confirmed cases in China by province')
fig.show()


# ## The situation of Hubei now?
# 
# Hubei record its new case peak on Feb 14. And finally, new case was not found on March 19.
# 
# To become no new case found, it took **about 2month after confirmed cases occured**, and **1 month after the peak has reached.** <br/>
# This term will be the reference for other country to how long we must lock-down the city.

# In[ ]:


china_df.query('(province == "Hubei") & (date > "2020-03-10")')


# <a id="id_converge"></a>
# # When will it converge? - Estimation by sigmoid fitting
# 
# I guess everyone is wondering when the coronavirus converges. Let's estimate it roughly using sigmoid fitting.<br/>
# I referenced below kernels for original ideas.
# 
#  - [Sigmoid per country](https://www.kaggle.com/group16/sigmoid-per-country-no-leakage) by @group16
#  - [COVID-19 growth rates per country](https://www.kaggle.com/mikestubna/covid-19-growth-rates-per-country) by @mikestubna

# In[ ]:


def sigmoid(t, M, beta, alpha, offset=0):
    alpha += offset
    return M / (1 + np.exp(-beta * (t - alpha)))

def error(x, y, params):
    M, beta, alpha = params
    y_pred = sigmoid(x, M, beta, alpha)
    loss_mse = np.mean((y_pred - y) ** 2)
    return loss_mse

def gen_random_color(min_value=0, max_value=256) -> str:
    """Generate random color for plotly"""
    r, g, b = np.random.randint(min_value, max_value, 3)
    return f'rgb({r},{g},{b})'


# In[ ]:


def fit_sigmoid(exclude_days=0):
    target_country_df_list = []
    pred_df_list = []
    for target_country in top30_countries:
        print('target_country', target_country)
        # --- Train ---
        target_country_df = country_df.query('country == @target_country')

        #train_start_date = target_country_df['date'].min()
        train_start_date = target_country_df.query('confirmed > 1000')['date'].min()
        train_end_date = pd.to_datetime(target_date) - pd.Timedelta(f'{exclude_days} days')
        target_date_df = target_country_df.query('(date >= @train_start_date) & (date <= @train_end_date)')
        if len(target_date_df) <= 7:
            print('WARNING: the data is not enough, use 7 more days...')
            train_start_date -= pd.Timedelta('7 days')
            target_date_df = target_country_df.query('(date >= @train_start_date) & (date <= @train_end_date)')

        confirmed = target_date_df['confirmed'].values
        x = np.arange(len(confirmed))

        lossfun = lambda params: error(x, confirmed, params)
        res = sp.optimize.minimize(lossfun, x0=[np.max(confirmed), 0.25, len(confirmed) / 2.], method='nelder-mead')
        M, beta, alpha = res.x
        # sigmoid_models[key] = (M, beta, alpha)
        # np.clip(sigmoid(list(range(len(data), len(data) + steps)), M, beta, alpha), 0, None).astype(int)

        # --- Pred ---
        pred_start_date = target_country_df['date'].min()
        pred_end_date = pd.to_datetime('2020-07-01')
        days = int((pred_end_date - pred_start_date) / pd.Timedelta('1 days'))
        # print('pred start', pred_start_date, 'end', pred_end_date, 'days', days)

        x = np.arange(days)
        offset = (train_start_date - pred_start_date) / pd.Timedelta('1 days')
        print('train_start_date', train_start_date, 'offset', offset)
        y_pred = sigmoid(x, M, beta, alpha, offset=offset)
        # target_country_df['confirmed_pred'] = y_pred

        all_dates = [pred_start_date + np.timedelta64(x, 'D') for x in range(days)]
        pred_df = pd.DataFrame({
            'date': all_dates,
            'country': target_country,
            'confirmed_pred': y_pred,
        })

        target_country_df_list.append(target_country_df)
        pred_df_list.append(pred_df)
    return target_country_df_list, pred_df_list


# In[ ]:


def plot_sigmoid_fitting(target_country_df_list, pred_df_list, title=''):
    n_countries = len(top30_countries)

    # --- Plot ---
    fig = go.Figure()

    for i in range(n_countries):
        target_country = top30_countries[i]
        target_country_df = target_country_df_list[i]
        pred_df = pred_df_list[i]
        color = gen_random_color(min_value=20)
        # Prediction
        fig.add_trace(go.Scatter(
            x=pred_df['date'], y=pred_df['confirmed_pred'],
            name=f'{target_country}_pred',
            line=dict(color=color, dash='dash')
        ))

        # Ground truth
        fig.add_trace(go.Scatter(
            x=target_country_df['date'], y=target_country_df['confirmed'],
            mode='markers', name=f'{target_country}_actual',
            line=dict(color=color),
        ))
    fig.update_layout(
        title=title, xaxis_title='Date', yaxis_title='Confirmed cases')
    fig.show()


# In[ ]:


target_country_df_list, pred_df_list = fit_sigmoid(exclude_days=0)


# In[ ]:


plot_sigmoid_fitting(target_country_df_list, pred_df_list, title='Sigmoid fitting with all latest data')


# If we believe above curve, confirmed cases is slowing down now and it will be converging **around mid of April** in most of the country.<br/>
# Some countries, Iran, Belgium, Sweden and Denmark are expected to converge later, **around May**.
# 
# However I'm not confident how this sigmoid fitting is accurate, it's just an estimation by some modeling.<br/>
# Let's try validation by excluding last 7 days data.

# In[ ]:


target_country_df_list, pred_df_list = fit_sigmoid(exclude_days=7)


# In[ ]:


plot_sigmoid_fitting(target_country_df_list, pred_df_list, title='Sigmoid fitting without last 7days data')


# Now I noticed that sigmoid fitting tend to **underestimate** the curve, and its actual value tend to be more than sigmoid curve estimation.<br/>
# Therefore, we need to be careful to see sigmoid curve fitting data, actual situation is likely to be worse than the previous figure trained with all data.

# <a id="id_ref"></a>
# # Further reading
# 
# That's all! Thank you for reading long kernel. I hope the world get back peace & usual daily life as soon as possible.
# 
# Here are the other information for further reading.
# 
# My other kernels:
#  - [COVID-19: Effect of temperature/humidity](https://www.kaggle.com/corochann/covid-19-effect-of-temperature-humidity)
#  - [COVID-19: Spread situation by prefecture in Japan](https://www.kaggle.com/corochann/covid-19-spread-situation-by-prefecture-in-japan)

# <h3 style="color:red">If this kernel helps you, please upvote to keep me motivated :)<br>Thanks!</h3>

# In[ ]:




