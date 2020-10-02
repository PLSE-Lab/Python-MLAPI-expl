#!/usr/bin/env python
# coding: utf-8

# # COVID-19 by Numbers & Charts
# 
# In this Notebook, we will try to understand the present state of the world as it is being affected by COVID-19.
# 
# * we will be using the dataset for covid-19 available [here...](https://github.com/CSSEGISandData/COVID-19)
# * this is a live notebook. Every time we run the notebook, we will be pulling in the latest data from the repository.
#  

# In[ ]:


import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from dateutil import parser

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from IPython.display import display, HTML, Markdown

import seaborn as sns
import altair as alt


# In[ ]:


# old url that stopped working as of 3-26-20
# url = 'https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv'


# # download data and prep
# - data is availabe in 2 different formats. I find it easier to work this csv instead of excell format also called time seried format ( which is a pivot of the csv format )
# - there is a daily file with data under ./csse_covid_19_daily_reports directory for each day.
# - we will iterate over all the files and build a dataframe, then preprocess some columns.

# In[ ]:


#url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/01-22-2020.csv'

url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/{date}.csv'


# * generate a list of dates from `2020-01-22` till `yesterday`
# * download the file corresponding to the date
# * create a final data frame by concating data frames for each day

# In[ ]:


dflist = []
yesterday = datetime.today() - timedelta(days=1)

for d in pd.date_range('2020-01-22', yesterday.strftime("%Y-%m-%d")):
    try:
        _df=pd.read_csv(url.format(date=d.strftime("%m-%d-%Y")))
        _df.columns = [ n.lower() for n in _df.columns ]
        _df.rename(columns={
                'country/region': 'country',
                'country_region':'country',
                'province/state': 'state',
                'province_state': 'state',
                'last_update' : 'ts',
                'last update' : 'ts',
                'long_' : 'lag',
                'longitude': 'lag',
                'latitude': 'lat'
        }, inplace=True)
        _df['ts'] = pd.to_datetime(_df.ts)
        _df['date'] = _df.ts.apply(lambda x : x.date())
        dflist.append(_df)
    except Exception as e:
        print(f"missing:{d} - {str(e)}")
dfn = pd.concat(dflist, sort=True)
display(dfn.shape)
dfn.head()


# # preprocessing
# 
# for our analysis here we will assume that any particular case either falls under `active`, `confirmed`, `recovered` or `death` state
# 1. fill missing values for confirmed cases. since we can assule that previously reported numbers for a missing day.
# 2. fill missing values for confirmed, deaths, recoreved

# In[ ]:


# drop rows that dont have confrimed cases
#dfn = dfn.drop(index=dfn[dfn.confirmed == 0].index)
dfn.confirmed.fillna(method='ffill', inplace=True)

# replace active, deaths, recovered with 0
[dfn[c].fillna(0, inplace=True) for c in ['confirmed','deaths','recovered'] ];


# 
# 3. although we do get reported active cases , we will calculate active to see if there is any discrepencies. (in case of discrepencies. we will fall back to calculated number for active cases)

# In[ ]:


# making sure there isnt any discrepency between active cases

#dfn['active'] = dfn['confirmed'] - (dfn['recovered'] + dfn['deaths'])

# they stopped sending active numbers
#dfn['active-off']= dfn.apply( lambda x : (x['comp-active'] - x['active']) != 0, axis=1)

# print(f"""
#     turns out we have {dfn['active-off'].sum()} discrepencies out of {dfn.shape[0]}  ( {round(dfn['active-off'].sum()/ dfn.shape[0]*100,2)}% )
# """)
# dfn['active'] = dfn['comp-active']
# dfn.drop(columns=['active-off'], inplace=True)


# 4. certain country name need preprocessing 
#     - we will compare the reported country names with ISO database, and update the discrepencies when needed
#     - eg: `china` and `Mainland china`  or our analysis we will count both as `chaina`

# In[ ]:


# get list of official country names 
# results here should alwasy should be empty

df_country_code_mapping = pd.read_csv("https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv")
_false_countries = dfn[['country']].merge(df_country_code_mapping, left_on=['country'], right_on=['name'], how='outer')


# In[ ]:


# process country names for accuracy

def preprocess_country(val):
    mapping = {
        'Mainland China':'China',
        'Vietnam' : 'Vietnam',
        'Taiwan*' : 'Taiwan',
        'Gambia, The' : 'The Gambia',
        'Bahamas, The' : 'The Bahamas',
        'Korea, South' : 'South Korea',
        'Macao SAR' : 'Macao',
        'Macau' : 'Macao',
        'Hong Kong SAR' : 'Hong Kong',
        'Republic of Korea':'South Korea',
        'Republic of Ireland' : 'Ireland',
        'Republic of Moldova' : 'Moldova',
        'Congo (Brazzaville)' : 'Republic of Congo',
        'West Bank and Gaza' : 'Palestinian territory',
        'occupied Palestinian territory' : 'Palestinian territory',
        'United Kingdom' : 'UK',
        'Iran (Islamic Republic of)' : 'Iran',
        'Viet Nam' : 'Vietnam',
        'Cruise Ship' : 'Others'
    }
    return mapping[val] if val in mapping.keys() else val.strip()

dfn['country'] = dfn.country.apply(preprocess_country)
dfn['country'] = dfn.apply(lambda x : 'Others' if x['country'] == 'Israel' and x['state'] == 'From Diamond Princess' else x['country'], axis=1)
dfn['state'  ] = dfn.state.apply(lambda x : 'Diamond Princess' if x == 'Diamond Princess cruise ship'  else x)
dfn['state'  ] = dfn.apply(lambda x : 'Diamond Princess' if x['country'] == 'Diamond Princess' else x['state'], axis=1)
dfn['country'] = dfn.apply(lambda x : 'Others' if x['state'] == 'Diamond Princess' else x['country'], axis=1)

# save original
dforig = dfn.copy()


# In[ ]:


dfn.head()


# * converting cululative numbers to daily numbers 

# In[ ]:


dfn.state.fillna('unknown',inplace=True)
_d = dfn.groupby(['country','state','date']).agg({'confirmed':'sum', 'recovered':'sum', 'deaths':'sum', 'active':'sum'})
_d = _d.merge( _d.groupby(['country','state']).diff() , left_index=True, right_index=True, suffixes=('','_diff'))

# fill na for 1 record 
_d['confirmed_diff'] = _d.apply(lambda x : x['confirmed'] if pd.isnull(x['confirmed_diff']) else x['confirmed_diff'], axis=1)
_d['active_diff'   ] = _d.apply(lambda x : x['active'   ] if pd.isnull(x['active_diff'])    else x['active_diff'],    axis=1)
_d['recovered_diff'] = _d.apply(lambda x : x['recovered'] if pd.isnull(x['recovered_diff']) else x['recovered_diff'], axis=1)
_d['deaths_diff'] = _d.apply(lambda x : x['deaths'] if pd.isnull(x['deaths_diff']) else x['deaths_diff'], axis=1)

dfn = _d.rename(columns={'confirmed':'confirmed_rep', 
                   'recovered':'recovered_rep',
                   'deaths': 'deaths_rep',
                   'active' : 'active_rep'})\
  .rename(columns={'confirmed_diff':'confirmed',
                   'recovered_diff':'recovered',
                   'deaths_diff':'deaths',
                   'active_diff':'active'})\
  .reset_index()


# In[ ]:


_d = dfn.groupby(['country','date']).agg({'confirmed':'sum', 'recovered':'sum', 'deaths':'sum', 'active':'sum'})
_d = _d.merge( _d.groupby(['country']).diff() , left_index=True, right_index=True, suffixes=('','_diff'))

# fill na for 1 record 
_d['confirmed_diff'] = _d.apply(lambda x : x['confirmed'] if pd.isnull(x['confirmed_diff']) else x['confirmed_diff'], axis=1)

dfc = _d.rename(columns={'confirmed':'confirmed_rep', 
                   'recovered':'recovered_rep',
                   'deaths': 'deaths_rep',
                   'active' : 'active_rep'})\
  .rename(columns={'confirmed_diff':'confirmed',
                   'recovered_diff':'recovered',
                   'deaths_diff':'deaths',
                   'active_diff':'active'})\
  .reset_index()


# # COVID-19 State across reporting countries in past 24 hr.

# In[ ]:


_df = dfc.copy()

# filter data with the last update date seen.
#_df['last_update_date'] = _df.ts.apply(lambda x : x.date())
_df = _df[_df.date == _df.date.max()]

# aggreagate and melt the columns. this will make it easier to split the data in the charts
data = _df.groupby(['country']).agg({'confirmed': 'sum',
                              'deaths'   : 'sum',
                              'active'   : 'sum',
                              'recovered': 'sum'})\
    .reset_index()\
    .sort_values(by='confirmed', ascending=False)\
    .melt(id_vars=['country'], value_vars=['confirmed','deaths','active','recovered'], var_name='status', value_name ='count')

confirmed = alt.Chart( data[data.status == 'confirmed'].sort_values(by=['count'], ascending=False).head(10) )    .mark_bar()    .encode(y=alt.Y('country',sort='-x'), x='count')    .properties(title='confirmed')
confirmed_text = confirmed.mark_text(dx = 15).encode(text='count')

recovered = alt.Chart( data[data.status == 'recovered'].sort_values(by=['count'], ascending=False).head(10) )    .mark_bar()    .encode(y=alt.Y('country',sort='-x'), x='count')    .properties(title='recovered')
recovered_text = recovered.mark_text(dx = 15).encode(text='count')

deaths = alt.Chart( data[data.status == 'deaths'].sort_values(by=['count'], ascending=False).head(10) )    .mark_bar()    .encode(y=alt.Y('country',sort='-x'), x='count')    .properties(title='deaths')
deaths_text = deaths.mark_text(dx = 15).encode(text='count')

active = alt.Chart( data[data.status == 'active'].sort_values(by=['count'], ascending=False).head(10) )    .mark_bar()    .encode(y=alt.Y('country',sort='-x'), x='count')    .properties(title='active')
active_text = active.mark_text(dx = 15).encode(text='count')

(
    ( (confirmed + confirmed_text) | (recovered + recovered_text) )
  & ( (deaths + deaths_text) | (active + active_text) )
).properties(title='last 24 hr stats')


# # Cumulative numbers across countries 
# - comparing countries by their reported confirmed cases on a cumulative basis
# - rank countries based on the number of cumulative reported numbers
# - last reported numbers is the cumulative number for each countries 
#     - we will try to find the daily numbers for each metric 

# In[ ]:


# dfg = dfcountry.reset_index().groupby(['country'])\
#          .agg({'confirmed_diff':'sum', 'recovered_diff':'sum', 'deaths_diff':'sum'})\
#          .rename(columns={'confirmed_diff':'confirmed','recovered_diff':'recovered', 'deaths_diff':'deaths'})\
#          .reset_index()

dfg = dfn.copy()
dfg = dfg.groupby(['country'])         .agg({'confirmed':'sum', 'recovered':'sum', 'deaths':'sum', 'active':'sum'})         .reset_index()

# calculate active counts
# calculate x-rates for each of confirmed, recovered , deaths and active counts

# dfg['active']         = dfg['confirmed'] - dfg['deaths'] - dfg['recovered']
dfg['death-rate']     = round(dfg['deaths']   /dfg['confirmed'] *100,2)
dfg['recovery-rate']  = round(dfg['recovered']/dfg['confirmed'] *100,2)
dfg['active-rate']    = round(dfg['active']   /dfg['confirmed'] *100,2)

# chart the heatmap table sorted by confirmed cases on a cumulative basis

display(
dfg.sort_values(by=['confirmed','active-rate'], ascending=False)\
    .head(20)\
    .set_index(pd.Index([i for i in range(1,21)]),'rank')\
    .style\
    .format(dict([(c,'{:.0f}')  for c in ['confirmed','deaths','recovered','active']]))\
    .format(dict([(c,'{:.0f}%') for c in ['death-rate','recovery-rate','active-rate']]))\
    .background_gradient( subset=['active-rate', 'death-rate', 'recovery-rate', 'deaths']
                         ,cmap=ListedColormap(sns.color_palette('Reds', desat=0.7)))\
    .background_gradient(subset=['recovery-rate', 'recovered']
                         ,cmap='Greens')\
    .background_gradient(subset=['active', 'confirmed']
                         ,cmap='Blues')\
    .set_caption(f"cumulative numbers from (date:{dfn.date.min()} to {dfn.date.max()}) for to 20 countries by confirmed cases")
)


# # Time Series of top 10 most affected countries
# - US has an exponential increase in confirmed cases. 
# - China's confirmed cases has started to decline.
# 

# In[ ]:


_df = dfn.copy()
_df['date'] = pd.to_datetime(_df.date)

# generate data by grouping and filtering top 10 countries
# plot area chart , this will show % share by country of the cumulative count
# plot line chart on the same data on the side , this will clearly show the rate over time for each country in top 10 

base = alt.Chart(
    data = _df.groupby(['date','country'], as_index=False).agg({'confirmed':'sum'})\
    .merge(
        # only look at top 10 countries
        dfg.sort_values(by=['confirmed'], ascending=False)\
           .head(10)\
           .country 
        , on='country')
    )

area = base.mark_area(opacity=.8)           .encode(  x='date:T'
                   , y='confirmed'
                   , tooltip=['confirmed']
                   , color=alt.Color('country'
                                     , sort=alt.EncodingSortField('confirmed'
                                                                  , op="sum"
                                                                  , order="descending")))\
        .properties(width=600, title='confirmed cases by country')\
        .interactive()

line = base.mark_line(opacity=.8, point=True)           .encode(  x='date:T'
                   , y='confirmed'
                   , tooltip=['confirmed']
                   , color=alt.Color('country'
                                     , sort=alt.EncodingSortField('confirmed'
                                                                  , op="sum"
                                                                  , order="descending")))\
        .properties(width=600, title='confirmed cases by country')\
        .interactive()
(area & line)


# # Start of Infection across Countries
# - look at how COVID-19 spread across the work in different countries
# - rate of spread and if it is still spreading..

# - first we will create base data frames. so we dont have to calculate the same layer again
#     - `_top_10_countries` : aggregate of top 10 countries by date
#     - `_top_20_countries` : agg. of top 20 countries by date
#     - `_df_base` : original df filtered for top 10 countries
#     - `_df_base_20` : original df filterd for top 20 countries
#     - `_df_base_with_states` : original df filtere for top 10 countries and includes states info

# In[ ]:


_df = dfn.copy()
_df['date'] = pd.to_datetime(_df.date)

dfg = dfn.groupby(['country'], as_index=False)         .agg({  'confirmed':'sum'
               , 'recovered':'sum'
               , 'deaths':'sum'})

# create a base df with to 10 countries by confirmed cases
# this will be later used to build rest of the thesis

_top_10_countries = dfg.sort_values(by=['confirmed'], ascending=False).head(10).country
_top_20_countries = dfg.sort_values(by=['confirmed'], ascending=False).head(20).country

_df_base = _df.groupby(['date'
                        ,'country'], as_index=False)\
              .agg({ 'confirmed':'sum'
                    ,'deaths'   :'sum'
                    ,'recovered':'sum'
                    , 'active':'sum'})\
              .merge( _top_10_countries , on='country') # filter for top 10 countries

_df_base_20 = _df.groupby(['date'
                           ,'country'], as_index=False)\
              .agg({  'confirmed':'sum'
                    , 'deaths'   :'sum'
                    , 'recovered':'sum'
                    , 'active'   :'sum'})\
              .merge( _top_20_countries , on='country') # filter for top 20 countries

_df_base_with_states = _df.groupby(['date'
                                    ,'country'
                                    ,'state'], as_index=False)\
              .agg({  'confirmed':'sum'
                    , 'deaths'   :'sum'
                    , 'recovered':'sum'
                    , 'active'   :'sum'})\
              .merge( _top_10_countries , on='country')


# - `grp` : create a grouper obj for top 20 countries aggregated over countries for confirmed cases and start date 

# In[ ]:


_df = dfn.copy()
_df['date'] = pd.to_datetime(_df.date)

# create a grouper obj
# agg on confirmed cases and collect start date (min date)
# grp = _df.groupby(['country']).agg({'confirmed':'sum', 'date':'min'})\
#          .sort_values(by=['confirmed'], ascending=False)\
#          .head(20)\
#          .groupby(['country'])

grp = _df.groupby(['country']).agg({'confirmed':'sum', 'date':'min'})         .sort_values(by=['confirmed'], ascending=False)         .groupby(['country'])

# calculate start date for reach country, the date when first confirmed case was reported
start_date_by_country = grp.agg({'date':min})                           .rename(columns={'date':'start_date'})

# calculate the date when first covid-19 case was seen in each of the top 10 countries
_dfsd = _df_base.merge(start_date_by_country, on=['country'])
_dfsd['days'] = _dfsd.apply(lambda x : (x['date'] - x['start_date']).days, axis=1)


# * chart based on start date of infection in top 20 countries

# In[ ]:


# create start date by country data frame  to show the order in the chart
# `p` : assign a column to count ( like select start_date , 1 as p from ... )
_dfsc = start_date_by_country            .sort_values(by=['start_date'])            .assign(p=True)            .reset_index()

# store the order of countries, useful to order country label on axis
_country_order = _dfsc.country

_df = _dfsc.pivot('start_date','country','p')[_country_order]

# ----------------------------------------------------------------- #
# (1) Start date of COVID-19 in top 20 countries by confirmed cases #
# ----------------------------------------------------------------- #

chart_1 = alt.Chart(_dfsc.iloc[200:])    .mark_point(stroke='red', filled=True, opacity=.6)    .encode(  x='yearmonthdate(start_date):T'
            , y=alt.Y('country', sort='-x')
            , tooltip=['country', 'start_date']
            , size='p')\
    .properties( title='(1) Start date of COVID-19 in top 20 countries by confirmed cases',
                 width=600)

# ----------------------------------------------- #
# (3) No. of new countries affected by start date #
# ----------------------------------------------- #

_df = dfn.copy()
_df['date'] = pd.to_datetime(_df.date)
base = alt.Chart(
        _df.groupby(['country'])\
           .agg({'date'     :'min'})\
           .assign(c = 1)\
           .reset_index()\
           .rename(columns={'date':'start_date'})
        )

flattened = base.mark_point(color='firebrick', opacity=0.7, thickness=2, filled=True)        .encode(  x='start_date:T'
                , size='sum(c)'
                , tooltip=['start_date', 'sum(c)'])\
        .properties(title="(3) No. of new countries affected by start date",
                    width=600)
breakdown = base.mark_bar(color='firebrick', opacity=.7)        .encode(  x='start_date:T'
                , y='sum(c)'
                , tooltip=['start_date', 'sum(c)'])\
        .properties(title="(3) No. of new countries infected",
                    width=600)

# --------------------------------------------- #
# (2) confirmed cases on start date for country #
# --------------------------------------------- #

_df20 = _df_base_20.merge(start_date_by_country, on=['country'])
_df20['days'] = _df20.apply(lambda x : (x['date'] - x['start_date']).days, axis=1)

base = alt.Chart( _df20[_df20['date'] == _df20['start_date']].reset_index(drop=True) )

chart20china = base.mark_bar()              .encode( x=alt.X('country')
                     , y='confirmed'
                     , tooltip=['country', 'confirmed']
                     , color='country')\
              .transform_filter( alt.datum.country == 'China' )\
text20china  = chart20china.mark_text(dx=15,angle=90).encode(text='confirmed')

chart20 = base.mark_bar()              .encode( y=alt.Y('country'
                               , sort=alt.EncodingSortField('start_date:T', order='ascending'))
                     , x='confirmed'
                     , tooltip=['country', 'confirmed']
                     , color='country')\
              .transform_filter( alt.datum.country != 'China' )\
              .properties(title='(2) No. of confirmed cases on start date for country',
                          width=450)\
text20  = chart20.mark_text(dx=15).encode(text='confirmed')

# ----------------------------------------- #
# (4) No. of new countries affected by week #
# ----------------------------------------- #

__df=_df.groupby(['country'])        .agg({'confirmed':'sum'
              , 'date':'min'})\
        .sort_values(by=['confirmed'], ascending=False)\
        .reset_index()\
        .groupby(['date'])\
            .agg({'country':'count'})\
            .reset_index()\
            .rename(columns={  'date':'start_date'
                             , 'country': 'country_count'})

__df['week-of-year'] = __df.start_date.apply(lambda x : f"{x.year} - {x.strftime('%U')}")
__df['grand_total']  = __df.country_count.sum()
__df['growth_rate']  = __df.apply(lambda x : round(x['country_count']/x['grand_total']*100,2), axis=1)

weekly_chart = alt.Chart(__df.groupby(['week-of-year']).agg({'country_count':'sum'}).reset_index())           .mark_bar(color='firebrick', opacity=.6)           .encode(  x=alt.X('week-of-year')
                   , y='country_count')\
           .properties(  title='(4) No. of new countries infected by week'
                       , width=600)

weeklytext = weekly_chart.mark_text(angle=0, dy=10).encode(text='country_count')

(     
 chart_1 
 & ( (chart20china + chart20china) | chart20+text20 ) 
 & ( flattened & breakdown )
 & (weekly_chart + weeklytext) 
)


# - as you can see. `US`, `China`, 'South Korea` and 'Brazil` has the first cases as per the data (us started collecging data from 01-22-20)
#     - interesting to me was `Brazil` had the cases at the same time as `US`
# - then it started spreading in Europe starting with `France`, `Germany`, `Italy`.. `UK`
# - at the start of the data set i.e. 01-22-02 there were 8 countries already affected 
# - if we look at spread across world, we see that it really started spreading between 02-22-02 and 03-22-02.
# - even as of 04-05-20 we are seeing it spread across countries. which is alarming..

# # Breakout curves
# 

# In[ ]:


_dfsd['confirmed-rolling-sum'] = _dfsd.groupby(['country']).confirmed.transform(lambda x : x.rolling(7).sum())
_dfsd['recovered-rolling-sum'] = _dfsd.groupby(['country']).recovered.transform(lambda x : x.rolling(7).sum())

confirmed = alt.Chart(_dfsd[['days','country','confirmed-rolling-sum']])    .mark_line(strokeOpacity=.7)    .encode(x='days:O', 
            y='confirmed-rolling-sum',
            tooltip=['days','confirmed-rolling-sum','country'],
            color=alt.Color('country', 
                            sort=alt.EncodingSortField('confirmed-rolling-sum', 
                                                       op="sum", 
                                                       order="descending")))\
    .properties(title="7 day sum of confirmed cases by days since reported",
                width=600)

recovered = alt.Chart(_dfsd[['days',
                             'country',
                             'recovered-rolling-sum']])\
    .mark_line(strokeOpacity=.7)\
    .encode(x='days', 
            y='recovered-rolling-sum',
            tooltip=['days','recovered-rolling-sum','country'],
            color='country')\
    .properties(title='7 day sum of recovered cases by day since reported',
                width=600)

(confirmed & recovered)
recovered


# - we can see the COVID-19 impact on `US` from the breakout curves. rolling 7 day difference in confirmed cases has spiked to 200K.
# - it has started to weaken for `Germany` by `65`th day.
# - `Spain`, `Italy` and `UK` has started recovering. they have reported fewer cases on a w/w basis.
# - `China` has already moved to -ve difference compared to previous weekly numbers.
# - finally , `US` has started to tip off on rolling 7 day diff.

# # Day of the week analysis

# - if you plot the day of the week by aggregated sum of various measures , we can see some interesting observation
#     - confirmed and deaths cases peak on wed!
#     - recoveries peak on Mon!

# In[ ]:


df_by_day = _df_base.copy()
df_by_day['dow']= df_by_day.date.apply(lambda x : x.strftime("%w-%a"))


# 1. calculate sum of all metrics by dow and country
# 2. calculate agg. of all metrics by country 
# 3. merge both dfs together
# 4. calculate % change on day of week over agg of country

# In[ ]:


cols_of_interest = ['confirmed','deaths','recovered','active']

_df = df_by_day.groupby(['dow', 
                         'country'])\
               .sum()\
               .reset_index()\
        .merge(
            df_by_day.groupby(['country'])\
                .sum()\
                .rename(columns=dict([(c,f"{c}-max") for c in cols_of_interest]))\
                .reset_index()
            , on='country'
            )\
        .set_index(['dow','country'])


for c in cols_of_interest:
    _df[f'{c}_pct'] = round(_df[c]/_df[f'{c}-max'], 2 )


# 1. get the largest `pct` percent change for all confirmed, active, recovered & deaths over country
#     - this is later used to display heat map
# 2. calculate prefered day of the week by metrics for the largest percent change
# 3. finally calculate break down by prefered day of the week across top 20 countries excluding `active`

# In[ ]:


dfc = _df.groupby(['country'])

df_by_pct =     dfc.apply(lambda x : x.nlargest(1,'confirmed_pct')
                          .reset_index()
                          .confirmed_pct)\
        .rename(columns={0:'confirmed'})\
        .assign(active=dfc.apply(lambda x : x.nlargest(1,'active_pct')
                                             .reset_index()
                                             .active_pct)\
                            .rename(columns={0:'active'}))\
        .assign(deaths=dfc.apply(lambda x : x.nlargest(1,'deaths_pct')
                                             .reset_index()
                                             .deaths_pct)\
                            .rename(columns={0:'deaths'}))\
        .assign(recovered=dfc.apply(lambda x : x.nlargest(1,'recovered_pct')
                                                .reset_index()
                                                .recovered_pct)\
                            .rename(columns={0:'recovered'}))\
        .unstack()\
        .to_frame()\
        .reset_index()\
        .rename(columns={0:'pct', 'confirmed_pct':'state'})

df_by_dow =     dfc.apply(lambda x : x.nlargest(1,'confirmed_pct')
                          .reset_index()
                          .dow)\
        .rename(columns={0:'confirmed'})\
        .assign(active=dfc.apply(lambda x : x.nlargest(1,'active_pct')
                                             .reset_index()
                                             .dow)\
                        .rename(columns={0:'active'}))\
        .assign(deaths=dfc.apply(lambda x : x.nlargest(1,'deaths_pct')
                                             .reset_index()
                                             .dow)\
                        .rename(columns={0:'deaths'}))\
        .assign(recovered=dfc.apply(lambda x : x.nlargest(1,'recovered_pct')
                                                .reset_index()
                                                .dow)\
                            .rename(columns={0:'recovered'}))\
        .unstack()\
        .to_frame()\
        .reset_index()\
        .rename(columns={0:'day', 'dow':'state'})

# Charts #
# chart the prefered day by state

prefered_dow_pct_breakdown =     alt.Chart( 
        df_by_dow.groupby(['day','state'], as_index=False)\
                .agg({'country':'count'})\
                .query('state != "active"') 
        )\
        .mark_bar()\
        .encode(y='day',
                x='country', 
                color='state', 
                column='state')

#display(prefered_dow_pct_breakdown)

cpct = alt.Chart(_df.reset_index())        .mark_bar()        .encode(y='dow', 
                x=alt.X('confirmed_pct', axis=alt.Axis(format=".0%")), 
                color='country',
                tooltip=['country','confirmed_pct','dow'])
cpct_txt = alt.Chart(_df.reset_index())              .mark_text(dx=-20)              .encode(
                y='dow',
                x=alt.X('confirmed_pct', stack='zero', axis=alt.Axis(format=".0%")),
                detail='country',
                text=alt.Text('confirmed_pct', format=".1%"))

dpct = alt.Chart(_df.reset_index())        .mark_bar()        .encode(y='dow', 
                x=alt.X('deaths_pct', axis=alt.Axis(format=".0%")), 
                color='country')
dpct_txt = alt.Chart(_df.reset_index())              .mark_text(dx=-20)              .encode(
                y='dow',
                x=alt.X('deaths_pct', stack='zero', axis=alt.Axis(format=".0%")),
                detail='country',
                text=alt.Text('deaths_pct', format=".1%"))

rpct = alt.Chart(_df.reset_index())        .mark_bar()        .encode(y='dow', 
                x=alt.X('recovered_pct', axis=alt.Axis(format=".0%")), 
                color='country')

rpct_txt = alt.Chart(_df.reset_index())              .mark_text(dx=-20)              .encode(
                y='dow',
                x=alt.X('recovered_pct', stack='zero', axis=alt.Axis(format=".0%")),
                detail='country',
                text=alt.Text('recovered_pct', format=".1%"))
display(
(    (cpct + cpct_txt).properties(width=600)
   & (dpct + dpct_txt).properties(width=600)
   & (rpct + rpct_txt).properties(width=600)
))



display(
    df_by_dow.merge( df_by_pct, on=['state','country'])\
             .pivot_table(index='country', 
                          columns=['state','day'], 
                          values=['pct'])\
             .fillna(0)\
             .style\
             .background_gradient(cmap='Blues')\
             .format("{:.0%}")
)


# - interesting to see that most deaths, recoveries and confirmed cases are reported on monday for 8 out of top 10 countries.

# # Major cities affected across top 10 countries

# 1. aggregate metrics across state and country
# 2. calculate % change across all metrics for each of the states in a country
# 3. filter top 5 states by % change in each of the countries.
# 4. chart

# In[ ]:


cols_of_interest = ['confirmed','deaths','recovered','active']

_df = dfn.copy()
_df = _df[_df.state != 'unknown']

all_countries_with_state = _df.groupby(['date'
                                    ,'country'
                                    ,'state'], as_index=False)\
              .agg({  'confirmed':'sum'
                    , 'deaths'   :'sum'
                    , 'recovered':'sum'
                    , 'active'   :'sum'})\
# calculate aggreate by country and state
# _df_cs = _df_base_with_states.groupby(['country'
#                                        ,'state'])\
#                              .agg(dict([ (c,'sum') for c in cols_of_interest ]))

_df_cs = all_countries_with_state.groupby(['country'
                                       ,'state'])\
                             .agg(dict([ (c,'sum') for c in cols_of_interest ]))


# calculate agg. by country
_dfc = _df.groupby(['date'
              ,'country'], as_index=False)\
              .agg({ 'confirmed':'sum'
                    ,'deaths'   :'sum'
                    ,'recovered':'sum'
                    , 'active':'sum'})\
                .groupby(['country'])\
                     .agg(dict([ (c,'sum') for c in cols_of_interest ]))


# filtere top 10 countries
# assign agg at country level to state level 

_dfm = _df_cs.reset_index()    .merge(_dfc, on='country')    .rename(columns={'confirmed_x':'confirmed',
                     'deaths_x'   : 'deaths',
                     'recovered_x': 'recovered',
                     'active_x'   : 'active',
                     
                     'confirmed_y': 'confirmed_total',
                     'deaths_y'   : 'deaths_total',
                     'recovered_y': 'recovered_total',
                     'active_y'   : 'active_total'})

# calculate percent change at country level for each of the matrics for each of the states
for c in cols_of_interest:
    _dfm[f'{c}_pct'] = round(_dfm[c]/_dfm[f'{c}_total'],2)

_dfg = _dfm.groupby('country')

# filter top 5 states in each country by % change
data = _dfg.apply(lambda x : x.nlargest(5,'confirmed').reset_index(drop=True))           .sort_values(by=['confirmed_total',
                            'confirmed_pct'], ascending=False)\
           .drop(columns=['country'])

chart_data = data.reset_index()
#chart_data = chart_data[chart_data.country != 'Germany']

chart = alt.Chart(chart_data)           .mark_bar(opacity=0.8, width=20)           .encode(x=alt.X('state',sort='-y'), 
                   y='confirmed', 
                   color='country')\
           .properties(width=150, height=150)

text = chart.mark_text(angle=90, 
                       align='right', 
                       dx=-1)\
            .encode(text=alt.Text('confirmed_pct', format='.1%'))

# display(
# data.style\
#     .format( dict((c,'{:.0f}')for c in ['confirmed','deaths','recovered','active','confirmed_total','deaths_total','recovered_total','active_total'] ))\
#     .bar(subset=['confirmed_pct','deths_pct','recovered_pct','active_pct'], axis=0)
# )

alt.layer(chart,
          text, 
          data=chart_data)\
.facet(facet='country',
       bounds='full', 
       columns=5)\
.resolve_scale(y='independent', 
               x='independent')\
.properties(title="top 5 states affected by COVID-19 across countries ( labels in %) (cumulative confirmed cases)")


# - as you can clearly see `Hubei` has `82`% of the cases reported in `China`.
# - and about `39`% of the cases reported in `US` are in `New York`. 
# - also about `41`% of the active cases in `US` are in `New York`.
# 
# ***NOTE***
# * ***`Germany` has only one city reported. so we can ignore that one from the table ***
# * ***some preprocessing need to be done for UK.***

# # Rate of infection over time around the world
# * calcualte growth rate of confirmed cases by country to its total
# * bucket growth rate into 10 bins
# * calculate number of countries that all into each bucket ( hist gram ) over date
# 
# * we can see that infection rate around the world and the rate of infaction over time

# In[ ]:


_df = dfn.copy()

_df = _df.groupby(['date','country'], as_index=False).agg({'confirmed':'sum'})
_df = _df        .merge(_df                   .groupby(['country'], as_index=False)                   .sum(),
              on=['country'])\
        .rename(columns={'confirmed_x':'confirmed',
                         'confirmed_y':'total_confirmed'})
_df['confirmed_pct'] = round(_df['confirmed']/_df['total_confirmed'],2)

def bucket_confirmed_pct(val):
    if val <  0.05 : return '0-5'
    if val >= 0.06 and val <=.10 : return '6-10'
    if val >=.11 and val <=.20 : return '11-20'
    if val >=.21 and val <=.50 : return '21-50'
    if val >=.51 and val <=.75 : return '51-75'
    if val >=.76 and val <=.100 : return '76-100'

_df['bucket'] = pd.Categorical( _df.confirmed_pct.apply(bucket_confirmed_pct),
                              ['0-5','6-10','11-20','21-50','51-75','76-100'])


# In[ ]:


_d = _df.groupby(['date','bucket'], as_index=True).agg({'country':pd.Series.nunique}).reset_index()
_d['date'] = pd.to_datetime(_d['date'])
alt.Chart(_d)    .mark_bar()    .encode(x='date:T', y='country', facet=alt.Facet('bucket', columns=1, sort=alt.EncodingSortField('bucket')), color='bucket')    .resolve_scale( y='independent')    .properties(height=100,width=600, title="No.of countries in the confirmed_pct bucket over time")


# In[ ]:



_d['date'] = _df['date'].apply(lambda x : x.strftime("%Y-%m-%d"))
_d['ym'] = _df['date'].apply(lambda x : x.strftime("%Y-%m"))
_d['d']   = _df['date'].apply(lambda x : x.strftime("%d"))

_dd = _d.sort_values(by=['date'])              .tail(80)              .groupby(['bucket','date'], as_index=False)              .sum()              .pivot('bucket','date','country')              .replace(0, np.nan)

plt.figure(figsize=(25,3))
plt.title("No. of countries by buckets of growth rate of confirmed_pct (% change) over time")
display(
    sns.heatmap(_dd
            , annot=True
            , linewidths=1
            , square=False
            , cbar=None
            , cmap='Reds'
            , fmt='.0f')
)


_dym_d = _d.sort_values(by=['date'])              .groupby(['bucket','ym','d'], as_index=False)              .sum()
display(
    pd.pivot_table(data=_dym_d, 
                   index=['bucket','ym'], 
                   columns=['d'], 
                   values=['country'])\
        .fillna(0)\
        .style\
        .background_gradient(cmap='Reds')\
        .format(lambda x : "" if x == 0 else x)\
        .set_caption('No. of countries by buckets of growth rate of confirmed_pct (% change) split by day and month matrix')
)


# * from above heat map and bar charts we can see that
#     * the infection is spread more recently across the world
#     * comparision ov m/m shows more countries are movign in the 2nd bucket of (5-10) from (0-5) % . which means the infections is actually spreading. instead of being contained.

# # Conclusion
# 1. from the chart (4) we can see that there are less newer countries that are being infected.
# 2. from chart above , we can see that although fewer new countries are being infected. infection is spreading among those countries.
# 3. infection rate is started to curve for US based on w/w comparision.

# In[ ]:





# In[ ]:




