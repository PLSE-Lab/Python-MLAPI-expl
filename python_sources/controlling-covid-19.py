#!/usr/bin/env python
# coding: utf-8

# I know a lot of professionals and hobbyists in the Data Sciences have looked to contribute to the public understanding of the COVID-19 Pandemic affecting the world and I have been proud to be part of such an amazing community driven by 'Data Science for Good'. In this notebook I try to provide some interesting plots, which aim to investigate how plots and metrics can mislead people about the state of the infection globally and propose a possible way in which we can control for these common pitfalls to provide possible insights on where we see the high growth in infections globally.  

# In[ ]:


get_ipython().system(' apt install libgeos-dev')
get_ipython().system(' pip uninstall -y shapely; pip install --no-binary :all: shapely==1.6.4')
get_ipython().system(' pip uninstall -y cartopy; pip install --no-binary :all: cartopy==0.17.0')
get_ipython().system(' pip install geoviews==1.6.6 hvplot==0.5.2 panel==0.8.0 bokeh==1.4.0')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from operator import add, mul
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import hvplot.pandas
import holoviews as hv
import cartopy.crs as ccrs
import geopandas as gpd
from toolz.curried import map, partial, pipe, reduce
from statsmodels.regression.linear_model import OLS

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

hv.extension('bokeh')


# Kaggle and its partners have done amazing job at getting data and compute into the hands of the Data Science community to try investigate and provide insights on the COVID-19 pandemic. While there are many well curated datasets tracking the pandemic on their platform, the one I have been interesting in exploring has been on the number of cases in different countries around the world through time. 

# In[ ]:


countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).replace('United States of America', 'US')
data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv', parse_dates=['Date'], index_col='Id')
data


# In[ ]:


weeks = (data
         .assign(dayofweek = lambda df: df.Date.dt.dayofweek)
         .set_index('Date')
         .drop(columns=['Province/State', 'Lat', 'Long'])
         .groupby(['Country/Region', pd.Grouper(freq='W')]).agg({'ConfirmedCases':'sum', 'Fatalities':'sum', 'dayofweek':'max'})
         .reset_index()
         .where(lambda df: df.ConfirmedCases > 0)
         .dropna(0)
         .groupby('Country/Region')
         .apply(lambda df: (df
                            .sort_values('Date')
                            .assign(week_of_infection = lambda df: pd.np.arange(df.shape[0]))))
         .where(lambda df: df.dayofweek >= 6)
         .dropna(0)
         .reset_index(drop=True)
         .merge(countries, left_on='Country/Region', right_on='name'))
weeks


# You have probably seen these two plots before: a plot of the number of confirmed cases in each country through time and a plot of the number of confirmed cases overlayed onto a map.  For many people this is an easy and intuitive view of the scale of the crisis each government has to deal with, but globally this can be misleading as it does not control for obvious factors like population size.  

# In[ ]:


weeks.hvplot.line(x='Date', y='ConfirmedCases', by='Country/Region', title='Confirmed Cases', width=800, height=400, legend=False)


# In[ ]:


gpd_weeks = gpd.GeoDataFrame(weeks, geometry='geometry')
gpd_weeks.hvplot(geo=True, c=gpd_weeks.ConfirmedCases, title='Confirmed Cases', cmap='Spectral_r') +gpd_weeks.hvplot(geo=True, c=gpd_weeks.Fatalities, title='Fatalities', cmap='Spectral_r')


# The obvious this we can do then is divide by the population estimates in these countries to get the number of confirmed cases and fatalities per capita, but the obvious mistake in this analysis is in not then accounting for population density as well.  We know for many countries population density is an important factor and I may argue that while we expect population density is proportional to the spread, we may also expect density to be proportional to access to health as people in rural areas may have to have healthcare professionals travel far to get tested. 

# In[ ]:


gpd_weeks.hvplot(geo=True, c=gpd_weeks.ConfirmedCases / gpd_weeks.pop_est, title='Confirmed Cases per Capita', cmap='Spectral_r') +gpd_weeks.hvplot(geo=True, c=gpd_weeks.Fatalities / gpd_weeks.pop_est, title='Fatalities per Capita', cmap='Spectral_r')


# If we control for this density, we then see a very different map of the world. While density can be misleading, as in Australia, Russia or Brazil where large parts of the country are very sparsely habbitted we assume that this clustering to large cities in large sparse countries is relatively consistent accross nations.  

# In[ ]:


gpd_weeks.hvplot(geo=True, color=(gpd_weeks.ConfirmedCases / gpd_weeks.pop_est) / (gpd_weeks.pop_est / gpd_weeks.area), title='Confirmed Cases per Capita, per Population Density', cmap='Spectral_r') +gpd_weeks.hvplot(geo=True, color=(gpd_weeks.Fatalities / gpd_weeks.pop_est) / (gpd_weeks.pop_est / gpd_weeks.area), title='Fatalities per Capita, per Population Density', cmap='Spectral_r')


# Another important factor to look at then is how this infections spreads within countries. Looking at its spread, we can see how new countries register infections through time.  One challenge in looking at this data and the growth in the infection in these countries is stage of the crisis the countries find themselves in.  In order to get a better sense of public responses to the virus, we can then look rather at the number of weeks since the first infection in a particular country rather than just the Infections per Capita, per Population Density at a given points in time. Here we see Iceland leading the pack. While this may signal the state of the infection in the country, we do that testing has been more availible in some countries that others and this may just reflect the unobserved availibility of testing not seen in the data. 

# In[ ]:


week_of_infection = (weeks)

percapita_perdensity = (gpd_weeks
                         .assign(infectionspercapita_populationdensity = lambda df: (df.ConfirmedCases / df.pop_est) / (df.pop_est / df.area))
                         .pipe(pd.DataFrame)
                         .groupby('Country/Region')
                         .apply(lambda df: (df
                                            .sort_values('Date')
                                            .assign(week_of_infection = lambda df: pd.np.arange(df.shape[0])))))

percapita_perdensity_top_mask = (percapita_perdensity
                            .groupby('Country/Region')
                            .max()
                            .nlargest(10, 'infectionspercapita_populationdensity')
                            .sort_values('infectionspercapita_populationdensity', ascending=True)
                            .index
                            .to_series())

percapita_perdensity_top = (percapita_perdensity
                            .merge(percapita_perdensity_top_mask,
                                   left_on='Country/Region',
                                   right_index=True, how='right'))

((percapita_perdensity_top
 .hvplot.line(x='Date', y='infectionspercapita_populationdensity',
              by='Country/Region',
              xlabel='Date', ylabel = 'Infections per Capita, per Population Density',
              title='Top 10 Confirmed Cases Per Capita, per Population Density',
              width=800, height=400,
              legend='right', logy=False)) + \

(percapita_perdensity_top
 .hvplot.line(x='week_of_infection', y='infectionspercapita_populationdensity',
              by='Country/Region',
              xlabel='Week of Local Infection', ylabel = 'Infections per Capita, per Population Density',
              title='Top 10 Confirmed Cases Per Capita, per Population Density by Week of Local Infection',
              width=800, height=400,
              legend='right', logy=False)))


# The next thing to control for in the dara was weeks since first infection in a given country, to get a better sense of how different countries we reponding the crisis. In order to do this I opted to do a regression model of growth in Infections per Capita, per Population Density against time and country to get a sense of Excess Growth Rate of Infections per Capita, per Population Density. While this is not a perfect model, as they are many factors such as recovery rates unaccounted for, this is still an interesting metrics to look at. 

# In[ ]:


X = pd.concat([gpd_weeks.week_of_infection.to_frame(), pd.get_dummies(gpd_weeks.loc[:,'Country/Region'])], axis=1).assign(const = 1)
y = ((gpd_weeks.ConfirmedCases / gpd_weeks.pop_est) / (gpd_weeks.pop_est / gpd_weeks.area)).apply(np.log).to_frame()

model = OLS(y, X).fit()
model.summary()


# In[ ]:


gpd_weeks_coef = gpd_weeks.merge(model.params.rename('coefficient').to_frame(), left_on='Country/Region', right_index=True)
gpd_weeks_coef.hvplot(geo=True, color='coefficient', 
                      title='Excess Growth in Confirmed Cases per Capita, per Population Density controlling for Week of Infection in Country', 
                      width=1000, height=600,
                      cmap='Spectral_r')


# I would love to hear from people on how this analysis could be improved and possible data science projects they have looking at the virus. 
