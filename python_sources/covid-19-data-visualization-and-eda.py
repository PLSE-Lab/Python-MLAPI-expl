#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import json
import geopandas as gpd
import seaborn as sns
from matplotlib import rcParams, pyplot as plt, style as style
from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar
from bokeh.palettes import viridis

train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv", index_col = 'Id')
test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv", index_col = 'ForecastId')
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)


# In[ ]:


# Data cleaning: removing typos

train_df.loc[(train_df['Country/Region'] =='Bahrain') & (train_df['Date'] == '2020-03-13'),'ConfirmedCases'] = 198
train_df.loc[(train_df['Country/Region'] =='Japan')& (train_df['Date'] == '2020-01-23'),'ConfirmedCases'] = 2
train_df.loc[(train_df['Country/Region'] =='Japan')& (train_df['Date'] == '2020-02-06'),'ConfirmedCases'] = 25
train_df.loc[(train_df['Country/Region'] =='Japan')& (train_df['Date'] == '2020-03-16'),'ConfirmedCases'] = 855
train_df.loc[(train_df['Country/Region'] =='Lebanon')& (train_df['Date'] == '2020-03-16'),'ConfirmedCases'] = 119
train_df.loc[(train_df['Country/Region'] =='Montenegro')& (train_df['Date'] == '2020-03-18'),'ConfirmedCases'] = 3
train_df.loc[(train_df['Country/Region'] =='Azerbaijan') & (train_df['Date'] == '2020-03-16'),'ConfirmedCases'] = 25
train_df.loc[(train_df['Country/Region'] =='Cruise Ship') & (train_df['ConfirmedCases'] == 696),'ConfirmedCases'] = 706


# ### Basic exploratory data analysis with visualizations using choropleth maps.
# 
#  This is an ongoing notebook that gets updated. 
# 
# ### Please upvote if you like this notebook
# 
# At first, we use `info()` function to look at the datatypes and to get an idea how many non-null values we have for each feature.
# 8788 Provinces are missing, since the majority of reports are broken down not by States or Provinces, but rather by country. Let's substitute them with 'N/A' so they don't get excluded in a `groupby` clause. Then we look at the most impacted countries and the total number of confirmed cases and deaths with Provinces/States breakdown.

# In[ ]:


train_df.info()


# In[ ]:


country_province = train_df.fillna('N/A').groupby(['Country/Region','Province/State'])['ConfirmedCases', 'Fatalities'].max().sort_values(by='ConfirmedCases', ascending=False)
country_province


# ### Top 10 countries with the highest number of cases with `seaborn.barblot` visualization. 
# 
# Let's look at the numbers of Confirmed cases and Fatalities for each country to compare. 

# In[ ]:


# Aggregate records by countries
countries = country_province.groupby('Country/Region')['ConfirmedCases','Fatalities'].sum().sort_values(by= 'ConfirmedCases',ascending=False)

countries['country'] = countries.index

# Unpivot the dataframe from wide to long format
df_long = pd.melt(countries, id_vars=['country'] , value_vars=['ConfirmedCases','Fatalities'])

#Top countries by confirmed cases
top_countries = countries.index[:10]

df_top_countries = df_long[df_long['country'].isin(top_countries)]

style.use('ggplot')
rcParams['figure.figsize'] = 15,10
ax = sns.barplot(x = 'country', hue="variable", y="value", data=df_top_countries)


# To create a static map we will utilize geospatial small scale data at 1:110m resolution. To render a world map we need a shapefile with world coordinates from Natural Earth domain:

# In[ ]:


shapefile = '/kaggle/input/110m-cultural/ne_110m_admin_0_countries.shp'

#Read shapefile using Geopandas
gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]

#Rename columns.
gdf.columns = ['country', 'code', 'geometry']


# ### Data preprocessing and cleaning

# Since the map data was gathered in 2016, some countries have been either renamed or deleted from the list. And moreover, we need to adjust namings, e.g. change 'United States of America' to 'US', etc. and delete the row corresponding to 'Antarctica'.

# In[ ]:


#Drop row corresponding to 'Antarctica'
gdf = gdf.drop(gdf.index[159])

#common = gdf.merge(df,left_on = 'country', right_on = 'Country/Region')
#train_df[(~train_df['Country/Region'].isin(common.country))]['Country/Region'].unique()
#gdf[(~gdf['country'].isin(common.country))]['country'].unique()
#gdf['country'].unique()

gdf.loc[gdf['country'] == 'Taiwan',['country']] = 'Taiwan*'
gdf.loc[gdf['country'] == 'Democratic Republic of the Congo',['country']] = 'Congo (Kinshasa)'
#gdf.loc[gdf['country'] == 'Republic of the Congo',['country']] = 'Congo (Brazzaville)'
gdf.loc[gdf['country'] == 'Ivory Coast',['country']] = "Cote d'Ivoire"
gdf.loc[gdf['country'] == 'eSwatini',['country']] = 'Eswatini'
gdf.loc[gdf['country'] == 'Gambia',['country']] = 'The Gambia'
gdf.loc[gdf['country'] == 'United Republic of Tanzania',['country']] = 'Tanzania'
gdf.loc[gdf['country'] == 'United States of America',['country']] = 'US' 
gdf.loc[gdf['country'] == 'Republic of Serbia',['country']] = 'Serbia'
gdf.loc[gdf['country'] == 'South Korea',['country']] = 'Korea, South'
gdf.loc[gdf['country'] == 'Macedonia',['country']] = 'North Macedonia'


# In[ ]:


#merge the dataset with the shapefile by country name
merged = gdf.merge(countries, left_on = 'country', right_on = 'country', how = 'left').fillna(0)
#Read data to json
merged_json = json.loads(merged.to_json())
#Convert to String like object.
json_data = json.dumps(merged_json)


# In[ ]:


from bokeh.io import curdoc, output_notebook
from bokeh.models import Slider, HoverTool
from bokeh.layouts import widgetbox, row, column

#Input GeoJSON source that contains features for plotting.
geosource = GeoJSONDataSource(geojson = json_data)

#Define a scale color palette with 250 colors.
palette = viridis(250)

#Reverse color order so that dark blue is the highest number.
palette = palette[::-1]
#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
color_mapper = LinearColorMapper(palette = palette, low = 0, high = countries.ConfirmedCases.max())

#Create color bar
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, border_line_color=None, width = 1300, height = 10, location = (0,0), orientation = 'horizontal')
#Add hover tool
hover = HoverTool(tooltips = [ ('Country/Region','@country'),('Confirmed Cases', '@ConfirmedCases')])

#Create figure object.
p = figure(title = 'Confirmed Cases of Coronavirus COVID-19', plot_height = 600 , plot_width = 950, toolbar_location = None, tools = [hover])
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

#Add patch renderer to figure. 
p.patches('xs','ys', source = geosource,fill_color = {'field' :'ConfirmedCases', 'transform' : color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)
#Specify layout
p.add_layout(color_bar, 'below')
 
# Make a column layout of widgetbox() and plot, and add it to the current document
layout = column(p,widgetbox())
curdoc().add_root(layout)
#Display plot inline in Jupyter notebook
output_notebook()
#Display plot
show(layout)


# In[ ]:


#Input GeoJSON source that contains features for plotting.
geosource = GeoJSONDataSource(geojson = json_data)
#Define a sequential multi-hue color palette.
palette = viridis(250)
#Reverse color order so that dark blue is highest confirmed cases.
palette = palette[::-1]
#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
color_mapper = LinearColorMapper(palette = palette, low = 0, high = countries.Fatalities.max())

#Create color bar
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, border_line_color=None, width = 1300, height = 10, location = (0,0), orientation = 'horizontal')
#Add hover tool
hover = HoverTool(tooltips = [ ('Country/Region','@country'),('Fatalities', '@Fatalities')])


#Create figure object.
p = figure(title = 'Fatalities of Coronavirus COVID-19', plot_height = 600 , plot_width = 950, toolbar_location = None, tools = [hover])
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
#Add patch renderer to figure. 
p.patches('xs','ys', source = geosource,fill_color = {'field' :'Fatalities', 'transform' : color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)
#Specify figure layout.
p.add_layout(color_bar, 'below')

# Make a column layout of widgetbox() and plot, and add it to the current document
layout = column(p,widgetbox())
curdoc().add_root(layout)

#Display figure inline in Jupyter Notebook.
output_notebook()
#Display figure.
show(p)


# ### Time series analysis 
# 

# In[ ]:


train_df.Date.min(), train_df.Date.max()


# In[ ]:


# Remove columns we do not need
cols = ['Lat', 'Long', 'Fatalities']
times_series_cntr = train_df.drop(cols, axis=1).fillna('N/A')

# Aggregate cases by date and country
times_series_cntr = times_series_cntr.groupby(['Date','Province/State','Country/Region'])['ConfirmedCases'].max()                    .groupby(['Date','Country/Region']).sum()                    .reset_index()

# Indexing with Time Series Data
times_series_cntr = times_series_cntr.set_index('Date')


# In[ ]:


times_series_df = times_series_cntr.groupby('Date')['ConfirmedCases'].sum().reset_index()
times_series_df = times_series_df.set_index('Date')


# #### Visualization of time series data

# In[ ]:


# Cumulative total of Confirmed cases

times_series_df.plot(figsize=(20, 10), title="The Cumulative total of Confirmed cases")
plt.legend(loc=2, prop={'size': 20})
plt.show()


# In[ ]:


# New Confirmed cases throughout the time

times_series_df.diff().fillna(0).plot(figsize=(20, 10), title="New Confirmed cases throughout the time")
plt.legend(loc=2, prop={'size': 20})
plt.show()


# In[ ]:


# What is the highest amount of cases reported on one day and when it happened?

times_series_df.diff().loc[times_series_df.diff()['ConfirmedCases']  == times_series_df.diff().fillna(0).max()['ConfirmedCases']]


# In[ ]:


top_countries_tm = times_series_cntr[times_series_cntr['Country/Region'].isin(top_countries)]
plt.xticks(rotation=45)

ax = sns.lineplot(x=top_countries_tm.index, y="ConfirmedCases", hue="Country/Region", data=top_countries_tm).set_title('Cumulative line')
plt.legend(loc=2, prop={'size': 12});


# In[ ]:


# Remove columns we do not need
cols = ['Province/State', 'Lat', 'Long', 'Fatalities']

times_series_cntr = times_series_cntr.sort_values(['Country/Region','Date'])
times_series_cntr['ConfirmedCases'] = times_series_cntr.ConfirmedCases.diff().fillna(0)
times_series_cntr.loc[times_series_cntr['ConfirmedCases'] < 0,['ConfirmedCases']] = 0 

top_countries_tm = times_series_cntr[times_series_cntr['Country/Region'].isin(top_countries)]
plt.xticks(rotation=45)
ax = sns.lineplot(x=top_countries_tm.index, y="ConfirmedCases", hue="Country/Region", data=top_countries_tm).set_title('Confirmed cases per day')
plt.legend(loc=2, prop={'size': 12});


# As it can be seen, the outlier dated Feb 13th and located in China and, actually, the day before no cases have been identified in China. Therefore we might assume that this large number could be the sum from two previous days. As well as for Mar 11th, most countries did not report any cases but the day after the number of cases rose sharply.

# We can also visualize our data using a time-series decomposition that allows us to decompose our time series into three distinct components: trend, seasonality, and noise.

# In[ ]:


from pylab import rcParams
import statsmodels.api as sm

times_series_df.index = pd.to_datetime(times_series_df.index, format='%Y-%m-%d')

rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(times_series_df.diff().fillna(0), model='additive')
fig = decomposition.plot()
plt.show()


# ### Applying Double exponential smoothing

# In[ ]:


# Remove columns we do not need
cols = ['Lat', 'Long','Date','Fatalities']
times_series_cntr_pr = train_df.drop(cols, axis=1)


# In[ ]:


provinces = train_df['Province/State'].unique()
countries = train_df['Country/Region'].unique()

def double_exponential_smoothing(series, alpha, beta):
    """
        series - dataset with timeseries
        alpha - float [0.0, 1.0], smoothing parameter for level
        beta - float [0.0, 1.0], smoothing parameter for trend
    """
    # first value is same as series
    series = list(series)
    result = [series[0]]
    for n in range(1, len(series)+14):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= (len(series)): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(int(level+trend))
    return result

def plotDoubleExponentialSmoothing(series, alphas, betas):
    """
        Plots double exponential smoothing with different alphas and betas
        
        series - dataset with timestamps
        alphas - list of floats, smoothing parameters for level
        betas - list of floats, smoothing parameters for trend
    """
    series = series.loc[series['Country/Region'] == 'Canada']
    series = series.loc[series['Province/State'] == 'Alberta']
    series = series.ConfirmedCases
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=(20, 8))
        for alpha in alphas:
            for beta in betas:
                plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
        plt.plot(series.values, label = "Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Double Exponential Smoothing")
        plt.grid(True)


# In[ ]:


plotDoubleExponentialSmoothing(times_series_cntr_pr, alphas=[0.15, 0.02], betas=[0.9, 0.09])


# In[ ]:


# Remove columns we do not need
cols = ['Lat', 'Long','Date','Fatalities']
times_series_cntr_pr = train_df.drop(cols, axis=1)

#Double exponential smoothing for Confirmed cases

countries = train_df['Country/Region'].unique()

def double_exponential_smoothing(df, alpha, beta):
    """
        series - dataset with timeseries
        alpha - float [0.0, 1.0], smoothing parameter for level
        beta - float [0.0, 1.0], smoothing parameter for trend
    """
    result =[]
    cntr = []
    prov=[]
    for c in countries:
        for p in df.loc[df['Country/Region'] == c]['Province/State'].unique():
            if p is not np.nan :
                series = df.loc[(df['Province/State'] == p) & (df['Country/Region'] == c)].ConfirmedCases
                series = list(series)
                #result.append(series[0])
                for n in range(1, len(series)+31):
                    if n == 1:
                        level, trend = series[0], series[1] - series[0]
                    if n >= len(series): # forecasting
                        value = result[-1]
                    else:
                        value = series[n]
                    last_level, level = level, alpha*value + (1-alpha)*(level+trend)
                    trend = beta*(level-last_level) + (1-beta)*trend
                    result.append(int(level+trend))
                    prov.append(p)
                    cntr.append(c)
            
            elif p is np.nan :
                series = df.loc[df['Country/Region'] == c].ConfirmedCases
                series = list(series)
                #result.append(series[0])
                for n in range(1, len(series)+31):
                    if n == 1:
                        level, trend = series[0], series[1] - series[0]
                    if n >= len(series): # forecasting
                        value = result[-1]
                    else:
                        value = series[n]
                    last_level, level = level, alpha*value + (1-alpha)*(level+trend)
                    trend = beta*(level-last_level) + (1-beta)*trend
                    result.append(int(level+trend))
                    prov.append(p)
                    cntr.append(c)

    return result, cntr, prov


# In[ ]:


t = double_exponential_smoothing(times_series_cntr_pr,0.15, 0.9)
full_cc = pd.DataFrame([t[0],t[1],t[2]], index = ['ConfirmedCases','Country/Region','Province/State'], columns= np.arange(1, len(t[0]) + 1)).T
full_cc.loc[(full_cc['ConfirmedCases'] < 0) ,'ConfirmedCases'] = 0
full_cc = full_cc.sort_values(['Country/Region','ConfirmedCases','Province/State'])


# In[ ]:


# Remove training data

total_days = len([x for x in train_df.Date.unique() if x not in test_df.Date.unique()]) + test_df.Date.nunique() #93
indeces = []
for j in range(0,284):
    for i in range(1,51):
        indeces.append((i+j*total_days))

pred_cc = full_cc.drop(indeces).reset_index().ConfirmedCases


# In[ ]:


# Remove columns we do not need
cols = ['Lat', 'Long','Date','ConfirmedCases']
times_series_cntr_f = train_df.drop(cols, axis=1)

# Double exponential smoothing for Confirmed cases

countries = train_df['Country/Region'].unique()

def double_exponential_smoothing_f(df, alpha, beta):
    """
        series - dataset with timeseries
        alpha - float [0.0, 1.0], smoothing parameter for level
        beta - float [0.0, 1.0], smoothing parameter for trend
    """
    result =[]
    cntr = []
    prov=[]
    for c in countries:
        for p in df.loc[df['Country/Region'] == c]['Province/State'].unique():
            if p is not np.nan :
                series = df.loc[(df['Province/State'] == p) & (df['Country/Region'] == c)].Fatalities
                series = list(series)
                #result.append(series[0])
                for n in range(1, len(series)+31):
                    if n == 1:
                        level, trend = series[0], series[1] - series[0]
                    if n >= len(series): # forecasting
                        value = result[-1]
                    else:
                        value = series[n]
                    last_level, level = level, alpha*value + (1-alpha)*(level+trend)
                    trend = beta*(level-last_level) + (1-beta)*trend
                    result.append(int(level+trend))
                    prov.append(p)
                    cntr.append(c)
            
            elif p is np.nan :
                series = df.loc[df['Country/Region'] == c].Fatalities
                series = list(series)
                #result.append(series[0])
                for n in range(1, len(series)+31):
                    if n == 1:
                        level, trend = series[0], series[1] - series[0]
                    if n >= len(series): # forecasting
                        value = result[-1]
                    else:
                        value = series[n]
                    last_level, level = level, alpha*value + (1-alpha)*(level+trend)
                    trend = beta*(level-last_level) + (1-beta)*trend
                    result.append(int(level+trend))
                    prov.append(p)
                    cntr.append(c)

    return result, cntr, prov


# In[ ]:


f = double_exponential_smoothing_f(times_series_cntr_f,0.15, 0.9)
full_f = pd.DataFrame([f[0],f[1],f[2]], index = ['Fatalities','Country/Region','Province/State'], columns= np.arange(1, len(f[0]) + 1)).T
full_f.loc[(full_f['Fatalities'] < 0) ,'Fatalities'] = 0

full_f = full_f.sort_values(['Country/Region','Fatalities','Province/State'])
pred_f = full_f.drop(indeces).reset_index().Fatalities


# In[ ]:


predicted_df = pd.DataFrame([pred_cc, pred_f], index = ['ConfirmedCases','Fatalities']).T
predicted_df.index += 1 
predicted_df.to_csv('submission.csv', index_label = "ForecastId")


# ### Random forest

# In[ ]:


from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import make_scorer


time_split = TimeSeriesSplit(n_splits=10)

# new cases (non cumulative)
#y_train_cc = np.array(train_df['ConfirmedCases'].diff().fillna(0).astype(int))
#y_train_ft = np.array(train_df['Fatalities'].diff().fillna(0).astype(int))

y_train_cc = np.array(train_df['ConfirmedCases'].astype(int))
y_train_ft = np.array(train_df['Fatalities'].astype(int))
cols = ['Lat', 'Long', 'ConfirmedCases', 'Fatalities']

full_df = pd.concat([train_df.drop(cols, axis=1), test_df.drop(['Lat', 'Long'],axis=1)])
index_split = train_df.shape[0]
full_df = pd.get_dummies(full_df, columns=full_df.columns)


# In[ ]:


from sklearn.metrics import mean_squared_log_error
def RMSLError(y_test, predictions):
    return np.sqrt(mean_squared_log_error(y_test, predictions))
    
rmsle_score = make_scorer(RMSLError, greater_is_better=False)


# In[ ]:


x_train = full_df[:index_split]
x_test= full_df[index_split:]
x_train.shape, x_test.shape


# In[ ]:


rf = RandomForestRegressor(n_estimators=100, n_jobs= -1, min_samples_leaf=3, random_state=17)

rf.fit(x_train,y_train_cc)


# In[ ]:


#rf_scores = cross_val_score(rf, x_train, y_train, cv=3, scoring='neg_mean_squared_log_error')
y_pred_cc = rf.predict(x_test)
y_pred_cc = y_pred_cc.astype(int)
y_pred_cc[y_pred_cc <0]=0


# In[ ]:


rf = RandomForestRegressor(n_estimators=100, n_jobs= -1, min_samples_leaf=3, random_state=17)

rf.fit(x_train,y_train_ft)


# In[ ]:


#rf_scores = cross_val_score(rf, x_train, y_train, cv=3, scoring='neg_mean_squared_log_error')
y_pred_ft = rf.predict(x_test)
y_pred_ft = y_pred_ft.astype(int)
y_pred_ft[y_pred_ft <0]=0


# In[ ]:


predicted_df_rf = pd.DataFrame([y_pred_cc, y_pred_ft], index = ['ConfirmedCases','Fatalities'], columns= np.arange(1, y_pred_cc.shape[0] + 1)).T
predicted_df_rf.to_csv('submission_rf.csv', index_label = "ForecastId")

