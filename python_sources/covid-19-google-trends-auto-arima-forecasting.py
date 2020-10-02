#!/usr/bin/env python
# coding: utf-8

# # **Covid-19 & Google Trends Auto-ARIMA Forecasting ** <br>
# TeYang<br>
# Created: 20/3/2020<br>
# Last update: 22/3/2020<br>
# 
# <img src="https://www.tvw.org/wp-content/uploads/2020/03/COVID-19_image.jpg" width="1000" height="300" align="center"/>
# 
# This kernel provides some exploratory analysis of the trajectory of the Covid-19 spread throughout the world using interactive plots from Plotly. As of now, data is still limited with few features but hopefully, that will improve when more people and organizations share their data to help fight the pandemic. As of now, the plots ignore states in countries and just sum up the total cases.
# 
# Update: I scraped data from Google Trends relating to serach queries of the virus 'Coronavirus' and 'COVID-19'. This is based on several research articles showing that google trends has the potential to help with the prediction and detection of disease outbreaks, such as the 2015 [Zika outbreak](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0165085), [Influenza](https://www.pnas.org/content/112/47/14473) and [Dengue fever](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0002713). Other search queries such as the symptoms of the virus can in turn be used to make predictions as they should be correlated with people feeling unwell and presenting with symptoms as well as physician visits. I have not yet scraped the data for search queries for symptoms keywords, but would highly encourage someone else to do it!
# 
# 
# ### What's in this kernel:
# 1. [Data Loading and Cleaning](#Data_loading_structure)
# 2. [Confirmed Cases and Deaths Across Countries/Cities](#Frequencies)
# 3. [Time Series Plots Per Country](#Line_Plots)
# 4. [Interactive Time Series Map](#Map_Data)
# 5. [Google Trends Exploration](#Google_Trends)
# 6. [Auto-ARIMA Modelling](#ARIMA)

# <a id='Data_loading_structure'></a>
# ## **1. Data Loading and Cleaning** ##

# In[ ]:


import numpy as np 
import pandas as pd 

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

# train = pd.read_csv(r'C:\Users\TeYan\OneDrive\Work\Kaggle\COVID19\Data\train.csv')
# test = pd.read_csv(r'C:\Users\TeYan\OneDrive\Work\Kaggle\COVID19\Data\test.csv')
# train = pd.read_csv('/Users/teyang/OneDrive/Work/Kaggle/COVID19/Data/train.csv')
# test = pd.read_csv('/Users/teyang/OneDrive/Work/Kaggle/COVID19/Data/test.csv')


# In[ ]:


# rename columns
train = train.rename(columns={'Province/State': 'Province_State', 'Country/Region': 'Country_Region'})
test = test.rename(columns={'Province/State': 'Province_State', 'Country/Region': 'Country_Region'})


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train['Date'].max(), test['Date'].min()


# In[ ]:


# Remove the overlapping train and test data

valid = train[train['Date'] >= test['Date'].min()] # set as validation data
train = train[train['Date'] < test['Date'].min()]
train.shape, valid.shape


# <a id='Frequencies'></a>
# ## **2. Confirmed Cases and Deaths Across Countries/Cities** ##

# In[ ]:


# Standard plotly imports
#import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.offline import iplot, init_notebook_mode, plot
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)


# In[ ]:


train_total = train[['Country_Region','Province_State','ConfirmedCases','Fatalities']]
train_total['Province_State'] = train_total['Province_State'].fillna(train_total['Country_Region']) # replace NaN States with country name
train_total = train_total.groupby(['Country_Region','Province_State'],as_index=False).agg({'ConfirmedCases': 'max', 'Fatalities': 'max'})


# In[ ]:


# pio.renderers.default = 'vscode'
pio.renderers.default = 'kaggle'

fig = px.treemap(train_total.sort_values(by='ConfirmedCases', ascending=False).reset_index(drop=True), 
                 path=["Country_Region", "Province_State"], values="ConfirmedCases", height=600, width=800,
                 title='Number of Confirmed Cases',
                 color_discrete_sequence = px.colors.qualitative.Prism)
fig.data[0].textinfo = 'label+text+value'
fig.show()

fig = px.treemap(train_total.sort_values(by='Fatalities', ascending=False).reset_index(drop=True), 
                 path=["Country_Region", "Province_State"], values="Fatalities", height=600, width=800,
                 title='Number of Deaths',
                 color_discrete_sequence = px.colors.qualitative.Prism)
fig.data[0].textinfo = 'label+text+value'
fig.show()


# <a id='Line_Plots'></a>
# ## **3. Time Series Plots Per Continent and Country** ##

# In[ ]:


# Sum countries with states, not dealing with states for now
train_agg= train[['Country_Region','Date','ConfirmedCases','Fatalities']].groupby(['Country_Region','Date'],as_index=False).agg({'ConfirmedCases': 'sum', 'Fatalities': 'sum'})

# change to datetime format
train_agg['Date'] = pd.to_datetime(train_agg['Date'])


# ### Time Series Bar Chart of Cases per Continent ###

# In[ ]:


get_ipython().system(' pip install pycountry_convert')
import pycountry_convert as pc
import pycountry
# function for getting the iso code through fuzzy search
def do_fuzzy_search(country):
    try:
        result = pycountry.countries.search_fuzzy(country)
    except Exception:
        return np.nan
    else:
        return result[0].alpha_2

train_continent = train_agg
# manually change name of some countries
train_continent.loc[train_continent['Country_Region'] == 'Korea, South', 'Country_Region'] = 'Korea, Republic of'
train_continent.loc[train_continent['Country_Region'] == 'Taiwan*', 'Country_Region'] = 'Taiwan'
# create iso mapping for countries in df
iso_map = {country: do_fuzzy_search(country) for country in train_continent['Country_Region'].unique()}
# apply the mapping to df
train_continent['iso'] = train_continent['Country_Region'].map(iso_map)
#train_continent['Continent'] = [pc.country_alpha2_to_continent_code(iso) for iso in train_continent['iso']]


# In[ ]:


def alpha2_to_continent(iso):
    try: cont = pc.country_alpha2_to_continent_code(iso)
    except: cont = float('NaN')
    return cont

train_continent['Continent'] = train_continent['iso'].apply(alpha2_to_continent) # get continent code
train_continent.loc[train_continent['iso'] == 'CN', 'Continent'] = 'CN' # Replace China's continent value as we want to keep it separate

train_continent = train_continent[['Continent','Date','ConfirmedCases','Fatalities']].groupby(['Continent','Date'],as_index=False).agg({'ConfirmedCases':'sum','Fatalities':'sum'})
train_continent['Continent'] = train_continent['Continent'].map({'AF':'Africa','AS':'Asia','CN':'China','EU':'Europe','NA':'North America','OC':'Oceania','SA':'South America'})


# In[ ]:


long = pd.melt(train_continent, id_vars=['Continent','Date'], value_vars=['ConfirmedCases','Fatalities'], var_name='Case', value_name='Count').sort_values(['Date','Count'])
long['Date'] = long['Date'].astype('str')


# In[ ]:


pio.renderers.default = 'kaggle' # does not work on vscode

# color palette
cnf = '#393e46' # confirmed - grey
dth = '#ff2e63' # death - red
# rec = '#21bf73' # recovered - cyan
# act = '#fe9801' # active case - yellow

fig = px.bar(long, y='Continent', x='Count', color='Case', barmode='group', orientation='h', text='Count', title='Counts by Continent', animation_frame='Date',
             color_discrete_sequence= [dth,cnf], range_x=[0, 100000])
fig.update_traces(textposition='outside')


# ### Time Series Bar Chart of Cases per Country ###

# In[ ]:


# Interactive time series plot of confirmed cases
fig = px.line(train_agg, x='Date', y='ConfirmedCases', color="Country_Region", hover_name="Country_Region")
fig.update_layout(autosize=False,width=1000,height=500,title='Confirmed Cases Over Time for Each Country')
fig.show()


# In[ ]:


# Interactive time series plot of fatalities
fig = px.line(train_agg, x='Date', y='Fatalities', color="Country_Region", hover_name="Country_Region")
fig.update_layout(autosize=False,width=1000,height=500,title='Fatalities Over Time for Each Country')
fig.show()


# <a id='Map_Data'></a>
# ## **4. Interactive Time Series Map** ##

# In[ ]:


## Load Natural Earth Map Data

import geopandas as gpd # for reading vector-based spatial data format
shapefile = '/kaggle/input/natural-earth-maps/ne_110m_admin_0_countries.shp'
#shapefile = r'C:\Users\TeYan\OneDrive\Work\Kaggle\COVID19\110m_cultural\ne_110m_admin_0_countries.shp'

# Read shapefile using Geopandas
#gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]
gdf = gpd.read_file(shapefile)

# Drop row corresponding to 'Antarctica'
gdf = gdf.drop(gdf.index[159])


# In[ ]:


## Get the ISO 3166-1 alpha-3 Country Codes

import pycountry
# function for getting the iso code through fuzzy search
def do_fuzzy_search(country):
    try:
        result = pycountry.countries.search_fuzzy(country)
    except Exception:
        return np.nan
    else:
        return result[0].alpha_3

# manually change name of some countries
train_agg.loc[train_agg['Country_Region'] == 'Korea, South', 'Country_Region'] = 'Korea, Republic of'
train_agg.loc[train_agg['Country_Region'] == 'Taiwan*', 'Country_Region'] = 'Taiwan'
# create iso mapping for countries in df
iso_map = {country: do_fuzzy_search(country) for country in train_agg['Country_Region'].unique()}
# apply the mapping to df
train_agg['iso'] = train_agg['Country_Region'].map(iso_map)


# In[ ]:


# # function for getting the better country name through fuzzy search
# def do_fuzzy_search_country(country):
#     try:
#         result = pycountry.countries.search_fuzzy(country)
#     except Exception:
#         return np.nan
#     else:
#         return result[0].name

# country_map = {country: do_fuzzy_search_country(country) for country in train_agg['Country_Region'].unique()}
# # apply the mapping to df
# train_agg['Country_Region'] = train_agg['Country_Region'].map(country_map)


# In[ ]:


# countries with no iso
noiso = train_agg[train_agg['iso'].isna()]['Country_Region'].unique()
# get other iso from natural earth data, create the mapping and add to our old mapping
otheriso = gdf[gdf['SOVEREIGNT'].isin(noiso)][['SOVEREIGNT','SOV_A3']]
otheriso = dict(zip(otheriso.SOVEREIGNT, otheriso.SOV_A3))
iso_map.update(otheriso)


# In[ ]:


# apply mapping and find countries with no iso again
train_agg['iso'] = train_agg['Country_Region'].map(iso_map)
train_agg[train_agg['iso'].isna()]['Country_Region'].unique()


# In[ ]:


# change date to string, not sure why plotly cannot accept datetime format
train_agg['Date'] = train_agg['Date'].dt.strftime('%Y-%m-%d')


# In[ ]:


# apply log10 so that color changes are more prominent
import numpy as np
train_agg['ConfirmedCases_log10'] = np.log10(train_agg['ConfirmedCases']).replace(-np.inf, 0) # log10 changes 0 to -inf so change back


# In[ ]:


# Interactive Map of Confirmed Cases Over Time

#pio.renderers.default = 'browser' # does not work on vscode
pio.renderers.default = 'kaggle'
fig = px.choropleth(train_agg, locations='iso', color='ConfirmedCases_log10', hover_name='Country_Region', animation_frame='Date', color_continuous_scale='reds')
fig.show()


# We can see that the virus originated in China, and spread across neighbouring Asia and Oceania in the beginning, followed by Europe and the Americas. It would be good to have travel and flight data to visualize how that influences the spread of the virus. Some countries/regions such as the Middle East were lagging perhaps due to the lack of proper virus detection measures. Much of Africa does not sufficient data at the moment.

# <a id='Google_Trends'></a>
# ## **5. Google Trends Exploration** ##
# 
# <img src="https://miro.medium.com/max/821/1*Fi6masemXJT3Q8YWekQCDQ.png" width="600" height="300" align="center"/>
# 
# 
# **Google Trends** presents a good opportunity to track the public's interest in a topic in real time and across time. It has been used in academic research to predict **Zika virus outbreak**, **Influenza** and **Dengue fever**. Here I will be using Google search queries of the keywords: **'coronavirus'** and **'COVID-19'** to explore the relationship between popularity of the virus in search queries and the actual confirmed cases for a few countries.
# 
# **NOTE:** The google trends data is **normalized** by Google and I do not think there is a way to look at the *absolute* counts of the search queries. They are on a scale of **0-100**, larger representing a higher **proportion** search or popularity of the keyword in the country. That means that for a country (e.g., San Marino) with very little search queries but a high proportion of those are related to the virus, the score will be higher than a country (e.g., Singapore), which has a high amounts of search qeuries but a lower proportion of coronavirus related searches. So although San Marino has a higher score than Singapore, it does not mean that it has a higher coronavirus-related search frequency compared to Singapore. Therefore, when comparing between countries, we are limited to looking at the **correlation** of Google search queries proportion and confirmed cases/fatalities rather than a direct comparison of the search queries counts.

# In[ ]:


# load google trends data
#cv = pd.read_csv(r'C:\Users\TeYan\OneDrive\Work\Kaggle\COVID19\GoogleTrends\coronavirus.csv', encoding = 'ISO-8859-1')
#covid = pd.read_csv(r'C:\Users\TeYan\OneDrive\Work\Kaggle\COVID19\GoogleTrends\covid19.csv', encoding = 'ISO-8859-1')

cv = pd.read_csv('/kaggle/input/covid19-googletrends/coronavirus.csv', encoding = 'ISO-8859-1')
covid = pd.read_csv('/kaggle/input/covid19-googletrends/covid19.csv', encoding = 'ISO-8859-1')


# In[ ]:


cv = cv.merge(covid, left_on=['Country','iso','date'],right_on=['Country','iso','date'],suffixes=('_cv', '_covid')) # merging removes some small countries but that's alright
cv['hits'] = cv[['hits_cv','hits_covid']].max(axis=1) # get whichever has a higher proportion between the 2 keywords

cv['iso'] = [pycountry.countries.get(alpha_2=a).alpha_3 for a in cv['iso']] # get the alpha_3 codes from the alpha_2 to merge with confirmed cases df
cc_google = train_agg.merge(cv, left_on=['iso','Date'], right_on=['iso','date']) # merge confirmed cases df with google trend df


# ### Relationship betwen Google search queries and Confirmed Cases

# In[ ]:


import seaborn as sns

sns.regplot(x='hits',y='ConfirmedCases_log10',data=cc_google,scatter_kws={'s':25},fit_reg=True, line_kws={"color": "black"})

# does not work on Kaggle
# p = sns.jointplot(x="hits", y="ConfirmedCases_log10", data=cc_google, kind='reg',
#                   joint_kws={'line_kws':{'color':'black'}}, bw=0.1)
# p.fig.set_figwidth(10)                    


# Originally, I used a jointplot to show the distribution of both features but that does not work on Kaggle because it cannot estimate the density, but somehow works on my VS Code. That plot shows lots of points that have 0 confirmed cases but with google search hits, which is difficult to see on the bottom portion of the scatter plot.

# ### Tracjectories of Google search queries and confirmed cases over time for 4 countries

# In[ ]:


popCountries = cc_google[cc_google['Country_Region'].isin(['Singapore','US','Italy','Iran'])] # select the countries

# separate confirmed cases (cc) and hits (h) columns to normalize them by group, then merge back the columns
pc_cc = popCountries[['Country_Region','Date','ConfirmedCases']] # popular countries confirmed cases
pc_f = popCountries[['Country_Region','Date','Fatalities']] # popular countries fatalities
pc_h = popCountries[['Country_Region','Date','hits']] # popular countries hits
# min-max normalization
pc_cc=pc_cc.assign(ConfirmedCases=pc_cc.groupby('Country_Region').transform(lambda x: (x - x.min()) / (x.max() - x.min()))) 
pc_f=pc_f.assign(Fatalities=pc_f.groupby('Country_Region').transform(lambda x: (x - x.min()) / (x.max() - x.min()))) 
pc_h=pc_h.assign(hits=pc_h.groupby('Country_Region').transform(lambda x: (x - x.min()) / (x.max() - x.min())))
# merge back the columns
popCountries = pc_cc.merge(pc_h, left_on=['Country_Region','Date'], right_on=['Country_Region','Date'])
popCountries = popCountries.merge(pc_f, left_on=['Country_Region','Date'], right_on=['Country_Region','Date'])
popCountries = popCountries[['Country_Region','Date','ConfirmedCases','Fatalities','hits']]
popCountries = popCountries.rename(columns={'ConfirmedCases':'val1','Fatalities':'val2','hits':'val3'})


# In[ ]:


# convert to long format for plotting
long = pd.wide_to_long(popCountries, stubnames='val', i=['Country_Region','Date'], j='CC_F_Hits').reset_index()

# Replace values with labels
def replaceVal(x):
    if x == 1: val = 'Confirmed Cases'
    elif x == 2: val = 'Fatalities'
    else: val = 'Hits'
    return val

long['CC_F_Hits'] = long['CC_F_Hits'].apply(replaceVal)


# In[ ]:


# plot facet line plots for each country
import matplotlib.pyplot as plt

g = sns.relplot(x="Date", y="val",
            hue="CC_F_Hits", col="Country_Region", col_wrap=2,
            height=4, aspect=1.45, facet_kws=dict(sharex=False),
            kind="line", legend="full", data=long)
g.set_xticklabels(rotation=45,fontsize=5,horizontalalignment='right')
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Confirmed Cases & Google Search Trajectories for 4 Countries', fontsize=16)
g.fig.set_figheight(10)
g.set_axis_labels(y_var="Normalized Confirmed Cases & Google Search Hits")


# We can see that for the 4 countries, google search queries for the keywords 'coronavirus' and 'COVID-19' tend to experience a sharp spike just before or during the start of the first few cases. This might happen because people generally only start to express interest or worry when they receive news that their region or neighbouring countries experience a sharp increase in confirmed cases. For example, in the case of Italy, although there was a brief moment of interest regarding the virus around the start of February, it died down quickly as there was no spike in confirmed cases in Europe. However, as more cases started to appear around the region, Google search queries to find out more about the virus increased quickly. We can also see that search queries will decrease after the virus has been around the country for a while, in the case of Singapore, and towards the end of the time series for Italy and Korea.
# 
# **NOTE:** Singapore experienced its first fatality on March 21 and so the it is not being plotted on the line chart above 
# 
# I am currently working on using ARIMA models to predict future cases........

# <a id='ARIMA'></a>
# ## **6. Auto-ARIMA Modelling** ##

# ### Auto-ARIMA on Iran Example

# In[ ]:


# get Iran
ir = cc_google[cc_google['Country_Region'] == 'Iran'].reset_index()
ir = ir[['Date','ConfirmedCases','Fatalities','hits']]
ir.Date = pd.to_datetime(ir.Date)
ir.index = ir.Date  # reassign the index.


# In[ ]:


# correlation of confirmed cases and google trend
import scipy.stats
sns.regplot(x='hits',y='ConfirmedCases',data=ir,scatter_kws={'s':25},fit_reg=True, line_kws={"color": "black"})

scipy.stats.pearsonr(ir.ConfirmedCases, ir.hits)


# In[ ]:


get_ipython().system(' pip install pmdarima')


# In[ ]:


# auto-arima
import pmdarima as pm

model = pm.auto_arima(ir[['ConfirmedCases']], exogenous=ir[['hits']], # include google trends data as external regressor
                        start_p=1, start_q=1,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=3, max_q=3, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0, D=0, trace=True, 
                        error_action='ignore', suppress_warnings=True, 
                        stepwise=True)
print(model.summary())


# The model with (p,d,q) = **(2,2,0)** performed the best with the lowest Akaike Information Criterion (AIC) score, which is a measure of the goodness-of-fit of the model. We see that hits, which is the external regressor (google search query) is significant.

# In[ ]:


model.plot_diagnostics(figsize=(7,5))
plt.show()


# In[ ]:


# get the exogeneous regressor values (google trend) for the forecasting period
exo = cv
exo = cv[(pd.to_datetime(cv['date']) > ir.Date.max()) & (cv['Country'] == 'Iran')]
exo.index = pd.to_datetime(exo['date'])

# get the validation confirmed cases for the forecasting period
ir_val = valid[valid.Country_Region == 'Iran']
ir_val.Date = pd.to_datetime(ir_val.Date)
ir_val.index = ir_val.Date


# In[ ]:


from datetime import timedelta

# Forecast
n_periods = 7 # validation only has 7 days now
fitted, confint = model.predict(n_periods=n_periods, 
                                  exogenous=np.tile(exo.hits, 1).reshape(-1,1), # predict using google trend
                                  return_conf_int=True)

index_of_fc = pd.date_range(ir.index[-1] + timedelta(days=1), periods = n_periods, freq='D') # get the date index range of the forecasting period

# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(ir.ConfirmedCases, label='Fitting')
plt.plot(fitted_series, color='darkgreen', label='Predicted')
plt.plot(ir_val.ConfirmedCases, color='red', label='Actual')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("ARIMA Forecast of Confirmed Cases for Iran")
plt.xticks(rotation=45, horizontalalignment='right')
plt.legend(loc="upper left")
plt.show()


# The actual confirmed cases appear to be just around the upper bound of the 95% confidence interval, which is not good! I also need to update my google trend data, and perhaps get more data of other search queries with keywords related to the virus symptoms.

# ### Questions to ask
# How long does it take from huge spike in search queries of symptoms keywords to huge increase in confirmed cases?
# 
# Can we use confirmed cases in combination with google search queries to predict subsequent cases? Using ARIMA models?
# 
# Perhaps fitting both a logistic/sigmoid function to the data as well as an exponential function and choose the one that is the best for each country
# 
# Can we predict the spread of the virus in the country/city based on the characteristics of the country/city, like its transport mobility system, cleaniness, population density etc?
# 
# How long does it take from first confirmed case to mass grocery panic?

# I hope that this notebook has been useful. I am still learning time series modelling and any advice is appreciated. 
# 
# **Please give this notebook an upvote if you like it!**
# 
# Stay safe everyone!
