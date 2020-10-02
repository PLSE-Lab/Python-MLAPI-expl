#!/usr/bin/env python
# coding: utf-8

# ### Articles and posts that inspired me to do this.
# 
# https://github.com/github/covid19-dashboard
# 
# https://towardsdatascience.com/when-and-how-to-use-weighted-least-squares-wls-models-a68808b1a89d

# 1. [EDA](#eda)
# 
# 2. [Extrapolation](#expl)

# NOTE: 
# 
# Of course there are Reporting biases
# 
# Perhaps the greatest bias is that cases can only be counted if they seek out medical care or are tested. COVID-19 appears to cause mild or no symptoms in a sizeable proportion of people, which means that the reported counts underestimate the true total number of infected persons. This could also cause biases between countries --- for example, if people are told to stay home unless their disease worsens, then fewer cases will be detected than if people are told to seek medical care for mild symptoms and receive testing for the virus. Also, some countries test systematically many individuals, while other countries only test individuals with severe symptoms. This testing strategy, as well as the diagnostic criteria, may vary across time in a given country.
# 
# We are simply using the data to project further growth. 

# In[ ]:


import numpy as np 
import pandas as pd
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Visualisation libraries

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
import pycountry
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import plotly.io as pls


py.init_notebook_mode(connected=True)
import folium 
from folium import plugins
plt.style.use("fivethirtyeight")

plt.rcParams['figure.figsize'] = 8, 5


# ### There are several datasets, lets start with the main one. 

# In[ ]:


data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

print(f"Dataset is of {data.shape[0]} rows and {data.shape[1]} columns")


# In[ ]:


#open = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")
# ll =  pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")


# In[ ]:


data.head()


# In[ ]:


data.ObservationDate = pd.to_datetime(data.ObservationDate)

data['Last Update'] = pd.to_datetime(data['Last Update'])


# In[ ]:


data['Total cases'] = data['Confirmed']

data['Active cases'] = data['Total cases'] - (data['Recovered'] + data['Deaths'])

print(f'Total number of Confirmed cases:', data['Total cases'].sum())

print(f'Total number of Active cases:', data['Active cases'].sum())


# In[ ]:


grouped_conf = data.groupby('ObservationDate')['Confirmed'].sum().reset_index()

grouped_Act = data.groupby('ObservationDate')['Active cases'].sum().reset_index()


# <a id="eda"></a>
# 
# ### Confirmed and Active cases over time globally

# In[ ]:


pls.templates.default = "plotly_dark"

fig = px.line(grouped_conf, x="ObservationDate", y="Confirmed", 
              title="Worldwide Confirmed Cases Over Time")
fig.show()

fig = px.line(grouped_Act, x="ObservationDate", y="Active cases", 
              title="Worldwide Active Cases Over Time")

fig.show()


# ### Plot of China vs Non-china active cases as of 21st March

# In[ ]:


cd_group = data.groupby(['Country/Region','ObservationDate'])['Confirmed'].sum().reset_index()
cd_act = data.groupby(['Country/Region', 'ObservationDate'])['Active cases'].sum().reset_index()

ch_data = data[data['Country/Region'].str.contains("China")] 
ch_data = ch_data.groupby(['ObservationDate'])['Active cases'].sum().reset_index()

row_data = data[~data['Country/Region'].str.contains("China")] 
row_data = row_data.groupby(['ObservationDate'])['Active cases'].sum().reset_index()


# In[ ]:


row_data['ObservationDate'] = row_data['ObservationDate'].dt.date
ch_data['ObservationDate'] = ch_data['ObservationDate'].dt.date

from datetime import date
today = date.today()


# In[ ]:


lrow = row_data.tail(1)
lc =  ch_data.tail(1)
all_act = lc.merge(lrow, on='ObservationDate')
all_act.columns = ['Date', 'China', 'Rest-of-the-World']


# In[ ]:


pls.templates.default = "ggplot2"

fig = px.line(ch_data, x="ObservationDate", y="Active cases", 
              title="Active Cases in CHINA Over Time")
fig.show()


# In[ ]:


pls.templates.default = "seaborn"

fig = px.line(row_data, x="ObservationDate", y="Active cases", 
              title="Active Cases in Rest of the world Over Time")
fig.show()


# ### Proportion of Active cases in China vs Rest of the World as of 21st March

# In[ ]:


colors = ['#7FB3D5', '#D5937F']

China = all_act['China'].sum()
RoW =  all_act['Rest-of-the-World'].sum()

fig = go.Figure(data=[go.Pie(labels=['China','Rest of the World'],
                             values= [China,RoW],hole =.3)])
                          
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)))

fig.show()


# ## China has dramatically brought it down. I Wonder how! 

# ### Map View of Confirmed & Active Cases over time globally

# In[ ]:


ggdf = data.groupby(['ObservationDate', 'Country/Region'])['Confirmed'].max()
ggdf = ggdf.reset_index()

ggdf['ObservationDate'] = pd.to_datetime(ggdf['ObservationDate'])
ggdf['date'] = ggdf['ObservationDate'].dt.strftime('%m/%d/%Y')
ggdf['size'] = ggdf['Confirmed'].pow(0.4)

fig = px.scatter_geo(ggdf, locations="Country/Region", locationmode='country names', 
                     color="Confirmed",  size='size', hover_name="Country/Region", 
                     range_color= [0, 1500], 
                     projection="natural earth", animation_frame="date", 
                     title='COVID-19 Confirmed Cases Spread Over Time Globally', color_continuous_scale=px.colors.sequential.Viridis)

fig.show()


# In[ ]:


ggdf_a = data.groupby(['ObservationDate', 'Country/Region'])['Active cases'].max()
ggdf_a = ggdf_a.reset_index()

ggdf_a['ObservationDate'] = pd.to_datetime(ggdf['ObservationDate'])
ggdf_a['date'] = ggdf_a['ObservationDate'].dt.strftime('%m/%d/%Y')
ggdf_a['size'] = ggdf_a['Active cases'].pow(0.4)

fig = px.scatter_geo(ggdf_a, locations="Country/Region", locationmode='country names', 
                     color="Active cases",  size='size', hover_name="Country/Region", 
                     range_color= [0, 1500], 
                     projection="natural earth", animation_frame="date", 
                     title='COVID-19 Active Cases Spread Over Time Globally', color_continuous_scale="Inferno")

fig.show()


# <a id="expl"></a>
# 
# ### Statistical model to Extrapolate Covid-19 active cases
# 
# 
# #### The model is made by fitting a weighted least square on the log of the number of cases over a window of the few last days.
# 
# 
# 
# How is the forecast made?
# 
# We use the data from the last 14 days in each country to estimate the rate of growth for that country. We then use that estimated rate of growth to project the future growth of cases in that country, assuming that it will continue to grow at the same rate as it has in recent days.
# 
# In this way, the model is unlike those used by epidemiologists as it only uses the reported case data, rather than trying to model the actual process of the epidemic. For more information about the more complex models used by epidemiologists, see [here](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology)
# 
# More on WLS model [here](https://www.statsmodels.org/0.6.1/generated/statsmodels.regression.linear_model.WLS.html)

# In[ ]:


# We model only the active cases

da =  data.groupby(['Country/Region','ObservationDate'])['Active cases'].sum().reset_index()
# data.groupby(['Country/Region'])['Active cases'].sum().sort_values(ascending=False).head(20).reset_index()

da_last_day = da[da['ObservationDate'] > '2020-02-20']
daa = da_last_day.sort_values(by='Active cases', ascending=False)


# In[ ]:


countries = ['Mainland China', 'Italy', 'South Korea','Iran', 'Spain', 'Germany' , 'France', 'US'
, 'Switzerland', 'UK','Netherlands', 'Norway','Japan', 'Singapore', 'India', 'Canada']

most_affected_countries = daa[daa['Country/Region'].isin(countries)]


# ### Active cases comparison across majorly hit countries

# In[ ]:


# t_countries = ['Mainland China', 'Italy', 'South Korea','Singapore', 'India']
# most_affected_countries = daa[daa['Country/Region'].isin(t_countries)]


# In[ ]:


fig = px.line(most_affected_countries, x="ObservationDate", y="Active cases", color='Country/Region')

fig.show()


# ### On log-scale
# 
# 
# 
# ### Why log-scale?
# 
# The plot of cases over time includes two different options: The linear plot above shows the actual count of cases, while the log plot shows the logarithm of the number of cases - which is basically the number of times one has to multiply the number 10 in order to get the number of cases. This logarithm view has a direct relationship with the exponential growth of the epidemic: in such a view, an exponential growth appears as a straight line. You can think of the logarithm as the opposite of the exponential.
# 
# Besides, the log plot lets us more easily see the relationships between trends over time when the actual numbers are very different. Because the logarithm increasingly compresses large numbers, it makes it easier to see whether the rate of increase is similar between two countries, even when one has many more cases than the other.

# In[ ]:


fig = px.line(most_affected_countries, x="ObservationDate", y="Active cases", color='Country/Region')
fig.update_layout(yaxis_type="log")

fig.show()


# ### Defining a weighted window

# In[ ]:


## The windows that we use are weighted: we give more weight to the last day, and less to the days further away in time.

import numpy as np

def ramp_window(start=14, middle=7):
    window = np.ones(start)
    window[:middle] = np.arange(middle) / float(middle)
    window /= window.sum()
    return window

def exp_window(start=14, growth=1.1):
    window = growth ** np.arange(start)
    window /= window.sum()
    return window


# ### The weights over the last few days

# In[ ]:


window_size = 15
weighted_window = exp_window(start=window_size, growth=1.6)


# In[ ]:


import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=weighted_window))
fig.show()


# ### Log of the number of active cases in the last 15 days

# In[ ]:


# We model only the active cases

da_lf =  data.groupby(['Country/Region','ObservationDate'])['Active cases'].sum().reset_index()
# data.groupby(['Country/Region'])['Active cases'].sum().sort_values(ascending=False).head(20).reset_index()

dalf_last_day = da_lf[da_lf['ObservationDate'] > '2020-03-05']
da_lff = dalf_last_day.sort_values(by='Active cases', ascending=False)


# In[ ]:


countries = ['Mainland China', 'Italy', 'South Korea','Iran', 'Spain', 'Germany' , 'France', 'US'
, 'Switzerland', 'UK','Netherlands', 'Norway','Japan', 'Singapore', 'India', 'Canada']

most_affected_countries_last_ft = da_lff[da_lff['Country/Region'].isin(countries)]


# In[ ]:


fig = px.line(most_affected_countries_last_ft, x="ObservationDate", y="Active cases", color='Country/Region')
fig.update_layout(yaxis_type="log")
fig.show()


# In[ ]:


newf = most_affected_countries_last_ft.pivot(index='ObservationDate', columns='Country/Region')


# The errors in the data are expected to be proportional to the value of the data: the more cases are present, the more tests are realized, and the more errors as well as the more cases are missed. This noise becomes additive after taking the log.

# In[ ]:


import statsmodels.api as sm

def fit_on_window(data, window):
    """ Fit the last window of the data
    """
    window_size = len(window)
    last_fortnight = data.iloc[-window_size:]
    log_last_fortnight = np.log(last_fortnight)
    log_last_fortnight[log_last_fortnight == -np.inf] = 0

    design = pd.DataFrame({'linear': np.arange(window_size),
                           'const': np.ones(window_size)})

    growth_rate = pd.DataFrame(data=np.zeros((1, len(data.columns))),
                               columns=data.columns)

    predicted_cases = pd.DataFrame()
    predicted_cases_lower = pd.DataFrame()
    predicted_cases_upper = pd.DataFrame()
    prediction_dates = pd.date_range(data.index[-window_size],
                                    periods=window_size + 7)
    
    
    for country in data.columns:
        mod_wls = sm.WLS(log_last_fortnight[country].values, design,
                         weights=window, hasconst=True)
        res_wls = mod_wls.fit()
        growth_rate[country] = np.exp(res_wls.params.linear)
        predicted_cases[country] = np.exp(res_wls.params.const +
                res_wls.params.linear * np.arange(len(prediction_dates))
            )
        
        # 1st and 3rd quartiles in the confidence intervals
        conf_int = res_wls.conf_int(alpha=.25)
        
        
        predicted_cases_lower[country] = np.exp(res_wls.params.const +
                conf_int[0].linear * np.arange(len(prediction_dates))
            )
        predicted_cases_upper[country] = np.exp(res_wls.params.const +
                conf_int[1].linear * np.arange(len(prediction_dates))
            )

    predicted_cases = pd.concat(dict(prediction=predicted_cases,
                                     lower_bound=predicted_cases_lower,
                                     upper_bound=predicted_cases_upper),
                                axis=1)
    predicted_cases['date'] = prediction_dates
    predicted_cases = predicted_cases.set_index('date')
    if window_size > 10:
        predicted_cases  = predicted_cases.iloc[window_size - 10:]
        
    return growth_rate, predicted_cases


# In[ ]:


growth_rate, predicted_cases = fit_on_window(newf, weighted_window)


# ### The Estimated Growth Rates in the coming days

# In[ ]:


pls.templates.default = "plotly_dark"
grt = growth_rate.T.sort_values(by=0).reset_index()
grt.columns = ['level', 'country', 'growth_rate']

fig = px.bar(grt, x="country", y="growth_rate", orientation='h')
fig.show()


# In[ ]:


mt = most_affected_countries_last_ft[most_affected_countries_last_ft['ObservationDate'] > '2020-03-10']


# In[ ]:


pcs = predicted_cases.stack().reset_index()

pcs = pcs[pcs['date'] > '2020-03-21']

pcs.columns = ['date', 'country', 'lower_bound', 'prediction', 'upper_bound']


# ### Last 15 days

# In[ ]:


fig = px.line(mt, x="ObservationDate", y="Active cases", color='Country/Region')

fig.update_layout(yaxis_type="log")

fig.show()


# ### Predictions

# In[ ]:


fig = px.line(pcs, x="date", y="prediction", color='country')

fig.update_layout(yaxis_type="log")

fig.show()


# In[ ]:


#plt.figure(figsize=(20,12))
#ax = sns.lineplot(x="date", y="prediction", data=pcs, hue='country')
#ax = sns.lineplot(x="date", y="prediction", data=pcs, hue='country')
#plt.legend(loc=(.8, -.6))
#ax.set_yscale('log')
#ax.set_title('Prediction of active cases in major countries + lower and upper bounds')


# In[ ]:




