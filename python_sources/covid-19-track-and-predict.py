#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Import" data-toc-modified-id="Import-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Import</a></span><ul class="toc-item"><li><span><a href="#Python-Libraries" data-toc-modified-id="Python-Libraries-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Python Libraries</a></span></li><li><span><a href="#Study-Settings" data-toc-modified-id="Study-Settings-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Study Settings</a></span></li><li><span><a href="#Data" data-toc-modified-id="Data-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Data</a></span></li></ul></li><li><span><a href="#Explore" data-toc-modified-id="Explore-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Explore</a></span></li><li><span><a href="#Model" data-toc-modified-id="Model-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Model</a></span><ul class="toc-item"><li><span><a href="#Global-view-only-(Country-level-analysis)" data-toc-modified-id="Global-view-only-(Country-level-analysis)-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Global view only (Country level analysis)</a></span></li><li><span><a href="#Cluster-View-(by-Province/State)" data-toc-modified-id="Cluster-View-(by-Province/State)-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Cluster View (by Province/State)</a></span><ul class="toc-item"><li><span><a href="#China" data-toc-modified-id="China-4.2.1"><span class="toc-item-num">4.2.1&nbsp;&nbsp;</span>China</a></span></li><li><span><a href="#Australia" data-toc-modified-id="Australia-4.2.2"><span class="toc-item-num">4.2.2&nbsp;&nbsp;</span>Australia</a></span></li></ul></li></ul></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Conclusion</a></span></li></ul></div>

# # Introduction

# This study uses the work from the following source: https://www.kaggle.com/alixmartin/covid-19-predictions. The previous study applied a model at a country level to predict the growth of confirmed cases. 
# 
# The previous approach could not adequately explain the odd behaviours of growth for countries like China. In the China data, the curve seems to grow exponentially, then tapers off, then picks up exponentially again, and then tapers off.  
# 
# This author reckons that this behaviour exists, because the growth is cluster-based. Each cluster should be treated as a newly infected 'country' and therefore modelled seperately with  their results rolled up to predict the growth at a country or global level.
# 
# The data does not identify the clusters per country explicitly (which is probably at a town or suburb level). Therefore the study will examine it by province/state to see whether a significant improvement in accuracy can be obtained.
# 
# In future work, the study could estimate the number of clusters, and provide parameters in the model that could assist in identifying clusters that are managing the COVID-19 contagion well or poorly.  

# # Import

# ## Python Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import plotly.graph_objects as go
import warnings
import datetime
import math
from scipy.optimize import minimize


# Configure the notebook (see https://jupyter-notebook.readthedocs.io/en/stable/config.html)

# In[ ]:


# Configure Jupyter Notebook
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', 500) 
pd.set_option('display.expand_frame_repr', False)
# pd.set_option('max_colwidth', -1)
display(HTML("<style>div.output_scroll { height: 35em; }</style>"))

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format ='retina'")

warnings.filterwarnings('ignore')


# ## Study Settings

# In[ ]:


# the number of days into the future for the forecast
days_forecast = 30


# ## Data

# In[ ]:


# download the latest data sets
conf_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')
recv_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')


# In[ ]:


# create full table
dates = conf_df.columns[4:]

conf_df_long = conf_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Confirmed')

deaths_df_long = deaths_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Deaths')

recv_df_long = recv_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Recovered')

full_table = pd.concat([conf_df_long, deaths_df_long['Deaths'], recv_df_long['Recovered']], 
                       axis=1, sort=False)


# In[ ]:


# avoid double counting
full_table = full_table[full_table['Province/State'].str.contains(',')!=True]


# In[ ]:


# cases 
cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']

# Active Case = confirmed - deaths - recovered
full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']

# replacing Mainland china with just China
full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')

# filling missing values 
full_table[['Province/State']] = full_table[['Province/State']].fillna('')


# # Explore

# This section does a brief exploration of the latest data set.
# 
# The first table shows a global summary with the latest data. 

# In[ ]:


# Display the number cases globally
df = full_table.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].max()
df = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
df =  df[df['Date']==max(df['Date'])].reset_index(drop=True)
df


# The table below shows the lastest values by country

# In[ ]:


# count the number cases per country
df = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()
df = df.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
df = df.sort_values(by='Confirmed', ascending=False)
df = df.reset_index(drop=True)
df.style.background_gradient(cmap='coolwarm')


# The table below counts the number of provinces/states for each country. These numbers will be the clusters used in Chapter 4.2. 

# In[ ]:


# count the number of provinces/states per country
countries = list(set(full_table['Country/Region']))
countries.sort()
df_cluster_count = pd.DataFrame(countries,columns=['Countries'])
df_cluster_count['Clusters']=0

for country in countries:
    df = full_table[(full_table['Country/Region'] == country)]
    clusters = len(set(df['Province/State']))
    df_cluster_count['Clusters'][df_cluster_count['Countries']==country]=clusters

df_cluster_count = df_cluster_count.sort_values(by=['Clusters'],ascending=False)
df_cluster_count.style.background_gradient(cmap='coolwarm')


# # Model

# The model is the same one as before (except with an 'offset'). The model is from a marketing paper by Emmanuelle Le Nagard and Alexandre Steyer, that attempts to reflect the social structure of a diffusion process. The paper is available (in French) [here](https://www.jstor.org/stable/40588987)
# 
# The model is also sensitive to when we define the origin of time for the epidemic process. I have added an offset to the original study so that the model ended up in the following form:
# 
# $$N(1 - e^{-a(t-t_0)})^{\alpha}$$
# 

# ## Global view only (Country level analysis)

# In this section we start by building and displaying a model for a country (ignoring the clusters). The model is simple to compare with the previous day's results. In the cases where the COVID-19 spread is recent and the number of confirmed cases are few the model is not accurate. In these low-number cases the predictions from yesterday to today may fluctate significantly. With the countries that have cases in the 100s or 1000s however, the model is fairly stable.  
# 
# This will be important to know later on when we model by clusters. The smaller the number of confirmed cases per cluster, the poorer the model should perform. 

# In[ ]:


countries = list(set(full_table['Country/Region']))
countries.sort()

# NOTE: the number of charts for all countries makes this notebook large. It is better to work with sub-sets
countries = ['South Africa', 'China','Italy','Korea, South', 'Iran', 'Germany', 'Spain', 'Australia']
#countries = ['South Africa']


for country in countries:
    def get_time_series(country):
        df = full_table[(full_table['Country/Region'] == country)]
        df = df.groupby(['Date','Country/Region']).sum().reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Date'])
        return df.set_index('Date')[['Confirmed']]

    df = get_time_series(country)

    # ensure that the model starts from when the first case is detected
    df = df[df[df.columns[0]]>0]

    # define the models to forecast the growth of cases
    def model(N, a, alpha, t):
        return N * (1 - math.e ** (-a * (t))) ** alpha

    def model_loss(params):
        N, a, alpha = params
        global df
        r = 0
        for t in range(len(df)):
            r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2
        return r 
    try:
        N = df['Confirmed'][-1]
    except:
        N = 10000
    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x

    # create series to be plotted 
    x_actual = pd.to_datetime(df.reset_index().iloc[:,0])
    x_actual =list(x_actual)
    y_actual = list(df.reset_index().iloc[:,1])

    start_date = pd.to_datetime(df.index[0])

    x_model = []
    y_model = []

    # get the model values for the same time series as the actuals
    for t in range(len(df) + days_forecast):
        x_model.append(start_date + datetime.timedelta(days=t))
        y_model.append(round(model(*opt,t)))

    # instantiate the figure and add the two series - actual vs modelled    
    fig = go.Figure()
    fig.update_layout(title=country,
                      xaxis_title='Date',
                      yaxis_title="nr People",
                      autosize=False,
                      width=700,
                      height=500,
                     )

    fig.add_trace(go.Line(x=x_actual,
                          y=y_actual,
                          mode='markers',
                          name='Actual',
                          marker=dict(symbol='circle-open-dot', 
                                      size=9, 
                                      color='black', 
                                      line_width=1.5,
                                     )
                         ) 
                 )    

    fig.add_trace(go.Line(x=x_model,
                          y=y_model,
                          mode='lines',
                          name="Today's Prediction",
                          line=dict(color='blue', 
                                    width=2.5
                                   )
                         ) 
                 ) 

    
    # drop the last row of dataframe to model yesterday's results
    df.drop(df.tail(1).index,inplace=True)

    # define the models to forecast the growth of cases
    def model(N, a, alpha, t):
        return N * (1 - math.e ** (-a * (t))) ** alpha

    def model_loss(params):
        N, a, alpha = params
        global df
        r = 0
        for t in range(len(df)):
            r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2
        return r 

    try:
        N = df['Confirmed'][-1]
    except:
        N = 10000
    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x
    
    try:
        start_date = pd.to_datetime(df.index[0])

        x_model = []
        y_model = []

        # get the model values for the same time series as the actuals
        for t in range(len(df) + days_forecast):
            x_model.append(start_date + datetime.timedelta(days=t))
            y_model.append(round(model(*opt,t)))


        # now plot the new series
        fig.add_trace(go.Line(x=x_model,
                              y=y_model,
                              mode='lines',
                              name="Yesterday's Prediction",
                              line=dict(color='Red', 
                                        width=1.5,
                                        dash='dot'
                                       )
                             ) 
                     )
    except:
        pass
    
    fig.show()


# ## Cluster View (by Province/State)

# This chapter treats each province/state as a cluster, and simulates the spread of the Corona virus for each one. This is still not quite accurate enough (as seen in a couple of provinces of China). But it does better with others such as Australia. 

# ### China

# In[ ]:


country = 'China'


# In[ ]:


clusters = list(set(full_table['Province/State'][(full_table['Country/Region'] == country)]))
clusters.sort()
print('there are ' + str(len(clusters)) + ' clusters (provinces/states) found for ' + country)


# In[ ]:


# print the results of each cluster

# first set up the country dataframe
df = full_table[(full_table['Country/Region'] == country)]
df = df.groupby(['Date','Country/Region']).sum().reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Date'])
df_country = df.copy()
df_country = df_country[['Date','Confirmed']]

df_length = len(df_country) + days_forecast

# then evaluate each cluster
for cluster in clusters:
    df = full_table[(full_table['Country/Region'] == country) & (full_table['Province/State'] == cluster)]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date'])
    df = df.set_index('Date')[['Confirmed']]
    df = df[df[df.columns[0]]>0]
    
    def model(N, a, alpha, t):
        return N * (1 - math.e ** (-a * t)) ** alpha

    def model_loss(params):
        N, a, alpha = params
        global df
        r = 0
        for t in range(len(df)):
            r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2
        return r 

#     opt = minimize(model_loss, x0=np.array([100000, 0.2, 30]), method='Nelder-Mead', tol=1e-5).x
    try:
        N = df['Confirmed'][-1]
    except:
        N = 10000
    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x
    
    # create actual series to be plotted 
    x_actual = pd.to_datetime(df.reset_index().iloc[:,0])
    x_actual = list(x_actual)
    y_actual = list(df.reset_index().iloc[:,1])
    
    # create modelled series to be plotted
    start_date = pd.to_datetime(df.index[0])
    x_model = []
    y_model = []
    for t in range(len(df) + days_forecast):
        x_model.append(start_date + datetime.timedelta(days=t))
        y_model.append(round(model(*opt,t)))

    # instantiate the figure and add the two series - actual vs modelled    
    fig = go.Figure()
    fig.update_layout(title=country + ' - ' + cluster,
                      xaxis_title='Date',
                      yaxis_title="nr People",
                      autosize=False,
                      width=700,
                      height=500,
                     )

    fig.add_trace(go.Line(x=x_actual,
                          y=y_actual,
                          mode='markers',
                          name='Actual',
                          marker=dict(symbol='circle-open-dot', 
                                      size=9, 
                                      color='black', 
                                      line_width=1.5,
                                     )
                         ) 
                 )    

    fig.add_trace(go.Line(x=x_model,
                          y=y_model,
                          mode='lines',
                          name='Modelled',
                          line=dict(color='blue', 
                                    width=2
                                   )
                         ) 
                 ) 

    fig.show()
    
    # now add the results of the cluster to the country's prediction
    df = pd.DataFrame(y_model,index=x_model,columns=[cluster])
    df.index.name = 'Date'
    df_country = pd.merge(df_country,
                          df,
                          how='outer',
                          left_on=['Date'],
                          right_on=['Date'])
   


#  In the charts above we see that the model still fails with some provinces/states such as Beijing, Gansu, Hong-Kong, Hubei (significant weight), Macau, Shandong, and Shanghai. These provinces/states should be modelled as though they have more than one cluster to get better accuracy. The others are modelled with sufficient accuracy.   

# In[ ]:


df_country['Predicted from Clusters']=0
for cluster in clusters:    
    df_country[cluster].fillna(method='ffill',inplace=True)
    df_country[cluster].fillna(method='bfill',inplace=True)
    df_country['Predicted from Clusters'] = df_country['Predicted from Clusters'] + df_country[cluster]

df_country = df_country[['Date','Confirmed','Predicted from Clusters']]


# In[ ]:


def get_time_series(country):
    df = full_table[(full_table['Country/Region'] == country)]
    df = df.groupby(['Date','Country/Region']).sum().reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date'])
    return df.set_index('Date')[['Confirmed']]

df = get_time_series(country)

# ensure that the model starts from when the first case is detected
df = df[df[df.columns[0]]>0]

# define the models to forecast the growth of cases
def model(N, a, alpha, t):
    # we enforce N, a and alpha to be positive numbers using min and max functions
    return N * (1 - math.e ** (-a * t)) ** alpha
    #return max(N, 0) * (1 - math.e ** (min(-a, 0) * t)) ** max(alpha, 0)

def model_loss(params):
    N, a, alpha = params
    global df
    r = 0
    for t in range(len(df)):
        r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2
    return r 

try:
    N = df['Confirmed'][-1]
except:
    N = 10000
opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x
#     opt = minimize(model_loss, x0=np.array([500000, 0.2, 30]), method='Nelder-Mead', tol=1e-5).x

# Plot the modelled vs actual - into the future

# create series to be plotted 
x_actual = pd.to_datetime(df.reset_index().iloc[:,0])
x_actual =list(x_actual)
y_actual = list(df.reset_index().iloc[:,1])

start_date = pd.to_datetime(df.index[0])

x_model = []
y_model = []

# get the model values for the same time series as the actuals
for t in range(len(df) + days_forecast):
    x_model.append(start_date + datetime.timedelta(days=t))
    y_model.append(round(model(*opt,t)))

# now add the results of the cluster to the country's prediction
df = pd.DataFrame(y_model,index=x_model,columns=['Predicted Global'])
df.index.name = 'Date'
df_country = pd.merge(df_country,
                      df,
                      how='outer',
                      left_on=['Date'],
                      right_on=['Date'])


# In[ ]:


df_country['Cluster error'] = (df_country['Confirmed']-df_country['Predicted from Clusters'])/df_country['Confirmed']*100
df_country['Global error'] = (df_country['Confirmed']-df_country['Predicted Global'])/df_country['Confirmed']*100

def highlight_max(s):
    '''
    highlight the absolute maximum value in a Series with red font.
    '''
    is_min = abs(s) == abs(s).max()
    return ['color: red' if v else '' for v in is_min]

df_country.drop(df_country.tail(days_forecast).index,inplace=False).style.apply(highlight_max,axis=1,subset=['Cluster error', 'Global error'])


# The dataframe above shows that the prediction from clusters tend to outperform the predictions from a global pov (red is the worst error). 

# In[ ]:


# now plot the prediction for the country
x_actual = pd.to_datetime(df_country['Date'])
x_actual = list(x_actual)
y_actual = list(df_country['Confirmed'])

x_model = x_actual
y_model_clus = list(df_country['Predicted from Clusters'])
y_model_glob = list(df_country['Predicted Global'])

# instantiate the figure and add the two series - actual vs modelled    
fig = go.Figure()

fig.update_layout(title=country,
                  xaxis_title='Date',
                  yaxis_title="nr People",
                  autosize=False,
                  width=700,
                  height=500,
#                   yaxis_type='log'
                 )

fig.add_trace(go.Line(x=x_actual,
                      y=y_actual,
                      mode='markers',
                      name='Actual',
                      marker=dict(symbol='circle-open-dot', 
                                  size=9, 
                                  color='black', 
                                  line_width=1.5,
                                 )
                     ) 
             )    

fig.add_trace(go.Line(x=x_model,
                      y=y_model_clus,
                      mode='lines',
                      name='Cluster Prediction',
                      line=dict(color='blue', 
                                width=2
                               )
                     ) 
             )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_glob,
                      mode='lines',
                      name='Global Prediction',
                      line=dict(color='red', 
                                width=1.5,
                                dash='dot'
                               )
                     ) 
             )

fig.show()


# The chart above shows that the cluster approach by province/state has very little advantage over the global model for China. The error margins are compared more clearly in the figure below.

# In[ ]:


# now plot the prediction for the country
x_actual = pd.to_datetime(df_country['Date'])
x_actual = list(x_actual)
y_actual = list(df_country['Confirmed'])

x_model = x_actual
y_model_clus = list(df_country['Cluster error'])
y_model_glob = list(df_country['Global error'])

# instantiate the figure and add the two series - actual vs modelled    
fig = go.Figure()

fig.update_layout(title=country,
                  xaxis_title='Date',
                  yaxis_title="% error",
                  autosize=False,
                  width=700,
                  height=500,
                  #yaxis_type='log'
                 )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_clus,
                      mode='lines',
                      name='Cluster error',
                      line=dict(color='blue', 
                                width=1
                               )
                     ) 
             )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_glob,
                      mode='lines',
                      name='Global error',
                      line=dict(color='red', 
                                width=1.0,
                                dash='dot'
                               )
                     ) 
             )

fig.show()


# ### Australia

# In[ ]:


country = 'Australia'


# In[ ]:


clusters = list(set(full_table['Province/State'][(full_table['Country/Region'] == country)]))
clusters.sort()
print('there are ' + str(len(clusters)) + ' clusters (provinces/states) found for ' + country)


# In[ ]:


# print the results of each cluster

# first set up the country dataframe
df = full_table[(full_table['Country/Region'] == country)]
df = df.groupby(['Date','Country/Region']).sum().reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Date'])
df_country = df.copy()
df_country = df_country[['Date','Confirmed']]

df_length = len(df_country) + days_forecast

# then evaluate each cluster
for cluster in clusters:
    df = full_table[(full_table['Country/Region'] == country) & (full_table['Province/State'] == cluster)]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date'])
    df = df.set_index('Date')[['Confirmed']]
    df = df[df[df.columns[0]]>0]
    
    def model(N, a, alpha, t):
        return N * (1 - math.e ** (-a * t)) ** alpha

    def model_loss(params):
        N, a, alpha = params
        global df
        r = 0
        for t in range(len(df)):
            r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2
        return r 

#     opt = minimize(model_loss, x0=np.array([100000, 0.2, 30]), method='Nelder-Mead', tol=1e-5).x
    try:
        N = df['Confirmed'][-1]
    except:
        N = 10000
    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x
    
    # create actual series to be plotted 
    x_actual = pd.to_datetime(df.reset_index().iloc[:,0])
    x_actual = list(x_actual)
    y_actual = list(df.reset_index().iloc[:,1])
    
    # create modelled series to be plotted
    start_date = pd.to_datetime(df.index[0])
    x_model = []
    y_model = []
    for t in range(len(df) + days_forecast):
        x_model.append(start_date + datetime.timedelta(days=t))
        y_model.append(round(model(*opt,t)))

    # instantiate the figure and add the two series - actual vs modelled    
    fig = go.Figure()
    fig.update_layout(title=country + ' - ' + cluster,
                      xaxis_title='Date',
                      yaxis_title="nr People",
                      autosize=False,
                      width=700,
                      height=500,
                     )

    fig.add_trace(go.Line(x=x_actual,
                          y=y_actual,
                          mode='markers',
                          name='Actual',
                          marker=dict(symbol='circle-open-dot', 
                                      size=9, 
                                      color='black', 
                                      line_width=1.5,
                                     )
                         ) 
                 )    

    fig.add_trace(go.Line(x=x_model,
                          y=y_model,
                          mode='lines',
                          name='Modelled',
                          line=dict(color='blue', 
                                    width=2
                                   )
                         ) 
                 ) 

    fig.show()
    
    # now add the results of the cluster to the country's prediction
    df = pd.DataFrame(y_model,index=x_model,columns=[cluster])
    df.index.name = 'Date'
    df_country = pd.merge(df_country,
                          df,
                          how='outer',
                          left_on=['Date'],
                          right_on=['Date'])
   


#  In the charts above we see that the model fails with some provinces/states that have low numbers.   

# In[ ]:


df_country['Predicted from Clusters']=0
for cluster in clusters:    
    df_country[cluster].fillna(method='ffill',inplace=True)
    df_country[cluster].fillna(method='bfill',inplace=True)
    df_country['Predicted from Clusters'] = df_country['Predicted from Clusters'] + df_country[cluster]

df_country = df_country[['Date','Confirmed','Predicted from Clusters']]


# In[ ]:


def get_time_series(country):
    df = full_table[(full_table['Country/Region'] == country)]
    df = df.groupby(['Date','Country/Region']).sum().reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date'])
    return df.set_index('Date')[['Confirmed']]

df = get_time_series(country)

# ensure that the model starts from when the first case is detected
df = df[df[df.columns[0]]>0]

# define the models to forecast the growth of cases
def model(N, a, alpha, t):
    # we enforce N, a and alpha to be positive numbers using min and max functions
    return N * (1 - math.e ** (-a * t)) ** alpha
    #return max(N, 0) * (1 - math.e ** (min(-a, 0) * t)) ** max(alpha, 0)

def model_loss(params):
    N, a, alpha = params
    global df
    r = 0
    for t in range(len(df)):
        r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2
    return r 

try:
    N = df['Confirmed'][-1]
except:
    N = 10000
opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x
#     opt = minimize(model_loss, x0=np.array([500000, 0.2, 30]), method='Nelder-Mead', tol=1e-5).x

# Plot the modelled vs actual - into the future

# create series to be plotted 
x_actual = pd.to_datetime(df.reset_index().iloc[:,0])
x_actual =list(x_actual)
y_actual = list(df.reset_index().iloc[:,1])

start_date = pd.to_datetime(df.index[0])

x_model = []
y_model = []

# get the model values for the same time series as the actuals
for t in range(len(df) + days_forecast):
    x_model.append(start_date + datetime.timedelta(days=t))
    y_model.append(round(model(*opt,t)))

# now add the results of the cluster to the country's prediction
df = pd.DataFrame(y_model,index=x_model,columns=['Predicted Global'])
df.index.name = 'Date'
df_country = pd.merge(df_country,
                      df,
                      how='outer',
                      left_on=['Date'],
                      right_on=['Date'])


# In[ ]:


df_country['Cluster error'] = (df_country['Confirmed']-df_country['Predicted from Clusters'])/df_country['Confirmed']*100
df_country['Global error'] = (df_country['Confirmed']-df_country['Predicted Global'])/df_country['Confirmed']*100

def highlight_max(s):
    '''
    highlight the absolute maximum value in a Series with red font.
    '''
    is_min = abs(s) == abs(s).max()
    return ['color: red' if v else '' for v in is_min]

df_country.drop(df_country.tail(days_forecast).index,inplace=False).style.apply(highlight_max,axis=1,subset=['Cluster error', 'Global error'])


# The dataframe above shows that the prediction from clusters tend to outperform the predictions from a global pov (red is the worst error). 

# In[ ]:


# now plot the prediction for the country
x_actual = pd.to_datetime(df_country['Date'])
x_actual = list(x_actual)
y_actual = list(df_country['Confirmed'])

x_model = x_actual
y_model_clus = list(df_country['Predicted from Clusters'])
y_model_glob = list(df_country['Predicted Global'])

# instantiate the figure and add the two series - actual vs modelled    
fig = go.Figure()

fig.update_layout(title=country,
                  xaxis_title='Date',
                  yaxis_title="nr People",
                  autosize=False,
                  width=700,
                  height=500,
                  # yaxis_type='log'
                 )

fig.add_trace(go.Line(x=x_actual,
                      y=y_actual,
                      mode='markers',
                      name='Actual',
                      marker=dict(symbol='circle-open-dot', 
                                  size=9, 
                                  color='black', 
                                  line_width=1.5,
                                 )
                     ) 
             )    

fig.add_trace(go.Line(x=x_model,
                      y=y_model_clus,
                      mode='lines',
                      name='Cluster Prediction',
                      line=dict(color='blue', 
                                width=2
                               )
                     ) 
             )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_glob,
                      mode='lines',
                      name='Global Prediction',
                      line=dict(color='red', 
                                width=1.5,
                                dash='dot'
                               )
                     ) 
             )

fig.show()


# The chart above shows that the cluster approach by province/state has some advantage over the global model for Australia. The error margins are compared more clearly in the figure below.

# In[ ]:


# now plot the prediction for the country
x_actual = pd.to_datetime(df_country['Date'])
x_actual = list(x_actual)
y_actual = list(df_country['Confirmed'])

x_model = x_actual
y_model_clus = list(df_country['Cluster error'])
y_model_glob = list(df_country['Global error'])

# instantiate the figure and add the two series - actual vs modelled    
fig = go.Figure()

fig.update_layout(title=country,
                  xaxis_title='Date',
                  yaxis_title="% error",
                  autosize=False,
                  width=700,
                  height=500,
                  #yaxis_type='log'
                 )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_clus,
                      mode='lines',
                      name='Cluster error',
                      line=dict(color='blue', 
                                width=1
                               )
                     ) 
             )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_glob,
                      mode='lines',
                      name='Global error',
                      line=dict(color='red', 
                                width=1.0,
                                dash='dot'
                               )
                     ) 
             )

fig.show()


# # Conclusion

# The study has shown that there may be some advantage to the clustering approach using provinces/states as clusters, but the benefit is not substantially greater or obvious. In some instances where the cluster populations are too small, the cluster approach does worse than the global view. When the model is applied to France for instance, the global approach outperforms the cluster-based approach. 
# 
# The cluster-based approach is more sensitive to upticks or downticks (per cluster) in the data than the global model. The global model will not react and adjust as much to the latest data point.
# 
# If more clusters are introduced to the model, then the model will become even more sensitive to the latest data points. Also the model itself will have substantial errors with small populations, therefore the model might need a better optimization algorithm. 

#  

# In[ ]:


<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Import" data-toc-modified-id="Import-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Import</a></span><ul class="toc-item"><li><span><a href="#Python-Libraries" data-toc-modified-id="Python-Libraries-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Python Libraries</a></span></li><li><span><a href="#Study-Settings" data-toc-modified-id="Study-Settings-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Study Settings</a></span></li><li><span><a href="#Data" data-toc-modified-id="Data-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Data</a></span></li></ul></li><li><span><a href="#Explore" data-toc-modified-id="Explore-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Explore</a></span></li><li><span><a href="#Model" data-toc-modified-id="Model-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Model</a></span><ul class="toc-item"><li><span><a href="#Global-view-only-(Country-level-analysis)" data-toc-modified-id="Global-view-only-(Country-level-analysis)-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Global view only (Country level analysis)</a></span></li><li><span><a href="#Cluster-View-(by-Province/State)" data-toc-modified-id="Cluster-View-(by-Province/State)-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Cluster View (by Province/State)</a></span><ul class="toc-item"><li><span><a href="#China" data-toc-modified-id="China-4.2.1"><span class="toc-item-num">4.2.1&nbsp;&nbsp;</span>China</a></span></li><li><span><a href="#Australia" data-toc-modified-id="Australia-4.2.2"><span class="toc-item-num">4.2.2&nbsp;&nbsp;</span>Australia</a></span></li></ul></li></ul></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Conclusion</a></span></li></ul></div>

# Introduction

This study uses the work from the following source: https://www.kaggle.com/alixmartin/covid-19-predictions. The previous study applied a model at a country level to predict the growth of confirmed cases. 

The previous approach could not adequately explain the odd behaviours of growth for countries like China. In the China data, the curve seems to grow exponentially, then tapers off, then picks up exponentially again, and then tapers off.  

This author reckons that this behaviour exists, because the growth is cluster-based. Each cluster should be treated as a newly infected 'country' and therefore modelled seperately with  their results rolled up to predict the growth at a country or global level.

The data does not identify the clusters per country explicitly (which is probably at a town or suburb level). Therefore the study will examine it by province/state to see whether a significant improvement in accuracy can be obtained.

In future work, the study could estimate the number of clusters, and provide parameters in the model that could assist in identifying clusters that are managing the COVID-19 contagion well or poorly.  

# Import

## Python Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import plotly.graph_objects as go
import warnings
import datetime
import math
from scipy.optimize import minimize


Configure the notebook (see https://jupyter-notebook.readthedocs.io/en/stable/config.html)

# Configure Jupyter Notebook
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', 500) 
pd.set_option('display.expand_frame_repr', False)
# pd.set_option('max_colwidth', -1)
display(HTML("<style>div.output_scroll { height: 35em; }</style>"))

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format ='retina'")

warnings.filterwarnings('ignore')

## Study Settings

# the number of days into the future for the forecast
days_forecast = 30

## Data

# download the latest data sets
conf_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')
recv_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')

# create full table
dates = conf_df.columns[4:]

conf_df_long = conf_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Confirmed')

deaths_df_long = deaths_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Deaths')

recv_df_long = recv_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Recovered')

full_table = pd.concat([conf_df_long, deaths_df_long['Deaths'], recv_df_long['Recovered']], 
                       axis=1, sort=False)


# avoid double counting
full_table = full_table[full_table['Province/State'].str.contains(',')!=True]

# cases 
cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']

# Active Case = confirmed - deaths - recovered
full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']

# replacing Mainland china with just China
full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')

# filling missing values 
full_table[['Province/State']] = full_table[['Province/State']].fillna('')

# Explore

This section does a brief exploration of the latest data set.

The first table shows a global summary with the latest data. 

# Display the number cases globally
df = full_table.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].max()
df = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
df =  df[df['Date']==max(df['Date'])].reset_index(drop=True)
df

The table below shows the lastest values by country

# count the number cases per country
df = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()
df = df.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
df = df.sort_values(by='Confirmed', ascending=False)
df = df.reset_index(drop=True)
df.style.background_gradient(cmap='coolwarm')

The table below counts the number of provinces/states for each country. These numbers will be the clusters used in Chapter 4.2. 

# count the number of provinces/states per country
countries = list(set(full_table['Country/Region']))
countries.sort()
df_cluster_count = pd.DataFrame(countries,columns=['Countries'])
df_cluster_count['Clusters']=0

for country in countries:
    df = full_table[(full_table['Country/Region'] == country)]
    clusters = len(set(df['Province/State']))
    df_cluster_count['Clusters'][df_cluster_count['Countries']==country]=clusters

df_cluster_count = df_cluster_count.sort_values(by=['Clusters'],ascending=False)
df_cluster_count.style.background_gradient(cmap='coolwarm')

# Model

The model is the same one as before (except with an 'offset'). The model is from a marketing paper by Emmanuelle Le Nagard and Alexandre Steyer, that attempts to reflect the social structure of a diffusion process. The paper is available (in French) [here](https://www.jstor.org/stable/40588987)

The model is also sensitive to when we define the origin of time for the epidemic process. I have added an offset to the original study so that the model ended up in the following form:

$$N(1 - e^{-a(t-t_0)})^{\alpha}$$


## Global view only (Country level analysis)

In this section we start by building and displaying a model for a country (ignoring the clusters). The model is simple to compare with the previous day's results. In the cases where the COVID-19 spread is recent and the number of confirmed cases are few the model is not accurate. In these low-number cases the predictions from yesterday to today may fluctate significantly. With the countries that have cases in the 100s or 1000s however, the model is fairly stable.  

This will be important to know later on when we model by clusters. The smaller the number of confirmed cases per cluster, the poorer the model should perform. 

countries = list(set(full_table['Country/Region']))
countries.sort()

# NOTE: the number of charts for all countries makes this notebook large. It is better to work with sub-sets
countries = ['South Africa', 'China','Italy','Korea, South', 'Iran', 'Germany', 'Spain', 'Australia']
#countries = ['South Africa']


for country in countries:
    def get_time_series(country):
        df = full_table[(full_table['Country/Region'] == country)]
        df = df.groupby(['Date','Country/Region']).sum().reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Date'])
        return df.set_index('Date')[['Confirmed']]

    df = get_time_series(country)

    # ensure that the model starts from when the first case is detected
    df = df[df[df.columns[0]]>0]

    # define the models to forecast the growth of cases
    def model(N, a, alpha, t):
        return N * (1 - math.e ** (-a * (t))) ** alpha

    def model_loss(params):
        N, a, alpha = params
        global df
        r = 0
        for t in range(len(df)):
            r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2
        return r 
    try:
        N = df['Confirmed'][-1]
    except:
        N = 10000
    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x

    # create series to be plotted 
    x_actual = pd.to_datetime(df.reset_index().iloc[:,0])
    x_actual =list(x_actual)
    y_actual = list(df.reset_index().iloc[:,1])

    start_date = pd.to_datetime(df.index[0])

    x_model = []
    y_model = []

    # get the model values for the same time series as the actuals
    for t in range(len(df) + days_forecast):
        x_model.append(start_date + datetime.timedelta(days=t))
        y_model.append(round(model(*opt,t)))

    # instantiate the figure and add the two series - actual vs modelled    
    fig = go.Figure()
    fig.update_layout(title=country,
                      xaxis_title='Date',
                      yaxis_title="nr People",
                      autosize=False,
                      width=900,
                      height=500,
                     )

    fig.add_trace(go.Line(x=x_actual,
                          y=y_actual,
                          mode='markers',
                          name='Actual',
                          marker=dict(symbol='circle-open-dot', 
                                      size=9, 
                                      color='black', 
                                      line_width=1.5,
                                     )
                         ) 
                 )    

    fig.add_trace(go.Line(x=x_model,
                          y=y_model,
                          mode='lines',
                          name="Today's Prediction",
                          line=dict(color='blue', 
                                    width=2.5
                                   )
                         ) 
                 ) 

    
    # drop the last row of dataframe to model yesterday's results
    df.drop(df.tail(1).index,inplace=True)

    # define the models to forecast the growth of cases
    def model(N, a, alpha, t):
        return N * (1 - math.e ** (-a * (t))) ** alpha

    def model_loss(params):
        N, a, alpha = params
        global df
        r = 0
        for t in range(len(df)):
            r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2
        return r 

    try:
        N = df['Confirmed'][-1]
    except:
        N = 10000
    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x
    
    try:
        start_date = pd.to_datetime(df.index[0])

        x_model = []
        y_model = []

        # get the model values for the same time series as the actuals
        for t in range(len(df) + days_forecast):
            x_model.append(start_date + datetime.timedelta(days=t))
            y_model.append(round(model(*opt,t)))


        # now plot the new series
        fig.add_trace(go.Line(x=x_model,
                              y=y_model,
                              mode='lines',
                              name="Yesterday's Prediction",
                              line=dict(color='Red', 
                                        width=1.5,
                                        dash='dot'
                                       )
                             ) 
                     )
    except:
        pass
    
    fig.show()

## Cluster View (by Province/State)

This chapter treats each province/state as a cluster, and simulates the spread of the Corona virus for each one. This is still not quite accurate enough (as seen in a couple of provinces of China). But it does better with others such as Australia. 

### China

country = 'China'

clusters = list(set(full_table['Province/State'][(full_table['Country/Region'] == country)]))
clusters.sort()
print('there are ' + str(len(clusters)) + ' clusters (provinces/states) found for ' + country)

# print the results of each cluster

# first set up the country dataframe
df = full_table[(full_table['Country/Region'] == country)]
df = df.groupby(['Date','Country/Region']).sum().reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Date'])
df_country = df.copy()
df_country = df_country[['Date','Confirmed']]

df_length = len(df_country) + days_forecast

# then evaluate each cluster
for cluster in clusters:
    df = full_table[(full_table['Country/Region'] == country) & (full_table['Province/State'] == cluster)]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date'])
    df = df.set_index('Date')[['Confirmed']]
    df = df[df[df.columns[0]]>0]
    
    def model(N, a, alpha, t):
        return N * (1 - math.e ** (-a * t)) ** alpha

    def model_loss(params):
        N, a, alpha = params
        global df
        r = 0
        for t in range(len(df)):
            r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2
        return r 

#     opt = minimize(model_loss, x0=np.array([100000, 0.2, 30]), method='Nelder-Mead', tol=1e-5).x
    try:
        N = df['Confirmed'][-1]
    except:
        N = 10000
    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x
    
    # create actual series to be plotted 
    x_actual = pd.to_datetime(df.reset_index().iloc[:,0])
    x_actual = list(x_actual)
    y_actual = list(df.reset_index().iloc[:,1])
    
    # create modelled series to be plotted
    start_date = pd.to_datetime(df.index[0])
    x_model = []
    y_model = []
    for t in range(len(df) + days_forecast):
        x_model.append(start_date + datetime.timedelta(days=t))
        y_model.append(round(model(*opt,t)))

    # instantiate the figure and add the two series - actual vs modelled    
    fig = go.Figure()
    fig.update_layout(title=country + ' - ' + cluster,
                      xaxis_title='Date',
                      yaxis_title="nr People",
                      autosize=False,
                      width=900,
                      height=500,
                     )

    fig.add_trace(go.Line(x=x_actual,
                          y=y_actual,
                          mode='markers',
                          name='Actual',
                          marker=dict(symbol='circle-open-dot', 
                                      size=9, 
                                      color='black', 
                                      line_width=1.5,
                                     )
                         ) 
                 )    

    fig.add_trace(go.Line(x=x_model,
                          y=y_model,
                          mode='lines',
                          name='Modelled',
                          line=dict(color='blue', 
                                    width=2
                                   )
                         ) 
                 ) 

    fig.show()
    
    # now add the results of the cluster to the country's prediction
    df = pd.DataFrame(y_model,index=x_model,columns=[cluster])
    df.index.name = 'Date'
    df_country = pd.merge(df_country,
                          df,
                          how='outer',
                          left_on=['Date'],
                          right_on=['Date'])
   

 In the charts above we see that the model still fails with some provinces/states such as Beijing, Gansu, Hong-Kong, Hubei (significant weight), Macau, Shandong, and Shanghai. These provinces/states should be modelled as though they have more than one cluster to get better accuracy. The others are modelled with sufficient accuracy.   

df_country['Predicted from Clusters']=0
for cluster in clusters:    
    df_country[cluster].fillna(method='ffill',inplace=True)
    df_country[cluster].fillna(method='bfill',inplace=True)
    df_country['Predicted from Clusters'] = df_country['Predicted from Clusters'] + df_country[cluster]

df_country = df_country[['Date','Confirmed','Predicted from Clusters']]

def get_time_series(country):
    df = full_table[(full_table['Country/Region'] == country)]
    df = df.groupby(['Date','Country/Region']).sum().reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date'])
    return df.set_index('Date')[['Confirmed']]

df = get_time_series(country)

# ensure that the model starts from when the first case is detected
df = df[df[df.columns[0]]>0]

# define the models to forecast the growth of cases
def model(N, a, alpha, t):
    # we enforce N, a and alpha to be positive numbers using min and max functions
    return N * (1 - math.e ** (-a * t)) ** alpha
    #return max(N, 0) * (1 - math.e ** (min(-a, 0) * t)) ** max(alpha, 0)

def model_loss(params):
    N, a, alpha = params
    global df
    r = 0
    for t in range(len(df)):
        r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2
    return r 

try:
    N = df['Confirmed'][-1]
except:
    N = 10000
opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x
#     opt = minimize(model_loss, x0=np.array([500000, 0.2, 30]), method='Nelder-Mead', tol=1e-5).x

# Plot the modelled vs actual - into the future

# create series to be plotted 
x_actual = pd.to_datetime(df.reset_index().iloc[:,0])
x_actual =list(x_actual)
y_actual = list(df.reset_index().iloc[:,1])

start_date = pd.to_datetime(df.index[0])

x_model = []
y_model = []

# get the model values for the same time series as the actuals
for t in range(len(df) + days_forecast):
    x_model.append(start_date + datetime.timedelta(days=t))
    y_model.append(round(model(*opt,t)))

# now add the results of the cluster to the country's prediction
df = pd.DataFrame(y_model,index=x_model,columns=['Predicted Global'])
df.index.name = 'Date'
df_country = pd.merge(df_country,
                      df,
                      how='outer',
                      left_on=['Date'],
                      right_on=['Date'])

df_country['Cluster error'] = (df_country['Confirmed']-df_country['Predicted from Clusters'])/df_country['Confirmed']*100
df_country['Global error'] = (df_country['Confirmed']-df_country['Predicted Global'])/df_country['Confirmed']*100

def highlight_max(s):
    '''
    highlight the absolute maximum value in a Series with red font.
    '''
    is_min = abs(s) == abs(s).max()
    return ['color: red' if v else '' for v in is_min]

df_country.drop(df_country.tail(days_forecast).index,inplace=False).style.apply(highlight_max,axis=1,subset=['Cluster error', 'Global error'])


The dataframe above shows that the prediction from clusters tend to outperform the predictions from a global pov (red is the worst error). 

# now plot the prediction for the country
x_actual = pd.to_datetime(df_country['Date'])
x_actual = list(x_actual)
y_actual = list(df_country['Confirmed'])

x_model = x_actual
y_model_clus = list(df_country['Predicted from Clusters'])
y_model_glob = list(df_country['Predicted Global'])

# instantiate the figure and add the two series - actual vs modelled    
fig = go.Figure()

fig.update_layout(title=country,
                  xaxis_title='Date',
                  yaxis_title="nr People",
                  autosize=False,
                  width=900,
                  height=500,
#                   yaxis_type='log'
                 )

fig.add_trace(go.Line(x=x_actual,
                      y=y_actual,
                      mode='markers',
                      name='Actual',
                      marker=dict(symbol='circle-open-dot', 
                                  size=9, 
                                  color='black', 
                                  line_width=1.5,
                                 )
                     ) 
             )    

fig.add_trace(go.Line(x=x_model,
                      y=y_model_clus,
                      mode='lines',
                      name='Cluster Prediction',
                      line=dict(color='blue', 
                                width=2
                               )
                     ) 
             )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_glob,
                      mode='lines',
                      name='Global Prediction',
                      line=dict(color='red', 
                                width=1.5,
                                dash='dot'
                               )
                     ) 
             )

fig.show()

The chart above shows that the cluster approach by province/state has very little advantage over the global model for China. The error margins are compared more clearly in the figure below.

# now plot the prediction for the country
x_actual = pd.to_datetime(df_country['Date'])
x_actual = list(x_actual)
y_actual = list(df_country['Confirmed'])

x_model = x_actual
y_model_clus = list(df_country['Cluster error'])
y_model_glob = list(df_country['Global error'])

# instantiate the figure and add the two series - actual vs modelled    
fig = go.Figure()

fig.update_layout(title=country,
                  xaxis_title='Date',
                  yaxis_title="% error",
                  autosize=False,
                  width=900,
                  height=500,
                  #yaxis_type='log'
                 )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_clus,
                      mode='lines',
                      name='Cluster error',
                      line=dict(color='blue', 
                                width=1
                               )
                     ) 
             )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_glob,
                      mode='lines',
                      name='Global error',
                      line=dict(color='red', 
                                width=1.0,
                                dash='dot'
                               )
                     ) 
             )

fig.show()

### Australia

country = 'Australia'

clusters = list(set(full_table['Province/State'][(full_table['Country/Region'] == country)]))
clusters.sort()
print('there are ' + str(len(clusters)) + ' clusters (provinces/states) found for ' + country)

# print the results of each cluster

# first set up the country dataframe
df = full_table[(full_table['Country/Region'] == country)]
df = df.groupby(['Date','Country/Region']).sum().reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Date'])
df_country = df.copy()
df_country = df_country[['Date','Confirmed']]

df_length = len(df_country) + days_forecast

# then evaluate each cluster
for cluster in clusters:
    df = full_table[(full_table['Country/Region'] == country) & (full_table['Province/State'] == cluster)]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date'])
    df = df.set_index('Date')[['Confirmed']]
    df = df[df[df.columns[0]]>0]
    
    def model(N, a, alpha, t):
        return N * (1 - math.e ** (-a * t)) ** alpha

    def model_loss(params):
        N, a, alpha = params
        global df
        r = 0
        for t in range(len(df)):
            r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2
        return r 

#     opt = minimize(model_loss, x0=np.array([100000, 0.2, 30]), method='Nelder-Mead', tol=1e-5).x
    try:
        N = df['Confirmed'][-1]
    except:
        N = 10000
    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x
    
    # create actual series to be plotted 
    x_actual = pd.to_datetime(df.reset_index().iloc[:,0])
    x_actual = list(x_actual)
    y_actual = list(df.reset_index().iloc[:,1])
    
    # create modelled series to be plotted
    start_date = pd.to_datetime(df.index[0])
    x_model = []
    y_model = []
    for t in range(len(df) + days_forecast):
        x_model.append(start_date + datetime.timedelta(days=t))
        y_model.append(round(model(*opt,t)))

    # instantiate the figure and add the two series - actual vs modelled    
    fig = go.Figure()
    fig.update_layout(title=country + ' - ' + cluster,
                      xaxis_title='Date',
                      yaxis_title="nr People",
                      autosize=False,
                      width=900,
                      height=500,
                     )

    fig.add_trace(go.Line(x=x_actual,
                          y=y_actual,
                          mode='markers',
                          name='Actual',
                          marker=dict(symbol='circle-open-dot', 
                                      size=9, 
                                      color='black', 
                                      line_width=1.5,
                                     )
                         ) 
                 )    

    fig.add_trace(go.Line(x=x_model,
                          y=y_model,
                          mode='lines',
                          name='Modelled',
                          line=dict(color='blue', 
                                    width=2
                                   )
                         ) 
                 ) 

    fig.show()
    
    # now add the results of the cluster to the country's prediction
    df = pd.DataFrame(y_model,index=x_model,columns=[cluster])
    df.index.name = 'Date'
    df_country = pd.merge(df_country,
                          df,
                          how='outer',
                          left_on=['Date'],
                          right_on=['Date'])
   

 In the charts above we see that the model fails with some provinces/states that have low numbers.   

df_country['Predicted from Clusters']=0
for cluster in clusters:    
    df_country[cluster].fillna(method='ffill',inplace=True)
    df_country[cluster].fillna(method='bfill',inplace=True)
    df_country['Predicted from Clusters'] = df_country['Predicted from Clusters'] + df_country[cluster]

df_country = df_country[['Date','Confirmed','Predicted from Clusters']]

def get_time_series(country):
    df = full_table[(full_table['Country/Region'] == country)]
    df = df.groupby(['Date','Country/Region']).sum().reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date'])
    return df.set_index('Date')[['Confirmed']]

df = get_time_series(country)

# ensure that the model starts from when the first case is detected
df = df[df[df.columns[0]]>0]

# define the models to forecast the growth of cases
def model(N, a, alpha, t):
    # we enforce N, a and alpha to be positive numbers using min and max functions
    return N * (1 - math.e ** (-a * t)) ** alpha
    #return max(N, 0) * (1 - math.e ** (min(-a, 0) * t)) ** max(alpha, 0)

def model_loss(params):
    N, a, alpha = params
    global df
    r = 0
    for t in range(len(df)):
        r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2
    return r 

try:
    N = df['Confirmed'][-1]
except:
    N = 10000
opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x
#     opt = minimize(model_loss, x0=np.array([500000, 0.2, 30]), method='Nelder-Mead', tol=1e-5).x

# Plot the modelled vs actual - into the future

# create series to be plotted 
x_actual = pd.to_datetime(df.reset_index().iloc[:,0])
x_actual =list(x_actual)
y_actual = list(df.reset_index().iloc[:,1])

start_date = pd.to_datetime(df.index[0])

x_model = []
y_model = []

# get the model values for the same time series as the actuals
for t in range(len(df) + days_forecast):
    x_model.append(start_date + datetime.timedelta(days=t))
    y_model.append(round(model(*opt,t)))

# now add the results of the cluster to the country's prediction
df = pd.DataFrame(y_model,index=x_model,columns=['Predicted Global'])
df.index.name = 'Date'
df_country = pd.merge(df_country,
                      df,
                      how='outer',
                      left_on=['Date'],
                      right_on=['Date'])

df_country['Cluster error'] = (df_country['Confirmed']-df_country['Predicted from Clusters'])/df_country['Confirmed']*100
df_country['Global error'] = (df_country['Confirmed']-df_country['Predicted Global'])/df_country['Confirmed']*100

def highlight_max(s):
    '''
    highlight the absolute maximum value in a Series with red font.
    '''
    is_min = abs(s) == abs(s).max()
    return ['color: red' if v else '' for v in is_min]

df_country.drop(df_country.tail(days_forecast).index,inplace=False).style.apply(highlight_max,axis=1,subset=['Cluster error', 'Global error'])


The dataframe above shows that the prediction from clusters tend to outperform the predictions from a global pov (red is the worst error). 

# now plot the prediction for the country
x_actual = pd.to_datetime(df_country['Date'])
x_actual = list(x_actual)
y_actual = list(df_country['Confirmed'])

x_model = x_actual
y_model_clus = list(df_country['Predicted from Clusters'])
y_model_glob = list(df_country['Predicted Global'])

# instantiate the figure and add the two series - actual vs modelled    
fig = go.Figure()

fig.update_layout(title=country,
                  xaxis_title='Date',
                  yaxis_title="nr People",
                  autosize=False,
                  width=900,
                  height=500,
                  # yaxis_type='log'
                 )

fig.add_trace(go.Line(x=x_actual,
                      y=y_actual,
                      mode='markers',
                      name='Actual',
                      marker=dict(symbol='circle-open-dot', 
                                  size=9, 
                                  color='black', 
                                  line_width=1.5,
                                 )
                     ) 
             )    

fig.add_trace(go.Line(x=x_model,
                      y=y_model_clus,
                      mode='lines',
                      name='Cluster Prediction',
                      line=dict(color='blue', 
                                width=2
                               )
                     ) 
             )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_glob,
                      mode='lines',
                      name='Global Prediction',
                      line=dict(color='red', 
                                width=1.5,
                                dash='dot'
                               )
                     ) 
             )

fig.show()

The chart above shows that the cluster approach by province/state has some advantage over the global model for Australia. The error margins are compared more clearly in the figure below.

# now plot the prediction for the country
x_actual = pd.to_datetime(df_country['Date'])
x_actual = list(x_actual)
y_actual = list(df_country['Confirmed'])

x_model = x_actual
y_model_clus = list(df_country['Cluster error'])
y_model_glob = list(df_country['Global error'])

# instantiate the figure and add the two series - actual vs modelled    
fig = go.Figure()

fig.update_layout(title=country,
                  xaxis_title='Date',
                  yaxis_title="% error",
                  autosize=False,
                  width=900,
                  height=500,
                  #yaxis_type='log'
                 )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_clus,
                      mode='lines',
                      name='Cluster error',
                      line=dict(color='blue', 
                                width=1
                               )
                     ) 
             )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_glob,
                      mode='lines',
                      name='Global error',
                      line=dict(color='red', 
                                width=1.0,
                                dash='dot'
                               )
                     ) 
             )

fig.show()

# Conclusion

The study has shown that there may be some advantage to the clustering approach using provinces/states as clusters, but the benefit is not substantially greater or obvious. In some instances where the cluster populations are too small, the cluster approach does worse than the global view. When the model is applied to France for instance, the global approach outperforms the cluster-based approach. 

The cluster-based approach is more sensitive to upticks or downticks (per cluster) in the data than the global model. The global model will not react and adjust as much to the latest data point.

If more clusters are introduced to the model, then the model will become even more sensitive to the latest data points. Also the model itself will have substantial errors with small populations, therefore the model might need a better optimization algorithm. 

 

