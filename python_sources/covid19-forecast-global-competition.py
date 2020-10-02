#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Import" data-toc-modified-id="Import-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Import</a></span><ul class="toc-item"><li><span><a href="#Python-Libraries" data-toc-modified-id="Python-Libraries-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Python Libraries</a></span></li><li><span><a href="#Study-Settings" data-toc-modified-id="Study-Settings-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Study Settings</a></span></li><li><span><a href="#Data" data-toc-modified-id="Data-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Data</a></span></li></ul></li><li><span><a href="#Explore" data-toc-modified-id="Explore-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Explore</a></span></li><li><span><a href="#Social-diffusion-Model" data-toc-modified-id="Social-diffusion-Model-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Social diffusion Model</a></span><ul class="toc-item"><li><span><a href="#Confirmed-Cases" data-toc-modified-id="Confirmed-Cases-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Confirmed Cases</a></span></li><li><span><a href="#Deaths" data-toc-modified-id="Deaths-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Deaths</a></span></li></ul></li><li><span><a href="#Competition" data-toc-modified-id="Competition-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Competition</a></span><ul class="toc-item"><li><span><a href="#ConfirmedCases" data-toc-modified-id="ConfirmedCases-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>ConfirmedCases</a></span></li><li><span><a href="#Fatalities" data-toc-modified-id="Fatalities-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Fatalities</a></span></li><li><span><a href="#Submission-File" data-toc-modified-id="Submission-File-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Submission File</a></span></li></ul></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Conclusion</a></span></li></ul></div>

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
days_forecast = 40


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


# # Social diffusion Model

# The model is the same one as before (except with an 'offset'). The model is from a marketing paper by Emmanuelle Le Nagard and Alexandre Steyer, that attempts to reflect the social structure of a diffusion process. The paper is available (in French) [here](https://www.jstor.org/stable/40588987)
# 
# The model is also sensitive to when we define the origin of time for the epidemic process. The model has an offset parameter included and better starting conditions for the optimization algorithm. The shape of the difusion can then be expressed in the following equation:
# 
# $$N(1 - e^{-a(t-t_0)})^{\alpha}$$
# 

# What i've learnt in the following sections is that the inclusion of an offset is fickle. It can outperform the non-offset model, but when it fails, it has a much larger error than the non-offset model. For the competition then, the non-offset model is used to calculate the results for all countries. 

# ## Confirmed Cases

# In this section we start by building and displaying a model for a country (ignoring the clusters). The model is simple to compare with the previous day's results. In the cases where the COVID-19 spread is recent and the number of confirmed cases are few the model is not accurate. In these low-number cases the predictions from yesterday to today may fluctate significantly. With the countries that have cases in the 100s or 1000s however, the model is fairly stable.  

# In[ ]:


# start a dataframe with the unique countries and provinces. 
# This will be used to keep track of the model performance for each one
df_model_select = full_table.groupby(['Country/Region','Province/State']).last().reset_index()
df_model_select = df_model_select[['Country/Region','Province/State','Confirmed']] 
df_model_select['Offset error'] = 100
df_model_select['No Offset error'] = 100


# In[ ]:


countries = list(set(full_table['Country/Region']))
countries.sort()

for country in countries:
    clusters = list(set(full_table['Province/State'][(full_table['Country/Region'] == country)]))
    clusters.sort()
    
    for cluster in clusters:
        print(' ')
        print('-----------------')
        print(country + ' - ' + cluster)
        
        df = full_table[(full_table['Country/Region'] == country)&(full_table['Province/State'] == cluster)]
        df = df.groupby(['Date','Country/Region']).sum().reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Date'])
        df = df.set_index('Date')[['Confirmed']]
        df.drop(df.tail(3).index,inplace=True)
        df_result = df.copy()
        # df_result = df_result[['Date','Confirmed']]

        # ensure that the model starts from when the first case is detected
        # NOTE: its better not to truncate the dataset like this 
        # df = df[df[df.columns[0]]>0]

        # define the models to forecast the growth of cases
        def model(N, a, alpha, t0, t):
            return N * (1 - math.e ** (-a * (t-t0))) ** alpha

        def model_loss(params):
            N, a, alpha, t0 = params
            global df
            r = 0
            for t in range(len(df)):
                r += (model(N, a, alpha, t0, t) - df.iloc[t, 0]) ** 2
            return r 
        try:
            N = df['Confirmed'][-1]
            T = -df['Confirmed'][0]
        except:
            N = 10000
            T = 0

        opt = minimize(model_loss, x0=np.array([N, 0.1, 5, T]), method='Nelder-Mead', tol=1e-7).x
        print('Offset' + str(opt))

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

        # now add the results of the model to the dataframe
        df2 = pd.DataFrame(y_model,index=x_model,columns=['Offset'])
        df2.index.name = 'Date'
        df_result = pd.merge(df_result,
                             df2,
                             how='outer',
                             left_on=['Date'],
                             right_on=['Date'])

        # define the models to forecast the growth of cases
        def model(N, a, alpha, t):
            return N * (1 - math.e ** (-a * (t))) ** alpha

        def model_loss(params):
            N, a, alpha = params
            global df
            r = 0
            # error minimization should prefer (weigh more on) the larger population in this case, therefore no normalization.
            for t in range(len(df)):
                r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2
            return r 

        try:
            N = df['Confirmed'][-1]
        except:
            N = 10000

        opt = minimize(model_loss, x0=np.array([N, 0.1, 5]), method='Nelder-Mead', tol=1e-7).x
        print('No offset' + str(opt))

        try:
            start_date = pd.to_datetime(df.index[0])

            x_model = []
            y_model = []

            # get the model values for the same time series as the actuals
            for t in range(len(df) + days_forecast):
                x_model.append(start_date + datetime.timedelta(days=t))
                y_model.append(round(model(*opt,t)))

            # now add the results of the model to the dataframe
            df2 = pd.DataFrame(y_model,index=x_model,columns=['No Offset'])
            df2.index.name = 'Date'
            df_result = pd.merge(df_result,
                                 df2,
                                 how='outer',
                                 left_on=['Date'],
                                 right_on=['Date'])
            
            df_result = df_result[df_result['Confirmed'].notnull()]
            err_offset = 0
            err_no_offset = 0
            for t in range(len(df_result)):
                err_offset += (math.log(df_result['Offset'].iloc[t]+1)-math.log(df_result['Confirmed'].iloc[t]+1))**2
                err_no_offset += (math.log(df_result['No Offset'].iloc[t]+1)-math.log(df_result['Confirmed'].iloc[t]+1))**2

            err_offset = math.sqrt(err_offset/len(df_result))
            err_no_offset = math.sqrt(err_no_offset/len(df_result))
            
            df_model_select['Offset error'][(df_model_select['Country/Region']==country)&(df_model_select['Province/State']==cluster)] = err_offset
            df_model_select['No Offset error'][(df_model_select['Country/Region']==country)&(df_model_select['Province/State']==cluster)] = err_no_offset
        except:
            pass


# In[ ]:


def highlight_max(s):
    '''
    highlight the absolute maximum value in a Series with red font.
    '''
    is_min = abs(s) == abs(s).max()
    return ['color: red' if v else '' for v in is_min]

df_model_select.style.apply(highlight_max,axis=1,subset=['Offset error', 'No Offset error'])


# In[ ]:


# plot the errors
x_model = list(df_model_select['Country/Region']+' - '+ df_model_select['Province/State'])
y_model_off = list(df_model_select['Offset error'])
y_model_noff = list(df_model_select['No Offset error'])

# instantiate the figure and add the two series - actual vs modelled    
fig = go.Figure()

fig.update_layout(title='Error comparison',
                  xaxis_title='Date',
                  yaxis_title="error",
                  autosize=False,
                  width=750,
                  height=800,
                  #yaxis_type='log'
                 )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_off,
                      mode='lines',
                      name='Offset error',
                      line=dict(color='blue', 
                                width=1.5
                               )
                     ) 
             )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_noff,
                      mode='lines',
                      name='No Offset error',
                      line=dict(color='red', 
                                width=1.0,
                                dash='dot'
                               )
                     ) 
             )

fig.show()


# The figure above compares the errors of the offset model vs the no-offset model. Where the error is zero, the actuals did not have a value>0 yet.

# ## Deaths

# In[ ]:


# start a dataframe with the unique countries and provinces. 
# This will be used to keep track of the model performance for each one
df_model_select = full_table.groupby(['Country/Region','Province/State']).last().reset_index()
df_model_select = df_model_select[['Country/Region','Province/State','Deaths']] 
df_model_select['Offset error'] = 100
df_model_select['No Offset error'] = 100


# In[ ]:


countries = list(set(full_table['Country/Region']))
countries.sort()

for country in countries:
    clusters = list(set(full_table['Province/State'][(full_table['Country/Region'] == country)]))
    clusters.sort()
    
    for cluster in clusters:
        print(' ')
        print('-----------------')
        print(country + ' - ' + cluster)
        
        df = full_table[(full_table['Country/Region'] == country)&(full_table['Province/State'] == cluster)]
        df = df.groupby(['Date','Country/Region']).sum().reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Date'])
        df = df.set_index('Date')[['Deaths']]
        df.drop(df.tail(3).index,inplace=True)
        df_result = df.copy()
        # df_result = df_result[['Date','Deaths']]

        # ensure that the model starts from when the first case is detected
        # NOTE: its better not to truncate the dataset like this 
        # df = df[df[df.columns[0]]>0]

        # define the models to forecast the growth of cases
        def model(N, a, alpha, t0, t):
            return N * (1 - math.e ** (-a * (t-t0))) ** alpha

        def model_loss(params):
            N, a, alpha, t0 = params
            global df
            r = 0
            for t in range(len(df)):
                r += (model(N, a, alpha, t0, t) - df.iloc[t, 0]) ** 2
            return r 
        try:
            N = df['Deaths'][-1]
            T = -df['Deaths'][0]
        except:
            N = 10000
            T = 0

        opt = minimize(model_loss, x0=np.array([N, 0.1, 5, T]), method='Nelder-Mead', tol=1e-7).x
        print('Offset' + str(opt))

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

        # now add the results of the model to the dataframe
        df2 = pd.DataFrame(y_model,index=x_model,columns=['Offset'])
        df2.index.name = 'Date'
        df_result = pd.merge(df_result,
                             df2,
                             how='outer',
                             left_on=['Date'],
                             right_on=['Date'])

        # define the models to forecast the growth of cases
        def model(N, a, alpha, t):
            return N * (1 - math.e ** (-a * (t))) ** alpha

        def model_loss(params):
            N, a, alpha = params
            global df
            r = 0
            # error minimization should prefer (weigh more on) the larger population in this case, therefore no normalization.
            for t in range(len(df)):
                r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2
            return r 

        try:
            N = df['Deaths'][-1]
        except:
            N = 10000

        opt = minimize(model_loss, x0=np.array([N, 0.1, 5]), method='Nelder-Mead', tol=1e-7).x
        print('No offset' + str(opt))

        try:
            start_date = pd.to_datetime(df.index[0])

            x_model = []
            y_model = []

            # get the model values for the same time series as the actuals
            for t in range(len(df) + days_forecast):
                x_model.append(start_date + datetime.timedelta(days=t))
                y_model.append(round(model(*opt,t)))

            # now add the results of the model to the dataframe
            df2 = pd.DataFrame(y_model,index=x_model,columns=['No Offset'])
            df2.index.name = 'Date'
            df_result = pd.merge(df_result,
                                 df2,
                                 how='outer',
                                 left_on=['Date'],
                                 right_on=['Date'])
            
            df_result = df_result[df_result['Deaths'].notnull()]
            err_offset = 0
            err_no_offset = 0
            for t in range(len(df_result)):
                err_offset += (math.log(df_result['Offset'].iloc[t]+1)-math.log(df_result['Deaths'].iloc[t]+1))**2
                err_no_offset += (math.log(df_result['No Offset'].iloc[t]+1)-math.log(df_result['Deaths'].iloc[t]+1))**2

            err_offset = math.sqrt(err_offset/len(df_result))
            err_no_offset = math.sqrt(err_no_offset/len(df_result))
            
            df_model_select['Offset error'][(df_model_select['Country/Region']==country)&(df_model_select['Province/State']==cluster)] = err_offset
            df_model_select['No Offset error'][(df_model_select['Country/Region']==country)&(df_model_select['Province/State']==cluster)] = err_no_offset
        except:
            pass


# In[ ]:


def highlight_max(s):
    '''
    highlight the absolute maximum value in a Series with red font.
    '''
    is_min = abs(s) == abs(s).max()
    return ['color: red' if v else '' for v in is_min]

df_model_select.style.apply(highlight_max,axis=1,subset=['Offset error', 'No Offset error'])


# In[ ]:


# plot the errors
x_model = list(df_model_select['Country/Region']+' - '+ df_model_select['Province/State'])
y_model_off = list(df_model_select['Offset error'])
y_model_noff = list(df_model_select['No Offset error'])

# instantiate the figure and add the two series - actual vs modelled    
fig = go.Figure()

fig.update_layout(title='Error comparison',
                  xaxis_title='Date',
                  yaxis_title="% error",
                  autosize=False,
                  width=750,
                  height=800,
                  #yaxis_type='log'
                 )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_off,
                      mode='lines',
                      name='Offset error',
                      line=dict(color='blue', 
                                width=1.5
                               )
                     ) 
             )

fig.add_trace(go.Line(x=x_model,
                      y=y_model_noff,
                      mode='lines',
                      name='No Offset error',
                      line=dict(color='red', 
                                width=1.0,
                                dash='dot'
                               )
                     ) 
             )

fig.show()


# # Competition

# In[ ]:


df_ca_train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')
df_ca_test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')
df_ca_submission = pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')


# In[ ]:


df_ca_train.tail(10)


# Well now, this is the same dataset that we've used in the previous chapter. Just up to 18 March. So the model can be used with just the last couple of days dropped. 

# ## ConfirmedCases

# In[ ]:


full_table = df_ca_train
full_table[['Province/State']] = full_table[['Province/State']].fillna('')
df_comp = df_ca_test
df_comp[['Province/State']] = df_comp[['Province/State']].fillna('')
df_comp['Date'] = pd.to_datetime(df_comp['Date'])
df_comp['ConfirmedCases']=0
df_comp['Fatalities']=0


# In[ ]:


countries = list(set(full_table['Country/Region']))
countries.sort()

for country in countries:
    clusters = list(set(full_table['Province/State'][(full_table['Country/Region'] == country)]))
    clusters.sort()
    
    for cluster in clusters:
        print(' ')
        print('-----------------')
        print(str(country) + ' - ' + str(cluster))
        
        df = full_table[(full_table['Country/Region'] == country)&(full_table['Province/State'] == cluster)]
        df = df.groupby(['Date','Country/Region']).sum().reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Date'])
        df = df.set_index('Date')[['ConfirmedCases']]

        df_result = df.copy()
        # df_result = df_result[['Date','Confirmed']]

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
            N = df['ConfirmedCases'][-1]
        except:
            N = 10000

        opt = minimize(model_loss, x0=np.array([N, 0.1, 5]), method='Nelder-Mead', tol=1e-7).x
        print(opt)
        
        x_actual = pd.to_datetime(df.reset_index().iloc[:,0])
        x_actual = list(x_actual)
        y_actual = list(df.reset_index().iloc[:,1])
        
        start_date = pd.to_datetime(df.index[0])
        days_forecast = len(df)+len(df_ca_test)-7

        x_model = []
        y_model = []

        for t in range(days_forecast):
            x_model.append(start_date + datetime.timedelta(days=t))
            y_model.append(round(model(*opt,t)))
        
        # now add the results of the model to the competition dataframe
        df2 = pd.DataFrame(y_model,index=x_model,columns=['ConfirmedCases'])
        df2.index.name = 'Date'
        df2['Country/Region']=country
        df2['Province/State']=cluster
        df_comp = pd.merge(df_comp,
                             df2,
                             how='left',
                             on=['Date','Country/Region','Province/State']
                          )
        
        df_comp = df_comp.rename(columns={'ConfirmedCases_y': 'ConfirmedCases'})
        df_comp['ConfirmedCases'] = df_comp['ConfirmedCases'].fillna(df_comp['ConfirmedCases_x'])
        df_comp = df_comp[['ForecastId','Province/State','Country/Region','Date','ConfirmedCases','Fatalities']]


# In[ ]:


df_comp.head()


# In[ ]:





# ## Fatalities

# In[ ]:


countries = list(set(full_table['Country/Region']))
countries.sort()

for country in countries:
    clusters = list(set(full_table['Province/State'][(full_table['Country/Region'] == country)]))
    clusters.sort()
    
    for cluster in clusters:
        print(' ')
        print('-----------------')
        print(str(country) + ' - ' + str(cluster))
        
        df = full_table[(full_table['Country/Region'] == country)&(full_table['Province/State'] == cluster)]
        df = df.groupby(['Date','Country/Region']).sum().reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Date'])
        df = df.set_index('Date')[['Fatalities']]

        df_result = df.copy()

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
            N = df['Fatalities'][-1]
        except:
            N = 10000

        opt = minimize(model_loss, x0=np.array([N, 0.1, 5]), method='Nelder-Mead', tol=1e-7).x
        print(opt)
        
        x_actual = pd.to_datetime(df.reset_index().iloc[:,0])
        x_actual = list(x_actual)
        y_actual = list(df.reset_index().iloc[:,1])
        
        start_date = pd.to_datetime(df.index[0])
        days_forecast = len(df)+len(df_ca_test)-7

        x_model = []
        y_model = []

        for t in range(days_forecast):
            x_model.append(start_date + datetime.timedelta(days=t))
            y_model.append(round(model(*opt,t)))
        
        # now add the results of the model to the competition dataframe
        df2 = pd.DataFrame(y_model,index=x_model,columns=['Fatalities'])
        df2.index.name = 'Date'
        df2['Country/Region']=country
        df2['Province/State']=cluster
        df_comp = pd.merge(df_comp,
                             df2,
                             how='left',
                             on=['Date','Country/Region','Province/State']
                          )
        
        df_comp = df_comp.rename(columns={'Fatalities_y': 'Fatalities'})
        df_comp['Fatalities'] = df_comp['Fatalities'].fillna(df_comp['Fatalities_x'])
        df_comp = df_comp[['ForecastId','Province/State','Country/Region','Date','ConfirmedCases','Fatalities']]


# ## Submission File

# In[ ]:


df_comp[df_comp['Fatalities']>0]


# In[ ]:


df_comp.head()


# In[ ]:


df_ca_test.head()


# In[ ]:


df_ca_test['Date'] = pd.to_datetime(df_ca_test['Date'])


# In[ ]:


df_ca_submission.info()


# In[ ]:


df_sub = df_comp[['ForecastId','ConfirmedCases','Fatalities']]


# In[ ]:


df_sub.head()


# In[ ]:


df_sub.to_csv('submission.csv',index=False)


# # Conclusion

# Hope the results was useful, enjoy!

#  
