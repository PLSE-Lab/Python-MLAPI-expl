#!/usr/bin/env python
# coding: utf-8

# The idea behind this notebook is to make an analysis on the CoronaVirus dataset using the packages **Plotly**(Data Visualisation) and **Prophet** (Prediction).

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load libraries and some utility functions

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.figure_factory as ff
from plotly import subplots
from plotly.subplots import make_subplots
import ipywidgets as widgets
init_notebook_mode(connected=True)


from datetime import datetime, date, timedelta

from fbprophet import Prophet

import warnings
warnings.filterwarnings('ignore')


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# In[ ]:


def get_data(path):
    tab_name = ['Deaths','Recovered']
    jc = ['Province/State','Country/Region']
    path_c = path+'Confirmed.csv'
    df = pd.read_csv(path_c)
    df = pd.melt(df, id_vars=['Province/State','Country/Region','Lat','Long'],
            var_name='Date', value_name= 'Confirmed')
    df['Date'] = pd.to_datetime(df['Date'])
    for name in tab_name:
        path_ = path+name+'.csv'
        data = pd.read_csv(path_)
        data = pd.melt(data, id_vars=['Province/State','Country/Region','Lat','Long'], 
                    var_name='Date', value_name= name)
        data['Date'] = pd.to_datetime(data['Date'])
        df[name] = data[name].values
        
        
    return(df)




def prepare_data(df):
    num_col = ['Confirmed','Deaths','Recovered'] 
    new_col = ['PS','Country','Lat','Long','Date','Confirmed','Deaths','Recovered']
    df.columns = new_col
    df[num_col] = df[num_col].apply(lambda x: x.fillna(value = 0))
    df[num_col] = df[num_col].astype(np.int32)
    df['Country'] = np.where(df['Country'] == 'Mainland China','China',df['Country'])
    df['PS'] = np.where(df['PS'].isnull(), df['Country'],df['PS'])
    
    return(df)
 
def check_anomalies(df):
    count_c = df.loc[(df['Confirmed_'] <0)].shape[0]
    count_d = df.loc[(df['Deaths_'] <0)].shape[0]
    count_r = df.loc[(df['Recovered_'] <0)].shape[0]
    
    print("Number of negative Confirmed_: {}\n".format(count_c))
    print("Number of negative Deaths_: {}\n".format(count_d))
    print("Number of negative Recovered_: {}\n".format(count_r))
    
def rebinnable_interactive_histogram(series, title,initial_bin_width=10):
    trace = go.Histogram(
        x=series,
        xbins={"size": initial_bin_width},
        marker_color = 'rgb(55, 83, 109)',
    )
    figure_widget = go.FigureWidget(
        data=[trace],
        layout=go.Layout(yaxis={"title": "Count"}, xaxis={"title": "x"}, bargap=0.05,
                        title = 'Histogram of Corfirmed Case - {}'.format(title)),
    )

    bin_slider = widgets.FloatSlider(
        value=initial_bin_width,
        min=5,
        max=24,
        step=2,
        description="Bin width:",
        readout_format=".0f", 
    )

    histogram_object = figure_widget.data[0]

    def set_bin_size(change):
        histogram_object.xbins = {"size": change["new"]}

    bin_slider.observe(set_bin_size, names="value")

    output_widget = widgets.VBox([figure_widget, bin_slider])
    return output_widget


# I am using the data extracted from the Time series google sheet, now on github https://github.com/CSSEGISandData/COVID-19. This is due to the fact that I wanted to use the data with the cumulative distribution of **Confirmed**, **Deaths** and **Recovered**.
# 
# I am leaving the **get_data** function, used to read the data directly from the repository on **GitHub**.
# 
# In this version of the notebook I am using the data without the time information, in order to have just $1$ sample each day, instead of having multiple records. I do not think that the time information is usefull since it is not the real time in which the case was discovered.

# In[ ]:


#path = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-'
df = pd.read_csv('/kaggle/input/timeseries/nCov_daily.csv',index_col = 0)


# In[ ]:


df.head(10)


# Since there were several NaN values in the last three columns, I decided to fill them with $0$, because they were the starting point of the cumulates. Moreover, I filled the **NaN** values in the **Province/State** column, entering the Country if the value was missing and changed the value Mainland China to China for semplicity.
# Furthermore I renamed the columns and cast some columns to the correct type.

# In[ ]:


df = prepare_data(df)


# In[ ]:


df.head(10)


# In[ ]:


df.isnull().sum()


# In[ ]:


print("Number of rows in the dataset: {}".format(df.shape[0]))
print("Number of Columns in the dataset: {}".format(df.shape[1]))


# I want to show how each day gone by the number of **Confirmed**,**Deaths** and **Recovered** changed, so I created a new Dataframe **sorted_df** in which I had $3$ columns, containing the number of cases, for each category, happened during that day.

# In[ ]:


sorted_df = df.sort_values(['Country','PS', 'Date'])
cols = ['Country', 'PS']
sorted_df['Confirmed_'] = np.where(np.all(sorted_df.shift()[cols]==sorted_df[cols], axis=1)
                                   , sorted_df['Confirmed'].diff(), sorted_df['Confirmed'])
sorted_df['Deaths_'] = np.where(np.all(sorted_df.shift()[cols]==sorted_df[cols], axis=1),
                                sorted_df['Deaths'].diff(), sorted_df['Deaths'])
sorted_df['Recovered_'] = np.where(np.all(sorted_df.shift()[cols]==sorted_df[cols], axis=1),
                                   sorted_df['Recovered'].diff(), sorted_df['Recovered'])


# Now I want to check if the computation of the daily **Confirmed**, **Deaths** and **Recovered** is correct, i.e if I only get positive values.

# In[ ]:


check_anomalies(sorted_df)


# I did not find a method to correct automatically these errors, if you have any idea please let me know!

# In[ ]:


sorted_df.loc[sorted_df['Confirmed_']<0]


# In[ ]:


# Queensland
df.loc[[651,801],'Confirmed'] = 2
# Japan
df.loc[107,'Confirmed'] = 2
df.loc[1157,'Confirmed'] = 25


# In[ ]:


sorted_df.loc[sorted_df['Recovered_']<0]


# In[ ]:


# Guangxi
df.loc[1506,'Recovered'] = 32
# Guizhou
df.loc[1057,'Recovered'] = 6
# Hainan
df.loc[1733,'Recovered'] = 37
# Heilongjiang
df.loc[1435,'Recovered'] = 21
# Ningxia
df.loc[1294,'Recovered'] = 9
# Shanxi
df.loc[849,'Recovered'] = 2


# As it's shown in previous cells, the values are incorrect. I changed manually, by leaving the value of the previous record.<br>
# Then I created again the **sorted_df**.

# In[ ]:


sorted_df = df.sort_values(['Country','PS', 'Date'])
cols = ['Country', 'PS']
sorted_df['Confirmed_'] = np.where(np.all(sorted_df.shift()[cols]==sorted_df[cols], axis=1)
                                   , sorted_df['Confirmed'].diff(), sorted_df['Confirmed'])
sorted_df['Deaths_'] = np.where(np.all(sorted_df.shift()[cols]==sorted_df[cols], axis=1),
                                sorted_df['Deaths'].diff(), sorted_df['Deaths'])
sorted_df['Recovered_'] = np.where(np.all(sorted_df.shift()[cols]==sorted_df[cols], axis=1),
                                   sorted_df['Recovered'].diff(), sorted_df['Recovered'])


# In[ ]:


check_anomalies(sorted_df)


# Now that data seems to be clean, I will start with some Exploratory Data Analysis.

# ## Exploratory Data Analysis

# Let's see the spread of the virus in China and in the rest of the world.

# Let's see if we can create some classes for the **Confirmed** cases, **Rest of World**.<br>
# This is done because I want to represent on a map the cases and I need some intervals to divide the number of cases.<br>
# By switching of the different number of bins of the widget, we can se how the distribution behave

# In[ ]:


df_world = df.loc[df['Country'] != 'China'].groupby(['Country'])[['PS','Long','Lat','Confirmed']].max().reset_index()
rebinnable_interactive_histogram(df_world.Confirmed,'Rest of the World')


# In[ ]:


limits = [0,13,27,41,83,df_world.Confirmed.max()+1]
df_world['text'] = 'Country: ' + df_world['Country'].astype(str) + '<br>Province/State ' + (df_world['PS']).astype(str) + '<br>Confirmed: ' + (df_world['Confirmed']).astype(str)
fig = go.Figure()

for i in range(len(limits)):
    lim = limits[i]
    df_sub = df_world.loc[(df_world['Confirmed'] < lim) & (df_world['Confirmed'] >= limits[i-1])]
    fig.add_trace(go.Scattergeo(
        locationmode = 'country names',
        lon = df_sub.Long,
        lat = df_sub.Lat,
        text = df_sub['text'],
        marker = dict(
            reversescale = True,
            size = df_sub.Confirmed*1.1,
            color = df_sub.Confirmed,
            colorscale = 'geyser',
            line_color='rgb(40,40,40)',
            line_width=0.5,
            sizemode = 'area'
        ),
        name = '{0}-{1}'.format(limits[i-1],limits[i])
    )
                 )

fig.update_layout(
        title_text = 'Confirmed cases Rest of the World',
        showlegend = True,
        geo = dict(
            scope = 'world',
            projection_type = 'natural earth',
            showcountries = True,
            showocean = False,
        )
    )

fig.show()


# You won't find the **Hubei** region in the next plot, giving the higher value with respect of the others.

# In[ ]:


df_china = df.loc[df['Country'] == 'China'].groupby(['PS'])[['Country','Long','Lat','Confirmed']].max().reset_index()
rebinnable_interactive_histogram(df_china.loc[df_china['PS'] != 'Hubei'].Confirmed,'China (not Hubei)',initial_bin_width=24)


# In[ ]:


limits = [0,100,200,300,400,500,600,1000,1500,df_china.Confirmed.max()+1]

df_china['text'] = 'Province/State: ' + (df_china['PS']).astype(str) + '<br>Confirmed: ' + (df_china['Confirmed']).astype(str)
fig = go.Figure()

for i in range(len(limits)):
    df_sub = df_china.loc[(df_china['Confirmed'] < limits[i]) & (df_china['Confirmed'] >= limits[i-1])]
    fig.add_trace(go.Scattergeo(
        locationmode = 'country names',
        lon = df_sub.Long,
        lat = df_sub.Lat,
        text = df_sub['text'],
        marker = dict(
            opacity = .7,
            size = df_sub.Confirmed/10,
            color = df_sub.Confirmed.max(),
            colorscale = 'geyser',
            line_color='rgb(40,40,40)',
            line_width=0.5,
            sizemode = 'area'
        ),
        name = '{0}-{1}'.format(limits[i-1],limits[i])
    )
                 )

fig.update_layout(
        title = {'text': 'Corona Virus spreading in Asia',
                                'y':0.98,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'},
        showlegend = True,
        geo = dict(
            scope = 'asia',
            projection = go.layout.geo.Projection(
            type = 'kavrayskiy7',
            scale=1.2
            ),
            showcountries = True,
            
        )
    )

fig.show()


# Now I will look more into the details of the data

# In[ ]:


fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=sorted_df.Country.loc[sorted_df['Country'] != 'China'],
        y=sorted_df.Confirmed_.loc[sorted_df['Country'] != 'China'],
        name='Rest of the world',
        marker_color='rgb(55, 83, 109)',
        text = sorted_df.Date.astype(str),
        hovertemplate =
        '<br><b>Country</b>: %{x} <br>' +
        '<b>Confirmed Cases:</b> %{y}<br>' +
        '<b>Date:</b> %{text}<br>'
    )
)

fig.update_layout(
    title={'text': 'Confirmed case all over the world',
           'y':0.95,
           'x':0.5,
           'xanchor': 'center',
           'yanchor': 'top'},
    xaxis_tickfont_size=14,
    xaxis=dict(tickangle=45),
    yaxis=dict(
        title='',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.1,
    bargroupgap=0.1,
    hoverlabel_align = 'left'
)
fig.show()


# By moving on the plot, it is shown how many cases happened during a specific day. It's easy to notice the abundance of cases mark as **Others** that were only confirmed in $3$ days. As far as the other contries, they did not have the same behaviour.

# In[ ]:


fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=sorted_df.PS.loc[sorted_df['Country'] == 'China'],
        y=sorted_df.Confirmed_.loc[sorted_df['Country'] == 'China'],
        name='China',
        marker_color='rgb(26, 118, 255)',
        text = sorted_df.loc[sorted_df['Country'] == 'China'].Date.astype(str),
        hovertemplate =
        '<br><b>Province</b>: %{x} <br>' +
        '<b>Confirmed Cases:</b> %{y}<br>' +
        '<b>Date:</b> %{text}<br>'
    )
)

fig.update_layout(
    title={'text': 'Confirmed case in China',
           'y':0.95,
           'x':0.5,
           'xanchor': 'center',
           'yanchor': 'top'},
    xaxis_tickfont_size=14,
    xaxis=dict(tickangle=45),
    yaxis=dict(
        title='',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
    hoverlabel_align = 'left',
)
fig.show()

fig2 = go.Figure()
fig2.add_trace(
    go.Bar(
        x=sorted_df.PS.loc[(sorted_df['Country'] == 'China') & (sorted_df['PS'] != 'Hubei')],
        y=sorted_df.Confirmed_.loc[(sorted_df['Country'] == 'China') & (sorted_df['PS'] != 'Hubei')],
        name='China',
        marker_color='rgb(26, 118, 255)',
        text = df.Date.loc[(sorted_df['Country'] == 'China') & (sorted_df['PS'] != 'Hubei')].astype(str),
        hovertemplate =
        '<br><b>Province</b>: %{x} <br>' +
        '<b>Confirmed Cases:</b> %{y}<br>' +
        '<b>Date:</b> %{text}<br>'
    )
)

fig2.update_layout(
    title={'text': 'Confirmed case in China (not Hubei)',
           'y':0.95,
           'x':0.5,
           'xanchor': 'center',
           'yanchor': 'top'},
    xaxis_tickfont_size=14,
    xaxis=dict(tickangle=45),
    yaxis=dict(
        title='',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
    hoverlabel_align = 'left',
)
fig2.show()


# Above you can see two barplots with only **Confirmed** case in **China**. I decided to use $2$ representation because the cases in the **Hubei** region are simply too much higher than the others.

# In[ ]:


fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=sorted_df.Country.loc[sorted_df['Country'] != 'China'],
        y=sorted_df.Deaths_.loc[sorted_df['Country'] != 'China'],
        name='Deaths',
        marker_color='rgb(55, 83, 109)',
        text = sorted_df.Date.astype(str),
        hovertemplate =
        '<br><b>Country</b>: %{x} <br>' +
        '<b>Death Cases:</b> %{y}<br>' +
        '<b>Date:</b> %{text}<br>'
    )
)
fig.add_trace(
    go.Bar(
        x=sorted_df.Country.loc[sorted_df['Country'] != 'China'],
        y=sorted_df.Recovered_.loc[sorted_df['Country'] != 'China'],
        name='Recovered',
        marker_color='rgb(26, 118, 255)',
        text = sorted_df.Date.astype(str),
        hovertemplate =
        '<br><b>Country</b>: %{x} <br>' +
        '<b>Recovered Cases:</b> %{y}<br>' +
        '<b>Date:</b> %{text}<br>'
    )
)
fig.update_layout(
    title={'text': 'Deaths & Recovered case all over the world',
           'y':0.95,
           'x':0.5,
           'xanchor': 'center',
           'yanchor': 'top'},
    xaxis_tickfont_size=14,
    xaxis=dict(tickangle=45),
    yaxis=dict(
        title='',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=1,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, 
    bargroupgap=0.1,
    hoverlabel_align = 'left',
)
fig.show()

df_not_hubei = sorted_df.loc[sorted_df['PS'] != 'Hubei']


fig2 = go.Figure()
fig2.add_trace(
    go.Bar(
        x=df_not_hubei.PS.loc[df_not_hubei['Country'] == 'China'],
        y=df_not_hubei.Deaths_.loc[df_not_hubei['Country'] == 'China'],
        name='Deaths',
        marker_color='rgb(55, 83, 109)',
        text = sorted_df.Date.astype(str),
        hovertemplate =
        '<br><b>Country</b>: %{x} <br>' +
        '<b>Death Cases:</b> %{y}<br>' +
        '<b>Date:</b> %{text}<br>'
    )
)
fig2.add_trace(
    go.Bar(
        x=df_not_hubei.PS.loc[df_not_hubei['Country'] == 'China'],
        y=df_not_hubei.Recovered_.loc[df_not_hubei['Country'] == 'China'],
        name='Recovered',
        marker_color='rgb(26, 118, 255)',
        text = sorted_df.Date.astype(str),
        hovertemplate =
        '<br><b>Country</b>: %{x} <br>' +
        '<b>Recovered Cases:</b> %{y}<br>' +
        '<b>Date:</b> %{text}<br>'
    )
)
fig2.update_layout(
    title={'text': 'Deaths & Recovered case in China (not Hubei)',
           'y':0.95,
           'x':0.5,
           'xanchor': 'center',
           'yanchor': 'top'},
    xaxis_tickfont_size=14,
    xaxis=dict(tickangle=45),
    yaxis=dict(
        title='',
        titlefont_size=16,
        tickfont_size=14
        ,range = [0, df_not_hubei['Recovered'].max() + 10]
    ),
    legend=dict(
        x=1,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, 
    bargroupgap=0.1 ,
    hoverlabel_align = 'left',
)
fig2.show()


# Above you can see the same barplot, but this time showing the **Deaths** and **Recovered** cases.

# In[ ]:


df_hubei = sorted_df.loc[sorted_df['PS'] == 'Hubei']

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=df_hubei.Date,
        y=df_hubei.Deaths,
        name='Deaths',
        mode='lines+markers',
        marker_color='rgb(55, 83, 109)',
         hovertemplate =
        '<br><b>Date</b>: %{x} <br>' +
        '<b>Death Cases:</b> %{y}<br>'
    )
)

fig2.add_trace(
    go.Scatter(
        x=df_hubei.Date,
        y=df_hubei.Recovered,
        name='Recovered',
        marker_color='rgb(26, 118, 255)',
         hovertemplate =
        '<br><b>Date</b>: %{x} <br>' +
        '<b>Recovered Cases:</b> %{y}<br>'
    )
)

fig2.update_traces(
    mode='lines+markers',
    marker_line_width=2,
    marker_size=5
)

fig2.update_layout(
    title={'text': 'Deaths and Recovered in Hubei (China)',
           'y':0.95,
           'x':0.5,
           'xanchor': 'center',
           'yanchor': 'top'},
    yaxis_zeroline=False,
    xaxis_zeroline=False,
    hoverlabel_align= 'left',
)

fig2.show()


# And now the plot of confirmed cases for the **Hubei** region. A before that magnitude of the data is really different from the other, so I decided to split it and represent only in one plot.

# In[ ]:


df_hubei['confirmed_case_world'] = df_not_hubei.groupby('Date').sum()['Confirmed'].values

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x = df_hubei.Date,
        y = df_hubei.Confirmed,
        name = 'Hubei',
        mode = 'lines+markers',
        marker_color = 'rgb(55,83,109)',
        hovertemplate =
        '<br><b>Date</b>: %{x} <br>' +
        '<b>Confirmed Cases:</b> %{y}<br>'
    )
)

fig.add_trace(
    go.Scatter(
        x=df_hubei.Date,
        y=df_hubei.confirmed_case_world,
        name='Other',
        marker_color='rgb(26, 118, 255)',
        hovertemplate =
        '<b>Date</b>: %{x} <br>' +
        '<b>Confirmed Cases:</b> %{y}<br>'
    )
)

fig.update_traces(mode='lines+markers',
                  marker_line_width=2,
                  marker_size=5)
fig.update_layout(
    title={'text': 'Confermed case in Hubei vs Rest of World',
           'y':0.95,
           'x':0.5,
           'xanchor': 'center',
           'yanchor': 'top'},
    yaxis_zeroline=False,
    xaxis_zeroline=False,
    hoverlabel_align = 'left',
)

fig.show()


# By looking at the plot, it is shown the difference between the confirmed cases of *Hubei* against the rest of the world.
# 
# Hence I decided to use only the data related to the *Hubei* region in my **Predictive Analisys**.

# ## Time Series Analysis - Prophet

# In[ ]:


from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric, add_changepoints_to_plot, plot_plotly


# In[ ]:


df_prophet = df_hubei[['Date','Confirmed']]
df_prophet.columns = ['ds','y']


# Just some thoughts.
# 
# The **Date** feature has equi-spaced intervals, since now it does not contains the timing information.I chose to not considered that because  to the fact that the data are stored at some time during the day and it is not real time.<br>
# <br>
# Moreover, we have a small sample of data.

# ### Basic Models

# Let's start by modeling a baseline model, including the daily trend. I do not think that this would be useful since the hour in the feature **Date** are not the real one in which the new confirmed case is registered.

# In[ ]:


m_d = Prophet(
    yearly_seasonality=False,
    weekly_seasonality = False,
    daily_seasonality = True,
    seasonality_mode = 'additive')
m_d.fit(df_prophet)
future_d = m_d.make_future_dataframe(periods=7)
fcst_daily = m_d.predict(future_d)


# In[ ]:


trace1 = {
  "fill": None, 
  "mode": "markers", 
  "name": "actual no. of Confirmed", 
  "type": "scatter", 
  "x": df_prophet.ds, 
  "y": df_prophet.y
}
trace2 = {
  "fill": "tonexty", 
  "line": {"color": "#57b8ff"}, 
  "mode": "lines", 
  "name": "upper_band", 
  "type": "scatter", 
  "x": fcst_daily.ds, 
  "y": fcst_daily.yhat_upper
}
trace3 = {
  "fill": "tonexty", 
  "line": {"color": "#57b8ff"}, 
  "mode": "lines", 
  "name": "lower_band", 
  "type": "scatter", 
  "x": fcst_daily.ds, 
  "y": fcst_daily.yhat_lower
}
trace4 = {
  "line": {"color": "#eb0e0e"}, 
  "mode": "lines+markers", 
  "name": "prediction", 
  "type": "scatter", 
  "x": fcst_daily.ds, 
  "y": fcst_daily.yhat
}
data = [trace1, trace2, trace3, trace4]
layout = {
  "title": "Confirmed - Time Series Forecast - Daily Trend", 
  "xaxis": {
    "title": "", 
    "ticklen": 5, 
    "gridcolor": "rgb(255, 255, 255)", 
    "gridwidth": 2, 
    "zerolinewidth": 1
  }, 
  "yaxis": {
    "title": "Confirmed nCov - Hubei", 
    "ticklen": 5, 
    "gridcolor": "rgb(255, 255, 255)", 
    "gridwidth": 2, 
    "zerolinewidth": 1
  }, 
}
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# Let's try now by removing the **daily_seasonality**

# In[ ]:


m_nd = Prophet(
    yearly_seasonality=False,
    weekly_seasonality = False,
    daily_seasonality = False,
    seasonality_mode = 'additive')
m_nd.fit(df_prophet)
future_nd = m_nd.make_future_dataframe(periods=7)
fcst_no_daily = m_nd.predict(future_nd)


# In[ ]:


trace1 = {
  "fill": None, 
  "mode": "markers", 
  "name": "actual no. of Confirmed", 
  "type": "scatter", 
  "x": df_prophet.ds, 
  "y": df_prophet.y
}
trace2 = {
  "fill": "tonexty", 
  "line": {"color": "#57b8ff"}, 
  "mode": "lines", 
  "name": "upper_band", 
  "type": "scatter", 
  "x": fcst_no_daily.ds, 
  "y": fcst_no_daily.yhat_upper
}
trace3 = {
  "fill": "tonexty", 
  "line": {"color": "#57b8ff"}, 
  "mode": "lines", 
  "name": "lower_band", 
  "type": "scatter", 
  "x": fcst_no_daily.ds, 
  "y": fcst_no_daily.yhat_lower
}
trace4 = {
  "line": {"color": "#eb0e0e"}, 
  "mode": "lines+markers", 
  "name": "prediction", 
  "type": "scatter", 
  "x": fcst_no_daily.ds, 
  "y": fcst_no_daily.yhat
}
data = [trace1, trace2, trace3, trace4]
layout = {
  "title": "Confirmed - Time Series Forecast", 
  "xaxis": {
    "title": "", 
    "ticklen": 5, 
    "gridcolor": "rgb(255, 255, 255)", 
    "gridwidth": 2, 
    "zerolinewidth": 1
  }, 
  "yaxis": {
    "title": "Confirmed nCov - Hubei", 
    "ticklen": 5, 
    "gridcolor": "rgb(255, 255, 255)", 
    "gridwidth": 2, 
    "zerolinewidth": 1
  }, 
}
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# It does not seem that the model are able to perfor very well.
# 
# Let's see how the two models perform in terms of **Mean Absolute Percentage Error**.

# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[ ]:


max_date = df_prophet.ds.max()
y_true = df_prophet.y.values
y_pred_daily = fcst_daily.loc[fcst_daily['ds'] <= max_date].yhat.values
y_pred_no_daily = fcst_no_daily.loc[fcst_no_daily['ds'] <= max_date].yhat.values


# In[ ]:


print('MAPE with daily seasonality: {}'.format(mean_absolute_percentage_error(y_true,y_pred_daily)))
print('MAPE without daily seasonality: {}'.format(mean_absolute_percentage_error(y_true,y_pred_no_daily)))


# It is pretty clear that the models perform very bad. This wa pretty obvious just by looking at the plot of the models. It seems that the model is not able to recognize the pattern of the data.

# Let's try to add some parameters into the both model and see if something changes, hoping for an improvement.

# In[ ]:


m_d = Prophet(
    changepoint_prior_scale=20,
    seasonality_prior_scale=20,
    n_changepoints=19,
    changepoint_range=0.9,
    yearly_seasonality=False,
    weekly_seasonality = False,
    daily_seasonality = True,
    seasonality_mode = 'additive')
m_d.fit(df_prophet)
future_d = m_d.make_future_dataframe(periods=7)
fcst_daily = m_d.predict(future_d)


# In[ ]:


trace1 = {
  "fill": None, 
  "mode": "markers", 
  "name": "actual no. of Confirmed", 
  "type": "scatter", 
  "x": df_prophet.ds, 
  "y": df_prophet.y
}
trace2 = {
  "fill": "tonexty", 
  "line": {"color": "#57b8ff"}, 
  "mode": "lines", 
  "name": "upper_band", 
  "type": "scatter", 
  "x": fcst_daily.ds, 
  "y": fcst_daily.yhat_upper
}
trace3 = {
  "fill": "tonexty", 
  "line": {"color": "#57b8ff"}, 
  "mode": "lines", 
  "name": "lower_band", 
  "type": "scatter", 
  "x": fcst_daily.ds, 
  "y": fcst_daily.yhat_lower
}
trace4 = {
  "line": {"color": "#eb0e0e"}, 
  "mode": "lines+markers", 
  "name": "prediction", 
  "type": "scatter", 
  "x": fcst_daily.ds, 
  "y": fcst_daily.yhat
}
data = [trace1, trace2, trace3, trace4]
layout = {
  "title": "Confirmed - Time Series Forecast - Daily Trend", 
  "xaxis": {
    "title": "", 
    "ticklen": 5, 
    "gridcolor": "rgb(255, 255, 255)", 
    "gridwidth": 2, 
    "zerolinewidth": 1
  }, 
  "yaxis": {
    "title": "Confirmed nCov - Hubei", 
    "ticklen": 5, 
    "gridcolor": "rgb(255, 255, 255)", 
    "gridwidth": 2, 
    "zerolinewidth": 1
  }, 
}
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


m_nd = Prophet(
    changepoint_range=0.90,
    changepoint_prior_scale=20,
    n_changepoints=19,
    yearly_seasonality=False,
    weekly_seasonality = False,
    daily_seasonality = False,
    seasonality_mode = 'additive')
m_nd.fit(df_prophet)
future_nd = m_nd.make_future_dataframe(periods=7)
fcst_no_daily = m_nd.predict(future_nd)


# In[ ]:


trace1 = {
  "fill": None, 
  "mode": "markers", 
  "name": "actual no. of Confirmed", 
  "type": "scatter", 
  "x": df_prophet.ds, 
  "y": df_prophet.y
}
trace2 = {
  "fill": "tonexty", 
  "line": {"color": "#57b8ff"}, 
  "mode": "lines", 
  "name": "upper_band", 
  "type": "scatter", 
  "x": fcst_no_daily.ds, 
  "y": fcst_no_daily.yhat_upper
}
trace3 = {
  "fill": "tonexty", 
  "line": {"color": "#57b8ff"}, 
  "mode": "lines", 
  "name": "lower_band", 
  "type": "scatter", 
  "x": fcst_no_daily.ds, 
  "y": fcst_no_daily.yhat_lower
}
trace4 = {
  "line": {"color": "#eb0e0e"}, 
  "mode": "lines+markers", 
  "name": "prediction", 
  "type": "scatter", 
  "x": fcst_no_daily.ds, 
  "y": fcst_no_daily.yhat
}
data = [trace1, trace2, trace3, trace4]
layout = {
  "title": "Confirmed - Time Series Forecast", 
  "xaxis": {
    "title": "", 
    "ticklen": 5, 
    "gridcolor": "rgb(255, 255, 255)", 
    "gridwidth": 2, 
    "zerolinewidth": 1
  }, 
  "yaxis": {
    "title": "Confirmed nCov - Hubei", 
    "ticklen": 5, 
    "gridcolor": "rgb(255, 255, 255)", 
    "gridwidth": 2, 
    "zerolinewidth": 1
  }, 
}
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


y_true = df_prophet.y.values
y_pred_daily = fcst_daily.loc[fcst_daily['ds'] <= max_date].yhat.values
y_pred_no_daily = fcst_no_daily.loc[fcst_no_daily['ds'] <= max_date].yhat.values


# In[ ]:


print('MAPE with daily seasonality: {}'.format(mean_absolute_percentage_error(y_true,y_pred_daily)))
print('MAPE without daily seasonality: {}'.format(mean_absolute_percentage_error(y_true,y_pred_no_daily)))


# The changes made seem to have brought a significant improvement on both models.<br>
# I try randomly to change the paramenters related to **changepoints** on both models. Obviously by incrementing the **prior_scale** we can get a more flexible model, which brought the major improvements. As for the **changepoints**, there is no logical reason for the paramaters that I chose.<br>
# <br>
# About the number of **changepoint** to pass to the model, you can look at the plot below.

# In[ ]:


df_ch_d = pd.DataFrame()
df_ch_nd = pd.DataFrame()

df_ch_d['deltas'] = m_d.params['delta'].mean(0)
df_ch_d['x'] = [x for x in range(19)]

df_ch_nd['deltas'] = m_nd.params['delta'].mean(0)
df_ch_nd['x'] = [x for x in range(19)]

fig = go.Figure()
fig2 = go.Figure()

fig.add_trace(
    go.Bar(
        x=df_ch_d.x,
        y=df_ch_d.deltas,
        name='# of changepoints',
        marker_color='rgb(55, 83, 109)',
        hovertemplate ="Change Rate: %{y: .2f}<extra></extra>",
        
    )
)

fig.update_layout(
    title={'text': 'Barplot of ChangePoints - Daily Model',
           'y':0.95,
           'x':0.5,
           'xanchor': 'center',
           'yanchor': 'top'},
    xaxis_tickfont_size=14,
    xaxis=dict(
        title = 'Potential ChangePoint'),
    yaxis=dict(
        title='Rate Change',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.1,
    bargroupgap=0.1
)


fig2.add_trace(
    go.Bar(
        x=df_ch_nd.x,
        y=df_ch_nd.deltas,
        name='# of changepoints',
        marker_color='rgb(55, 83, 109)',
        hovertemplate ="Change Rate: %{y: .2f}<extra></extra>",
    )
)

fig2.update_layout(
    title={'text': 'Barplot of ChangePoints - Non Daily Model',
           'y':0.95,
           'x':0.5,
           'xanchor': 'center',
           'yanchor': 'top'},
    xaxis_tickfont_size=14,
    xaxis=dict(
        title = 'Potential ChangePoint'),
    yaxis=dict(
        title='Rate Change',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.1,
    bargroupgap=0.1
)

fig.show()
fig2.show()


# Here there is the best model.

# In[ ]:


fig = plot_plotly(m_nd, fcst_no_daily) 
fig.update_layout(
    title={'text': 'Prediction Confermed cases in Hubei',
           'y':0.95,
           'x':0.5,
           'xanchor': 'center',
           'yanchor': 'top'},
    yaxis=dict(
        title='Confirmed Cases',
        titlefont_size=16,
        tickfont_size=14,
    )
)
fig.show()


# ### Deaths Prediction

# Let's try to see now how the **Deaths** are predicted.

# In[ ]:


df_death = df_hubei[['Date','Deaths']]
df_death.columns = ['ds','y']


# In[ ]:


m_death = Prophet(
    changepoint_range=0.90,
    changepoint_prior_scale=20,
    n_changepoints=17,
    yearly_seasonality=False,
    weekly_seasonality = False,
    daily_seasonality = False,
    seasonality_mode = 'additive')
m_death.fit(df_death)
future_death = m_death.make_future_dataframe(periods=7)
fcst_death = m_death.predict(future_death)


# In[ ]:


trace1 = {
  "fill": None, 
  "mode": "markers",
  "marker_size": 10,
  "name": "actual no. of Confirmed", 
  "type": "scatter", 
  "x": df_death.ds, 
  "y": df_death.y
}
trace2 = {
  "fill": "tonexty", 
  "line": {"color": "#57b8ff"}, 
  "mode": "lines", 
  "name": "upper_band", 
  "type": "scatter", 
  "x": fcst_death.ds, 
  "y": fcst_death.yhat_upper
}
trace3 = {
  "fill": "tonexty", 
  "line": {"color": "#57b8ff"}, 
  "mode": "lines", 
  "name": "lower_band", 
  "type": "scatter", 
  "x": fcst_death.ds, 
  "y": fcst_death.yhat_lower
}
trace4 = {
  "line": {"color": "#eb0e0e"}, 
  "mode": "lines+markers",
  "marker_size": 4,
  "name": "prediction", 
  "type": "scatter", 
  "x": fcst_death.ds, 
  "y": fcst_death.yhat
}
data = [trace1, trace2, trace3, trace4]
layout = {
  "title": "Deaths - Time Series Forecast", 
  "xaxis": {
    "title": "Monthly Dates", 
    "ticklen": 5, 
    "gridcolor": "rgb(255, 255, 255)", 
    "gridwidth": 2, 
    "zerolinewidth": 1
  }, 
  "yaxis": {
    "title": "Deaths nCov - Hubei", 
    "ticklen": 5, 
    "gridcolor": "rgb(255, 255, 255)", 
    "gridwidth": 2, 
    "zerolinewidth": 1
  }, 
}
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


max_date = df_death.ds.max()
y_true = df_death.y.values
y_pred_death = fcst_death.loc[fcst_death['ds'] <= max_date].yhat.values


# In[ ]:


print('MAPE with daily seasonality: {}'.format(mean_absolute_percentage_error(y_true,y_pred_death)))


# Will follow further analysis, waiting for more data.<br>
# <br>
# Feel free to leave any suggestions!!!
