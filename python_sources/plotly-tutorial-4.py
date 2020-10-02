#!/usr/bin/env python
# coding: utf-8

# <h1><center>PLOTLY TUTORIAL - 4</center></h1>
# 
# ***
# 
# In this notebook, my aim is to investigate Google Merchandise Store customer dataset by using plotly time-series features and custom buttons while generating a comprehensive tutorial for plotly enthusiasts. You may want to check my other Plotly tutorials.
# 
# **PLOTLY TUTORIAL - 1 (Kaggle Survey 2017): https://www.kaggle.com/hakkisimsek/plotly-tutorial-1**
# 
# **PLOTLY TUTORIAL - 2 (2015 Flight Delays and Cancellations): https://www.kaggle.com/hakkisimsek/plotly-tutorial-2**
# 
# **PLOTLY TUTORIAL - 3 (S&P 500 Stock Data): https://www.kaggle.com/hakkisimsek/plotly-tutorial-3**
# 
# **PLOTLY TUTORIAL - 5 (Kaggle Survey 2018): https://www.kaggle.com/hakkisimsek/plotly-tutorial-5**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
color = sns.color_palette()

import os
import datetime

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings('ignore')

print(os.listdir("../input"))
df = pd.read_csv('../input/train-set/extracted_train/extracted_train', 
                    dtype={'date': str, 'fullVisitorId': str, 'sessionId':str},
                    nrows=None)


# **Before starting the explanatory data analysis, we should change type of transaction revenue to float and create date-related features.**

# In[ ]:


df["totals.transactionRevenue"] = df["totals.transactionRevenue"].astype('float')
df['date'] = df['date'].apply(lambda x: datetime.date(int(str(x)[:4]), 
                                    int(str(x)[4:6]), int(str(x)[6:])))
df['time'] = pd.to_datetime(df['visitStartTime'], unit='s')
df['month'] = df['time'].dt.month
df['dow'] = df['time'].dt.dayofweek
df['day'] = df['time'].dt.day
df['hour'] = df['time'].dt.hour

df['day_frame'] = 0
df['day_frame'] = np.where((df["hour"]>=0) & (df["hour"]<4), 'overnight', 
                           df['day_frame'])
df['day_frame'] = np.where((df["hour"]>=4) & (df["hour"]<8), 'dawn', 
                           df['day_frame'])
df['day_frame'] = np.where((df["hour"]>=8) & (df["hour"]<12), 'morning', 
                           df['day_frame'])
df['day_frame'] = np.where((df["hour"]>=12) & (df["hour"]<14), 'lunch', 
                           df['day_frame'])
df['day_frame'] = np.where((df["hour"]>=14) & (df["hour"]<18), 'afternoon', 
                           df['day_frame'])
df['day_frame'] = np.where((df["hour"]>=18) & (df["hour"]<21), 'evening', 
                           df['day_frame'])
df['day_frame'] = np.where((df["hour"]>=21) & (df["hour"]<24), 'night', 
                           df['day_frame'])


# ### Check missing values

# In[ ]:


miss = pd.concat([df.isnull().sum(), 100 * df.isnull().sum()/len(df)], 
              axis=1).rename(columns={0:'Missing Records', 
                        1:'Percentage (%)'}).sort_values(by='Percentage (%)',
                                                         ascending=False)[:8]
trace = go.Bar(
        y=miss.index[::-1],
        x=miss['Percentage (%)'][::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color='firebrick',
        )
    )

data = [trace]
layout = dict(
    title = 'Percentage (%) missing values',
    margin  = dict(l = 200
                                      )
    )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# **As a starting point, it would be good to understand behavior of the target variable revenue transaction. Then, comparing behavior of total visitors, non-zero revenue visitors and total revenue  will be more meaningful.**

# In[ ]:


def scatter_plot(data, color, name, mode=None):
    trace = go.Scatter(
        x=data.index[::-1],
        y=data.values[::-1],
        showlegend=False,
        name = name,
        mode = mode,
        marker=dict(
        color = color
        )
    )
    return trace


# **On the left graph, we show transaction revenues by visitors while on the right we calculate log of transaction revenues. It can be seen easily that most visitors do not have an effect on revenue. **
# 
# **By following section, our aim is to understand which features are critical for seperating non-zero revenue visitors from others.**

# In[ ]:


f, ax = plt.subplots(1,2, figsize=(18,5))
rev = df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

ax[0].scatter(range(rev.shape[0]), 
              np.sort(rev["totals.transactionRevenue"].values), color='navy')
ax[0].set_xlabel('index')
ax[0].set_ylabel('Revenue')
ax[0].set_title('Transaction revenue by visitors')

ax[1].scatter(range(rev.shape[0]), 
              np.sort(np.log1p(rev["totals.transactionRevenue"].values)), 
              color='navy')
ax[1].set_xlabel('index')
ax[1].set_ylabel('Revenue (log)')
ax[1].set_title('Transaction revenue (log) by visitors')
plt.show()

visit_group = df.groupby('fullVisitorId')['fullVisitorId'].count()

for i in [2, 10, 30, 50]:
    print('Visitors that appear less than {} times: {}%'.format(i, 
                                 round((visit_group < i).mean() * 100, 2)))


# **There are only 9996 non-zero revenue visitors out of 714167 visitors (1.3%). Let's analyze contributions of non-zero revenue visitors.**

# In[ ]:


rev = df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()
rev1 = np.sort(rev["totals.transactionRevenue"].values)
rev2 = rev1[rev1>0]
rev3 = np.sort(np.log1p(rev["totals.transactionRevenue"].values))
rev4 = rev3[rev3>0]

trace0 = scatter_plot(pd.DataFrame(rev2)[0],'red','revenue', mode='markers')
trace1 = scatter_plot(pd.DataFrame(rev4)[0],'red','revenue', mode='markers')

fig = tools.make_subplots(rows=1, cols=2, 
                          subplot_titles=('Total revenue by user level',
                                          'Total revenue (log) by user level'
                                         )
                         )

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(showlegend=False)
py.iplot(fig)


# **Now, we could apply groupby function on date and resample based on weekly and monthly basis. **

# In[ ]:


stats = df.groupby('date')['totals.transactionRevenue'].agg(['size', 'count'])
stats.columns = ["count total", "count non-zero"]
stats.index = pd.to_datetime(stats.index)
stats['per'] = stats['count non-zero'] / stats['count total']

df1dm = stats.reset_index().resample('D', on='date').mean()
df1wm = stats.reset_index().resample('W', on='date').mean()
df1mm = stats.reset_index().resample('M', on='date').mean()

trace0 = scatter_plot(df1mm['count total'].round(0), 'red', 'count')
trace1 = scatter_plot(df1wm['count total'].round(0), 'orange', 'count')
trace2 = scatter_plot(df1dm['count total'], 'indigo', 'count')

trace3 = scatter_plot(df1mm['count non-zero'].round(0), 'red', 'count')
trace4 = scatter_plot(df1wm['count non-zero'].round(0), 'orange', 'count')
trace5 = scatter_plot(df1dm['count non-zero'], 'indigo', 'count')

fig = tools.make_subplots(rows=2, cols=3, 
                          subplot_titles=('Monthly total','Weekly', 'Daily',
                                          'Monthly non-zero','Weekly', 'Daily'
                                         )
                         )

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 1, 3)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)
fig.append_trace(trace5, 2, 3)

fig['layout'].update(showlegend=False, 
                     title='Total visitors vs non-zero revenue visitors')
py.iplot(fig)


# **It is very interesting that trends in total visitors and non-zero revenue visitors do not overlap. Let's make analysis of non-zero visitors/total visitors.**

# In[ ]:


trace0 = scatter_plot(df1mm['per'].round(4), 'red', 'count', 'markers')
trace1 = scatter_plot(df1wm['per'].round(4), 'orange', 'count', 'markers')
trace2 = scatter_plot(df1dm['per'].round(4), 'indigo', 'count', 'markers')

fig = tools.make_subplots(rows=2, cols=2, specs=[[{}, {}], 
                          [{'colspan': 2}, None]],
                          subplot_titles=('Monthly','Weekly', 'Daily'
                                         )
                         )
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)

fig['layout'].update(showlegend=False, title='Non-zero/Total visitors')
py.iplot(fig)


# **Rather than creating graphs seperately as above, we could create custom buttons and make our work more interactive.**

# In[ ]:


trace = go.Scatter(
                   x=list(df1dm.index),
                   y=list(df1dm.per.round(4)), 
                   line=dict(color='red'))

data = [trace]
layout = dict(
    title='Non-zero/Total visitors',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=7,
                     label='7d',
                     step='day',
                     stepmode='backward'),
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=3,
                     label='3m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                    label='YTD',
                    step='year',
                    stepmode='todate')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
)

fig = dict(data=data, layout=layout)
py.iplot(fig)


# **Using custom buttons, we could compare total visits, non-zero revenue visits and zero revenue visits across hours on a single graph easily.**

# In[ ]:


trace = [
    go.Histogram(x=df['hour'],
                opacity = 0.7,
                 name="Total Visits",
                 hoverinfo="y",
                 marker=dict(line=dict(width=1.6),
                            color='grey')
                ),
    
    go.Histogram(x=df[df['totals.transactionRevenue'].notnull()]['hour'],
                 visible=False,
                 opacity = 0.7,
                 name = "Non-zero revenue visits",
                 hoverinfo="y",
                 marker=dict(line=dict(width=1.6),
                            color='red')
                ),
    
    go.Histogram(x=df[df['totals.transactionRevenue'].isnull()]['hour'],
                 visible=False,
                opacity = 0.7,
                 name = "Zero revenue visits",
                 hoverinfo="y",
                 marker=dict(line=dict(width=1.6),
                            color='aqua')         
                )
]

layout = go.Layout(title='Visiting hours',
    paper_bgcolor = 'rgb(240, 240, 240)',
     plot_bgcolor = 'rgb(240, 240, 240)',
    autosize=True, xaxis=dict(tickmode="linear"),
                   yaxis=dict(title="# of Visits",
                             titlefont=dict(size=17)),
                  )

updatemenus = list([
    dict(
    buttons=list([
        dict(
            args = [{'visible': [True, False, False]}],
            label="Total visits",
            method='update',
        ),
        dict(
            args = [{'visible': [False, True, False]}],
            label="Non-zero revenue visits",
            method='update',
        ),
        dict(
            args = [{'visible': [False, False, True]}],
            label="Zero revenue visits",
            method='update',
        ),
        
    ]),
        direction="down",
        pad = {'r':10, "t":10},
        x=0.1,
        y=1.25,
        yanchor='top',
    )
])
layout['updatemenus'] = updatemenus

fig = dict(data=trace, layout=layout)
py.iplot(fig)


# **We can say that it would be good to focus on graphs related to non-zero revenue visits rather than the hype.**
# 
# **Now, we try to understand the relation between usage of operating system and revenue on different day frames. For doing such a task, a heatmap is a very good choice.**

# In[ ]:


fv = df.pivot_table(index="device.operatingSystem",columns="day_frame",
                    values="totals.transactionRevenue",aggfunc=lambda x:x.sum())
fv = fv[['morning', 'lunch', 'afternoon', 'evening','night','overnight', 'dawn']]
fv = fv.sort_values(by='morning', ascending=False)[:6]

trace = go.Heatmap(z=[fv.values[0],fv.values[1],fv.values[2],fv.values[3],
                      fv.values[4],fv.values[5]],
                   x=['morning', 'lunch', 'afternoon', 'evening', 'night',
                      'overnight','dawn'],
                   y=fv.index.values, colorscale='Blues', reversescale = True
                  )

data=[trace]
layout = go.Layout(
    title='Total Revenue by Device OS<br>(parts of the day)')

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# **Now, we can compare general statistics of operating systems like total count, non-zero revenue count and total revenue. Bar charts are common way to do such a task but I prefer to use pie chart now. **

# In[ ]:


color = ['tomato',  'bisque','lightgreen', 'gold', 'tan', 'lightgrey', 'cyan']

def PieChart(column, title, limit):
    revenue = "totals.transactionRevenue"
    count_trace = df.groupby(column)[revenue].size().nlargest(limit).reset_index()
    non_zero_trace = df.groupby(column)[revenue].count().nlargest(limit).reset_index()
    rev_trace = df.groupby(column)[revenue].sum().nlargest(limit).reset_index()    

    trace1 = go.Pie(labels=count_trace[column], 
                    values=count_trace[revenue], 
                    name= "Visit", 
                    hole= .5, textfont=dict(size=10),
                    domain= {'x': [0, .32]},
                   marker=dict(colors=color))

    trace2 = go.Pie(labels=non_zero_trace[column], 
                    values=non_zero_trace[revenue], 
                    name="Revenue", 
                    hole= .5,  textfont=dict(size=10),
                    domain= {'x': [.34, .66]})
    
    trace3 = go.Pie(labels=rev_trace[column], 
                    values=rev_trace[revenue], 
                    name="Revenue", 
                    hole= .5,  textfont=dict(size=10),
                    domain= {'x': [.68, 1]})

    layout = dict(title= title, font=dict(size=15), legend=dict(orientation="h"),
                  annotations = [
                      dict(
                          x=.10, y=.5,
                          text='<b>Number of<br>Visitors', 
                          showarrow=False,
                          font=dict(size=12)
                      ),
                      dict(
                          x=.50, y=.5,
                          text='<b>Number of<br>Visitors<br>(non-zero)', 
                          showarrow=False,
                          font=dict(size=12)
                      ),
                      dict(
                          x=.88, y=.5,
                          text='<b>Total<br>Revenue', 
                          showarrow=False,
                          font=dict(size=12)
                      )
        ])
    
    fig = dict(data=[trace1, trace2,trace3], layout=layout)
    py.iplot(fig)


# In[ ]:


PieChart("device.operatingSystem", "Operating System", 4)


# **Although Windows users create a lot of traffic at the end of the day Macintosh users create non-zero revenue traffic and total revenue.**
# 
# **Let's apply same steps on traffic source in terms of day of week. We analyze which sources give us more traffic, number of non-zero revenue traffic and total revenue. By doing so, we understand the real effect of a source on revenues.  **

# In[ ]:


fv = df.pivot_table(index="trafficSource.source",columns="dow",values="totals.transactionRevenue",aggfunc=lambda x:x.sum())
fv = fv.sort_values(by=0, ascending=False)[:6]

trace = go.Heatmap(z=[fv.values[0],fv.values[1],fv.values[2], fv.values[3],fv.values[4],fv.values[5]],
                   x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday','Sunday'],
                   y=fv.index.values, colorscale='Reds'
                  )

data=[trace]
layout = go.Layout(dict(
    title='Total Revenue by Traffic Source<br>(day of week)'),
                margin  = dict(l = 150
                                      )  )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


PieChart("trafficSource.source", "Traffic Source", 5)


# **Although youtube creates a huge traffic it does not have a signifact effect on total revenues**
# 
# **Google plex that create only 8.9% total traffic contributes to 41.5% of total revenue.**
# 
# **Let's apply same steps on source medium in terms of months.  **

# In[ ]:


fv = df.pivot_table(index="trafficSource.medium",columns="month",values="totals.transactionRevenue",aggfunc=lambda x:x.sum())
fv = fv.sort_values(by=1, ascending=False)[:5]

trace = go.Heatmap(z=[fv.values[0],fv.values[1],fv.values[2], fv.values[3],fv.values[4]],
                   x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                   y=fv.index.values, colorscale='Reds'
                  )

data=[trace]
layout = go.Layout(dict(
    title='Total Revenue by Source Medium<br>(months)'),
                margin  = dict(l = 150
                                      ))

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


PieChart("trafficSource.medium", "Source Medium", 5)


# **Let's finalize the work with a world map!**

# In[ ]:


count_geo = df.groupby('geoNetwork.country')['geoNetwork.country'].count()

data = [dict(
        type = 'choropleth',
        locations = count_geo.index,
        locationmode = 'country names',
        z = count_geo.values,
        text = count_geo.index,
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Number of calls'),
      ) ]

layout = dict(
    title = 'Number of Visits by Country',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict(data=data, layout=layout )
py.iplot(fig, validate=False)


# In[ ]:


count = df[['geoNetwork.continent','totals.transactionRevenue',
            'day']].groupby('geoNetwork.continent', as_index=False)['day'].\
                    count().sort_values(by='day', ascending=False)

sumrev = df[['geoNetwork.continent','totals.transactionRevenue']].groupby('geoNetwork.continent', 
                                                       as_index=False)['totals.transactionRevenue'].\
                    sum().sort_values(by='totals.transactionRevenue', ascending=False)

meanrev = df[['geoNetwork.continent','totals.transactionRevenue']].groupby('geoNetwork.continent', 
                                                       as_index=False)['totals.transactionRevenue'].\
                    mean().sort_values(by='totals.transactionRevenue', ascending=False)

trace = [
    go.Bar(x=count['geoNetwork.continent'],
    y=count['day'],
                opacity = 0.7,
                 name="COUNT",
                 hoverinfo="y",
                 marker=dict(line=dict(width=1.6),
                            color='aqua')
                ),
    go.Bar(x=sumrev['geoNetwork.continent'],
    y=sumrev['totals.transactionRevenue'],
                 visible=False,
                 opacity = 0.7,
                 name = "TOTAL",
                 hoverinfo="y",
                 marker=dict(line=dict(width=1.6),
                            color='navy')
                ),
    go.Bar(x=meanrev['geoNetwork.continent'],
    y=meanrev['totals.transactionRevenue'],
                 visible=False,
                opacity = 0.7,
                 name = "MEAN",
                 hoverinfo="y",
                 marker=dict(line=dict(width=1.6),
                            color='red')
                
                )
]

layout = go.Layout(title = 'Revenue Statistics of Continents',
    paper_bgcolor = 'rgb(240, 240, 240)',
     plot_bgcolor = 'rgb(240, 240, 240)',
    autosize=True,
                   xaxis=dict(title="",
                             titlefont=dict(size=20),
                             tickmode="linear")
                  )

updatemenus = list([
    dict(
    buttons=list([
        dict(
            args = [{'visible': [True, False, False]}],
            label="Count",
            method='update',
        ),
        dict(
            args = [{'visible': [False, True, False]}],
            label="Total Revenue",
            method='update',
        ),
        dict(
            args = [{'visible': [False, False, True]}],
            label="Mean Revenue",
            method='update',
        ),
        
    ]),
        direction="down",
        pad = {'r':10, "t":10},
        x=0.1,
        y=1.25,
        yanchor='top',
    )
])

layout['updatemenus'] = updatemenus

fig = dict(data=trace, layout=layout)
py.iplot(fig)

