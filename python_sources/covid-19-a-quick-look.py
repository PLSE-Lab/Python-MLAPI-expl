#!/usr/bin/env python
# coding: utf-8

# There has been only one thing on everyone's mind for the last few weeks - COVID-19. It has been hard to miss all the disucssions surrounding it. There has been a lot of panic and some of them are justifibale. The downside is there has been a lot of rumor mongerinmg as well. This is a very basic exploratory data analysis that will be built upon in the coming weeks. 
# 
# The primary source for the data is from : https://www.kaggle.com/imdevskp/corona-virus-report. More infiormation about the dataset is also available at the same link.

# ![Virus](https://behavioralscientist.org/wp-content/uploads/2020/03/coronavirus_links.png)
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Standard plotly imports
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode

import plotly.express as px
import plotly.io as pio

# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)


# In[ ]:


# read clean datatset that is updated every 24 hours
clean_data = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv',parse_dates=['Date'])


# In[ ]:


# rename column
clean_data = clean_data.rename(columns={'Country/Region':'Country'})


# In[ ]:


# overall cases growth 
overall_cases_death = clean_data.groupby('Date')['Date','Confirmed','Deaths','Recovered'].sum().reset_index()
india_cases_death = clean_data[clean_data.Country == 'India'].groupby('Date')['Date','Confirmed','Deaths','Recovered'].sum().reset_index()
rest_of_world_cases_death  = clean_data[clean_data.Country != 'India'].groupby('Date')['Date','Confirmed','Deaths','Recovered'].sum().reset_index()
italy_cases_death = clean_data[clean_data.Country == 'Italy'].groupby('Date')['Date','Confirmed','Deaths','Recovered'].sum().reset_index()


# # Confirmed Cases

# In[ ]:


# plot overall confirmed cases
trace1 = go.Scatter(
                    x = overall_cases_death.Date,
                    y = overall_cases_death.Confirmed,
                    mode = "lines",
                    name = "",
                    text= 'Confirmed Cases',
                    line=dict(color='#4cff00', width=2),
                    fill='tozeroy')

data = [trace1]
layout = dict(title = {'text':'Growth of Confirmed Cases - Overall'},font=dict(color='white',family='Arial'),
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),
              yaxis =  {'showgrid': False},paper_bgcolor='#273746',
              plot_bgcolor='#273746')

fig = dict(data = data, layout = layout)

iplot(fig)


# In[ ]:


# India 
trace1 = go.Scatter(
                    x = india_cases_death.Date,
                    y = india_cases_death.Confirmed,
                    mode = "lines",
                    name = "India",
                    text = 'Confirmed Cases',
                    line=dict(color='#00ccff', width=2),
                    fill='tozeroy')

data = [trace1]
layout = dict(title = 'Growth of Confirmed Cases - India',font=dict(color='white',family='Arial'),
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),
              yaxis =  {'showgrid': False},paper_bgcolor='#273746',
              plot_bgcolor='#273746')
fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


# Italy
trace1 = go.Scatter(
                    x = italy_cases_death.Date,
                    y = italy_cases_death.Confirmed,
                    mode = "lines",
                    name = "Italy",
                    text = 'Confirmed Cases',
                    line=dict(color='#ff3300', width=2),fill='tozeroy')

data = [trace1]
layout = dict(title = 'Growth of Confirmed Cases - Italy',font=dict(color='white',family='Arial'),
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),
              yaxis =  {'showgrid': False},paper_bgcolor='#273746',
              plot_bgcolor='#273746')
        
fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


# top countries 10 countries with maximum confirmed cases 
last_date =  clean_data[clean_data.Date == max(clean_data.Date)]
top_10_countries_confirmed_cases = last_date.groupby('Country')['Confirmed','Deaths'].sum().reset_index().sort_values('Confirmed',ascending=False)[:10]


# In[ ]:


# Creating two subplots
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                    shared_yaxes=False, vertical_spacing=0.001)

fig.append_trace(go.Bar(
    x=top_10_countries_confirmed_cases[::-1].Confirmed,
    y=top_10_countries_confirmed_cases[::-1].Country,
    marker=dict(
        color='rgba(50, 171, 96, 0.6)',
        line=dict(
            color='rgba(50, 171, 96, 1.0)',
            width=1),
    ),
    name='Total Number of Confirmed Cases',
    orientation='h',
), 1, 1)

fig.append_trace(go.Scatter(
    x=top_10_countries_confirmed_cases[::-1].Deaths, y=top_10_countries_confirmed_cases[::-1].Country,
    mode='lines+markers',
    line_color='rgb(128, 0, 128)',
    name='Total Number of Deaths',
), 1, 2)

fig.update_layout(
    title='Covid-19 : Confirmed Cases & Total Deaths',
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
        domain=[0, 0.85],
    ),
    yaxis2=dict(
        showgrid=False,
        showline=True,
        showticklabels=False,
        linecolor='rgba(102, 102, 102, 0.8)',
        linewidth=2,
        domain=[0, 0.85],
    ),
    xaxis=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0, 0.42],
    ),
    xaxis2=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0.47, 1],
        side='top',
        dtick=1000,
    ),
    legend=dict(x=0.029, y=1.038, font_size=10),
    margin=dict(l=100, r=20, t=70, b=70),
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

annotations = []

y_s = np.round(top_10_countries_confirmed_cases.Confirmed, decimals=0)
y_nw = np.rint(top_10_countries_confirmed_cases.Deaths)

# Adding labels
for ydn, yd, xd in zip(y_nw, y_s, top_10_countries_confirmed_cases.Country):
    # labeling the scatter deaths
    annotations.append(dict(xref='x2', yref='y2',
                            y=xd, x=ydn + 250,
                            text='{:,.0f}'.format(ydn),
                            font=dict(family='Arial', size=12,
                                      color='rgb(128, 0, 128)'),
                            showarrow=False))
    # labeling the bar confirmed cases
    annotations.append(dict(xref='x1', yref='y1',
                            y=xd, x=yd + 5000,
                            text='{:,.0f}'.format(yd),
                            font=dict(family='Arial', size=12,
                                      color='rgb(50, 171, 96)'),
                            showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()


# * The overall number of confirmed cases has been on the rise steadily across the globe. We have already crossed the 330K mark and the increase has been exponential. This is expected to increase more as testing increases in countries like India and US increase.
# * China has the most number of reported cases followed by Italy and the US. 
# * The number of deaths in China is condiderably lower than that of Italy. The US has reported 417 deaths already.

# # Confirmed Fatalities

# In[ ]:


# plot overall confirmed fatalities
trace1 = go.Scatter(
                    x = overall_cases_death.Date,
                    y = overall_cases_death.Deaths,
                    mode = "lines",
                    name = "",
                    text= 'Confirmed Fatalities', line=dict(color='#4cff00', width=2),fill='tozeroy')

data = [trace1]
layout = dict(title = 'Growth of Confirmed Fatalities - Overall',font=dict(color='white',family='Arial'),
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),
              yaxis =  {'showgrid': False},paper_bgcolor='#273746',
              plot_bgcolor='#273746')

fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


# plot India fatalities
trace1 = go.Scatter(
                    x = india_cases_death.Date,
                    y = india_cases_death.Deaths,
                    mode = "lines",
                    name = "",
                    text= 'Confirmed Fatalities',line=dict(color='#00ccff', width=2),fill='tozeroy')

data = [trace1]
layout = dict(title = 'Growth of Confirmed Fatalities - India',font=dict(color='white',family='Arial'),
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),
              yaxis =  {'showgrid': False},paper_bgcolor='#273746',
              plot_bgcolor='#273746')

fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


# plot Italy fatalities
trace1 = go.Scatter(
                    x = italy_cases_death.Date,
                    y = italy_cases_death.Deaths,
                    mode = "lines",
                    name = "",
                    text= 'Confirmed Fatalities', line=dict(color='#ff3300', width=2),fill='tozeroy')

data = [trace1]
layout = dict(title = 'Growth of Confirmed Fatalities - Italy',font=dict(color='white',family='Arial'),
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),
              yaxis =  {'showgrid': False},paper_bgcolor='#273746',
              plot_bgcolor='#273746')
fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


latest_deaths = clean_data[clean_data.Date == max(clean_data.Date)]
top_10_countries_confirmed_deaths = latest_deaths.groupby('Country')['Deaths'].sum().reset_index().sort_values('Deaths', ascending=False)[:10]


# In[ ]:


fig = px.bar(top_10_countries_confirmed_deaths[::-1], x= 'Deaths', y='Country', orientation='h',text='Deaths',
             title='Confirmed Fatalities - Top 10 Countries',template="plotly_dark")
fig.show()


# * The confirmed number of fatalities is also growing across the globe and the rise again is exponential, which is concerning. 
# * Italy has the most cases of deaths and has been in the news for this reason unfortunately.
# * India has so far reported only 7 deaths, not so alarming quite yet, but need to see what happens in the next few weeks.

# # Recoveries

# In[ ]:


# plot overall recovered cases
trace1 = go.Scatter(
                    x = overall_cases_death.Date,
                    y = overall_cases_death.Recovered,
                    mode = "lines",
                    name = "",
                    text= 'Recoveries',line=dict(color='#4cff00', width=2),fill='tozeroy')

data = [trace1]
layout = dict(title = 'Recovered Cases - Overall',font=dict(color='white',family='Arial'),
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),
              yaxis =  {'showgrid': False},paper_bgcolor='#273746',
              plot_bgcolor='#273746')

fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


# plot recovered cases in India
trace1 = go.Scatter(
                    x = india_cases_death.Date,
                    y = india_cases_death.Recovered,
                    mode = "lines",
                    name = "",
                    text= 'Recoveries',line=dict(color='#00ccff', width=2),fill='tozeroy')

data = [trace1]
layout = dict(title = 'Recovered Cases - India',font=dict(color='white',family='Arial'),
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),
              yaxis =  {'showgrid': False},paper_bgcolor='#273746',
              plot_bgcolor='#273746')

fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


# plot recovered cases in Italy
trace1 = go.Scatter(
                    x = italy_cases_death.Date,
                    y = italy_cases_death.Recovered,
                    mode = "lines",
                    name = "",
                    text= 'Recoveries',line=dict(color='#ff3300', width=2),fill='tozeroy')

data = [trace1]
layout = dict(title = 'Recovered Cases - Italy',font=dict(color='white',family='Arial'),
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),
              yaxis =  {'showgrid': False},paper_bgcolor='#273746',
              plot_bgcolor='#273746')

fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


last_recovery = clean_data[clean_data.Date == max(clean_data.Date)]
top_10_countries_recovery = last_recovery.groupby('Country')['Recovered'].sum().reset_index().sort_values('Recovered', ascending=False)[:10]


# In[ ]:


fig = px.bar(top_10_countries_recovery[::-1], x= 'Recovered', y='Country', orientation='h',text='Recovered',
             title='Confirmed Recoveries - Top 10 Countries',template="plotly_dark")
fig.show()


# * The numnber of recoevries has also been on the rise, which gives us some hope that we can ride this tide.
# * The number of recoveries in China is quite outstanding. Wonder if they have a vaacine already?

# # Active Cases

# Active cases are all confirmed cases without the recovered cases and deaths.

# In[ ]:


# add active cases columns
clean_data['Active'] = clean_data['Confirmed'] - (clean_data['Recovered'] + clean_data['Deaths'])


# In[ ]:


# active cases growth
active_case_growth = clean_data.groupby('Date')['Active'].sum().reset_index()


# In[ ]:


# plot active case growth
trace1 = go.Scatter(
                    x = active_case_growth.Date,
                    y = active_case_growth.Active,
                    mode = "lines",
                    name = "",
                    text= 'Active Cases',line=dict(color='#4cff00', width=2),fill='tozeroy')

data = [trace1]

layout = dict(title = 'Active Case Growth - Overall',font=dict(color='white',family='Arial'),
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),
              yaxis =  {'showgrid': False},paper_bgcolor='#273746',
              plot_bgcolor='#273746')

fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


india_active_cases = clean_data[clean_data.Country == 'India'].groupby('Date')['Active'].sum().reset_index()
italy_active_cases = clean_data[clean_data.Country == 'Italy'].groupby('Date')['Active'].sum().reset_index()
china_active_cases = clean_data[clean_data.Country == 'China'].groupby('Date')['Active'].sum().reset_index()


# In[ ]:


# plot active case growth - India
trace1 = go.Scatter(
                    x = india_active_cases.Date,
                    y = india_active_cases.Active,
                    mode = "lines",
                    name = "",
                    text= 'Active Cases', line=dict(color='#00ccff', width=2),fill='tozeroy')

data = [trace1]
layout = dict(title = 'Active Case Growth - India',font=dict(color='white',family='Arial'),
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),
              yaxis =  {'showgrid': False},paper_bgcolor='#273746',
              plot_bgcolor='#273746')

fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


# plot active case growth - Italy
trace1 = go.Scatter(
                    x = italy_active_cases.Date,
                    y = italy_active_cases.Active,
                    mode = "lines",
                    name = "",
                    text= 'Active Cases', line=dict(color='#ff3300', width=2),fill='tozeroy')

data = [trace1]
layout = dict(title = 'Active Case Growth - Italy',font=dict(color='white',family='Arial'),
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),
              yaxis =  {'showgrid': False},paper_bgcolor='#273746',
              plot_bgcolor='#273746')


fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


# plot active case growth - India
trace1 = go.Scatter(
                    x = china_active_cases.Date,
                    y = china_active_cases.Active,
                    mode = "lines",
                    name = "",
                    text= 'Active Cases', line=dict(color='#F1C40F', width=2),fill='tozeroy')

data = [trace1]
layout = dict(title = 'Active Case Growth - China',font=dict(color='white',family='Arial'),
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid= False),
              yaxis =  {'showgrid': False},paper_bgcolor='#273746',
              plot_bgcolor='#273746')
fig = dict(data = data, layout = layout)
iplot(fig)


# * As stated earlier the number of active cases will increase once countries like India and US start to increase their testing.
# * On the other hand, countries like India have imposed mandatory country-wide lockdowns on March 25th and maybe the curve will flatten in the coming weeks. It will be intresting to watch. 
# * The number of active cases in China has dropped significantly in the last few weeks. This is a positive sign that policies such as social distancing do break the chain and will stop the spread of the infection.

# # Recovery & Death Rates

# In[ ]:


last_rec_death = clean_data[clean_data.Date == max(clean_data.Date)]
recovery_death_rates = last_rec_death.groupby('Country')['Confirmed','Deaths','Recovered'].sum().reset_index()


# In[ ]:


# add death rate and recovery rate
recovery_death_rates['Death_Rate'] = (recovery_death_rates['Deaths']/recovery_death_rates['Confirmed']) * 100
recovery_death_rates['Recovery_Rate'] = (recovery_death_rates['Recovered']/recovery_death_rates['Confirmed']) * 100


# In[ ]:


top_10_death_rates = recovery_death_rates.sort_values('Confirmed', ascending=False)[:10]
top_10_recovery_rates = recovery_death_rates.sort_values('Confirmed', ascending=False)[:10]


# In[ ]:


# countries with recovery rates

fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                    shared_yaxes=False, vertical_spacing=0.001)

fig.append_trace(go.Bar(
    x=top_10_recovery_rates[::-1].Confirmed,
    y=top_10_recovery_rates[::-1].Country,
    marker=dict(
        color='rgba(50, 171, 96, 0.6)',
        line=dict(
            color='rgba(50, 171, 96, 1.0)',
            width=1),
    ),
    name='Total Number of Confirmed Cases',
    orientation='h',
), 1, 1)

fig.append_trace(go.Scatter(
    x=top_10_recovery_rates[::-1].Recovery_Rate, 
    y=top_10_recovery_rates[::-1].Country,
    mode='lines+markers',
    line_color='rgb(128, 0, 128)',
    name='Recovery Rate',
), 1, 2)

fig.update_layout(
    title='Covid-19 : Confirmed Cases & Recovery Rates',
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
        domain=[0, 0.85],
    ),
    yaxis2=dict(
        showgrid=False,
        showline=True,
        showticklabels=False,
        linecolor='rgba(102, 102, 102, 0.8)',
        linewidth=2,
        domain=[0, 0.85],
    ),
    xaxis=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0, 0.42],
    ),
    xaxis2=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0.47, 1],
        side='top',
        dtick=20,
    ),
    legend=dict(x=0.029, y=1.038, font_size=10),
    margin=dict(l=100, r=20, t=70, b=70),
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

annotations = []

y_s = np.round(top_10_recovery_rates.Confirmed, decimals=0)
y_nw = np.rint(top_10_recovery_rates.Recovery_Rate)

# Adding labels
for ydn, yd, xd in zip(y_nw, y_s, top_10_countries_confirmed_cases.Country):
    # labeling the scatter deaths
    annotations.append(dict(xref='x2', yref='y2',
                            y=xd, x=ydn + 10,
                            text= str(ydn) + '%',
                            font=dict(family='Arial', size=12,
                                      color='rgb(128, 0, 128)'),
                            showarrow=False))
    # labeling the bar confirmed cases
    annotations.append(dict(xref='x1', yref='y1',
                            y=xd, x=yd + 5000,
                            text='{:,.0f}'.format(yd),
                            font=dict(family='Arial', size=12,
                                      color='rgb(50, 171, 96)'),
                            showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()


# In[ ]:


# confirmed cases & mortality rates

fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                    shared_yaxes=False, vertical_spacing=0.001)

fig.append_trace(go.Bar(
    x=top_10_death_rates[::-1].Confirmed,
    y=top_10_death_rates[::-1].Country,
    marker=dict(
        color='rgba(50, 171, 96, 0.6)',
        line=dict(
            color='rgba(50, 171, 96, 1.0)',
            width=1),
    ),
    name='Total Number of Confirmed Cases',
    orientation='h',
), 1, 1)

fig.append_trace(go.Scatter(
    x=top_10_death_rates[::-1].Death_Rate, 
    y=top_10_death_rates[::-1].Country,
    mode='lines+markers',
    line_color='rgb(128, 0, 128)',
    name='Recovery Rate',
), 1, 2)

fig.update_layout(
    title='Covid-19 : Confirmed Cases & Mortality Rates',
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
        domain=[0, 0.85],
    ),
    yaxis2=dict(
        showgrid=False,
        showline=True,
        showticklabels=False,
        linecolor='rgba(102, 102, 102, 0.8)',
        linewidth=2,
        domain=[0, 0.85],
    ),
    xaxis=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0, 0.42],
    ),
    xaxis2=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0.47, 1],
        side='top',
        dtick=2,
    ),
    legend=dict(x=0.029, y=1.038, font_size=10),
    margin=dict(l=100, r=20, t=70, b=70),
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

annotations = []

y_s = np.round(top_10_death_rates.Confirmed, decimals=0)
y_nw = np.rint(top_10_death_rates.Death_Rate)

# Adding labels
for ydn, yd, xd in zip(y_nw, y_s, top_10_death_rates.Country):
    # labeling the scatter deaths
    annotations.append(dict(xref='x2', yref='y2',
                            y=xd, x=ydn + 0.8,
                            text= str(ydn) + '%',
                            font=dict(family='Arial', size=12,
                                      color='rgb(128, 0, 128)'),
                            showarrow=False))
    # labeling the bar confirmed cases
    annotations.append(dict(xref='x1', yref='y1',
                            y=xd, x=yd + 5000,
                            text='{:,.0f}'.format(yd),
                            font=dict(family='Arial', size=12,
                                      color='rgb(50, 171, 96)'),
                            showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()


# In[ ]:


# best recovery rates - more than 100 cases 
best_rec_rate = recovery_death_rates[recovery_death_rates.Confirmed >= 100].sort_values('Recovery_Rate', ascending=False)[:10]
best_rec_rate['Recovery_Rate'] = np.round(best_rec_rate['Recovery_Rate'],1)

# Worst Mortality Rates
worst_mort_rate = recovery_death_rates[recovery_death_rates.Confirmed >= 100].sort_values('Death_Rate', ascending=False)[:10]
worst_mort_rate['Death_Rate'] = np.round(worst_mort_rate['Death_Rate'],1)


# In[ ]:


fig = px.bar(best_rec_rate[::-1], x= 'Recovery_Rate', y='Country', 
             orientation='h',text='Recovery_Rate',title='Countries with best recovery rates (> 100 Confirmed Cases)',
             labels={'Recovery_Rate':'Recovery Rate %'}, template="plotly_dark")
fig.show()


# In[ ]:


fig = px.bar(worst_mort_rate[::-1], x= 'Death_Rate', y='Country', orientation='h',text='Death_Rate',
             title='Countries with worst mortality rates (> 100 Confirmed Cases)', 
             labels={'Death_Rate':'Mortality Rate %'}, template="plotly_dark")
fig.show()


# * San Marino (12.5%) has the worst mortality rate followed by Indonesia and Italy (9.3%). 
# * China has a mind-boggling recvovery rate of 89%. Sonething is going to right for them.
# * Surprisingly the cruise ship Diamond Princess also had a significant 45.6% recovery rate. So all hope is not lost. 

# **Conclusions**
# 
# The number of active cases is most likely expected to increase in the coming weeks, especially in countries like India and the US as they increase testing. Also, countries like India have imposed mandatory curfews to break the chain. So it may help flatten the curve to a certain extent as evidenced by the drop in cases in China.
# 
# The recovery rates have been significant in China and the cruise ship. This is a positive sign that things will turn around eventually.
# 
# The next steps will be to analyze coutry specific data (I am inetersted in India and US) and also study the impact of demography on the recovery and mortatlity rates.
