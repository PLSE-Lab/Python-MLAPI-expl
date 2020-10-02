#!/usr/bin/env python
# coding: utf-8

# # COVID - 19 Updates

# Here are the corona virus update for different countries.

# **Importing Libraries**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
pd.set_option('display.max_rows', None)
import datetime
from plotly.subplots import make_subplots


# **Importing dataset**

# In[ ]:


data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')


# In[ ]:


data.head(5)


# I have done some preprocessing steps i.e. converting all float types into int types and make a column of active cases.

# In[ ]:


data[["Confirmed","Deaths","Recovered"]] =data[["Confirmed","Deaths","Recovered"]].astype(int)


# In[ ]:


data['Country/Region'] = data['Country/Region'].replace('Mainland China', 'China')


# In[ ]:


data.head()


# In[ ]:


data['Active_case'] = data['Confirmed'] - data['Deaths'] - data['Recovered']


# In[ ]:


data.head()


# In[ ]:


def dark_confirmed(data,lockdown,month_lockdown,country):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["ObservationDate"], y=data['Confirmed'],
                    mode="lines",
                    name='Confirmed cases in ' + country,
                    marker_color='yellow',
                        ))

    fig.add_annotation(
            x=lockdown,
            y=data['Confirmed'].max(),
            text="COVID-19 pandemic lockdown in "+ country,
             font=dict(
            family="Courier New, monospace",
            size=16,
            color="red"
            ),
    )


    fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0=lockdown,
            y0=data['Confirmed'].max(),
            x1=lockdown,
    
            line=dict(
                color="red",
                width=4
            )
    ))
    fig.add_annotation(
            x=month_lockdown,
            y=data['Confirmed'].min(),
            text="Month after lockdown",
             font=dict(
            family="Courier New, monospace",
            size=16,
            color="#00FE58"
            ),
    )

    fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0=month_lockdown,
            y0=data['Confirmed'].max(),
            x1=month_lockdown,
    
            line=dict(
                color="#00FE58",
                width=3
            )
    ))
    fig
    fig.update_layout(
        title='Evolution of Confirmed cases over time in '+ country,
        template='plotly_dark'

    )

    fig.show()


# In[ ]:


def dark_active(data,lockdown,month_lockdown,country):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["ObservationDate"], y=data['Active_case'],
                    mode="lines",
                    name='Active cases in' + country,
                    marker_color='#00FE58',
                        ))

    fig.add_annotation(
            x=lockdown,
            y=data['Active_case'].max(),
            text="COVID-19 pandemic lockdown in "+ country,
             font=dict(
            family="Courier New, monospace",
            size=16,
            color="red"
            ),
    )


    fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0=lockdown,
            y0=data['Active_case'].max(),
            x1=lockdown,
    
            line=dict(
                color="red",
                width=3
            )
    ))
    fig.add_annotation(
            x=month_lockdown,
            y=data['Active_case'].min(),
            text="Month after lockdown",
             font=dict(
            family="Courier New, monospace",
            size=16,
            color="rgb(255,217,47)"
            ),
    )

    fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0=month_lockdown,
            y0=data['Active_case'].max(),
            x1=month_lockdown,
    
            line=dict(
                color="rgb(255,217,47)",
                width=3
            )
    ))
    fig
    fig.update_layout(
        title='Evolution of Active cases over time in '+ country,
        template='plotly_dark'

    )

    fig.show()


# In[ ]:


def dark_recovered(data,lockdown,month_lockdown,country):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["ObservationDate"], y=data['Recovered'],
                    mode="lines",
                    name='Recovered cases in' + country,
                    marker_color='rgb(192,229,232)',
                        ))

    fig.add_annotation(
            x=lockdown,
            y=data['Recovered'].max(),
            text="COVID-19 pandemic lockdown in "+ country,
             font=dict(
            family="Courier New, monospace",
            size=16,
            color="red"
            ),
    )


    fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0=lockdown,
            y0=data['Recovered'].max(),
            x1=lockdown,
    
            line=dict(
                color="red",
                width=3
            )
    ))
    fig.add_annotation(
            x=month_lockdown,
            y=data['Active_case'].min(),
            text="Month after lockdown",
             font=dict(
            family="Courier New, monospace",
            size=16,
            color="rgb(103,219,165)"
            ),
    )

    fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0=month_lockdown,
            y0=data['Recovered'].max(),
            x1=month_lockdown,
    
            line=dict(
                color="rgb(103,219,165)",
                width=3
            )
    ))
    fig
    fig.update_layout(
        title='Evolution of Recovered cases over time in '+ country,
        template='plotly_dark'

    )

    fig.show()


# Updates in **Tunisia**

# In[ ]:


Data_tunisia = data [(data['Country/Region'] == 'Tunisia') ].reset_index(drop=True)


# In[ ]:


dark_confirmed(Data_tunisia,"03/22/2020","04/22/2020","Tunisia")


# In[ ]:


dark_active(Data_tunisia,"03/22/2020","04/22/2020","Tunisia")


# In[ ]:


dark_recovered(Data_tunisia,"03/22/2020","04/22/2020","Tunisia")


# Updates in **Italy**

# In[ ]:


Data_Italy = data [(data['Country/Region'] == 'Italy') ].reset_index(drop=True)
Data_italy_op= Data_Italy.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)


# In[ ]:


dark_confirmed(Data_italy_op,"03/09/2020","04/09/2020","Italy")


# In[ ]:


dark_active(Data_italy_op,"03/09/2020","04/09/2020","Italy")


# In[ ]:


dark_recovered(Data_italy_op,"03/09/2020","04/09/2020","Italy")


# Updates in **France**

# In[ ]:


Data_France = data [(data['Country/Region'] == 'France') ].reset_index(drop=True)
Data_France_op= Data_France.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)


# In[ ]:


dark_confirmed(Data_France_op,"03/17/2020","04/17/2020","France")


# In[ ]:


dark_active(Data_France_op,"03/17/2020","04/17/2020","France")


# In[ ]:


dark_recovered(Data_France_op,"03/17/2020","04/17/2020","France")


# Updates in **UK**

# In[ ]:


Data_UK = data [(data['Country/Region'] == 'UK') ].reset_index(drop=True)
Data_UK_op= Data_UK.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)


# In[ ]:


dark_confirmed(Data_UK_op,"03/23/2020","04/23/2020","United Kingdom")


# In[ ]:


dark_active(Data_UK_op,"03/23/2020","04/23/2020","United Kingdom")


# In[ ]:


dark_recovered(Data_UK_op,"03/23/2020","04/23/2020","United Kingdom")


# Updates in **India**

# In[ ]:


Data_India = data [(data['Country/Region'] == 'India') ].reset_index(drop=True)
Data_India_op= Data_India.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)


# In[ ]:


dark_confirmed(Data_India_op,"03/24/2020","04/24/2020","India")


# In[ ]:


dark_active(Data_India_op,"03/24/2020","04/24/2020","India")


# In[ ]:


dark_recovered(Data_India_op,"03/24/2020","04/24/2020","India")


# Updates in **Germany**

# In[ ]:


Data_Germany = data [(data['Country/Region'] == 'Germany') ].reset_index(drop=True)
Data_Germany_op= Data_Germany.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)


# In[ ]:


dark_confirmed(Data_Germany_op,"03/23/2020","04/23/2020","Germany")


# In[ ]:


dark_active(Data_Germany_op,"03/23/2020","04/23/2020","Germany")


# In[ ]:


dark_recovered(Data_Germany_op,"03/23/2020","04/23/2020","Germany")


# Here I displayed some countries updates of corona virus.

# Till then Enjoy Machine Learning!!!!!!!!!!!!!!
