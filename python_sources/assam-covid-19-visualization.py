#!/usr/bin/env python
# coding: utf-8

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


import pandas as pd

import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# In[ ]:


Assam_data = pd.read_csv('../input/AssamCovid19.csv',parse_dates=['Date'], dayfirst=True)
Assam_Place_data = pd.read_csv('../input/Assam_state.csv')


# In[ ]:


Assam_data.head()


# In[ ]:


Assam_Place_data.head()


# In[ ]:


Assam_data['Date']=pd.to_datetime(Assam_data.Date,dayfirst=True)
Assam_daily= Assam_data.groupby(['Date'])['Confirmed'].sum().reset_index().sort_values('Confirmed',ascending=True)

fig=go.Figure()
fig.add_trace(go.Bar(x=Assam_daily['Date'],y=Assam_daily['Confirmed'],marker_color='red'))
fig.update_layout(
    title="Overall COVID-19 cases in Assam",
    xaxis_title="Date",
    yaxis_title="Total Confirmed cases",
    font=dict(
#     family="Airel, monospace",
    size=18,
    color="black"),
    annotations=[             #----------- Annotation section to add specific details on plots
        dict(
            x='2020-03-22',
            y=0,
            xref="x",
            yref="y",
            text="India Lockdown",
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-300
        )
    ])


# In[ ]:


Assam_Place_data = pd.read_csv('../input/Assam_age.csv'


# In[ ]:


fig = go.Figure()
Assam_daily= Assam_data.groupby(['Date'])['Confirmed'].sum().reset_index().sort_values('Confirmed',ascending=True)
fig.add_trace(go.Scatter(x=Assam_daily['Date'],y=Assam_daily['Confirmed'] fill='tozeroy', line_shape='spline',
                        hovertext=["Percentage= 3.18%", "Percentage= 3.90%", "Percentage= 24.86%", "Percentage= 21.10%", "Percentage= 16.18%", "Percentage= 11.13%", "Percentage= 12.86%", "Percentage= 16.18%"],
                        hoverinfo='text',)) 
fig.update_layout(
    width=1200,
    height=900,
    title="Age-wise confirmed cases",
    yaxis_title="Total cases",
    xaxis_title="Age")


# # **will continue.................. work under process**
