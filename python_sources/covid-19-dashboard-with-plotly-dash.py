#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Dashboard

# **This notebook is about how to build a visually pleasant Dashboard using plotly dash libraries. Although tools like Tableau/Power BI are good and powerful for data visualizaiton and dashboarding but it would also be nice to build something in programming language like python using standard libraries. It is just a guideline for the starters and lot of things can be done to build nice and standard dashboards.**

# **Importing some useful standard libraries**

# In[ ]:


#loading dash libraries
get_ipython().system('pip install dash==1.11.0;')
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

import pandas as pd
import numpy as np


# **Loading Input Datasets**

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


df=pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")


# **Preprocessing for Dashboard**

# In[ ]:


df_ts=df.groupby(['ObservationDate']).sum()
df_bd=df.where(df['Country/Region']=='Bangladesh')


# In[ ]:


df1=df.where(df.ObservationDate==max(df.ObservationDate))
df2=df1.groupby('Country/Region').sum()
df3=df2.sort_values(by='Deaths',ascending=False)
df3=df3.head(20);


# **Building the Dashboard**

# In[ ]:


app=dash.Dash()
app.layout= html.Div([
    html.H1('COVID-19 Dashboard'),
    
    dcc.Graph(
        id='line_bd',
        figure={            
                'data':[ 
                {'x': df_bd.ObservationDate, 'y': df_bd.Confirmed,'type':'bar','name':'Confirmed_bd'},
                {'x': df_bd.ObservationDate, 'y': df_bd.Deaths,'type':'bar','name':'Deaths_bd', 'marker' : { "color" : 'rgb(255,0,0)'}},
                {'x': df_bd.ObservationDate, 'y': df_bd.Recovered,'type':'bar','name':'Recovered_bd', 'marker' : { "color" : 'rgb(0,128,0)'}}
                        ],
                'layout':go.Layout(title='Bangladesh Time Series Cases')           
                }
            ),
    
     dcc.Graph(
        id='line',
        figure={            
                'data':[ 
                {'x': df_ts.index, 'y': df_ts.Confirmed,'name':'Confirmed'},
                {'x': df_ts.index, 'y': df_ts.Deaths,'name':'Deaths', 'marker' : { "color" : 'rgb(255,0,0)'}},
                {'x': df_ts.index, 'y': df_ts.Recovered,'name':'Recovered', 'marker' : { "color" : 'rgb(0,128,0)'}}
                        ],
                'layout':go.Layout(title='Worldwide Time Series Cases')           
                }
            ),
    
    dcc.Graph(
        id='confirmed',
        figure={            
                'data':[
                {'x': df3.index, 'y': df3.Confirmed, 'type':'bar','name':'Confirmed'}
                        ],
                'layout':go.Layout(title='Confirmed Cases by Country')           
                }
            ),
    
    dcc.Graph(
        id='death',
        figure={            
                'data':[                
                {'x': df3.index, 'y': df3.Deaths, 'type':'bar','name':'Deaths', 'marker' : { "color" : 'rgb(255,0,0)'}}
                        ],
                'layout':go.Layout(title='Death Cases by Country')            
                }
            ),
    
    dcc.Graph(
        id='recovered',
        figure={            
                'data':[              
                {'x': df3.index, 'y': df3.Recovered, 'type':'bar','name':'Recovered', 'marker' : { "color" : 'rgb(0,128,0)'}}
                        ],
                'layout':go.Layout(title='Recovered Cases by Country')            
                }
            )
])


# **Run the Dashboard in the server**

# In[ ]:


if __name__=="__main__":
    app.run_server(port=4051)


# **Although the server is not running in kaggle and not showing the link to go to the Dashboard, it will work fine in you own python/anaconda environment. Thank you. Have Fun!.**
