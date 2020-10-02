#!/usr/bin/env python
# coding: utf-8

# The data set i have taken is from a Indian govt website https://data.gov.in/ the data set contains information different indian airline operators and their operational data in aggerated form.  

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly
import plotly.graph_objs as go
import plotly.plotly as py
plotly.offline.init_notebook_mode(connected=True)


# i have importing different packages like plotly, seaborn and matplotlib for data Visualization  

# In[ ]:


airline_data_india=pd.read_csv('../input/Table_1_Airlinewise_DGCA_Q4_OCT-DEC_2017.csv')


# In[ ]:


airline_data_india.head()


# In[ ]:


airline_data_india.tail()


# As we can see in the above line of code that the last row has to be dropped from the data set, a simple slicing will do the trick but we use dataframe operation to drop the row data point  

# In[ ]:


airline_data_india=airline_data_india[airline_data_india.Category!='TOTAL (DOMESTIC & FOREIGN CARRIERS)']


# First lets use seaborn package 
# i will be using count plot 

# In[ ]:


sns.set
size=sns.countplot(x='Category',data=airline_data_india)
size.figure.set_size_inches(10,8)
plt.show()


# Now lets ask ourselves some basic question which category fly more PASSENGERS TO INDIA 

# In[ ]:


airline_category=airline_data_india.groupby(['Category'], as_index=False).agg({"PASSENGERS TO INDIA": "sum"})


# In[ ]:


fig = {
  "data": [
    {
      "values":airline_category['PASSENGERS TO INDIA'],
      "labels": airline_category.Category,
      "hoverinfo":"label+percent",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Flyers to India"
  }
}
plotly.offline.iplot(fig, filename='airline_pie_chart')


# Lets see which Foreign carriers brings more passengers 

# In[ ]:


airline_foreign=airline_data_india[airline_data_india.Category=='FOREIGN CARRIERS']


# In[ ]:


import ipywidgets as widgets


# In[ ]:


g=widgets.Dropdown(
    options=list(airline_foreign['NAME OF THE AIRLINE'].unique()),
    value='AEROFLOT',
    description='Option:'
)
x_value=widgets.IntSlider(min=0,max=0)
y_value=widgets.IntSlider(min=0,max=4)
ui = widgets.HBox([g, x_value, y_value])


# In[ ]:


def on_change(change,change_x,change_y):
    sample_data=airline_foreign[airline_foreign['NAME OF THE AIRLINE']==change]
    numeric_data=sample_data.select_dtypes(exclude='object')
    x=numeric_data.columns.tolist(),
    y=numeric_data.values.tolist()
    trace = [go.Bar(
            x=x[0][change_x:change_y],
            y=y[0][change_x:change_y],
            marker=dict(
        color=['rgba(158, 21, 30, 1)', 'rgba(26, 118, 255,0.8)',
               'rgba(107, 107, 107,1)', 'rgba(255, 140, 0, 1)',
               'rgba(0, 191, 255, 1)']),
    )]
    plotly.offline.iplot(trace, filename='basic-bar')
out=widgets.interactive_output(on_change,{'change':g,'change_x':x_value,'change_y':y_value})
display(ui, out)


# Lets see which domestic Airline is preferred by people

# In[ ]:


sorted_data=airline_data_india.sort_values(by=['PASSENGERS TO INDIA'],ascending=False)


# In[ ]:


sorted_data=sorted_data[sorted_data['NAME OF THE AIRLINE']!="TOTAL (FOREIGN CARRIERS)"]
sorted_data=sorted_data[sorted_data['NAME OF THE AIRLINE']!="TOTAL (DOMESTIC CARRIERS)"]


# In[ ]:


gs=widgets.Dropdown(
    options=sorted_data.iloc[:,3:].columns,
    value='PASSENGERS TO INDIA',
    description='Option:'
)
Max=widgets.IntSlider(min=2,max=40)
us = widgets.HBox([Max,gs])


# In[ ]:


def ms(x,y):
    sorteds=sorted_data.iloc[:x,]
    bar=sorteds['NAME OF THE AIRLINE'].tolist()
    value=sorteds[y].tolist()
    trace = [go.Bar(
            x=bar,
            y=value,
    )]
    plotly.offline.iplot(trace, filename='basic-bar')
out=widgets.interactive_output(ms,{'x':Max,'y':gs})
display(us, out)


# 
