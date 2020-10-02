#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_19 = pd.read_csv('../input/flight-delay-prediction/Jan_2019_ontime.csv')
df_19.drop('Unnamed: 21',axis=1,inplace=True)
df_19.head()


# In[ ]:


df_19.info()


# # Visualizations

# In[ ]:


plot1 = df_19.groupby('DAY_OF_MONTH')['CANCELLED'].count()
fig = go.Figure()

fig.add_trace(go.Bar(x=plot1.index, y=plot1.values, name='Cancel bar',opacity=0.9,marker_color='#a5afff'))

fig.add_trace(go.Scatter(x=plot1.index, y=plot1.values, line=dict(color='red'), name='Cancel trend'))
fig.update_layout(
    title="Cancelled flights vs day of month",
    xaxis_title="Day of month",
    yaxis_title="Cancel count",
)
fig.show()


# In[ ]:


plot2 = df_19.groupby('DAY_OF_MONTH')['DIVERTED'].count()
fig2 = go.Figure()

fig2.add_trace(go.Bar(x=plot2.index, y=plot2.values, name='Diverted bar',opacity=0.6,marker_color='#ff0000'))

fig2.add_trace(go.Scatter(x=plot2.index, y=plot2.values, line=dict(color='#4200ff'), name='Diverted trend'))
fig2.update_layout(
    title="Diverted flights vs day of month",
    xaxis_title="Day of month",
    yaxis_title="Diverted count",
)
fig2.show()


# ### Days such as 1,5,12... have least cancelled flights whereas days such as 2,11,18... have most cancelled flights, let's now check the day name of these dates

# In[ ]:


import calendar
yy = 2019 
mm = 1    
print(calendar.month(yy, mm))


# ### There is a seasonality in the pattern of cancelled flights
# * Days with least cancelled flights are Saturday
# * Days with most cancelled flights are Monday and Friday
# 
# ### Also the plots seem to be equal, let's check it out

# In[ ]:


all(plot1 == plot2)


# ### We'll be now using just CANCELLED feature because DIVERTED does not provide different observations.

# In[ ]:


plot3 = df_19.groupby('ORIGIN')['CANCELLED'].count().sort_values(ascending=False)
#cap to above 400
plot3 = plot3[plot3>400]
fig3 = px.bar(plot3,x=plot3.index,y=plot3.values,color_discrete_sequence=['#6ec8ba'],
             title="Cancelled flights by origin airport",labels={"ORIGIN":"Origin airport","y":"Count"})
fig3.layout.template = 'plotly_white'
fig3.update_xaxes(tickangle=45)
fig3.show()


# In[ ]:


plot4 = df_19.groupby('DEST')['CANCELLED'].count().sort_values(ascending=False)
#cap to above 400
plot4 = plot4[plot4>400]
fig4 = px.bar(plot4,x=plot4.index,y=plot4.values,color_discrete_sequence=['#1A7065'],
             title="Cancelled flights by destination airport",labels={"DEST":"Destination airport","y":"Count"})
fig4.layout.template = 'plotly_white'
fig4.update_xaxes(tickangle=45)
fig4.show()


# ### After looking at the regions affected by tornados, ALT and CLT i.e airports with most cancelled flights lie within high risk region of tornados.
# ![](https://strangesounds.org/wp-content/uploads/2014/04/tornado-risk-map.jpg)

# In[ ]:


plot5 = df_19.groupby(['ORIGIN','DEST'])['CANCELLED'].count().sort_values(ascending=False)
plot5.index = [i+'-'+j for i,j in plot5.index] 
#cap to above 500 for clear visualization
plot5 = plot5[plot5>500]
fig5 = px.bar(plot5,x=plot5.index,y=plot5.values,color_discrete_sequence=['#B93795'],
             title="Cancelled flights by origin-dest",labels={"index":"Origin-Destination","y":"Count"})
fig5.layout.template = 'plotly_white'
fig5.update_xaxes(tickangle=45)
fig5.show()


# ## The case of delayed/cancelled flights and be explained due to a storm occurence in san francisco on 16th Jan, 2019
# https://www.sfchronicle.com/bayarea/article/Flood-warnings-issued-for-San-Francisco-region-14980399.php

# ### Also the fact that the airports with most cancelled/delayed flights are amongst most busiest in USA makes the picture clearer.
# https://www.world-airport-codes.com/us-top-40-airports.html

# In[ ]:


plot6 = df_19.groupby('OP_CARRIER')['DEP_DEL15'].sum().sort_values()
fig6 = px.pie(names=plot6.index,values=list(map(int,plot6.values)),
              color_discrete_sequence =px.colors.qualitative.T10, hole=0.5, title='Airlines with most delayed flights')
fig6.show()


# In[ ]:


plot6 = df_19.groupby('OP_CARRIER')['DISTANCE'].mean().sort_values()
fig6 = px.pie(names=plot6.index,values=list(map(int,plot6.values)),
              color_discrete_sequence =px.colors.qualitative.Pastel, hole=0.5, 
              title='Airlines with most long distance flights')
fig6.show()


# In[ ]:


plt.Figure(figsize=(30,22))
sns.distplot(df_19[df_19.CANCELLED==0.0].DISTANCE, hist=True)
sns.distplot(df_19[df_19.CANCELLED==1.0].DISTANCE, hist=True)
# cancelled orange
# not cancelled blue


# In[ ]:


cor = df_19.corr().fillna(0)
fig = px.imshow(cor,
            labels=dict(color="Viridis"),
            x=cor.index,
            y=cor.columns)
fig.update_layout(width=800,height=800)
fig.show()


# In[ ]:




