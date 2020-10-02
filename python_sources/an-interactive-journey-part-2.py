#!/usr/bin/env python
# coding: utf-8

# Hi everyone! Given the plethora of notebooks and analysis available on the recent covid19 situations, I took a step back to think about something comprehensive and novel that could be done to make it more interactive to the viewers. So I came up with the ideas (which you'll see below) and went on to implement it. Here is my interactive plots part 2. You can find the part 1 [here](https://www.kaggle.com/twinkle0705/an-interactive-eda-of-electricity-consumption). 
# 
# I was overwhelmed by the responses I got and it kept me going with this notebook. I hope you guys enjoy it while I keep updating the plots in future versions of it.
# Stay tuned, stay safe!
# 
# Do comment your suggestions and upvote if you like it!

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


# # World Count

# In[ ]:


df = pd.read_csv('../input/corona-virus-report/worldometer_data.csv')

fig = go.Figure(go.Funnel(
    y = ["Total Cases", "Total Recovered", "Active Cases", "Deaths"],
    x = [df['TotalCases'].sum(),df['TotalRecovered'].sum(),df['ActiveCases'].sum(),df['TotalDeaths'].sum()],
    textposition = "inside",
    textinfo = "value+percent initial",
    opacity = 0.9, 
    marker = {"color": ["Red", "Green", "Yellow", "Crimson"],"line": {"width": 2, "color": 'Black'}},
    connector = {"line": {"color": "Black", "dash": "dot", "width": 3}} 
                             ))
fig.update_layout(
    template="plotly_white",
    title={
        'text': "Cases around the world",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_layout( width=700,height=600)

fig.show()


# The figure shows the total number of cases around the world along with recovered, active and deaths. Hover over the plot to see the values along with percentages of different categories.

# In[ ]:


c_df = df.groupby([df.Continent])['TotalCases','TotalDeaths','TotalRecovered','ActiveCases'].sum()
c_df.index = c_df.index.set_names(['Continent'])
c_df = c_df.reset_index()
c_df = c_df.melt(id_vars=['Continent','TotalCases'],var_name= 'kind', value_name='Cases')


# In[ ]:


fig = px.sunburst(c_df, path=['Continent','TotalCases','kind'], values='Cases',title='Continent-wise overview',color='Cases',
                  color_continuous_scale='rdbu')
fig.update_layout( width=800,height=600)

fig.show()


# The above figure will give you an overall number for the cases continent-wise. Click on any continent to get a view of active cases, recovered cases and deaths.

# In[ ]:


world_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
world_data.sort_values('6/22/20', ascending = False, inplace = True)
world_data.rename(columns={'6/22/20':'Date'}, inplace = True)
world_data['Province/State'].fillna('.', inplace = True)

world_data1 = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
world_data1.sort_values('6/22/20', ascending = False, inplace = True)
world_data1.rename(columns={'6/22/20':'Date'}, inplace = True)
world_data1['Province/State'].fillna('.', inplace = True)

world_data2 = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
world_data2.sort_values('6/22/20', ascending = False, inplace = True)
world_data2.rename(columns={'6/22/20':'Date'}, inplace = True)
world_data2['Province/State'].fillna('.', inplace = True)


# In[ ]:


world_data['text'] ='Confirmed Cases ' + (world_data['Date']).astype(str) + '<br>' + world_data['Country/Region']  + '<br>' + world_data['Province/State']
world_data1['text'] = 'Recovered Cases ' + (world_data1['Date']).astype(str)+ '<br>' + world_data1['Country/Region'] + '<br>' + world_data1['Province/State']
world_data2['text'] = 'Deaths ' + (world_data2['Date']).astype(str) + '<br>' + world_data2['Country/Region'] + '<br>' + world_data2['Province/State']


fig = go.Figure()


fig.add_trace(
    go.Scattergeo(lat=world_data["Lat"],
                  lon=world_data["Long"],
                  name = 'Total Case',
                  mode="markers",
                  text = world_data['text'],
                  showlegend=True,
                  marker=dict(color='yellow', size=10, opacity=0.8,line=dict(width=1, color='Black')))
)

fig.add_trace(
    go.Scattergeo(lat=world_data1["Lat"],
                  lon=world_data1["Long"],
                  name = 'Recovered',
                  mode="markers",
                  text = world_data1['text'],
                  showlegend=True,
                  marker=dict(color='darkblue', size=10, opacity=0.8,line=dict(width=1, color='Black')))
)

fig.add_trace(
    go.Scattergeo(lat=world_data2["Lat"],
                  lon=world_data2["Long"],
                  name = 'Deaths',
                  mode="markers",
                  text = world_data2['text'],
                  showlegend=True,
                  marker=dict(color='red', size=10, opacity=0.8,line=dict(width=1, color='Black')))
)

fig.update_geos(
    projection_type="orthographic",
    landcolor="seagreen",
    oceancolor="skyblue",
    showocean=True,
    lakecolor="LightBlue",
    center_lon=-180,
    center_lat=0,
    projection_rotation_lon=-180,
)


fig.update_layout(
    template="plotly_white",
    title={
        'text': "Cases in different countries around the world",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig.update_layout( width=800,height=600)

fig.show()


# **Double click on the side legend and press reset from above to view a particular trace.**
# 
# The fully rotatable world map presents to us the overall scenario in all of the countries around the world. You can zoom in/out, hover over the points, and select any of the 3 categories from the side labels.

# # Trends in Different Countries

# By playing the animations below, we can see the trends in every country over time. Hovering over the map will tell us the exact number and the country that we are pointing. Zoom in to observe the trends of a particular region closely.

# In[ ]:


df_country = pd.read_csv('../input/corona-virus-report/full_grouped.csv') 


# In[ ]:


fig = px.choropleth(df_country, locations=df_country['Country/Region'],
                    color=df_country['Confirmed'], locationmode='country names',
                    hover_name=df_country['Country/Region'],hover_data =['Country/Region','Confirmed'],
                    color_continuous_scale='ylgnbu',template='plotly_white', animation_frame = 'Date')
fig.update_layout(
    title='Confirmed Cases In Each Country over Time',
)
fig.update_layout( width=700,height=600)
fig.show()


# In[ ]:


fig = px.choropleth(df_country, locations=df_country['Country/Region'],
                    color=df_country['Recovered'], locationmode='country names',
                    hover_name=df_country['Country/Region'],hover_data =['Country/Region','Recovered'],
                    color_continuous_scale='speed',template='plotly_white', animation_frame = 'Date')
fig.update_layout(
    title='Recovered Cases In Each Country over Time',
)
fig.update_layout( width=700,height=600)
fig.show()


# In[ ]:


fig = px.choropleth(df_country, locations=df_country['Country/Region'],
                    color=df_country['Deaths'], locationmode='country names',
                    hover_name=df_country['Country/Region'],hover_data =['Country/Region','Deaths'],
                    color_continuous_scale='ylorrd',template='plotly_white', animation_frame = 'Date')
fig.update_layout(
    title='Deaths In Each Country over Time',
)
fig.update_layout( width=700,height=600)
fig.show()


# In[ ]:


fig = px.choropleth(df_country, locations=df_country['Country/Region'],
                    color=df_country['Active'], locationmode='country names',
                    hover_name=df_country['Country/Region'],hover_data =['Country/Region','Active'],
                    color_continuous_scale='agsunset',template='plotly_white', animation_frame = 'Date')
fig.update_layout(
    title='Active Cases In Each Country Over time',
)
fig.update_layout(width=700,height=600)
fig.show()


# # Checking top 10 affected countries

# Next we are going to have a look at the top 10 affected countries. I have included china as well so that we can compare its numbers with other countries. It looks like other countries have been severly affected. We can include/exclude any country by simply clicking on it from the side labels. 
# 
# 

# In[ ]:


new = df_country[(df_country['Country/Region']=='India') | (df_country['Country/Region']=='China') | (df_country['Country/Region']=='US') | (df_country['Country/Region']=='Brazil')| (df_country['Country/Region']=='Russia') | (df_country['Country/Region']=='United Kingdom') | (df_country['Country/Region']=='Spain') | (df_country['Country/Region']=='Italy') | (df_country['Country/Region']=='Peru') | (df_country['Country/Region']=='Iran') | (df_country['Country/Region']=='Germany')]
new['New deaths'] = new['New deaths'].abs()
new['New recovered'] = new['New recovered'].abs()


# In[ ]:


fig = px.bar(new, x="Country/Region", y="Confirmed",color='Country/Region',animation_frame = 'Date')
fig.update_traces(marker_line_width=2, opacity=0.9, marker_line_color = 'black')
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.update_layout(title='Top 10 countries Confirmed cases',width=700,height=500)
fig.show()


# In[ ]:


fig = px.bar(new, x="Country/Region", y="Active",color='Country/Region',animation_frame = 'Date')
fig.update_traces(marker_line_width=2, opacity=0.9, marker_line_color = 'black')
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.update_layout(title='Top 10 countries Active cases',width=700,height=500)
fig.show()


# In[ ]:


fig = px.bar(new, x="Country/Region", y="Recovered",color='Country/Region',animation_frame = 'Date')
fig.update_traces(marker_line_width=2, opacity=0.9, marker_line_color = 'black')
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.update_layout(title='Top 10 countries Recovered cases',width=700,height=500)
fig.show()


# In[ ]:


fig = px.bar(new, x="Country/Region", y="Deaths",color='Country/Region',animation_frame = 'Date')
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.update_traces(marker_line_width=2, opacity=0.9, marker_line_color = 'black')
fig.update_layout(title='Top 10 countries Deaths',width=700,height=500)
fig.show()


# Who doesn't love plotting a bar chart race? Never knew it could be done so easily!

# In[ ]:


# topn = 10 ## Top 10 confirmed cases to get
# cols = list(df.where(df['Date']==df['Date'].max()).dropna().sort_values(by='Confirmed', ascending=False)[:topn]['Country/Region']); print(cols)


# # Taking a look at different curves

# Instead of looking at curves of different countries separately, I decided to put them all in the same plot. Now we have the option to make comparisons as per our wish and it can be done by simply clicking on the country name. We can also select the range by clicking on the buttons (1m/2m/3m) and slide through the dates by using the slider below. 

# **SELECT ANY REGION TO ZOOM IT, DOUBLE CLICK ON ANY COUNTRY ON THE SIDE LEGEND TO ISOLATE IT'S CURVE**

# In[ ]:



fig = px.scatter(new,x="Date", y="Confirmed",color = 'Country/Region')
fig.update_traces(mode='markers + lines', marker_line_width=1.5, marker_size=15, marker_line_color = 'black')
fig.update_layout(title='Curve - Confirmed',
    legend=dict(x=0,y=1, traceorder="normal",
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2
    ))


fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=2, label="2m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(step="all")
        ])
    )
)

fig.show()


# In[ ]:



fig = px.scatter(new,x="Date", y="Active",color = 'Country/Region')
fig.update_traces(mode='markers + lines', marker_line_width=1.5, marker_size=15, marker_line_color = 'black')
fig.update_layout(title='Curve - Active',
    legend=dict(x=0,y=1, traceorder="normal",
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2
    ))


fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=2, label="2m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(step="all")
        ])
    )
)

fig.show()


# In[ ]:



fig = px.scatter(new,x="Date", y="Recovered",color = 'Country/Region')
fig.update_traces(mode='markers + lines', marker_line_width=1.5, marker_size=15, marker_line_color = 'black')
fig.update_layout(title='Curve - Recovered',
    legend=dict(x=0,y=1, traceorder="normal",
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2
    ))


fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=2, label="2m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show()


# In[ ]:



fig = px.scatter(new,x="Date", y="Deaths",color = 'Country/Region')
fig.update_traces(mode='markers + lines', marker_line_width=1.5, marker_size=15, marker_line_color = 'black')
fig.update_layout(title='Curve - Deaths',
    legend=dict(x=0,y=1, traceorder="normal",
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2
    ))

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=2, label="2m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show()


# # NEW CASES/ NEW RECOVERED/ NEW DEATHS

# For this analysis , I have used two types of plots. One which shows the trends over time, and the other that shows linear curves. 

# In[ ]:



fig = px.scatter(new,
    x='New cases',
    y='Country/Region',
    animation_frame= 'Date',
    range_x = (0,50000),
    color = 'Country/Region',
    text = 'New cases'
   
    )

fig.update_traces(mode='markers', marker_line_width=2, marker_size=40, marker_line_color = 'black')

fig.update_layout(title="Number of new cases over time",
                  xaxis_title="Number of new cases",
                  yaxis_title="Country")

fig.show()


# **SELECT ANY REGION TO ZOOM IT, DOUBLE CLICK ON ANY COUNTRY ON THE SIDE LEGEND TO ISOLATE IT'S CURVE**

# In[ ]:



fig = px.scatter(new,x="Date", y="New cases",color = 'Country/Region')
fig.update_traces(mode='markers+lines', marker_size=5)
fig.update_layout(title='Curve - New cases',
    legend=dict(x=0,y=1, traceorder="normal",
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2
    ))


fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=2, label="2m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(step="all")
        ])
    )
)

fig.show()


# Where on one hand Spain,Germany,Italy has shown decrease in number of new cases, Countries like US and India continue to see a rise in new cases.

# In[ ]:


fig = px.scatter(new,
    x='New recovered',
    y='Country/Region',
    animation_frame= 'Date',
    range_x = (0,30000),
    color = 'Country/Region',
    text = 'New recovered'
   
    )

fig.update_traces(mode='markers', marker_line_width=2, marker_size=40, marker_line_color = 'black')

fig.update_layout(title="Number of new recovered cases over time",
                  xaxis_title="Number of new recovered cases",
                  yaxis_title="Country")

fig.show()


# **SELECT ANY REGION TO ZOOM IT, DOUBLE CLICK ON ANY COUNTRY ON THE SIDE LEGEND TO ISOLATE IT'S CURVE**

# In[ ]:


fig = px.scatter(new,x="Date", y="New recovered",color = 'Country/Region')
fig.update_traces(mode='markers+lines', marker_size=5)
fig.update_layout(title='Curve - New Recovered',
    legend=dict(x=0,y=1, traceorder="normal",
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2
    ))

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=2, label="2m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show()


# In comparison with other countries, UK has shown almost zero recovery.

# In[ ]:


fig = px.scatter(new,
    x='New deaths',
    y='Country/Region',
    animation_frame= 'Date',
    range_x = (0,2600),
    color = 'Country/Region',
    text = 'New deaths'
   
    )

fig.update_traces(mode='markers', marker_line_width=2, marker_size=30, marker_line_color = 'black')

fig.update_layout(title="Number of new deaths over time",
                  xaxis_title="Number of new deaths",
                  yaxis_title="Country")

fig.show()


# **SELECT ANY REGION TO ZOOM IT, DOUBLE CLICK ON ANY COUNTRY ON THE SIDE LEGEND TO ISOLATE IT'S CURVE**

# In[ ]:


fig = px.scatter(new,x="Date", y="New deaths",color = 'Country/Region')
fig.update_traces(mode='markers+lines', marker_size=5)
fig.update_layout(title='Curve - New deaths',
    legend=dict(x=0,y=1, traceorder="normal",
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2
    ))

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=2, label="2m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show()


# # Recovered vs Deaths of most affected countries

# Making a comparison between Recovered vs Deaths of individual countries, it is shocking to see that countries like UK have far more deaths compared to recovered cases!

# In[ ]:


def mostaffected(df,topn):
    """
    Returns list of names of 'topn' most affected Country/Region
    till date.
    """
    return list(df.where(df['Date']==df['Date'].max()).dropna().sort_values(by='Confirmed', ascending=False)[:topn]['Country/Region']); print(cols)

def getsubdf(df, colname):
    """
    Returns sorted time series sub-dataframe 
    for the passed on column name.
    """
    
    return df.groupby('Country/Region').get_group(colname).sort_values(by='Date', ascending=True).reset_index(drop=True)


# In[ ]:


cols = mostaffected(df_country, 10)


for i in range (1,11):
    data = getsubdf(df_country, cols[i-1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data['Date'],y = data['Recovered'],name = cols[i-1]+' Recovered',marker_color='rgba(50, 300, 50, .9)',fill='tonexty'))
    fig.add_trace(go.Scatter(x = data['Date'],y = data['Deaths'],name= cols[i-1]+' Deaths',marker_color='rgba(300, 0, 0, .9)',fill='tozeroy'))
    fig.update_traces(mode='markers + lines', marker_line_width=1, marker_size=10)
    fig.update_layout(width = 600,height=500)
    fig.update_layout(title = "Recovered vs Deaths in " + cols[i-1])
    fig.update_layout(
    legend=dict(x=0,y=1, traceorder="normal",
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2
    ))
    fig.show()


# # Number of people affected wrt population

# Although we have seen the most affected countries by considering the most number of cases, we should also have a look at the most affected regions by considering the population.

# In[ ]:


df = pd.read_csv('../input/corona-virus-report/worldometer_data.csv')


# In[ ]:


df1 = df.sort_values('Tot Cases/1M pop', ascending = False).iloc[:25,:]
df1['text'] = 'Population:' + (df1['Population']).astype(str)+ '<br>'+ 'Confirmed:' + (df1['TotalCases']).astype(str)
fig = go.Figure()

fig.add_trace(go.Scatter(x=df1['Country/Region'], y=df1['Tot Cases/1M pop'], name='Cases/1M pop',mode = 'markers',
                         hovertext = df1['text'],
                         marker=dict(line_width = 2, size=df1['Tot Cases/1M pop']*0.01, color = df1['Tot Cases/1M pop'])))
fig.update_layout(title = "Countries with most cases per million population " )
fig.update_layout( width=700,height=600)
fig.show()


# In[ ]:


df2 = df.sort_values('Deaths/1M pop', ascending = False).iloc[:25,:]
df2['text'] = 'Population:' + (df2['Population']).astype(str)+ '<br>'+ 'Deaths:' + (df2['TotalDeaths']).astype(str)
fig = go.Figure()

fig.add_trace(go.Scatter(x=df2['Country/Region'], y=df2['Deaths/1M pop'], name='Deaths/1M pop',mode = 'markers',
                         hovertext = df2['text'],
                         marker=dict(line_width = 2, size=df2['Deaths/1M pop']*0.1, color = df2['Deaths/1M pop'])))
fig.update_layout(title = "Countries with most deaths per million population " )
fig.update_layout( width=700,height=600)
fig.show()


# In[ ]:


df3 = df.sort_values('Tests/1M pop', ascending = False).iloc[:25,:]
df3['text'] = 'Population:' + (df3['Population']).astype(str)+ '<br>'+ 'Tests:' + (df3['TotalTests']).astype(str)
fig = go.Figure()

fig.add_trace(go.Scatter(x=df3['Country/Region'], y=df3['Tests/1M pop'], name='Tests/1M pop',mode = 'markers',
                         hovertext = df3['text'],
                         marker=dict(line_width = 2, size=df3['Tests/1M pop']*0.0005, color = df3['Tests/1M pop'])))
fig.update_layout(title = "Countries with most Tests per million population " )
fig.update_layout( width=700,height=600)
fig.show()


# In[ ]:





# **WORK IN PROGRESS. STAY TUNED FOR MORE AND DO COMMENT YOUR SUGGESTIONS :)**
