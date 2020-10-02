#!/usr/bin/env python
# coding: utf-8

# # Economic Freedom World Wide 2016
# 
# Economic Freedom of the World measures the degree to which the policies and institutions of countries are supportive of economic freedom. The cornerstones of economic freedom are personal choice, voluntary exchange, freedom to enter markets and compete, and security of the person and privately owned property. Reference [Fraser Institude Economic Freedom Report]  (https://www.fraserinstitute.org/economic-freedom)

# # Objective:
# 
# The objective of this notebook is to analyse the Economic Freedom of the World data by the Fraser Institute and take a closer look at different countries in  2016. I will also be analysing as to what might be the major measure for the economic freedom and how countries can focus on some areas as well what major downturn of events has resulted in lower economic index for some countries.
# 
# Here are the steps I'm going to perform:  
# 1) Data Preparation & Analysis: This is in general looking at the data to figure out whats going on. Inspect the data: Check whether there is any missing data, irrelevant data and do a cleanup.  
# 2) Data Visualization  
# 3) Document the insights  

# # 1) Data Preparation

# In[ ]:


#Import all required libraries for reading data, analysing and visualizing data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
import os
print(os.listdir("../input"))


# In[ ]:


#Read the data
efw = pd.read_csv('../input/efw_cc.csv')
efw.shape


# In[ ]:


efw.head(3)


# In[ ]:


efw.info()


# In[ ]:


efw.year.value_counts().sort_index().index


# There is data for 162 countries ranging from 1970 till 2016. 1970 to 2000 has data once in for 5 years and there is data for every year from 2000.

# In[ ]:


efw.ISO_code.unique()


# The above are the list of 162 countries.

# As of now, my focus is only on the data for year 2016. Lets move on to creating a new df for year 2016 and work on the data.

# # 2) Data Processing

# In[ ]:


efw2016 = efw[efw.year == 2016]
efw2016.shape


# As per [Fraser Institude Economic Freedom approach ](https://www.fraserinstitute.org/economic-freedom/approach), the economic freedom  index measures the degree of economic freedom present in five major areas that is made up of several subcomponents. The average of all sub-components result in the following five major areas score(s): 
# 1. Size of Government
# 1. Legal System and Security of Property Rights
# 1. Sound Money
# 1. Freedom to Trade Internationally
# 1. Regulation.
# 

# * Hence, lets replace the subcomponents null information with zeroes as the values of the major areas are the average of these and I validated random data to make sure the average is derived based on zeroes.

# In[ ]:


efw2016 = efw2016.fillna(0)
efw2016.isnull().sum()[efw2016.isnull().sum()>0]


# In[ ]:


efw2016.head()


# In[ ]:


efw2016_x = efw2016[['ISO_code', 'countries', 'ECONOMIC FREEDOM', 'rank', 'quartile','1_size_government', 
                     '2_property_rights', '3_sound_money', '4_trade', '5_regulation']]


# # 3) Top countries- Overall Economic Freedom

# In[ ]:


top10_efw = efw2016_x.sort_values('ECONOMIC FREEDOM', ascending=False).head(11)
top10_efw.head()


# In[ ]:


import plotly.graph_objs as go
trace1 = go.Bar(
    x=top10_efw['countries'],
    y=top10_efw['1_size_government'],
    name='Size of Govt'
)
trace2 = go.Bar(
    x=top10_efw['countries'],
    y=top10_efw['2_property_rights'],
    name='Property Rights'
)
trace3 = go.Bar(
    x=top10_efw['countries'],
    y=top10_efw['3_sound_money'],
    name='Sound Money'
)
trace4 = go.Bar(
    x=top10_efw['countries'],
    y=top10_efw['4_trade'],
    name='Freedom to Trade'
)
trace5 = go.Bar(
    x=top10_efw['countries'],
    y=top10_efw['5_regulation'],
    name='Regulation'
)

data = [trace1, trace2, trace3, trace4, trace5]
layout = go.Layout(
    title='Top 10 countries & Economic indicators',
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='Top 10 countries and different indicators')


# In[ ]:


import plotly.graph_objs as go
trace0 = go.Bar(
    x=top10_efw['ECONOMIC FREEDOM'],
    y=top10_efw['ISO_code'],
    marker=dict(
        color='rgba(66, 244, 146, 0.6)',
        line=dict(
            color='rgba(66, 244, 146, 1.0)',
            width=1),
    ),
    name='ECONOMIC FREEDOM',
    orientation='h',    
)
trace1 = go.Scatter(
    x=top10_efw['1_size_government'],
    y=top10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(214, 53, 25)'),
    name='Government',
)
trace2 = go.Scatter(
    x=top10_efw['2_property_rights'],
    y=top10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(192, 26, 221)'),
    name='Property Rights',
)
trace3 = go.Scatter(
    x=top10_efw['3_sound_money'],
    y=top10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(114, 78, 22)'),
    name='Sound Money',
)
trace4 = go.Scatter(
    x=top10_efw['4_trade'],
    y=top10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(14, 94, 160)'),
    name='Trade',
)
trace5 = go.Scatter(
    x=top10_efw['5_regulation'],
    y=top10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(237, 135, 147)'),
    name='Regulation',
)
layout = dict(
    title='World Economic Freedom 2016 - Top10 countries',
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
        domain=[0, 0.32],
    ),
    yaxis2=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0, 0.32],
    ),
    yaxis3=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0.34, .66],
    ),
    yaxis4=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0.34, .66],
    ),    
    yaxis5=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0.68, 1],
    ),
    yaxis6=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0.68, 1],
    ),        
    xaxis=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0, 0.45],
    ),
    xaxis2=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0.55, 1],
        side='top',
        dtick=25000,
    ),
    xaxis3=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0, 0.45],
    ),
    xaxis4=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0.55, 1],
        side='top',
        dtick=25000,
    ),  
    xaxis5=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0, 0.40],
        side='top',
        dtick=25000,
    ),
    xaxis6=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0.55, 1],
        side='top',
        dtick=25000,
    ),    

    legend=dict(
        x=0.029,
        y=1.038,
        font=dict(
            size=10,
        ),
    ),
    margin=dict(
        l=100,
        r=20,
        t=70,
        b=70,
    ),    
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)


# Creating two subplots
fig = tools.make_subplots(rows=3, cols=2, specs=[[{}, {}],[{}, {}],[{}, {}]], shared_xaxes=False, shared_yaxes=False,
                         subplot_titles=('Economic Freedom', 'Govt spending, decision making','Legal System & Property Rights',
                                         'Sound Money', 'Freedom to Trade', 'Regulation'), vertical_spacing=0.1)

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 2, 2)
fig.append_trace(trace4, 3, 1)
fig.append_trace(trace5, 3, 2)


fig['layout'].update(height=1000, width=1000,  title='World Economic Freedom 2016 - Top10 countries')
iplot(fig, filename='ECONOMIC FREEDOM Vs Govt')


# ## Observations from the Economic freedom in 5 areas:
# - Area 1 - Size of Government: HongKong, Singapore, Georgia, Mauritius has more govt spending. This includes govt spending, govt-controlled Enterprises, govt decision making etc. The more govt is involved in decision, the less economic freedom is available for the nation.
# - Area 2 - Legal System and Property Rights: Georgia and Mauritius has very low score in legal system. The lower the score, it  indicates that the property rights is not great in these 2 top10 nations. Newzealand has the highest score.
# - Area 3 - Sound Money: Switzerland, USA and United Kingdom tops this aspect. This indicates that there is stability in the economy of the nation (includes money growth, inflation volatility)
# - Area 4 - Freedom to Trade Internationally: HongKong and Singapore scores the best. While Australia, USA and Switzerland scores the least. This measures the freedom to buy, sell and make contracts internationally.
# - Area 5 - Regulation: HongKong and Newzealand tops this measure. Mauritius has the lowest score in top10. This indicator measures the ability for the government to limit the international exchanges and the free operation of any businesses. 

# ## Something to think about... 
# Size of government indicate that the more goverment is involved in any decisions related to enterprises/businesses, economic freedom is reduced. Though Hongkong and Singapore is more controlled by the government compared to other countries, their econmoic freedom is still more with more freedom to trade, regulation...

# # 4) Lowest rated countries and their economic freedom

# In[ ]:


low10_efw = efw2016_x.sort_values('ECONOMIC FREEDOM', ascending=False).tail(10)
low10_efw.head()


# In[ ]:


import plotly.graph_objs as go

trace1 = go.Bar(
    x=low10_efw['countries'],
    y=low10_efw['1_size_government'],
    name='Size of Govt'
)
trace2 = go.Bar(
    x=low10_efw['countries'],
    y=low10_efw['2_property_rights'],
    name='Property Rights'
)
trace3 = go.Bar(
    x=low10_efw['countries'],
    y=low10_efw['3_sound_money'],
    name='Sound Money'
)
trace4 = go.Bar(
    x=low10_efw['countries'],
    y=low10_efw['4_trade'],
    name='Freedom to Trade'
)
trace5 = go.Bar(
    x=low10_efw['countries'],
    y=low10_efw['5_regulation'],
    name='Regulation'
)

data = [trace1, trace2, trace3, trace4, trace5]
layout = go.Layout(
    title='Lowest 10 countries & Economic indicators',    
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='Lowest rated countries and different indicators')


# In[ ]:


import plotly.graph_objs as go

trace0 = go.Bar(
    x=low10_efw['ECONOMIC FREEDOM'],
    y=low10_efw['ISO_code'],
    marker=dict(
        color='rgba(66, 244, 146, 0.6)',
        line=dict(
            color='rgba(66, 244, 146, 1.0)',
            width=1),
    ),
    name='ECONOMIC FREEDOM',
    orientation='h',    
)
trace1 = go.Scatter(
    x=low10_efw['1_size_government'],
    y=low10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(214, 53, 25)'),
    name='Government',
)
trace2 = go.Scatter(
    x=low10_efw['2_property_rights'],
    y=low10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(192, 26, 221)'),
    name='Property Rights',
)
trace3 = go.Scatter(
    x=low10_efw['3_sound_money'],
    y=low10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(114, 78, 22)'),
    name='Sound Money',
)
trace4 = go.Scatter(
    x=low10_efw['4_trade'],
    y=low10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(14, 94, 160)'),
    name='Trade',
)
trace5 = go.Scatter(
    x=low10_efw['5_regulation'],
    y=low10_efw['ISO_code'],
    mode='lines+markers',
    line=dict(
        color='rgb(237, 135, 147)'),
    name='Regulation',
)
layout = dict(
    title='World Economic Freedom 2016 - Lowest rated countries',
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
        domain=[0, 0.32],
    ),
    yaxis2=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0, 0.32],
    ),
    yaxis3=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0.34, .66],
    ),
    yaxis4=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0.34, .66],
    ),    
    yaxis5=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0.68, 1],
    ),
    yaxis6=dict(
        showgrid=False,
        showline=True,
        showticklabels=True,
        linecolor='rgb(255, 252, 252)',
        linewidth=2,
        domain=[0.68, 1],
    ),        
    xaxis=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0, 0.45],
    ),
    xaxis2=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0.55, 1],
        side='top',
        dtick=25000,
    ),
    xaxis3=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0, 0.45],
    ),
    xaxis4=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0.55, 1],
        side='top',
        dtick=25000,
    ),  
    xaxis5=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0, 0.40],
        side='top',
        dtick=25000,
    ),
    xaxis6=dict(
        zeroline=False,
        showline=False,
        showticklabels=False,
        showgrid=True,
        domain=[0.55, 1],
        side='top',
        dtick=25000,
    ),    

    legend=dict(
        x=0.029,
        y=1.038,
        font=dict(
            size=10,
        ),
    ),
    margin=dict(
        l=100,
        r=20,
        t=70,
        b=70,
    ),    
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)


# Creating two subplots
fig = tools.make_subplots(rows=3, cols=2, specs=[[{}, {}],[{}, {}],[{}, {}]], shared_xaxes=False, shared_yaxes=False,
                         subplot_titles=('Economic Freedom', 'Govt spending, decision making','Legal System & Property Rights',
                                         'Sound Money', 'Freedom to Trade', 'Regulation'), vertical_spacing=0.1)

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 2, 2)
fig.append_trace(trace4, 3, 1)
fig.append_trace(trace5, 3, 2)


fig['layout'].update(height=1000, width=1000,  title='World Economic Freedom 2016 - Lowest rated countries')
iplot(fig, filename='ECONOMIC FREEDOM Vs Govt')


# ## Lowest rated of all.
# Venezuela seems to be the country with the lowest economic freedom rate. The details regarding what has happened in this nation can be found in Wikipedia https://en.wikipedia.org/wiki/Venezuela#Economy

# # 5) Lets talk about the whole worlds economic freedom...

# In[ ]:


import plotly.graph_objs as go
data = [ dict(
        type = 'choropleth',
        locations = efw2016['ISO_code'],
        z = efw2016['quartile'],
        text = efw2016['countries'],
        colorscale = 'Rainbow',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'World Economic Index'),
      ) ]

layout = dict(
    title = '2016 World Economic Freedom Index',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='Free Economies' )


# #### Its pretty imminent that majority of South America and Africa, some of middle east is in the lowest ranking of the economic freedom. From the map, the following countries are economically bad due to the recent events that are still happening
# - Myanmar - Rohingya genocide
# - Venezuela - Economic war
# - Ukraine - Russian & Ukraine issues
# - Turkey, Iraq, Iran

# # 6) Government Controlled

# In[ ]:


import plotly.graph_objs as go
data = [ dict(
        type = 'choropleth',
        locations = efw2016['ISO_code'],
        z = efw2016['1_size_government'],
        text = efw2016['countries'],
        colorscale = 'Rainbow',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'World Economic Index'),
      ) ]

layout = dict(
    title = '2016 World Economic Freedom - Government Controlled Index',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='Free Economies' )


# #### Guatemala and Sudan tops the list.
# Sudan is in the lowest rated countries (4th quartile), while Guatemala is in the 1st quartile. Guatemala has improved scores in sound money, trade & regulation aspect of the economy.

# # 7) Property Rights

# In[ ]:


import plotly.graph_objs as go
data = [ dict(
        type = 'choropleth',
        locations = efw2016['ISO_code'],
        z = efw2016['2_property_rights'],
        text = efw2016['countries'],
        colorscale = 'Hot',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Money'),
      ) ]

layout = dict(
    title = '2016 World Economic Freedom - Property Rights & Legal',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='Free Economies' )


# #### Central African Republic & Venezuela are lowest in the list. Most of South America & Africa have low scores 

# # 7) Sound Money

# In[ ]:


import plotly.graph_objs as go
data = [ dict(
        type = 'choropleth',
        locations = efw2016['ISO_code'],
        z = efw2016['3_sound_money'],
        text = efw2016['countries'],
        colorscale = 'Hot',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Money'),
      ) ]

layout = dict(
    title = '2016 World Economic Freedom - Sound Money',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='Free Economies' )


# #### Venezuela is the lowest in the list.

# # 8) Trade

# In[ ]:


import plotly.graph_objs as go
data = [ dict(
        type = 'choropleth',
        locations = efw2016['ISO_code'],
        z = efw2016['4_trade'],
        text = efw2016['countries'],
        colorscale = 'Hot',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'International Trade'),
      ) ]

layout = dict(
    title = '2016 World Economy - Freedom to trade internationally',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='Free Economies' )


# #### Sudan & Venezuela are the lowest in the list.

# # 9) Regulation

# In[ ]:


import plotly.graph_objs as go
data = [ dict(
        type = 'choropleth',
        locations = efw2016['ISO_code'],
        z = efw2016['5_regulation'],
        text = efw2016['countries'],
        colorscale = 'Hot',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'REgulations'),
      ) ]

layout = dict(
    title = '2016 World Economy - Regulations',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='Free Economies' )


# #### Venezuela is the lowest in the list.

# # To be continued....
# - Analysis of few specific countries in Africa, Asia, North America, Middle East
# - Broad areas of the economic freedom and their different measures.
# - Analysis of data from 2000-2016. 
# - Gender disparity index
# - Analysis of countries impacted with major events like genocide/war etc... 

# In[ ]:




