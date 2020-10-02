#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#plotly
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('/kaggle/input/coronavirus-covid19-dataset/corona_latest.csv')


# In[ ]:


df


# In[ ]:


df.drop("Unnamed: 0",axis=1,inplace=True)
df


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=df["Country,Other"][df.TotalCases>=5000],
    y=df.TotalCases[df.TotalCases>=5000],
    name='TotalCases',
    marker_color='#636EFA',
    text=df.TotalCases[df.TotalCases>=5000],
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=df["Country,Other"][df.TotalCases>=5000],
    y=df["TotalRecovered"][df.TotalCases>=5000],
    name='TotalRecovered',
    marker_color='#2ca02c',
    text=df.TotalRecovered[df.TotalCases>=5000],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=df["Country,Other"][df.TotalCases>=5000],
    y=df.TotalDeaths[df.TotalCases>=5000],
    name='TotalDeaths',
    marker_color='#EF553B',
    text=df.TotalDeaths[df.TotalCases>=5000],
    textposition='auto'
))
fig.update_layout(title_text='Corona virus TotalCase greater than 5000',xaxis_tickfont_size=14,
    yaxis=dict(
        title='COUNT',
        titlefont_size=16,
        tickfont_size=14))
fig.show()



# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=df["Country,Other"][df.TotalCases>1000],
    y=df.TotalCases[df.TotalCases>1000],
    name='TotalCases',
    marker_color='#636EFA',
    text=df.TotalCases[df.TotalCases>1000],
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=df["Country,Other"][df.TotalCases>1000],
    y=df["TotalRecovered"][df.TotalCases>1000],
    name='TotalRecovered',
    marker_color='#2ca02c',
    text=df.TotalRecovered[df.TotalCases>1000],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=df["Country,Other"][df.TotalCases>1000],
    y=df.TotalDeaths[df.TotalCases>1000],
    name='TotalDeaths',
    marker_color='#EF553B',
    text=df.TotalDeaths[df.TotalCases>1000],
    textposition='auto'
))
fig.update_layout(title_text='Corona virus TotalCase greater than 1000',xaxis_tickfont_size=14,
    yaxis=dict(
        title='COUNT',
        titlefont_size=16,
        tickfont_size=14))
fig.show()



# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<5000 )&(df.TotalCases>1000))],
    y=df.TotalCases[((df.TotalCases<5000 )&(df.TotalCases>1000))],
    name='TotalCases',
    marker_color='#636EFA',
    text=df.TotalCases[((df.TotalCases<5000 )&(df.TotalCases>1000))],
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<5000 )&(df.TotalCases>1000))],
    y=df["TotalRecovered"][((df.TotalCases<5000 )&(df.TotalCases>1000))],
    name='TotalRecovered',
    marker_color='#2ca02c',
    text=df.TotalRecovered[((df.TotalCases<5000 )&(df.TotalCases>1000))],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<5000 )&(df.TotalCases>1000))],
    y=df.TotalDeaths[((df.TotalCases<5000 )&(df.TotalCases>1000))],
    name='TotalDeaths',
    marker_color='#EF553B',
    text=df.TotalDeaths[((df.TotalCases<5000 )&(df.TotalCases>1000))],
    textposition='auto'
))
fig.update_layout(title_text='Corona virus TotalCase greater than 1000 and less than 5000',xaxis_tickfont_size=14,
    yaxis=dict(
        title='COUNT',
        titlefont_size=16,
        tickfont_size=14))
fig.show()



# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<1000 )&(df.TotalCases>500))],
    y=df.TotalCases[((df.TotalCases<1000 )&(df.TotalCases>500))],
    name='TotalCases',
    marker_color='#636EFA',
    text=df.TotalCases[((df.TotalCases<1000 )&(df.TotalCases>500))],
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<1000 )&(df.TotalCases>500))],
    y=df["TotalRecovered"][((df.TotalCases<1000 )&(df.TotalCases>500))],
    name='TotalRecovered',
    marker_color='#2ca02c',
    text=df.TotalRecovered[((df.TotalCases<1000 )&(df.TotalCases>500))],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<1000 )&(df.TotalCases>500))],
    y=df.TotalDeaths[((df.TotalCases<1000 )&(df.TotalCases>500))],
    name='TotalDeaths',
    marker_color='#EF553B',
    text=df.TotalDeaths[((df.TotalCases<1000 )&(df.TotalCases>500))],
    textposition='auto'
))
fig.update_layout(title_text='Corona virus TotalCase greater than 500 and less than 1000',xaxis_tickfont_size=14,
    yaxis=dict(
        title='COUNT',
        titlefont_size=16,
        tickfont_size=14))
fig.show()



# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<500 )&(df.TotalCases>250))],
    y=df.TotalCases[((df.TotalCases<500 )&(df.TotalCases>250))],
    name='TotalCases',
    marker_color='#636EFA',
    text=df.TotalCases[((df.TotalCases<500 )&(df.TotalCases>250))],
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<500 )&(df.TotalCases>250))],
    y=df["TotalRecovered"][((df.TotalCases<500 )&(df.TotalCases>250))],
    name='TotalRecovered',
    marker_color='#2ca02c',
    text=df.TotalRecovered[((df.TotalCases<500 )&(df.TotalCases>250))],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<500 )&(df.TotalCases>250))],
    y=df.TotalDeaths[((df.TotalCases<500 )&(df.TotalCases>250))],
    name='TotalDeaths',
    marker_color='#EF553B',
    text=df.TotalDeaths[((df.TotalCases<500 )&(df.TotalCases>250))],
    textposition='auto'
))
fig.update_layout(title_text='Corona virus TotalCase greater than 250 and less than 500',xaxis_tickfont_size=14,
    yaxis=dict(
        title='COUNT',
        titlefont_size=16,
        tickfont_size=14))
fig.show()



# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<250 )&(df.TotalCases>100))],
    y=df.TotalCases[((df.TotalCases<250 )&(df.TotalCases>100))],
    name='TotalCases',
    marker_color='#636EFA',
    text=df.TotalCases[((df.TotalCases<250 )&(df.TotalCases>100))],
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<250 )&(df.TotalCases>100))],
    y=df["TotalRecovered"][((df.TotalCases<250 )&(df.TotalCases>100))],
    name='TotalRecovered',
    marker_color='#2ca02c',
    text=df.TotalRecovered[((df.TotalCases<250 )&(df.TotalCases>100))],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<250 )&(df.TotalCases>100))],
    y=df.TotalDeaths[((df.TotalCases<250 )&(df.TotalCases>100))],
    name='TotalDeaths',
    marker_color='#EF553B',
    text=df.TotalDeaths[((df.TotalCases<250 )&(df.TotalCases>100))],
    textposition='auto'
))
fig.update_layout(title_text='Corona virus TotalCase greater than 100 and less than 250',xaxis_tickfont_size=14,
    yaxis=dict(
        title='COUNT',
        titlefont_size=16,
        tickfont_size=14))
fig.show()



# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<100 )&(df.TotalCases>50))],
    y=df.TotalCases[((df.TotalCases<100 )&(df.TotalCases>50))],
    name='TotalCases',
    marker_color='#636EFA',
    text=df.TotalCases[((df.TotalCases<100 )&(df.TotalCases>50))],
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<100 )&(df.TotalCases>50))],
    y=df["TotalRecovered"][((df.TotalCases<100 )&(df.TotalCases>50))],
    name='TotalRecovered',
    marker_color='#2ca02c',
    text=df.TotalRecovered[((df.TotalCases<100 )&(df.TotalCases>50))],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=df["Country,Other"][((df.TotalCases<100 )&(df.TotalCases>50))],
    y=df.TotalDeaths[((df.TotalCases<100 )&(df.TotalCases>50))],
    name='TotalDeaths',
    marker_color='#EF553B',
    text=df.TotalDeaths[((df.TotalCases<100 )&(df.TotalCases>50))],
    textposition='auto'
))
fig.update_layout(title_text='Corona virus TotalCase greater than 50 and less than 100',xaxis_tickfont_size=14,
    yaxis=dict(
        title='COUNT',
        titlefont_size=16,
        tickfont_size=14))
fig.show()



# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=df["Country,Other"][df.TotalCases<50],
    y=df.TotalCases[df.TotalCases<50],
    name='TotalCases',
    marker_color='#636EFA',
    text=df.TotalCases[df.TotalCases<50],
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=df["Country,Other"][df.TotalCases<50],
    y=df["TotalRecovered"][df.TotalCases<50],
    name='TotalRecovered',
    marker_color='#2ca02c',
    text=df.TotalRecovered[df.TotalCases<50],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=df["Country,Other"][df.TotalCases<50],
    y=df.TotalDeaths[df.TotalCases<50],
    name='TotalDeaths',
    marker_color='#EF553B',
    text=df.TotalDeaths[df.TotalCases<50],
    textposition='auto'
))
fig.update_layout(title_text='Corona virus TotalCase less than 50',xaxis_tickfont_size=14,
    yaxis=dict(
        title='COUNT',
        titlefont_size=16,
        tickfont_size=14))
fig.show()



# In[ ]:


fig = go.Figure()
fig.add_trace(go.Choropleth(
        locationmode = 'country names',
        locations = df["Country,Other"],
        z = df.TotalCases,
        text = df["Country,Other"],
        #colorscale = [[0,'rgb(0, 0, 0)'],[1,'rgb(0, 0, 0)']],
        colorscale='Rainbow',
        autocolorscale = False,
        showscale = True,
        geo = 'geo'
    ))


# In[ ]:


df1=df.copy()


# In[ ]:


df1.loc['total'] = df1.select_dtypes(np.number).sum()
df1


# In[ ]:


df1.fillna("world",inplace=True)
df1


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=df["Country,Other"][:10],
    y=df["TotalCases"][:10],
    name='TotalCases: '+str(df1.loc["total"]["TotalCases"]),
    marker_color='#636EFA',
    text=df["TotalCases"][:10],
    textposition='auto'
))


fig.add_trace(go.Bar(
    x=df["Country,Other"][:10],
    y=df["ActiveCases"][:10],
    name='ActiveCases: '+str(df1.loc["total"]["ActiveCases"]),
    marker_color='LightSkyBlue',
    text=df["ActiveCases"][:10],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=df["Country,Other"][:10],
    y=df["TotalDeaths"][:10],
    name='TotalDeaths: '+str(df1.loc["total"]["TotalDeaths"]),
    marker_color='#EF553B',
    text=df["TotalDeaths"][:10],
    textposition='auto'
))
fig.update_layout(title_text='Top 10 affected Countries',xaxis_tickfont_size=14,
    yaxis=dict(
        title='COUNT',
        titlefont_size=16,
        tickfont_size=14))
fig.show()



# In[ ]:


df2=df[df.TotalCases>5000].copy()


# In[ ]:


df2.sort_values("Tests/ 1M pop", axis = 0, ascending = False, 
                 inplace = True, na_position ='last')
df2


# In[ ]:


fig = go.Figure()
data=[go.Bar(
    x=df2["Country,Other"][:10],
    y=df2["Tests/ 1M pop"][:10],
    name='Tests/ 1M pop',
    text=df2["Tests/ 1M pop"][:10],
    textposition='auto',
    marker={
        'color': df2["Tests/ 1M pop"],
        'colorscale': 'balance'
    }
)]



fig = go.FigureWidget(data=data)

fig.update_layout(title_text=("Countries with most no of tests done per 1 Million population:"),xaxis_tickfont_size=14,
    yaxis=dict(
        title='COUNT',
        titlefont_size=16,
        tickfont_size=14))
fig.show()



# In[ ]:


fig = go.Figure()
data=[go.Bar(
    x=df2["Country,Other"][df2["Country,Other"]!="China"].tail(10),
    y=df2["Tests/ 1M pop"][df2["Country,Other"]!="China"].tail(10),
    name='Tests/ 1M pop',
    text=df2["Tests/ 1M pop"][df2["Country,Other"]!="China"].tail(10),
    textposition='auto',
    marker={
        'color':df2["Tests/ 1M pop"][df2["Country,Other"]!="China"].tail(10),
        'colorscale': 'balance'
    }
)]



fig = go.FigureWidget(data=data)

fig.update_layout(title_text=("Countries with least no of tests done per 1 Million population:"),xaxis_tickfont_size=14,
    yaxis=dict(
        title='COUNT',
        titlefont_size=16,
        tickfont_size=14))
fig.show()



# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1585206810614' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;co&#47;corona_global&#47;corona-Global&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='corona_global&#47;corona-Global' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;co&#47;corona_global&#47;corona-Global&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1585206810614');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='977px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:




