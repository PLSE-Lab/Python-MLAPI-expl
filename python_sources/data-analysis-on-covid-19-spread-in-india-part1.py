#!/usr/bin/env python
# coding: utf-8

# # Data Analysis on COVID-19 Spread in India

# **Coronaviruses (CoV)** are a large family of viruses that cause illness ranging from the common cold to more severe diseases such as Middle East Respiratory Syndrome (MERS-CoV) and Severe Acute Respiratory Syndrome (SARS-CoV). A novel coronavirus (nCoV) is a new strain that has not been previously identified in humans.
# 
# **Coronaviruses** are zoonotic, meaning they are transmitted between animals and people. Detailed investigations found that SARS-CoV was transmitted from civet cats to humans and MERS-CoV from dromedary camels to humans. Several known coronaviruses are circulating in animals that have not yet infected humans.
# 
# 
# ![](https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60)
# 
# 
# **Common signs of infection** include respiratory symptoms, fever, cough, shortness of breath and breathing difficulties. In more severe cases, infection can cause pneumonia, severe acute respiratory syndrome, kidney failure and even death.
# 
# **Standard recommendations** to prevent infection spread include regular hand washing, covering mouth and nose when coughing and sneezing, thoroughly cooking meat and eggs. Avoid close contact with anyone showing symptoms of respiratory illness such as coughing and sneezing.

# ## <font color = 'tomato'>Do Upvote the Notebook, If You Like the Work.<font/>

# <div class="list-group" id="list-tab" role="tablist">
#   <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Notebook Content</h3>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#Spread-of-COVID-19-Around-World " role="tab" aria-controls="profile">Spread of COVID-19 aroud world<span class="badge badge-primary badge-pill">1</span></a>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#Census-of-India-2011" role="tab" aria-controls="messages">Census of India 2011<span class="badge badge-primary badge-pill">2</span></a>
#   <a class="list-group-item list-group-item-action"  data-toggle="list" href="#Confirmed-Cases-in-India" role="tab" aria-controls="settings">Confirmed Cases in India<span class="badge badge-primary badge-pill">3</span></a>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#Confirmed-Cases-w.r.t-State-or-UnionTerritory " role="tab" aria-controls="settings">Confirmed Cases w.r.t State or UnionTerritory <span class="badge badge-primary badge-pill">4</span></a> 
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#Confirmed-Cases-w.r.t-Date-in-States-of-India " role="tab" aria-controls="settings">Confirmed Cases w.r.t Date in States of India <span class="badge badge-primary badge-pill">5</span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#Cured-Cases-w.r.t-Date-in-States-of-India" role="tab" aria-controls="settings">Cured Cases w.r.t Date in States of India<span class="badge badge-primary badge-pill">6</span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#Count-of-Cured-Cases-w.r.t-State-or-UnioinTerritory" role="tab" aria-controls="settings">Count of Cured Cases w.r.t State or UnioinTerritory <span class="badge badge-primary badge-pill">7</span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#Deaths-Occured-w.r.t-Date-in-State-or-UnioinTerritory" role="tab" aria-controls="settings">Deaths Occured w.r.t Date in State or UnioinTerritory<span class="badge badge-primary badge-pill">8</span></a>  
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#Count-of-Deaths-Occured-w.r.t-State-or-UnioinTerritory" role="tab" aria-controls="settings">Count of Deaths Occured w.r.t Date in State or UnioinTerritory <span class="badge badge-primary badge-pill">9</span></a>
#      <a class="list-group-item list-group-item-action" data-toggle="list" href="#Mortality-Rate-w.r.t-Data-in-State-or-UnionTerritory" role="tab" aria-controls="settings">Mortality Rate w.r.t Data in State or UnionTerritory<span class="badge badge-primary badge-pill">10</span></a>
#      <a class="list-group-item list-group-item-action" data-toggle="list" href="#Mortality-Rate-w.r.t-State" role="tab" aria-controls="settings">Mortality Rate w.r.t State<span class="badge badge-primary badge-pill">11</span></a>
#      <a class="list-group-item list-group-item-action" data-toggle="list" href="#Correlation-Between-Confirmed-Cases,-Cured-and-Deaths" role="tab" aria-controls="settings">Correlation Between Confirmed Cases, Cured and Deaths<span class="badge badge-primary badge-pill">12</span></a>
#      <a class="list-group-item list-group-item-action" data-toggle="list" href="#20-Reasons-for-the-spread-of-COVID-19 " role="tab" aria-controls="settings">20 Reasons for the spread of COVID 19 <span class="badge badge-primary badge-pill">13</span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#Status-of-COVID-19-Patients" role="tab" aria-controls="settings">Status of COVID 19 Patients<span class="badge badge-primary badge-pill">14</span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#Gender-Affected-Most-By-COVID-19" role="tab" aria-controls="settings">Gender Affected Most By COVID-19<span class="badge badge-primary badge-pill">15</span></a>
#     
# 

# In[ ]:


# Standard plotly imports
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import iplot, init_notebook_mode
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
import pandas as pd
import numpy as np

path = '../input/covid19-in-india/'


# ## Spread of COVID 19 Around World 
# 

# In[ ]:


w = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')
x = w.groupby('Country/Region')['Confirmed'].max()

w_x = pd.DataFrame(data={'Country':x.index, 'Confirmed':x.values})
# w_x.head()
data = dict(type='choropleth',
            locations = w_x.Country.values.tolist(),
            locationmode = 'country names',
            colorscale = 'sunsetdark',
            text = w_x.Country.values.tolist(),
            z = w_x.Confirmed.values.tolist(),
            colorbar = {'title':"Count"}
            )
layout = dict(title = 'COVID-19 Positive Cases around the world',
              geo = dict(scope='world')
             )
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)


# ## Census of India 2011 
# <a id = 'census'/>

# In[ ]:


df3 = pd.read_csv(path+'population_india_census2011.csv')
(df3.style
 .hide_index()
 .bar(color='#00FFFF', vmin=df3.Population.min(), subset=['Population'])
 .bar(color='#FF6F61', vmin=df3['Rural population'].min(), subset=['Rural population'])
 .bar(color='mediumspringgreen', vmin=df3['Urban population'].min(), subset=['Urban population'])
 .bar(color='orange', vmin=df3['Gender Ratio'].min(), subset=['Gender Ratio'])
 .set_caption(''))


# ## Confirmed Cases in India
# <a id = 'cindia'/>

# In[ ]:


df = pd.read_csv(path+'covid_19_india.csv')
df['country'] = 'India'
df.head()


# In[ ]:


data = dict(type='choropleth',
            locations = ['india'],
            locationmode = 'country names',
            colorscale = 'Reds',
            text = ['Total Cases'],
            z = [df.groupby('State/UnionTerritory')['Confirmed'].max().sum()],
#             colorbar = {'title':"Stores Count"}
            )
layout = dict(title = 'Total COVID-19 Positive Cases in India',
              geo = dict(scope='asia')
             )
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)


# ## Confirmed Cases w.r.t State or UnionTerritory 
# <a id = 'cst'/>

# In[ ]:


df.groupby('State/UnionTerritory')['Confirmed'].sum().iplot(kind = 'bar', xTitle = 'State/UnionTerritory', 
                                                              yTitle = 'Confirmed Cases', color ='tomato')


# In[ ]:


fig = px.bar(df, x="State/UnionTerritory", y="Confirmed", color='State/UnionTerritory', 
             color_continuous_scale=px.colors.sequential.Plasma, )
fig.show()


# # Confirmed Cases w.r.t Date in States of India 
# <a id ='stin'>

# ### Line Plot

# In[ ]:


fig = px.line(df, x="Date", y="Confirmed", color='State/UnionTerritory')
fig.show()


# ### Bar Graph

# In[ ]:


fig = px.bar(df, x="Date", y="Confirmed", color='State/UnionTerritory')
fig.show()


# ### Bubble Chart

# In[ ]:


fig = px.scatter(df, x="Date", y="Confirmed", color="State/UnionTerritory",
                 size='Confirmed', color_continuous_scale=px.colors.sequential.Plasma,)
fig.show()


# ## Cured Cases w.r.t Date in States of India 
# <a id='dstin'/>

# In[ ]:


fig = px.scatter(df, x='Date', y='Cured', color='State/UnionTerritory', title='Cases Cured').update_traces(mode='lines+markers')
fig.show()


# ## Count of Cured Cases w.r.t State or UnioinTerritory
# <a id ='cin'>

# In[ ]:


df.groupby('State/UnionTerritory')['Cured'].sum().iplot(kind='bar', color='green', title='Cured Cases w.r.t Sate/UnionTerritory')


# ## Deaths Occured w.r.t Date in State or UnioinTerritory 
# <a id='din'/>

# In[ ]:


fig = px.scatter(df, x='Date', y='Deaths', color='State/UnionTerritory', title='Deaths occured').update_traces(mode='lines+markers')
fig.show()


# ## Count of Deaths Occured w.r.t State or UnioinTerritory 
# <a id='cdin'>

# In[ ]:


df.groupby('State/UnionTerritory')['Deaths'].sum().iplot(kind='bar', color='red', title='Deaths Occured w.r.t Sate/UnionTerritory')


# In[ ]:


fig = px.scatter_matrix(df,
    dimensions=["Confirmed", "Cured"],
    color="State/UnionTerritory", title='Plot between Confirmed and Cured')
fig.show()
fig = px.scatter_matrix(df,
    dimensions=["Confirmed", "Deaths"],
    color="State/UnionTerritory", title='Plot between Confirmed and Deaths')
fig.show()


# ## Mortality Rate w.r.t Data in State or UnionTerritory
# <a id = 'mort'/>

# In[ ]:


def div(x,y): 
    if y==0:
        return 0
    return x/y
df['mortality_rate'] = df.apply(lambda row: div(row['Deaths'], row['Confirmed']) , axis = 1)
fig = px.line(df, x="Date", y="mortality_rate", color='State/UnionTerritory')
fig.show()


# ## Mortality Rate w.r.t State

# In[ ]:


df.groupby('State/UnionTerritory')['mortality_rate'].sum().iplot(kind='bar', color='darkblue')


# ## Correlation Between Confirmed Cases, Cured and Deaths 
# <a id = 'corr'/>

# In[ ]:


corr = df[['Confirmed', 'Cured', 'Deaths']].corr()
ff.create_annotated_heatmap(
    z=corr.values,
    x=list(corr.columns),
    y=list(corr.index),
    annotation_text=corr.round(2).values,
    showscale=True, colorscale='emrld')


# ## Status of COVID 19 Patients 
# <a id = 'sts'/>

# In[ ]:


df2 = pd.read_csv(path+'IndividualDetails.csv')
df2.head()


# In[ ]:


x = df2.current_status.value_counts()
x = pd.DataFrame(data={'Current_satus': x.index.tolist(), 'Count': x.values.tolist()})
fig = px.pie(x, values='Count', names='Current_satus', title='Current Staus of COVID-19 Victims')
fig.show()


# ## 20 Reasons for the spread of COVID 19 

# In[ ]:


x = df2.notes.map(lambda x:str(x).title()).value_counts()[2:].head(20)
x = pd.DataFrame(data={'Current_satus': x.index.tolist(), 'Count': x.values.tolist()})
fig = px.pie(x, values='Count', names='Current_satus', title='Top 20 Reasons for COVID-19 Spread')
fig.show()


# ## Gender Affected Most By COVID 19
# <a id ='gndr'/>

# In[ ]:


x = df2['gender'].value_counts()
x = pd.DataFrame(data={'Gender': x.index.tolist(), 'Count': x.values.tolist()})
fig = px.pie(x, values='Count', names='Gender', title='Who are affected by COVID-19(M/F)?')
fig.show()


# ## <font color = 'tomato'>Do Upvote the Notebook, It Motivates to Provide More Content<font/>
