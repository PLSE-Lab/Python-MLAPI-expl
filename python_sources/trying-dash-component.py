#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import folium


# In[ ]:


indicators = pd.read_csv('../input/world-development-indicators/Indicators.csv')
indicators.sample(5)


# Selecting all the Indicator's starting with a indicator code "IC", since all the code starting with IC has observations which gives information related to new business kickoff

# In[ ]:


buss = indicators[indicators['IndicatorCode'].str.match(r'(^IC.*)')]
buss.head()


# In[ ]:


buss.IndicatorCode.nunique()


# Check for Null Values

# In[ ]:


sns.heatmap(buss.isnull(),yticklabels=False,cbar=False,cmap='GnBu_r')


# In[ ]:


#To view the full content in the column
pd.set_option('display.max_colwidth',None)


# Checking for Ease of doing business in a country 

# In[ ]:


xq = buss.query('IndicatorCode=="IC.BUS.EASE.XQ"')
top_10 = xq.groupby('CountryName',as_index=False)[['Value']].mean().sort_values(by='Value',ascending=True).iloc[:10]


# In[ ]:


sns.barplot(top_10['Value'],top_10['CountryName'])
plt.xlabel('Index')
plt.title('Top 10 countries under Ease of doing business index')


# In[ ]:


# xq.Year.unique()


# Below is the Folium world map depecting the color code based on the index for ease of doing business where 1 being the most friendly

# In[ ]:


tomap = xq.groupby(['CountryCode','CountryName'],as_index=False)['Value'].mean()
tomap


# In[ ]:


country_geo = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json'

world_map = folium.Map(location=[100, 0], zoom_start=2)

folium.Choropleth(
    geo_data=country_geo,
    data=tomap,
    columns=['CountryCode','Value'],
    key_on='feature.id',
    fill_color='YlGnBu', 
    fill_opacity=0.6, 
    line_opacity=1,
    legend_name="Ease of doing Business"
).add_to(world_map)
world_map


# Comparing all the parameters for 2 nations which are in top 10 list of ease of doing business, considered the GBR ie Great Britian and Singapore(SGP) which are at the 6th and 1st position respectively

# In[ ]:


gbr = buss.loc[(buss['CountryCode']=="GBR")] 
gbr.IndicatorCode.nunique()


# In[ ]:


sgp = buss.loc[(buss['CountryCode']=="SGP")] 
sgp.IndicatorCode.nunique()


# In[ ]:


a = gbr.groupby('IndicatorName')[['Value']].mean()


# In[ ]:


b = sgp.groupby('IndicatorName')[['Value']].mean()


# In[ ]:


a.merge(b,left_index=True,right_index=True,suffixes=('_United Kingdom', '_Singapore'))


# In[ ]:


reg_proc_gb = gbr.query('IndicatorCode=="IC.REG.PROC"')


# In[ ]:


reg_proc_sg = sgp.query('IndicatorCode=="IC.REG.PROC"')


# In[ ]:


sns.lineplot(reg_proc_gb['Year'],reg_proc_gb['Value'])
sns.lineplot(reg_proc_sg['Year'],reg_proc_sg['Value'])


# Below is a dashboard where a user can compare any two countries for ease of doing business based on the parameters/Indicators

# In[ ]:


pip install dash


# In[ ]:


import dash
import plotly
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


# In[ ]:


ext_styles = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=ext_styles)


# In[ ]:


available_indicators = buss['IndicatorName'].unique()
available_country = buss['CountryName'].unique()


# In[ ]:


app.layout = html.Div([ 
    html.H1('Ease Of Doing Business',style={'textAlign':'center','font-size': '35px','color':'#343434'}),
    html.Div([
        dcc.Graph(id='wld_map',
                  figure = px.choropleth(tomap,locations='CountryCode',color='Value',projection='natural earth',
                                         title='Rating Index for ease of doing business',hover_name=tomap['CountryName']
                                         ,color_continuous_scale=px.colors.sequential.Blues,range_color=(1,190)
                                        ) 
                       )],className='twelve columns'),
    html.Div([
    html.Div('Select any two countries',style={'color':'#343434','margin-bottom':'10px','font-weight':'bold','font-size':'18px'}), 
    dcc.Dropdown(id='cout1',options=[{'label':i,'value':i} for i in available_country],value='Singapore',style={'width':'50%','margin-bottom':'10px'}),
    dcc.Dropdown(id='cout2',options=[{'label':i,'value':i}for i in available_country],value='United Kingdom',style={'width':'50%'}),
    html.Div('Select a parameter to compare',style={'color':'#343434','margin-top':'10px','font-weight':'bold','font-size':'18px'}),
    dcc.Dropdown(id='indi',options=[{'label': i,'value':i} for i in available_indicators],
                value='Start-up procedures to register a business (number)',style={'width':'70%'})],className='nine columns') ,
    html.Div([
        html.Div([dcc.Graph('one_grap')],className='six columns'),
        html.Div([dcc.Graph(
            id='top_grap',
             figure={
                 'data':[{'x':top_10['CountryName'],'y':top_10['Value'],'type':'bar'}],
                 'layout': 
                     dict(title=('Top 10 countries under Ease of doing business index'),
                     xaxis={'title':'Countries'},
                     yaxis={'title':'Ratings'}
                            )
                     }
             )],className='six columns'),
             ],className='row')
])
@app.callback(Output(component_id='one_grap',component_property='figure'),
              [Input(component_id='cout1',component_property='value'),
              Input('cout2',component_property='value'),
              Input('indi','value')]
              )
def to_pltng(cou1,cou2,ind1):
    fir = buss[buss.CountryName == cou1]
    sec = buss[buss.CountryName == cou2]
    fir_ind = fir[fir.IndicatorName == ind1]
    sec_ind = sec[sec.IndicatorName == ind1]
    fir_dic = {'x':fir_ind.Year,'y':fir_ind.Value,'name':cou1}
    sec_dic = {'x':sec_ind.Year,'y':sec_ind.Value,'name':cou2}
    return {
        'data':[fir_dic,sec_dic],
        'layout': dict(title='Comparing two countries as per Indicator selection',
            xaxis={'title':'Years'},
            yaxis={'title':'Value as per Indicator selection'}
        )
    }

if __name__=='__main__':
    app.run_server(debug=True)


# ![Dashboard.gif](attachment:Dashboard.gif)

# 

# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




