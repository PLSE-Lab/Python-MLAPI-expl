#!/usr/bin/env python
# coding: utf-8

# # Analysing India's fight against COVID19
# <img src='https://media.giphy.com/media/kgsBIWtPd5Q5Pw11Rq/giphy.gif' height=400 width=300>
# <br>
# <div class="list-group" id="list-tab" role="tablist">
#   <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Table of Contents</h3>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#in" role="tab" aria-controls="profile">Where India stands currently?<span class="badge badge-primary badge-pill">1</span></a>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#dr" role="tab" aria-controls="messages">Studying Doubling rates<span class="badge badge-primary badge-pill">2</span></a>
#   <a class="list-group-item list-group-item-action"  data-toggle="list" href="#geo" role="tab" aria-controls="settings">Geospatial Analysis<span class="badge badge-primary badge-pill">3</span></a>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#test" role="tab" aria-controls="settings">Testing in India<span class="badge badge-primary badge-pill">4</span></a> 
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#state" role="tab" aria-controls="settings">Statewise Analysis<span class="badge badge-primary badge-pill">5</span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#pop" role="tab" aria-controls="settings">Does Population Density has to do anything with the spread?<span class="badge badge-primary badge-pill">6</span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#trend" role="tab" aria-controls="settings"> Studying the trend<span class="badge badge-primary badge-pill">7</span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#heat" role="tab" aria-controls="settings">Heatmaps to showcase daily trends<span class="badge badge-primary badge-pill">8</span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#top" role="tab" aria-controls="settings">Top 5 worst affected states<span class="badge badge-primary badge-pill">9</span></a> 

# In[ ]:


get_ipython().system('pip install pycountry_convert')


# In[ ]:


get_ipython().system('pip install GoogleMaps')


# In[ ]:


import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import googlemaps
import re 
import pycountry
import pycountry_convert as pc
import requests


# # <a id='in'>1. Where India stands currently? </a>
# COVID19 outbreak has started a bit late in India as compared to other countries. But, it has started to pick up pace. With limited testing and not a well funded healthcare system, India is surely up for a challenge. Still the fight is on after 2 lockdowns and the virus shows no signs of slowing down. 

# In[ ]:


class country_utils():
    def __init__(self):
        self.d = {}
    
    def get_dic(self):
        return self.d
    
    def get_country_details(self,country):
        """Returns country code(alpha_3) and continent"""
        try:
            country_obj = pycountry.countries.get(name=country)
            if country_obj is None:
                c = pycountry.countries.search_fuzzy(country)
                country_obj = c[0]
            continent_code = pc.country_alpha2_to_continent_code(country_obj.alpha_2)
            continent = pc.convert_continent_code_to_continent_name(continent_code)
            return country_obj.alpha_3, continent
        except:
            if 'Congo' in country:
                country = 'Congo'
            if country == 'Mainland China':
                country = 'China'
            elif country == 'Diamond Princess' or country == 'Laos' or country == 'MS Zaandam'            or country == 'Holy See' or country == 'Timor-Leste':
                return country, country
            elif country == 'Korea, South' or country == 'South Korea':
                country = 'Korea, Republic of'
            elif country == 'Taiwan*':
                country = 'Taiwan'
            elif country == 'Burma':
                country = 'Myanmar'
            elif country == 'West Bank and Gaza':
                country = 'Gaza'
            else:
                return country, country
            country_obj = pycountry.countries.search_fuzzy(country)
            continent_code = pc.country_alpha2_to_continent_code(country_obj[0].alpha_2)
            continent = pc.convert_continent_code_to_continent_name(continent_code)
            return country_obj[0].alpha_3, continent
    
    def get_iso3(self, country):
        return self.d[country]['code']
    
    def get_continent(self,country):
        return self.d[country]['continent']
    
    def add_values(self,country):
        self.d[country] = {}
        self.d[country]['code'],self.d[country]['continent'] = self.get_country_details(country)
    
    def fetch_iso3(self,country):
        if country in self.d.keys():
            return self.get_iso3(country)
        else:
            self.add_values(country)
            return self.get_iso3(country)
        
    def fetch_continent(self,country):
        if country in self.d.keys():
            return self.get_continent(country)
        else:
            self.add_values(country)
            return self.get_continent(country)


# In[ ]:


df_world = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df_world.ObservationDate = pd.to_datetime(df_world.ObservationDate, format='%m/%d/%Y')
max_date = df_world.ObservationDate.max()
df_world = df_world[df_world.ObservationDate==max_date]
df_world.rename(columns={'Country/Region':'Country'},inplace=True)
df_cont = df_world.copy()
df_world = df_world.groupby(['Country'],as_index=False)['Confirmed','Deaths','Recovered'].sum()
df_world['rank_c'] = df_world['Confirmed'].rank(ascending=False)
df_world['rank_d'] = df_world['Deaths'].rank(ascending=False)
df_world['rank_r'] = df_world['Recovered'].rank(ascending=False)
world_stat = (df_world.loc[df_world['Country']=='India'])
world_stat.set_index('Country',inplace=True)
world_stat = world_stat.astype(int)


# In[ ]:


obj = country_utils()
df_cont['continent'] = df_cont.apply(lambda x: obj.fetch_continent(x['Country']), axis=1)
df_cont = df_cont.groupby(['continent','Country'],as_index=False)['Confirmed','Deaths','Recovered'].sum()
df_cont = df_cont[df_cont['continent']=='Asia']
df_cont['rank_c'] = df_cont['Confirmed'].rank(ascending=False)
df_cont['rank_d'] = df_cont['Deaths'].rank(ascending=False)
df_cont['rank_r'] = df_cont['Recovered'].rank(ascending=False)
cont_stat = (df_cont.loc[df_cont['Country']=='India'])
cont_stat.set_index('Country',inplace=True)
cont_stat.drop('continent',inplace=True,axis=1)
cont_stat = cont_stat.astype(int)


# The table below shows the current figures and ranking for India in the World and in Asia.

# In[ ]:


def make_538(fig,title=None,xtext=None,ytext=None,hovermode='x',width=700,height=400,margin=dict(t=50,b=10,l=10,r=10),legend=None,annotations=None):
    fig.update_layout(
        template='simple_white',
        title=title,
        hovermode=hovermode,
        xaxis=dict(title=xtext,showline=False,showgrid=True,ticks='',gridcolor=colors['grid']),
        yaxis=dict(title=ytext,showline=False,showgrid=True,ticks='',gridcolor=colors['grid']),
        paper_bgcolor=colors['bg'],
        plot_bgcolor=colors['bg'],
        width=width,
        height=height,
        margin=margin,
        legend=legend,
        annotations=annotations
    )
    return fig


# In[ ]:


colors = dict(bg='#f0f0f0',
              grid='#d7d7d7',
              cases='#30a2da',
              deaths='#fc4f30',
              recoveries='#6d904f',
              shape='#8b8b8b')


# In[ ]:


import plotly.graph_objects as go

values = [['Figures','World Ranking','Asia Ranking'],
          [world_stat.Confirmed['India'],world_stat.rank_c['India'],cont_stat.rank_c['India']],
          [world_stat.Deaths['India'],world_stat.rank_d['India'],cont_stat.rank_d['India']],
          [world_stat.Recovered['India'],world_stat.rank_r['India'],cont_stat.rank_r['India']]]


fig = go.Figure(data=[go.Table(
  columnorder = [1,2,3,4],
  columnwidth = [300,400],
  header = dict(
    values = [['<b>STATISTICS</b><br>as of '+str(max_date.day)+' '+ max_date.month_name()],
              ['<b>CASES</b>'],['<b>DEATHS</b>'],['<b>RECOVERIES</b>']],
    line_color=colors['grid'],
    fill_color=colors['cases'],
    align=['left','center'],
    font=dict(color='white', size=12),
    height=40
  ),
  cells=dict(
    values=values,
    line_color=colors['grid'],
    fill=dict(color=[colors['shape'], 'white']),
    align=['left', 'center'],
    font_size=12,
    height=30)
    )
])
fig = make_538(fig,height=250,title=dict(text='<b>Report from Ground Zero</b>',font=dict(family='Helvetica')))
fig.show()


# In[ ]:


df = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
df['Date'] = pd.to_datetime(df['Date'],format='%d/%m/%y')
df['Date'] = df['Date'].dt.date
df.rename(columns={'Date':'date','State/UnionTerritory':'state','ConfirmedIndianNational':'confirmed_in',                   'ConfirmedForeignNational':'confirmed_fr'}, inplace=True)
df.drop(['Sno','Time'],axis=1,inplace=True)
df['state'] = df.apply(lambda x: 'Nagaland' if x['state']=='Nagaland#' else 'Jharkhand' if x['state']=='Jharkhand#' else x['state'], axis=1)
df = df[df['state']!='Unassigned']
df.reset_index(inplace=True)


# In[ ]:


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("gmaps")
secret_value_1 = user_secrets.get_secret("mapboxtoken")
gmaps = googlemaps.Client(key=secret_value_0)


# In[ ]:


def add_daily_measures(df):
    has_state=False
    if 'state' in df.columns:
        states = []
        has_state = True
    df.loc[0,'Daily Cases'] = df.loc[0,'Confirmed']
    df.loc[0,'Daily Deaths'] = df.loc[0,'Deaths']
    df.loc[0,'Daily Cured'] = df.loc[0,'Cured']
    for i in range(1,len(df)):
        if has_state:
            if df.loc[i,'state'] in states:
                df.loc[i,'Daily Cases'] = df.loc[i,'Confirmed'] - df.loc[i-1,'Confirmed']
                df.loc[i,'Daily Deaths'] = df.loc[i,'Deaths'] - df.loc[i-1,'Deaths'] 
                df.loc[i,'Daily Cured'] = df.loc[i,'Cured'] - df.loc[i-1,'Cured']
            else:
                states.append(df.loc[i,'state'])
                df.loc[i,'Daily Cases'] = df.loc[i,'Confirmed']
                df.loc[i,'Daily Deaths'] = df.loc[i,'Deaths']
                df.loc[i,'Daily Cured'] = df.loc[i,'Cured']
        else:
            df.loc[i,'Daily Cases'] = df.loc[i,'Confirmed'] - df.loc[i-1,'Confirmed']
            df.loc[i,'Daily Deaths'] = df.loc[i,'Deaths'] - df.loc[i-1,'Deaths'] 
            df.loc[i,'Daily Cured'] = df.loc[i,'Cured'] - df.loc[i-1,'Cured']
    #Make the first row as 0 because we don't know the previous value
    df.loc[0,'Daily Cases'] = 0
    df.loc[0,'Daily Deaths'] = 0
    df.loc[0,'Daily Cured'] = 0
    return df


# The plot below shows the total cases, deaths and recoveries reported on a daily basis. The worrying point is that even after almost 2 nation-wide lockdowns the case are increasing.

# In[ ]:


df.loc[1428,'state'] = 'Madhya Pradesh'
df.loc[1428,'Deaths'] = '119'
df.fillna(0,inplace=True)
df.loc[df.Deaths=='0#','Deaths'] = 0
df.Deaths = df.Deaths.astype(np.int16)


# In[ ]:


imp_dates = [dict(date='2020-03-23',event="Lockdown Phase 1<br><b>23<sup>rd</sup> March</b>"),
             dict(date='2020-04-15',event="Lockdown Phase 2<br><b>15<sup>th</sup> April</b>"),
             dict(date='2020-05-04',event="Lockdown Phase 3<br><b>4<sup>th</sup> May</b>"),
             dict(date='2020-05-18',event="Lockdown Phase 4<br><b>18<sup>th</sup> May</b>"),
             dict(date='2020-06-01',event="Unlock 1.0<br><b>1<sup>st</sup> June</b>"),
             dict(date='2020-07-01',event="Unlock 2.0<br><b>1<sup>st</sup> July</b>"),
             dict(date='2020-08-01',event="Unlock 3.0<br><b>1<sup>st</sup> August</b>")]


# In[ ]:


df_ind = df.copy()
df_ind = df_ind.groupby('date',as_index=False)['Cured','Deaths','Confirmed'].sum()
df_ind = add_daily_measures(df_ind)
fig = go.Figure(data=[
    go.Scatter(name='Cases', x=df_ind['date'], y=df_ind['Daily Cases'],mode='lines',
               line=dict(color=colors['cases'],width=5)),
    go.Scatter(name='Recoveries', x=df_ind['date'], y=df_ind['Daily Cured'],mode='lines',
               line=dict(color=colors['recoveries'],width=5)),
    go.Scatter(name='Deaths', x=df_ind['date'], y=df_ind['Daily Deaths'],mode='lines',
               line=dict(color=colors['deaths'],width=5)),
])

annotations = []

for date in imp_dates:
    fig.add_shape(type='line',xref='x',yref='y',layer='below',
                  x0=date['date'] ,y0=0,x1=date['date'],y1=60000,
                  line=dict(dash='dot',color=colors['shape'],width=3))
    annotations.append(dict(x=date['date'], y=50000, xref="x", yref="y",textangle=-45, 
                            text=date['event'], font=dict(size=10), showarrow=False))
annotations[-1]['y'] = 15000
legend=dict(orientation='h',x=0.5,y=1.1,bgcolor=colors['bg'])
fig = make_538(fig,title='Daily Cases, Deaths & Recoveries',legend=legend,annotations=annotations)
fig.show()


# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/2061549" data-url="https://flo.uri.sh/visualisation/2061549/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')


# The above Bar chart race is an attempt to show the progression of total cases in India.
# > Please note that this plot is created using [this service](https://app.flourish.studio/@flourish/bar-chart-race/11) and has to be manually updated.

# In[ ]:


#Code to get the dataset for Bar chart race
df_br = df.copy()
df_br = df_br.pivot(index='state',columns='date',values='Confirmed')
df_br.fillna(0,inplace=True)
df_br.reset_index(level=0, inplace=True)
df_br.columns.name = ''
df_br.to_csv(r'confirmed_cases_india.csv',index=False)


# # <a id='dr'>2. Studying Doubling rates</a>
# The doubling time is time it takes for a population to double in size/value. When the relative growth rate (not the absolute growth rate) is constant, the quantity undergoes exponential growth and has a constant doubling time or period, which can be calculated directly from the growth rate.This time can be calculated by dividing the natural logarithm of 2 by the exponent of growth, or approximated by dividing 70 by the percentage growth rate (more roughly but roundly, dividing 72).
# 
# For a constant growth rate of r% within time t, the formula for the doubling time Td is given by
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/44fa9e83cc6807214d065fe22aa46c041e45482e)
# 
# Given two measurements of a growing quantity, q1 at time t1 and q2 at time t2, and assuming a constant growth rate, you can calculate the doubling time as
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/137261413a0a52ccba07032f1abc0d6338e906ff)
# 
# >In this section I'll be studying the doubling rates of COVID19 across whole india and its states respectively.

# In[ ]:


def doubling_rate_india(x):
    dr = np.log(2)/np.log(x['Confirmed']/x['Confirmed_prev'])
    return np.round(dr,2)

df_ind['Confirmed_prev'] = df_ind['Confirmed'].shift(1)
df_ind['Doubling rate'] = df_ind.apply(lambda x: doubling_rate_india(x),axis=1)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_ind['date'],y=df_ind['Doubling rate'],mode='lines',
                         line=dict(color=colors['cases'],width=5)))
fig = make_538(fig,ytext='Days',title="India's Daily Doubling rate variation")
fig.show()


# In[ ]:


doubling_rate = 0
state = df.state.unique().tolist()[0]
def calc_doubling_rate(x):
    global growth_rate
    global state
    if x['state']!=state:
        doubling_rate=0
        state = x['state']
    try:
        dr = np.log(2)/np.log(x['Confirmed']/x['Confirmed_prev'])
    except ZeroDivisionError:
        dr = 0
    return np.round(dr,2)

df_dr = df.copy()
df_dr.sort_values(['state','date'],inplace=True)
df_dr.reset_index(drop=True,inplace=True)
df_dr['Confirmed_prev'] = df_dr.groupby('state')['Confirmed'].transform(lambda x: x.shift(1))
df_dr['Doubling rate'] = df_dr.apply(lambda x: calc_doubling_rate(x),axis=1)
df_curr_dr = df_dr[df_dr['date']==df_dr.date.max()]
df_curr_dr.sort_values('Confirmed',ascending=False,inplace=True)
df_curr_dr=df_curr_dr[['state','Confirmed','Doubling rate']]
df_curr_dr=df_curr_dr.nlargest(15,'Confirmed')
df_curr_dr.reset_index(drop=True,inplace=True)

# Set CSS properties for th elements in dataframe
th_props = [
  ('font-size', '11px'),
  ('text-align', 'center'),
  ('font-weight', 'bold'),
  ('color', '#6d6d6d'),
  ('background-color', '#f7f7f9')
  ]

# Set CSS properties for td elements in dataframe
td_props = [
  ('font-size', '11px')
  ]

# Set table styles
styles = [
  dict(selector="th", props=th_props),
  dict(selector="td", props=td_props)
  ]

(df_curr_dr.style
 .background_gradient(cmap='Reds_r',subset='Doubling rate')
 .background_gradient(cmap='Blues',subset='Confirmed')
 .set_caption('Doubling Rates of 15 Indian States with most cases')
 .set_table_styles(styles))


# > **NOTE: ** The the Doubling rate that I have calculated is in days. The lesser the doubling rate, the more the spread of virus.

# In[ ]:


df_dr = df_dr.merge(df_curr_dr['state'],how='inner',on='state')
rows=5
cols=3
states = df_curr_dr.state.tolist()
fig = make_subplots(rows,cols,shared_xaxes=True,subplot_titles=states)
for r in range(1,rows+1):
    for c in range(1,cols+1):
        ind = r+c-2
        df_s = df_dr[df_dr['state']==states[ind]]
        fig.add_trace(go.Scatter(x=df_s['date'],y=np.abs(df_s['Doubling rate'])),r,c)
        fig.update_xaxes(showline=False,ticks='',showgrid=True,gridcolor=colors['grid'],row=r,col=c)
        fig.update_yaxes(showline=False,ticks='',showgrid=True,gridcolor=colors['grid'],tickfont=dict(size=8),row=r,col=c)
fig.update_layout(template='simple_white',showlegend=False,margin=dict(t=100),
                 title='<b>State-wise Doubling Rate variation</b>',
                 width=700,paper_bgcolor=colors['bg'],plot_bgcolor=colors['bg'])
fig.show()


# # <a id='geo'>3. Geospatial Analysis</a>
# In this section I have used a time slider on a map to show the progression of cases in India.

# In[ ]:


d = {}
states = df.state.unique().tolist()
for state in states:
    if state!='Cases being reassigned to states':
        data = gmaps.geocode(state)
        if data:
            d[state] = data[0]['geometry']['location']

def get_coords(state):
    if state!='Cases being reassigned to states':
        lat = d[state]['lat']
        lng = d[state]['lng']
    else:
        return (np.float('nan'),np.float('nan'))
    return (lat,lng)


# In[ ]:


df_map = df.copy()
df_map['latitude'] = df_map.apply(lambda x: get_coords(x['state'])[0],axis=1)
df_map['longitude'] = df_map.apply(lambda x: get_coords(x['state'])[1],axis=1)
px.set_mapbox_access_token(secret_value_1)
df_map['date'] = df_map['date'].astype('str')
fig = px.scatter_mapbox(df_map,
                        lat="latitude",
                        lon="longitude",
                        size="Confirmed",
                        color='Confirmed',
                        mapbox_style='streets',
                        color_continuous_scale=px.colors.cyclical.IceFire,
                        range_color=(0, df_map['Confirmed'].max()),
                        animation_frame='date',
                        hover_name='state',
                        hover_data=['Deaths','Cured'],
                        zoom=2.5,
                        size_max=50,
                        title= 'India:Spread of COVID19')
fig.show()


# In[ ]:


def change_state_name(state):
    if state == 'Odisha':
        return 'Orissa'
    elif state == 'Telengana':
        return 'Telangana'
    elif state == 'Andaman and Nicobar Islands':
        return 'Andaman and Nicobar'
    return state
r = requests.get(url='https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson')
geojson = r.json()
df_map['state'] = df_map.apply(lambda x: change_state_name(x['state']), axis=1)
last_date = df_map.date.max()
df_map = df_map[df_map['date']==last_date]
fig = px.choropleth_mapbox(df_map, geojson=geojson, color="Confirmed",
                    locations="state", featureidkey="properties.NAME_1",
                    hover_name='state',
                    hover_data=['Cured','Deaths'],
                    center={"lat": 20.5937, "lon": 78.9629},
                    mapbox_style="carto-positron",
                    zoom=2.75,
                    color_continuous_scale=px.colors.qualitative.Vivid,
                    title='Total Cases per State'
                   )
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.show()


# In[ ]:


df_ind_det = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')
r = requests.get(url='https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson')
geojson = r.json()
df_dist = df_ind_det.groupby('detected_district',as_index=False)['id'].count()
df_dist.rename(columns={'detected_district':'District','id':'Cases Reported'},inplace=True)
fig = px.choropleth_mapbox(df_dist, geojson=geojson, color="Cases Reported",
                    locations="District", featureidkey="properties.NAME_2",
                    hover_name='District',
                    center={"lat": 20.5937, "lon": 78.9629},
                    mapbox_style="carto-positron",
                    zoom=2.75,
                    color_continuous_scale=px.colors.qualitative.Vivid,
                    title='Total Cases per District'
                   )
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig = make_538(fig,title='Total Cases per District')
fig.show()


# In[ ]:


df_zone = pd.read_csv('/kaggle/input/covid19indiazones/India-District-Zones.csv')
df_zone = df_zone.groupby(['State','Zone'],as_index=False)['District'].count()
df_zone.sort_values('District',inplace=True)
fig = go.Figure()
df_g = df_zone[df_zone['Zone']=='Green Zone']
fig.add_trace(go.Bar(name='Green zone',x=df_g['State'],y=df_g['District'], marker_color='Green',
                    marker_line_color='black',marker_line_width=1.5))
df_o = df_zone[df_zone['Zone']=='Orange Zone']
fig.add_trace(go.Bar(name='Orange zone',x=df_o['State'],y=df_o['District'], marker_color='Orange',
                    marker_line_color='black',marker_line_width=1.5))
df_r = df_zone[df_zone['Zone']=='Red Zone']
fig.add_trace(go.Bar(name='Red zone',x=df_r['State'],y=df_r['District'], marker_color='Red',
                    marker_line_color='black',marker_line_width=1.5))
fig.update_layout(barmode='stack')
fig = make_538(fig,title='Number of Zones per State from 4th May',margin=dict(l=0,r=0,t=50,b=170))
fig.show()


# # <a id='test'>4. Testing in India</a>

# In[ ]:


df_lab = pd.read_csv('/kaggle/input/icmrtestinglabs/ICMRTestingLabsWithCoords.csv')
fig = px.scatter_mapbox(df_lab,
                        lat="latitude",
                        lon="longitude",
                        mapbox_style='streets',
                        hover_name='lab',
                        hover_data=['city','state','pincode'],
                        zoom=2.5,
                        size_max=15,
                        title= 'COVID19 Testing Labs in India')
fig.show()


# In[ ]:


df_lab_group = df_lab.groupby('state',as_index=False)['lab'].count().sort_values('lab',ascending=False)
fig = px.bar(df_lab_group,x='state',y='lab')
fig.update_traces(marker_color=colors['cases'],marker_line_color=colors['cases'],text=df_lab_group['lab'],textposition='outside')
fig.update_layout(template='plotly_white')
fig.update_yaxes(title_text = 'Number of Labs')
fig.update_xaxes(title_text = '')
fig = make_538(fig,height=500,title='Number of COVID19 testing labs per State/Union Territory',margin=dict(b=280))
fig.show()


# In[ ]:


df_tes_st = pd.read_csv('/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv')
fig = go.Figure()
for state in df_tes_st.State.unique():
    df_s = df_tes_st[df_tes_st['State']==state]
    fig.add_trace(go.Scatter(x=df_s['Date'],y=df_s['TotalSamples'],mode='lines',name=state))
    #fig.add_trace(go.Bar(x=df_s['Date'],y=df_s['Positive']))
fig = make_538(fig,height=500,title='State-wise number of tests')
fig.show()


# # <a id='state'>5. Statewise Analysis</a>

# In[ ]:


df_pop = pd.read_csv('/kaggle/input/covid19-in-india/population_india_census2011.csv')
df_bed = pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')
df_test = pd.read_csv('/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv',parse_dates=True)
df_bed['Total beds'] = df_bed.NumPublicBeds_HMIS + df_bed.NumRuralBeds_NHP18 + df_bed.NumUrbanBeds_NHP18


# In[ ]:


def get_area(s):
    if pd.isnull(s):
        return s
    temp = re.findall(r'\d+.?\d+', s)
    temp = [x.replace(',','') for x in temp]
    res = list(map(int, temp))
    return res

def get_density(s):
    if pd.isnull(s):
        return s
    temp = re.findall(r'\d+[.\.]?\d+', s)
    temp = [x.replace(',','') for x in temp]
    res = list(map(float, temp))
    return res


# In[ ]:


df_latest = df[df['date']==df.date.max()]
df_latest = df_latest.merge(df_pop,how='left',left_on='state',right_on='State / Union Territory')
df_latest = df_latest.merge(df_bed,how='left',left_on='state',right_on='State/UT')
df_latest = df_latest[['state','Cured','Deaths','Confirmed','Population','Rural population','Area','Density','Gender Ratio','Total beds','TotalPublicHealthFacilities_HMIS']]
df_latest = df_latest[df_latest['state']!='Cases being reassigned to states']
df_latest.drop([7,30],inplace=True)
df_latest['Area(sq km)'] = df_latest.apply(lambda x:get_area(x['Area'])[0],axis=1)
df_latest['Area(sq miles)'] = df_latest.apply(lambda x:get_area(x['Area'])[1],axis=1)
df_latest['Density(sq km)'] = df_latest.apply(lambda x:get_density(x['Density'])[0],axis=1)
df_latest['Density(sq miles)'] = df_latest.apply(lambda x:get_density(x['Density'])[1],axis=1)
df_latest.drop(['Area','Density'],axis=1,inplace=True)
df_latest.rename(columns={'state':'State'},inplace=True)
df_latest['Cases/million'] = round((df_latest.Confirmed/df_latest.Population)*1000000).astype(int)
df_latest.fillna(0,inplace=True)
df_latest['Beds/million'] = round((df_latest['Total beds']/df_latest.Population)*1000000).astype(int)
df_latest['Health Facilities/100sq.km'] = round((df_latest['TotalPublicHealthFacilities_HMIS']/df_latest['Area(sq km)'])*1000).astype(int)
df_latest['Mortality Rate %'] = np.round((df_latest.Deaths/df_latest.Confirmed)*100,2)
df_latest['Recovery Rate %'] = np.round((df_latest.Cured/df_latest.Confirmed)*100,2)
df_latest.sort_values('Confirmed',ascending=False,inplace=True)
df_latest.reset_index(inplace=True,drop=True)

df_table = df_latest[['State','Cured','Deaths','Confirmed','Mortality Rate %','Recovery Rate %','Cases/million','Beds/million','Health Facilities/100sq.km']]
(df_table.style
 .background_gradient(cmap='Blues',subset=['Confirmed','Cases/million','Beds/million','Health Facilities/100sq.km'])
 .background_gradient(cmap='Greens', subset=['Cured','Recovery Rate %'])
 .background_gradient(cmap='Reds', subset=['Deaths','Mortality Rate %'])
 .set_caption('COVID19: Statistics about India')
 .set_table_styles(styles))


# # <a id='pop'> 6. Does Population Density has to do anything with the spread?</a>
# Here, I will be studing the impact of population density(per km square) on the total confirmed cases, deaths and recoveries. I'll be using a scatter plot with Total Cases, Deaths or Recoveries on the X-asis, population density on the Y-axis and population representing the size of the bubble.

# In[ ]:


fig = px.scatter(df_latest,x='Confirmed',y='Density(sq km)',size='Population',color='State')
fig.update_yaxes(title_text='Population Density(per sq km)')
fig.update_xaxes(title_text='Total Confirmed Cases')
fig=make_538(fig,title='Total Confirmed Cases vs Population Density(per sq km)')
fig.show()


# In[ ]:


fig = px.scatter(df_latest,x='Deaths',y='Density(sq km)',size='Population',color='State')
fig.update_yaxes(title_text='Population Density(per sq km)')
fig.update_xaxes(title_text='Total Deaths')
fig = make_538(fig, title='Total Deaths vs Population Density(per sq km)')
fig.show()


# In[ ]:


fig = px.scatter(df_latest,x='Cured',y='Density(sq km)',size='Population',color='State')
fig.update_yaxes(title_text='Population Density(per sq km)')
fig.update_xaxes(title_text='Total Recoveries')
fig = make_538(fig, title='Total Recoveries vs Population Density(per sq km)')
fig.show()


# Uttar Pradesh with highest population density has not reported as many cases as Maharashtra or Gujarat have. Although, there population density is not even less than half of that of Uttar Pradesh's. So, there seems to be so concrete releation.

# # <a id='trend'>7. Studying the trend</a>
# For the first three plots, I'm plotting the cumulative cases, deaths and recoveries reported against the day it's first instance was reported.

# In[ ]:


df_states = df.copy()
def add_days(df,new_col,basis):
    states = {}
    df[new_col] = 0
    for i in range(len(df_states)):
        if df_states.loc[i,'state'] in states:
            df_states.loc[i,new_col] = (df_states.loc[i,'date'] - states[df_states.loc[i,'state']]).days
        else:
            if df_states.loc[i,basis] > 0:
                states[df_states.loc[i,'state']] = df_states.loc[i,'date']
    return df
df_states = add_days(df_states,'day_since_inf','Confirmed')
df_states = add_days(df_states,'day_since_death','Deaths')
df_states = add_days(df_states,'day_since_cure','Cured')


# In[ ]:


fig = px.line(df_states,x='day_since_inf',y='Confirmed',color='state')
fig = make_538(fig,title='Cumulative cases over time')
fig.update_xaxes(title_text='Days since first infection was reported')
fig.update_yaxes(title_text='No. of Confirmed cases')
fig.show()


# In[ ]:


fig = px.line(df_states,x='day_since_death',y='Deaths',color='state')
fig = make_538(fig,title='Cumulative deaths over time')
fig.update_xaxes(title_text='Days since first death was reported')
fig.update_yaxes(title_text='No. of Confirmed deaths')
fig.show()


# In[ ]:


fig = px.line(df_states,x='day_since_cure',y='Cured',color='state')
fig = make_538(fig,title='Cumulative recoveries over time')
fig.update_xaxes(title_text='Days since first recovery was reported')
fig.update_yaxes(title_text='No. of Confirmed recoveries')
fig.show()


# In the below three plots, I'm plotting the 7-day rolling mean of daily cases, deaths and recoveries reported against the day it's first instance was reported.

# In[ ]:


df_states.sort_values(by=['state','date'],inplace=True)
df_states.reset_index(inplace=True,drop=True)
df_states_daily = add_daily_measures(df_states)
df_states_daily.fillna(0,inplace=True)


# In[ ]:


states = df_states_daily['state'].unique().tolist()
df_roll = pd.DataFrame()
for state in states:
    df_state = df_states_daily[df_states_daily['state']==state]
    df_state['roll_avg_c'] = np.round(df_state['Daily Cases'].rolling(7).mean())
    df_state['roll_avg_d'] = np.round(df_state['Daily Deaths'].rolling(7).mean())
    df_state['roll_avg_r'] = np.round(df_state['Daily Cured'].rolling(7).mean())
    df_roll = df_roll.append(df_state,ignore_index=True)


# In[ ]:


fig = px.line(df_roll,x='day_since_inf',y='roll_avg_c',color='state',title='Daily cases over time')
fig.update_layout(template="simple_white")
fig.update_xaxes(title_text='Days since first infection was reported')
fig.update_yaxes(title_text='7-Day Rolling average')
fig.show()


# In[ ]:


fig = px.line(df_roll,x='day_since_inf',y='roll_avg_d',color='state',title='Daily deaths over time')
fig.update_layout(template="simple_white")
fig.update_xaxes(title_text='Days since first infection was reported')
fig.update_yaxes(title_text='7-Day Rolling average')
fig.show()


# In[ ]:


fig = px.line(df_roll,x='day_since_inf',y='roll_avg_r',color='state',title='Daily recoveries over time')
fig.update_layout(template="simple_white")
fig.update_xaxes(title_text='Days since first infection was reported')
fig.update_yaxes(title_text='7-Day Rolling average')
fig.show()


# So, from the above plots, It's pretty evident that out of all states only **Kerala** has been able to faltten the curve. While **Maharashtra, Gujarat, Delhi and Madhya Pradesh** are still fighting hard to curb the spread of infection. The positive is that the recoveries have started to pick up pace in all the states whiles death rate is slowing down.

# # <a id='heat'>8. Heatmaps to showcase daily trends</a>

# In[ ]:


fig = go.Figure(data=go.Heatmap(
        z=df_states_daily['Daily Cases'],
        x=df_states_daily.date,
        y=df_states_daily.state,
        colorscale=px.colors.sequential.Blues))
fig.update_layout(width=700,height=800,template="plotly_white",title='Heatmap: Daily cases per day',yaxis_nticks=50)
fig.show()


# In[ ]:


fig = go.Figure(data=go.Heatmap(
        z=df_states_daily['Daily Deaths'],
        x=df_states_daily.date,
        y=df_states_daily.state,
        colorscale=px.colors.sequential.Reds))
fig.update_layout(width=700,height=800,template="plotly_white",title='Heatmap: Daily deaths per day',yaxis_nticks=56)
fig.show()


# In[ ]:


fig = go.Figure(data=go.Heatmap(
        z=df_states_daily['Daily Cured'],
        x=df_states_daily.date,
        y=df_states_daily.state,
        colorscale=px.colors.sequential.Greens))
fig.update_layout(height=800,width=700,template="plotly_white",title='Heatmap: Daily recoveries per day',yaxis_nticks=56)
fig.show()


# # <a id='top'>9. Top 5 worst affected states</a>
# In this section, I'll take a look at the top 5 worst affected states in India. I'm going to plot daily cases, deaths and recoveries along with their rolling average go get a sense of the trend: increasing or decreasing.

# In[ ]:


n = 5
df_curr = df_states[df_states['date']==df_states.date.max()]
states = df_curr.nlargest(n,'Confirmed')['state'].values.tolist()


# In[ ]:


def plot_state(state=state):
    df_state = df_states[df_states['state']==state]
    df_state['roll_avg_c'] = df_state['Daily Cases'].rolling(7).mean()
    df_state['roll_avg_d'] = df_state['Daily Deaths'].rolling(7).mean()
    df_state['roll_avg_r'] = df_state['Daily Cured'].rolling(7).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Cases',x=df_state['date'],y=df_state['Daily Cases'],marker_color='royalblue',yaxis='y'))
    fig.add_trace(go.Scatter(name='7-day rolling avg',x=df_state['date'],y=df_state['roll_avg_c'],marker_color='black',yaxis='y'))
    fig.add_trace(go.Bar(name='Deaths',x=df_state['date'],y=df_state['Daily Deaths'],marker_color='crimson',yaxis='y2'))
    fig.add_trace(go.Scatter(name='7-day rolling avg',x=df_state['date'],y=df_state['roll_avg_d'],marker_color='black',yaxis='y2'))
    fig.add_trace(go.Bar(name='Recoveries',x=df_state['date'],y=df_state['Daily Cured'],marker_color='limegreen',yaxis='y3'))
    fig.add_trace(go.Scatter(name='7-day rolling avg',x=df_state['date'],y=df_state['roll_avg_r'],marker_color='black',yaxis='y3'))
    # Update axes
    fig.update_layout(
        xaxis=dict(
        autorange=True,
        rangeslider=dict(
            autorange=True,
        ),
        type="date"
    ),
    yaxis=dict(
        anchor="x",
        autorange=True,
        domain=[0, 0.3],
        linecolor="royalblue",
        mirror=True,
        showline=True,
        side="right",
        tickfont={"color": "royalblue"},
        tickmode="auto",
        ticks="",
        title='Cases',
        titlefont={"color": "royalblue","size":10},
        type="linear",
        zeroline=False
    ),
    yaxis2=dict(
        anchor="x",
        autorange=True,
        domain=[0.35, 0.65],
        linecolor="crimson",
        mirror=True,
        showline=True,
        side="right",
        tickfont={"color": "crimson"},
        tickmode="auto",
        ticks="",
        title = 'Deaths',
        titlefont={"color": "crimson","size":10},
        type="linear",
        zeroline=False
    ),
    yaxis3=dict(
        anchor="x",
        autorange=True,
        domain=[0.7, 1],
        linecolor="limegreen",
        mirror=True,
        showline=True,
        side="right",
        tickfont={"color": "limegreen"},
        tickmode="auto",
        ticks="",
        title="Recoveries",
        titlefont={"color": "limegreen","size":10},
        type="linear",
        zeroline=False
    )
    )
    fig.update_layout(title=state,showlegend=False,template='plotly_white')
    fig.show()


# In[ ]:


plot_state(states[0])


# In[ ]:


plot_state(states[1])


# In[ ]:


plot_state(states[2])


# In[ ]:


plot_state(states[3])


# In[ ]:


plot_state(states[4])

