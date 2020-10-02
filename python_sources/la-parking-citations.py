#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Define the column types in a dictionary
file_dtypes = {'Ticket number': np.str, 'Issue time': np.float64, 'Meter Id': np.str, 'Marked Time': np.str,
       'RP State Plate': np.str, 'Plate Expiry Date': np.str, 'VIN': np.str, 'Make': np.str, 'Body Style': np.str,
       'Color': np.str, 'Location': np.str, 'Route': np.str, 'Agency': np.float64, 'Violation code': np.str,
       'Violation Description': np.str, 'Fine amount': np.float32, 'Latitude': np.float32, 'Longitude': np.float32}

# pass dictionary to read_csv, parse date for Issue Date
df = pd.read_csv('../input/parking-citations.csv', 
                 dtype=file_dtypes, 
                 parse_dates=['Issue Date'], 
                 index_col=['Issue Date'])

# From http://code.activestate.com/recipes/577305-python-dictionary-of-us-states-and-territories/
STATES = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}


# In[ ]:


df_2016 = df.loc[(df.index>='2016-01-01') & df['RP State Plate'].isin(STATES.keys())].copy()
daily_state = df_2016.groupby(by=[df_2016.index,'RP State Plate']).agg({'Ticket number':np.size})
daily_state.columns = ['Count']
ca = daily_state.loc[daily_state.index.get_level_values('RP State Plate') == 'CA'].reset_index(level=1,drop=True)
non_ca = daily_state.loc[daily_state.index.get_level_values('RP State Plate') != 'CA'].reset_index(level=1,drop=True).reset_index().groupby('Issue Date').sum()

daily_ratio = ca / non_ca
daily_ratio.head()


# In[ ]:


top_state_citations = daily_state.groupby('RP State Plate').sum().sort_values(by='Count', ascending=False)

top = top_state_citations.index[0]
top_rest = top_state_citations.index[1:6]
rest = top_state_citations.index[6:]

top_state_df = daily_state.loc[daily_state.index.get_level_values('RP State Plate')==top].reset_index(level=1,drop=True).rename(columns={'Count':top}).fillna(0).rolling(window=28).mean().round(1)
top_rest_df = daily_state.loc[daily_state.index.get_level_values('RP State Plate').isin(top_rest)].pivot_table(index='Issue Date', columns='RP State Plate')['Count'].fillna(0).rolling(window=28).mean().round(1)
rest_df = daily_state.loc[daily_state.index.get_level_values('RP State Plate').isin(rest)].groupby('Issue Date').sum().rename(columns={'Count':'All Other States'}).fillna(0).rolling(window=28).mean().round(1)


# In[ ]:


# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

trace1 = go.Scatter(
    x=top_state_df.index,
    y=top_state_df.iloc[:,0],
    name=STATES[top_state_df.columns[0]]
)

trace2 = go.Scatter(
    x=rest_df.index,
    y=rest_df.iloc[:,0],
    name=rest_df.columns[0],
    yaxis='y2'
)

trace3 = go.Scatter(
    x=top_rest_df.index,
    y=top_rest_df.iloc[:,0],
    name=STATES[top_rest_df.columns[0]],
    yaxis='y2'
)
trace4 = go.Scatter(
    x=top_rest_df.index,
    y=top_rest_df.iloc[:,1],
    name=STATES[top_rest_df.columns[1]],
    yaxis='y2'
)
trace5 = go.Scatter(
    x=top_rest_df.index,
    y=top_rest_df.iloc[:,2],
    name=STATES[top_rest_df.columns[2]],
    yaxis='y2'
)
trace6 = go.Scatter(
    x=top_rest_df.index,
    y=top_rest_df.iloc[:,3],
    name=STATES[top_rest_df.columns[3]],
    yaxis='y2'
)
trace7 = go.Scatter(
    x=top_rest_df.index,
    y=top_rest_df.iloc[:,4],
    name=STATES[top_rest_df.columns[4]],
    yaxis='y2'
)

data = [trace1, trace3, trace4, trace5, trace6, trace7, trace2]

layout = go.Layout(
    title='Citation Volume by Top 6 States (28d Moving Avg.)',
    yaxis=dict(
        title='{} Citations'.format(STATES[top_state_df.columns[0]])
    ),
    yaxis2=dict(
        title='Other States',
        overlaying='y',
        side='right'
    )
)
fig = go.Figure(data=data, layout=layout)

iplot(fig)


# In[ ]:


# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data = [go.Scatter(x=daily_ratio.index, y=daily_ratio['Count'])]

# specify the layout of our figure
layout = dict(title = "Ratio of Total California Issued Citations vs. Non-California Issued Citations (excl. non-USA)",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


total_by_non_ca = daily_state.loc[daily_state.index.get_level_values('RP State Plate')!='CA'].groupby('RP State Plate').sum()
total_by_non_ca.head()


# In[ ]:


# specify what we want our map to look like
data = [ dict(
        type='choropleth',
        autocolorscale = False,
        locations = total_by_non_ca.index,
        z = total_by_non_ca.Count,
        locationmode = 'USA-states'
)]

# chart information
layout = dict(
        title = 'Total Citations by State (excl. California) (since 2016)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'
        ),
)
   
# actually show our figure
fig = dict(data=data, layout=layout)
iplot(fig)


# In[ ]:


daily_state_2018 = daily_state.loc[daily_state.index.get_level_values('Issue Date')>=(max(daily_state.index.get_level_values('Issue Date')) - pd.to_timedelta(1, unit='M'))].groupby('RP State Plate').sum()
daily_state_2018 = daily_state_2018.loc[daily_state_2018.index!='CA']
daily_state_2018['pct'] = daily_state_2018['Count'] / sum(daily_state_2018['Count'])


# In[ ]:


daily_state_20162017 = daily_state.loc[(daily_state.index.get_level_values('Issue Date') >= (max(daily_state.index.get_level_values('Issue Date')) - pd.to_timedelta(4, unit='M'))) &
                                      (daily_state.index.get_level_values('Issue Date') < (max(daily_state.index.get_level_values('Issue Date')) - pd.to_timedelta(1, unit='M')))].groupby('RP State Plate').sum()
daily_state_20162017 = daily_state_20162017.loc[daily_state_20162017.index!='CA']
daily_state_20162017['pct'] = daily_state_20162017['Count'] / sum(daily_state_20162017['Count'])


# In[ ]:



# specify what we want our map to look like
data = [dict(
        type='choropleth',
        autocolorscale = False,
        locations = (daily_state_2018['pct'] - daily_state_20162017['pct']).index,
        z = (daily_state_2018['pct'] - daily_state_20162017['pct']) * 100,
        locationmode = 'USA-states'
)]

# chart information
layout = dict(
        title = 'Pts. Change Citations by State (excl. California) (Last 30 days vs 3 months prior)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'
        ),
)
   
# actually show our figure
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )


# In[ ]:


df_2018 = df.loc[df.index >= (max(df.index) - pd.to_timedelta(12, unit='M'))].copy()
df_2018.head()


# In[ ]:


df_2018['Issue hour'] = round(df_2018.loc[:, 'Issue time'],-2) / 100


# In[ ]:


hours_violation =     pd.DataFrame(df_2018.loc[df_2018['Violation Description']                .isin(df_2018['Violation Description'].value_counts()[0:9].index)]                .groupby(['Violation Description','Issue hour']).size())                .pivot_table(index='Issue hour',columns='Violation Description', aggfunc='sum')

hours_violation.columns = hours_violation.columns.get_level_values(1)
hours_violation.head()


# In[ ]:


# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode()
# new_list = [expression(i) for i in old_list if filter(i)]

# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data = [go.Scatter(x=hours_violation.index,
                   y=hours_violation[hours_violation.columns[ind]],
                   name=hours_violation.columns[ind]) \
        for ind, col in enumerate(hours_violation.columns)]

# specify the layout of our figure
layout = dict(title = "Top 9 Violation Reasons by Hour (TTM)",
              xaxis= dict(title= 'Hour',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


df_2018['Issue Week'] = df_2018.index - pd.to_timedelta(df_2018.index.dayofweek, unit='d')
df_2018['Issue Day of Week'] = df_2018.index.weekday_name
cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_2018['Issue Day of Week'] = df_2018['Issue Day of Week'].astype(pd.api.types.CategoricalDtype(categories = cats, ordered=True))


# In[ ]:


weekly_violations = df_2018.loc[(df_2018['Violation Description']                .isin(df_2018['Violation Description'].value_counts()[0:9].index))]                .groupby(['Violation Description','Issue Day of Week','Issue hour']).size()

weekly_violations = pd.DataFrame(weekly_violations, columns=['Citations']).pivot_table(index='Issue hour',
                                 columns=['Violation Description','Issue Day of Week'],
                                 aggfunc=np.sum)


# In[ ]:


from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

citations = df_2018['Violation Description'].value_counts()[0:9].index

colors = ['rgb(31, 119, 180)', 
          'rgb(67,67,67)', 
          'rgb(148, 103, 189)', 
          'rgb(49,130,189)', 
          'rgb(214, 39, 40)',
          'rgb(44, 160, 44)',
          'rgb(255, 127, 14)']

## Citation
objs = []
for e, description in enumerate(citations):
    objs.append(
        [go.Scatter(           
            x=weekly_violations.index,
            y=weekly_violations['Citations'][description][day_of_week],
            name=day_of_week,
            showlegend=False if e >= 1 else True,
            legendgroup=day_of_week,
            line = dict(color = (colors[ind]))
        )
            for ind, day_of_week in enumerate(weekly_violations['Citations'][description].columns)]
    )

fig = tools.make_subplots(rows=3, 
                          cols=3, 
                          shared_yaxes=False, 
                          subplot_titles=list(citations),
                          print_grid=False)

fig.append_trace(objs[0][0], 1, 1)
fig.append_trace(objs[0][1], 1, 1)
fig.append_trace(objs[0][2], 1, 1)
fig.append_trace(objs[0][3], 1, 1)
fig.append_trace(objs[0][4], 1, 1)
fig.append_trace(objs[0][5], 1, 1)
fig.append_trace(objs[0][6], 1, 1)

fig.append_trace(objs[1][0], 1, 2)
fig.append_trace(objs[1][1], 1, 2)
fig.append_trace(objs[1][2], 1, 2)
fig.append_trace(objs[1][3], 1, 2)
fig.append_trace(objs[1][4], 1, 2)
fig.append_trace(objs[1][5], 1, 2)
fig.append_trace(objs[1][6], 1, 2)

fig.append_trace(objs[2][0], 1, 3)
fig.append_trace(objs[2][1], 1, 3)
fig.append_trace(objs[2][2], 1, 3)
fig.append_trace(objs[2][3], 1, 3)
fig.append_trace(objs[2][4], 1, 3)
fig.append_trace(objs[2][5], 1, 3)
fig.append_trace(objs[2][6], 1, 3)

fig.append_trace(objs[3][0], 2, 1)
fig.append_trace(objs[3][1], 2, 1)
fig.append_trace(objs[3][2], 2, 1)
fig.append_trace(objs[3][3], 2, 1)
fig.append_trace(objs[3][4], 2, 1)
fig.append_trace(objs[3][5], 2, 1)
fig.append_trace(objs[3][6], 2, 1)

fig.append_trace(objs[4][0], 2, 2)
fig.append_trace(objs[4][1], 2, 2)
fig.append_trace(objs[4][2], 2, 2)
fig.append_trace(objs[4][3], 2, 2)
fig.append_trace(objs[4][4], 2, 2)
fig.append_trace(objs[4][5], 2, 2)
fig.append_trace(objs[4][6], 2, 2)

fig.append_trace(objs[5][0], 2, 3)
fig.append_trace(objs[5][1], 2, 3)
fig.append_trace(objs[5][2], 2, 3)
fig.append_trace(objs[5][3], 2, 3)
fig.append_trace(objs[5][4], 2, 3)
fig.append_trace(objs[5][5], 2, 3)
fig.append_trace(objs[5][6], 2, 3)

fig.append_trace(objs[6][0], 3, 1)
fig.append_trace(objs[6][1], 3, 1)
fig.append_trace(objs[6][2], 3, 1)
fig.append_trace(objs[6][3], 3, 1)
fig.append_trace(objs[6][4], 3, 1)
fig.append_trace(objs[6][5], 3, 1)
fig.append_trace(objs[6][6], 3, 1)

fig.append_trace(objs[7][0], 3, 2)
fig.append_trace(objs[7][1], 3, 2)
fig.append_trace(objs[7][2], 3, 2)
fig.append_trace(objs[7][3], 3, 2)
fig.append_trace(objs[7][4], 3, 2)
fig.append_trace(objs[7][5], 3, 2)
fig.append_trace(objs[7][6], 3, 2)

fig.append_trace(objs[8][0], 3, 3)
fig.append_trace(objs[8][1], 3, 3)
fig.append_trace(objs[8][2], 3, 3)
fig.append_trace(objs[8][3], 3, 3)
fig.append_trace(objs[8][4], 3, 3)
fig.append_trace(objs[8][5], 3, 3)
fig.append_trace(objs[8][6], 3, 3)

xaxis_title = 'Hour of Day'
yaxis_title = 'Citations'
fig['layout']['xaxis1'].update(title=xaxis_title)
fig['layout']['yaxis1'].update(title=yaxis_title)
fig['layout']['xaxis2'].update(title=xaxis_title)
fig['layout']['xaxis3'].update(title=xaxis_title)
fig['layout']['xaxis4'].update(title=xaxis_title)
fig['layout']['yaxis4'].update(title=yaxis_title)
fig['layout']['xaxis5'].update(title=xaxis_title)
fig['layout']['xaxis6'].update(title=xaxis_title)
fig['layout']['xaxis7'].update(title=xaxis_title)
fig['layout']['yaxis7'].update(title=yaxis_title)
fig['layout']['xaxis8'].update(title=xaxis_title)
fig['layout']['xaxis9'].update(title=xaxis_title)

fig['layout'].update(height=900, width=900, title='Top Citation Violation Descriptions' +
                                                   ' (TTM)')

iplot(fig)


# In[ ]:


temp = df_2018.groupby(by=['Issue Date','Issue hour','Issue Day of Week']).size()
temp = pd.DataFrame(temp)
temp = temp.pivot_table(index=['Issue Day of Week','Issue hour'],aggfunc=[np.mean])['mean']


# In[ ]:


l7d = df_2018[['Issue hour','Issue Day of Week']].loc[df_2018.index >= (max(df_2018.index) - pd.to_timedelta(6, unit='d'))]
l7d = l7d.groupby(by=['Issue hour','Issue Day of Week']).size()
l7d = pd.DataFrame(l7d)


# In[ ]:


merged = temp.merge(l7d, how='left', left_on=['Issue hour','Issue Day of Week'], right_on=['Issue hour','Issue Day of Week'])
merged = merged.pivot_table(columns='Issue Day of Week', index='Issue hour')
merged = merged.reorder_levels([1,0],axis=1)
merged = merged.rename(columns={'0_x':'TTM Avg.', '0_y':'Current'})


# In[ ]:


from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

metrics = ['Current', 'TTM Avg.'][::-1]
last_7dates = df_2018.loc[df_2018.index >= (max(df_2018.index) - pd.to_timedelta(6, unit='d'))]              .index.unique()[::-1]
weekdays = last_7dates.weekday_name

objs = []
ind = 0

colors = ['rgb(67,67,67)','rgb(0, 215, 180)'][::-1]

for day, weekday in zip(last_7dates.date, weekdays):
    for color, metric in enumerate(metrics):
        objs.append(
            go.Scatter(           
                x=merged.index,
                y=merged[weekday][metric],
                name=metric,
                showlegend=False if ind > 0 else True,
                legendgroup=metric,
                line = dict(color = (colors[color]))
            )
        )
    ind+=1

fig = tools.make_subplots(
    rows=2, 
    cols=4, 
    shared_yaxes=False, 
    subplot_titles=[(a[0] + ' ' + a[1]) for a in \
                  zip([str(date) for date in last_7dates.date], weekdays)],
    print_grid=False
)

fig.append_trace(objs[0], 1, 1)
fig.append_trace(objs[1], 1, 1)

fig.append_trace(objs[2], 1, 2)
fig.append_trace(objs[3], 1, 2)

fig.append_trace(objs[4], 1, 3)
fig.append_trace(objs[5], 1, 3)

fig.append_trace(objs[6], 1, 4)
fig.append_trace(objs[7], 1, 4)

fig.append_trace(objs[8], 2, 1)
fig.append_trace(objs[9], 2, 1)

fig.append_trace(objs[10], 2, 2)
fig.append_trace(objs[11], 2, 2)

fig.append_trace(objs[12], 2, 3)
fig.append_trace(objs[13], 2, 3)

fig['layout'].update(height=500, width=900, title='Citations in Last 7 Days')

for i in fig['layout']['annotations']:
    i['font'] = dict(size=10,color='#000000')

iplot(fig)


# In[ ]:


from pyproj import Proj, transform

# x1,y1 = 6439997.9,1802686.4
# inProj = Proj(init='epsg:2229', preserve_units=True)
# outProj = Proj(init='epsg:4326')
# LONGITUDE,LATITUDE = transform(inProj,outProj,x1,y1)
# print(LATITUDE,',',LONGITUDE)

inProj = Proj(init='epsg:2229', preserve_units=True)
outProj = Proj(init='epsg:4326')

def df_transform(df):
    x, y = transform(inProj, outProj, df['Latitude'], df['Longitude'])
    return x, y


# In[ ]:


stateplane = df.loc[df['Latitude']!=99999.0][['Latitude','Longitude']].copy().reset_index(drop=True)
stateplane = stateplane.groupby(by=['Latitude','Longitude']).size()
stateplane = stateplane.reset_index()
stateplane.rename({0:'Count'},inplace=True, axis=1)
stateplane = stateplane.join(pd.DataFrame(stateplane.apply(axis=1, func=df_transform).tolist(),                                           columns=['Lon','Lat'], index=stateplane.index))


# In[ ]:


from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

total_coords = 3000
scl = [0,"rgb(150,0,90)"],[0.125,"rgb(0, 0, 200)"],[0.25,"rgb(0, 25, 255)"],[0.375,"rgb(0, 152, 255)"],[0.5,"rgb(44, 255, 150)"],[0.625,"rgb(151, 255, 0)"],[0.75,"rgb(255, 234, 0)"],[0.875,"rgb(255, 111, 0)"],[1,"rgb(255, 0, 0)"]

data = [ dict(
    lat = stateplane.sort_values(by='Count',ascending=False)['Lat'][0:total_coords],
    lon = stateplane.sort_values(by='Count',ascending=False)['Lon'][0:total_coords],
    text = stateplane.sort_values(by='Count',ascending=False)['Count'][0:total_coords].astype(str) + ' citations',
    marker = dict(
        color = stateplane.sort_values(by='Count',ascending=False)['Count'][0:10],
        colorscale = scl,
        reversescale = True,
        opacity = 0.7,
        size = 2,
        colorbar = dict(
            thickness = 10,
            titleside = "right",
            outlinecolor = "rgba(68, 68, 68, 0)",
            ticks = "outside",
            ticklen = 3,
            showticksuffix = "last",
            ticksuffix = " citations",
            dtick = 0.1
        ),
    ),
    type = 'scattergeo'
) ]

layout = dict(
    geo = dict(
        scope = 'north america',
        showland = True,
        landcolor = "rgb(212, 212, 212)",
        subunitcolor = "rgb(255, 255, 255)",
        countrycolor = "rgb(255, 255, 255)",
        showlakes = False,
        lakecolor = "rgb(255, 255, 255)",
        showsubunits = True,
        showcountries = True,
        resolution = 50,
        projection = dict(
            type = 'conic conformal',
            rotation = dict(
                lon = -100
            )
        ),
        
#[[[-118.8947883188,33.4539248384],[-117.3633873133,33.4539248384],[-117.3633873133,34.4545130192],[-118.8947883188,34.4545130192],[-118.8947883188,33.4539248384]]]        
        lonaxis = dict(
            showgrid = True,
            gridwidth = 0.5,
            range= [ -118.895, -117.363 ],
            dtick = 5
        ),
        lataxis = dict (
            showgrid = True,
            gridwidth = 0.1,
            range= [ 33.454, 34.454 ],
            dtick = 5
        )
    ),
    title = 'LA Citations (Top {} citations)'.format(total_coords),
)
fig = { 'data':data, 'layout':layout }
iplot(fig)


# In[ ]:


stateplane.to_csv('stateplane.csv')

