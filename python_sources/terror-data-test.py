# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory




terror_data=pd.read_csv('../input/globalterrorismdb_0616dist.csv',encoding='ISO-8859-1',usecols=[0, 1, 2, 3, 8, 11, 13, 14, 35, 84, 100, 103])
terror_data=terror_data.rename(columns={'eventid':'id', 'iyear':'year', 'imonth':'month', 'iday':'day',
             'country_txt':'country', 'provstate':'state', 'targtype1_txt':'target',
             'weaptype1_txt':'weapon', 'nkill':'fatalities', 'nwound':'injuries'})
terror_data['fatalities'] = terror_data['fatalities'].fillna(0).astype(int)
terror_data['injuries'] = terror_data['injuries'].fillna(0).astype(int)
terror_ch=terror_data[(terror_data.country=='China')]
terror_ch['day'][terror_ch.day == 0] = 1
terror_ch['date'] = pd.to_datetime(terror_ch[['day', 'month', 'year']])
terror_ch = terror_ch[['id', 'date', 'year', 'state', 'latitude', 'longitude',
                         'target', 'weapon', 'fatalities', 'injuries']]
terror_ch = terror_ch.sort_values(['fatalities', 'injuries'], ascending = False)
terror_ch = terror_ch.drop_duplicates(['date', 'latitude', 'longitude', 'fatalities'])

terror_ch['text'] = terror_ch['date'].dt.strftime('%B %-d, %Y') + '<br>' +\
                     terror_ch['fatalities'].astype(str) + ' Killed, ' +\
                     terror_ch['injuries'].astype(str) + ' Injured'

fatality=dict(
    type='scattergeo',
    locationmode='China',
    lon=terror_ch[terror_ch.fatalities>0]['longitude'],
    lat=terror_ch[terror_ch.fatalities>0]['latitude'],
    text=terror_ch[terror_ch.fatalities>0]['text'],
    mode='markers',
    name='fatality',
    hoverinfo='text+name',
    marker = dict(
               size = terror_ch[terror_ch.fatalities > 0]['fatalities'] ** 0.255 * 8,
               opacity = 0.95,
               color = 'rgb(240, 140, 45)')

)

injury =dict(
    type='scattergeo',
    locationmode='China',
    lon=terror_ch[terror_ch.injuries >0]['longitude'],
    lat=terror_ch[terror_ch.injuries >0]['latitude'],
    text=terror_ch[terror_ch.injuries >0]['text'],
    mode='markers',
    name='injury ',
    hoverinfo='text+name',
    marker=dict(
        size=(terror_ch[terror_ch.fatalities == 0]['injuries'] + 1) ** 0.245 * 8,
        opacity=0.85,
        color='rgb(20, 150, 187)')
)


layout = dict(
         title = 'Terrorist Attacks by Latitude/Longitude in China (1970-2015)',
         showlegend = True,
         legend = dict(
             x = 0.85, y = 0.4
         ),
         geo = dict(
             scope = 'china',
             projection = dict(type = 'china'),
             showland = True,
             landcolor = 'rgb(250, 250, 250)',
             subunitwidth = 1,
             subunitcolor = 'rgb(217, 217, 217)',
             countrywidth = 1,
             countrycolor = 'rgb(217, 217, 217)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)')
         )

data = [fatality, injury]
figure = dict(data = data, layout = layout)
iplot(figure)















# Any results you write to the current directory are saved as output.