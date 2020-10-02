#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The aim of this set of three notebooks is to export a graph for each one of the 1449 buildings in PNG format:
# 
# - [ASHRAE - PNG graphs for buildings 0-482](https://www.kaggle.com/juanmah/ashrae-png-graphs-for-buildings-0-482)
# - [ASHRAE - PNG graphs for buildings 483-965](https://www.kaggle.com/juanmah/ashrae-png-graphs-for-buildings-483-965)
# - [ASHRAE - PNG graphs for buildings 966-1448](https://www.kaggle.com/juanmah/ashrae-png-graphs-for-buildings-966-1448)
# 
# This notebook can also be configured to export the graphs to SVG format. The format to export can be changed in the EXPORT_SVG and EXPORT_PNG constants in the next cell.
# 
# There are two methods to export the graph, depending if it is used in edition mode or committing mode:
# 
# - `orca` for commits. It saves the graphs into files. The files can be downloaded in the 'Output Files' section, clicking on the 'Download All' button.
# - `iplot` for edition mode. It downloads the graphs into the computer.
# 
# The method can be changed in the METHOD constant in the next cell.

# In[ ]:


EXPORT_PNG = True
EXPORT_SVG = False
METHOD = 'orca' # 'iplot', 'orca'

f = open('README.md', 'a')
f.write('If all the saved files are images, the output section of the notebook is shown only as *Output Visualizations*.')
f.write('')
f.write('This file is created to have the possibility of download all files in a zip file.')
f.write('')
f.write('The image files can be downloaded in the *Output Files* section, clicking on the *Download All* button, despite not being shown in the output files list.')
f.close()


# In[ ]:


if METHOD == 'orca':
    get_ipython().system('conda install -y -c plotly plotly-orca')
    get_ipython().system('/usr/bin/apt-get --yes install libxss1 libgconf2-4')


# In[ ]:


import numpy as np
import pandas as pd
import pickle
import re

import plotly.graph_objects as go
from ipywidgets import widgets
from plotly.offline import iplot
from tqdm.auto import trange


# In[ ]:


data_path = '../input/ashrae-data-wrangling-csv-to-pickle/'
with open(data_path + 'X_train.pickle', 'rb') as f:
    X_train = pickle.load(f)
with open(data_path + 'nan.pickle', 'rb') as f:
    nan = pickle.load(f)    
with open(data_path + 'building_metadata.pickle', 'rb') as f:
    building_metadata = pickle.load(f)
with open(data_path + 'weather_train.pickle', 'rb') as f:
    weather_train = pickle.load(f)    


# In[ ]:


def get_data(building_id, meter, xy):
    return X_train[(X_train['building_id']==str(building_id)) &                   (X_train['meter']==meter)][xy]

electricity = go.Scatter(x=[],
                         y=[],
                         line=dict(
                             width=2
                         ),
                         name='Electricity')

hotwater = go.Scatter(x=[],
                      y=[],
                     line=dict(
                         width=2
                     ),
                      name='Hot water')

chilledwater = go.Scatter(x=[],
                          y=[],
                         line=dict(
                             width=2
                         ),
                          name='Chilled water')

steam = go.Scatter(x=[],
                   y=[],
                 line=dict(
                     width=2
                 ),
                   name='Steam')

g = go.FigureWidget(data=[electricity, hotwater, chilledwater, steam],
                    layout=go.Layout(
                        showlegend=True,
                        height=800,
                        xaxis=dict(
                            dtick='M1'
                        ),
                        yaxis=dict(
                            ticksuffix=' '
                        ),
                        margin=dict(
                            l=100,
                            r=300
                        ),
                        font=dict(
                            family='droid sans',
                            size=32
                        )
                    )
                   )
display(g)
    
for building_id in range(966, 1449):
    electricity_ts = get_data(building_id, 'electricity', 'timestamp')
    electricity_reading = get_data(building_id, 'electricity', 'meter_reading')
    hotwater_ts = get_data(building_id, 'hotwater', 'timestamp')
    hotwater_reading = get_data(building_id, 'hotwater', 'meter_reading')
    chilledwater_ts = get_data(building_id, 'chilledwater', 'timestamp')
    chilledwater_reading = get_data(building_id, 'chilledwater', 'meter_reading')
    steam_ts = get_data(building_id, 'steam', 'timestamp')
    steam_reading = get_data(building_id, 'steam', 'meter_reading')
    with g.batch_update():
        g.layout.title.text = f'Energy for building {building_id}'
        g.data[0].x = electricity_ts
        g.data[0].y = electricity_reading
        g.data[1].x = hotwater_ts
        g.data[1].y = hotwater_reading
        g.data[2].x = chilledwater_ts
        g.data[2].y = chilledwater_reading
        g.data[3].x = steam_ts
        g.data[3].y = steam_reading
    zero_nan = pd.DataFrame(columns=['Energy aspect', 'Zero count', 'NaN count'])
    energy_aspects = []
    if len(electricity_ts) > 0:
        energy_aspects.append('electricity')
        zero_nan = zero_nan.append({'Energy aspect': 'Electricity',
                                    'Zero count': (electricity_reading == 0).sum(),
                                    'NaN count': 366 * 24 - len(electricity_ts)},
                                   ignore_index=True)
    if len(hotwater_ts) > 0:
        energy_aspects.append('hotwater')
        zero_nan = zero_nan.append({'Energy aspect': 'Hot water',
                                    'Zero count': (hotwater_reading == 0).sum(),
                                    'NaN count': 366 * 24 - len(hotwater_ts)},
                                   ignore_index=True)
    if len(chilledwater_ts) > 0:
        energy_aspects.append('chilledwater')
        zero_nan = zero_nan.append({'Energy aspect': 'Chilled water ',
                                    'Zero count': (chilledwater_reading == 0).sum(),
                                    'NaN count': 366 * 24 - len(chilledwater_ts)},
                                   ignore_index=True)
    if len(steam_ts) > 0:
        energy_aspects.append('steam')
        zero_nan = zero_nan.append({'Energy aspect': 'Steam ',
                                    'Zero count': (steam_reading == 0).sum(),
                                    'NaN count': 366 * 24 - len(steam_ts)},
                                   ignore_index=True)
#     results.value = f"{zero_nan.style.hide_index().set_table_attributes('class=''table''').render()}"    
#     display(g)
    filename = f"building_id-{building_id:0>4}-site_id-{building_metadata['site_id'][building_id]:0>2}-{re.sub('/', '-', building_metadata['primary_use'].unique()[1])}-{'-'.join(energy_aspects)}"
    if METHOD == 'iplot':
        if EXPORT_SVG:
            iplot(g, image='svg', filename=filename, image_width=3840, image_height=2160)
        if EXPORT_PNG:
            iplot(g, image='png', filename=filename, image_width=3840, image_height=2160)
    if METHOD == 'orca':
        if EXPORT_SVG:
            g.write_image(filename + '.svg', width=3840, height=2160)
        if EXPORT_PNG:
            g.write_image(filename + '.png', width=3840, height=2160)

