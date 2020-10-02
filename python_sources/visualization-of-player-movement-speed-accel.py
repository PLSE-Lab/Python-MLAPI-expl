#!/usr/bin/env python
# coding: utf-8

# The purpose of this notebook is to provide a method for visualizing player movement, focusing on injury plays. The thought is that this may help generate ideas of how to analyze the data.
# 
# An animated plot function using Plotly is provided that shows player movement on the field, with accompanying speed and acceleration subplots.

# In[ ]:


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot
init_notebook_mode(connected=True)

pd.options.mode.chained_assignment = None


# In[ ]:


# Read the input files
playlist = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')
inj = pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv')
trk = pd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv')


# Define a function to filter the data for a single play, and preprocess.
# 
# Note that player movement prior to 'ball_snap' or 'kickoff' is removed.

# In[ ]:


def preprocess_samp_play(samp_play):

    play_dt = trk[trk.PlayKey==samp_play]
    dataset = play_dt.iloc[np.flatnonzero((play_dt.event == 'ball_snap') | (play_dt.event == 'kickoff'))[0]:]
    dataset['event'].ffill(inplace=True)

    # Calculate instantaneous acceleration
    dataset['a'] = (dataset.s - dataset.s.shift(1)) / (dataset.time - dataset.time.shift(1))
    dataset.a.iloc[0] = 0 # At the moment of ball_snap or kickoff, acceleration is likely 0
    
    return dataset


# Define the animated plot function.

# In[ ]:


def plot_injury_play(samp_play, dataset):

    # Define the basic figure consisting in 6 traces: two in the field subplot,
    # 2 in the speed subplot, and 2 in the accel subplot. These will be updated
    # by frames:

    x = dataset.x.values
    y = dataset.y.values
    N = dataset.shape[0] - 1
    title_string = (samp_play+', '+
                   inj[inj.PlayKey==samp_play].BodyPart.values[0]+', '+
                   'M1: '+str(inj[inj.PlayKey==samp_play].DM_M1.values[0])+', '+
                   'M7: '+str(inj[inj.PlayKey==samp_play].DM_M7.values[0])+', '+
                   'M28: '+str(inj[inj.PlayKey==samp_play].DM_M28.values[0])+', '+
                   'M42: '+str(inj[inj.PlayKey==samp_play].DM_M42.values[0])+', '+
                   playlist[playlist.PlayKey==samp_play].RosterPosition.values[0]+', '+
                   str(playlist[playlist.PlayKey==samp_play].StadiumType.values[0])+', '+
                   playlist[playlist.PlayKey==samp_play].FieldType.values[0])

    fig = dict(
        layout = dict(height=400,
            xaxis1 = {'domain': [0.0, 0.75], 'anchor': 'y1', 'range': [0, 120], 'tickmode': 'array',
                      'tickvals': [0, 10, 35, 60, 85, 110, 120],
                      'ticktext': ['End', 'G', '25', '50', '25', 'G', 'End']},
            yaxis1 = {'domain': [0.0, 1], 'anchor': 'x1', 'range': [0, 160/3], 
                      'scaleanchor': 'x1', 'scaleratio': 1, 'tickmode': 'array',
                      'tickvals': [0, 23.583, 29.75, 160/3],
                      'ticktext': ['Side', 'Hash', 'Hash', 'Side']},
            xaxis2 = {'domain': [0.8, 1], 'anchor': 'y2', 'range': [0, N]},
            yaxis2 = {'domain': [0.0, 0.475], 'anchor': 'x2', 'range': [0, 10]},
            xaxis3 = {'domain': [0.8, 1], 'anchor': 'y3', 'range': [0, N],
                      'showticklabels': False},
            yaxis3 = {'domain': [0.525, 1], 'anchor': 'x3', 'range': [-10, 10]},
            title = {'text': title_string, 'y':0.92, 'x':0, 'xanchor': 'left', 'yanchor': 'top',
                     'font': dict(size=12)},
            annotations= [{"x": 0.9, "y": 0.425, "font": {"size": 12}, "text": "Speed",
                           "xref": "paper", "yref": "paper", "xanchor": "center",
                           "yanchor": "bottom", "showarrow": False},
                          {"x": 0.9, "y": 0.95, "font": {"size": 12}, "text": "Accel",
                           "xref": "paper", "yref": "paper", "xanchor": "center",
                           "yanchor": "bottom", "showarrow": False}],
            plot_bgcolor = 'rgba(181, 226, 141, 1)', # https://www.hexcolortool.com/#b5e28d
            margin = {'t': 50, 'b': 50, 'l': 50, 'r': 50},
        ),

        data = [
            {'type': 'scatter', # This trace is identified inside frames as trace 0
             'name': 'f1', 
             'x': x, 
             'y': y, 
             'hoverinfo': 'name+text', 
             'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}},
             'line': {'color': 'rgba(255,79,38,1.000000)'}, 
             'mode': 'lines', 
             'fillcolor': 'rgba(255,79,38,0.600000)', 
             'legendgroup': 'f1',
             'showlegend': False, 
             'xaxis': 'x1', 'yaxis': 'y1'},
            {'type': 'scatter', # This trace is identified inside frames as trace 1
             'name': 'f12', 
             'x': [x[0]],
             'y': [y[0]],
             'mode': 'markers+text',
             'text': dataset.event.iloc[0],
             'textposition': 'middle left' if x[0] >= 60 else 'middle right', #'middle right',
             'showlegend': False,
             'marker': {'size': 10, 'color':'black'},
             'xaxis': 'x1', 'yaxis': 'y1'},
            {'type': 'scatter', # # This trace is identified inside frames as trace 2
             'name': 'f2', 
             'x': list(range(N)), 
             'y': dataset.s, 
             'hoverinfo': 'name+text', 
             'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}},
             'line': {'color': 'rgba(255,79,38,1.000000)'}, 
             'mode': 'lines', 
             'fillcolor': 'rgba(255,79,38,0.600000)', 
             'legendgroup': 'f2',
             'showlegend': False, 
             'xaxis': 'x2', 'yaxis': 'y2'},
            {'type': 'scatter', # This trace is identified inside frames as trace 3
             'name': 'f22', 
             'x': [0],
             'y': [dataset.s.iloc[0]],
             'mode': 'markers',
             'showlegend': False,
             'marker': {'size': 7, 'color':'black'},
             'xaxis': 'x2', 'yaxis': 'y2'},
            {'type': 'scatter', # # This trace is identified inside frames as trace 4
             'name': 'f3', 
             'x': list(range(N)), 
             'y': dataset.a, 
             'hoverinfo': 'name+text', 
             'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}},
             'line': {'color': 'rgba(255,79,38,1.000000)'}, 
             'mode': 'lines', 
             'fillcolor': 'rgba(255,79,38,0.600000)', 
             'legendgroup': 'f2',
             'showlegend': False, 
             'xaxis': 'x3', 'yaxis': 'y3'},
            {'type': 'scatter', # This trace is identified inside frames as trace 5
             'name': 'f33', 
             'x': [0],
             'y': [dataset.a.iloc[0]],
             'mode': 'markers',
             'showlegend': False,
             'marker': {'size': 7, 'color':'black'},
             'xaxis': 'x3', 'yaxis': 'y3'},
        ]


    )


    frames = [dict(name=k,
                   data=[dict(x=x, y=y),
                         dict(x=[x[k]], y=[y[k]], text=dataset.event.iloc[k]),
                         dict(x=list(range(N)), y=dataset.s),
                         dict(x=[k], y=[dataset.s.iloc[k]]),
                         dict(x=list(range(N)), y=dataset.a),
                         dict(x=[k], y=[dataset.a.iloc[k]]),
                       ],
                   traces=[0,1,2,3,4,5]) for k in range(N)]



    updatemenus = [dict(type='buttons',
                        buttons=[dict(label='Play',
                                      method='animate',
                                      args=[[f'{k}' for k in range(N)], 
                                             dict(frame=dict(duration=25, redraw=False), 
                                                  transition=dict(duration=0),
                                                  easing='linear',
                                                  fromcurrent=True,
                                                  mode='immediate'
                                                                     )]),
                                 dict(label='Pause',
                                      method='animate',
                                      args=[[None],
                                            dict(frame=dict(duration=0, redraw=False), 
                                                 transition=dict(duration=0),
                                                 mode='immediate' )])],
                        direction= 'left', 
                        pad=dict(r= 10, t=85), 
                        showactive =True, x= 0.1, y= 0, xanchor= 'right', yanchor= 'top')
                ]



    sliders = [{'yanchor': 'top',
                'xanchor': 'left', 
                'currentvalue': {'font': {'size': 16}, 'prefix': 'Frame: ', 'visible': True, 'xanchor': 'right'},
                'transition': {'duration': 25.0, 'easing': 'linear'},
                'pad': {'b': 10, 't': 50}, 
                'len': 0.9, 'x': 0.1, 'y': 0, 
                'steps': [{'args': [[k], {'frame': {'duration': 25.0, 'easing': 'linear', 'redraw': False},
                                          'transition': {'duration': 0, 'easing': 'linear'}}], 
                           'label': k, 'method': 'animate'} for k in range(N)       
                        ]}]



    fig.update(frames=frames),
    fig['layout'].update(updatemenus=updatemenus,
              sliders=sliders)



    iplot(fig)


# Select a random injury play, and plot.  Note that injury plays with PlayKey=nan are ignored.
# 
# Some notes on the plot:
# * The text that follows the player's marker is the most recent event
# * The default animation speed is very slow.  I found it more useful to press Play, and/or to manually move the slider back and forth to investigate relationships of direction, speed, and acceleration.
# * The plot begins with 'ball_snap' or 'kickoff', thinking no injury will happen before those events...surely!
# * Each 'frame' of the plot is 0.1 seconds.
# * The plotting process takes more time than a typical plot because it is basically a separate plot for each of the many frames (sometimes 100+) that the animation cycles through.
# * The scaling looks great on my laptop, but looks stretched out in the width direction on my desktop.  Not sure how this can be fixed.

# In[ ]:


samp_play = np.random.choice(inj.PlayKey[~inj.PlayKey.isna()])
dataset = preprocess_samp_play(samp_play)
plot_injury_play(samp_play, dataset)


# Hope this is a useful tool. Please let me know of any questions, comments, or suggestions!

# In[ ]:




