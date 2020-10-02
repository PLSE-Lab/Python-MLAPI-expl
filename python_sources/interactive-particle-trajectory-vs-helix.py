#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import numpy as np
import pandas as pd

from trackml.dataset import load_event
from trackml.randomize import shuffle_hits
from trackml.score import score_event



# In[ ]:


event_prefix = 'event000001000'
hits, cells, particles, truth = load_event(os.path.join('../input/train_1', event_prefix))


# ### Particle Trajectory
# Use mouse to move the axis. The following graph reconstructs a single particle trajectory as well as a synthetic trajectory using the right helix equation. The blue trajectory is from TrackML data and the red is from the helix equation
# $$x(t) = \cos(t),\,$$
# $$y(t) = \sin(t),\,$$
# $$z(t) = t.\,$$

# In[ ]:


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

init_notebook_mode(connected=True)

particle = particles.loc[particles.nhits == particles.nhits.max()].iloc[0]

p_traj = hits[truth.particle_id == particle.particle_id][['x', 'y', 'z']]

t = np.asarray(range(10))
x = 400*np.cos(t*3.1415926536/10)
y = 400*np.sin(t*3.1415926536/10)
z = 20*t


track = go.Scatter3d(
    x=p_traj.x, y=p_traj.y, z=p_traj.z,
    marker=dict(
        size=4,
        color='blue',
        colorscale='Viridis',
    ),
    line=dict(
        color='#1f77b4',
        width=1
    )
)
helix = go.Scatter3d(
    x=x, y=y, z=z,
    marker=dict(
        size=4,
        color='red',
        colorscale='Viridis',
    ),
    line=dict(
        color='#1f77b4',
        width=1
    )
)
data = [track,helix]

layout = dict(
    width=800,
    height=700,
    autosize=False,
    title='TrackML',
    scene=dict(
        xaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        yaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        zaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        camera=dict(
            up=dict(
                x=0,
                y=0,
                z=1
            ),
            eye=dict(
                x=-1.7428,
                y=1.0707,
                z=0.7100,
            )
        ),
        aspectratio = dict( x=1, y=1, z=0.7 ),
        aspectmode = 'manual'
    ),
)

fig = dict(data=data, layout=layout)

iplot(fig, validate=False)


# In[ ]:




