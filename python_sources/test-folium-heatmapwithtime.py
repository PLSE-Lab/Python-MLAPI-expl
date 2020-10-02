#!/usr/bin/env python
# coding: utf-8

# ## This kernel is just to test folium heatmapwithtime

# ## dependencies

#  ## Install branca from github -
# (Keep internet option on in settings for this to work)

# In[ ]:


get_ipython().system('pip install git+https://github.com/python-visualization/branca')


# In[ ]:


import pandas as pd 
import folium
import folium.plugins as plugins
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


folium.branca.__version__


# ## Monkey patch code - 

# In[ ]:


"""
import base64

import folium

def _repr_html_(self, **kwargs):
    html = base64.b64encode(self.render(**kwargs).encode('utf8')).decode('utf8')
    onload = (
        'this.contentDocument.open();'
        'this.contentDocument.write(atob(this.getAttribute(\'data-html\')));'
        'this.contentDocument.close();'
    )
    if self.height is None:
        iframe = (
            '<div style="width:{width};">'
            '<div style="position:relative;width:100%;height:0;padding-bottom:{ratio};">'
            '<iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;'
            'border:none !important;" '
            'data-html={html} onload="{onload}" '
            'allowfullscreen webkitallowfullscreen mozallowfullscreen>'
            '</iframe>'
            '</div></div>').format
        iframe = iframe(html=html, onload=onload, width=self.width, ratio=self.ratio)
    else:
        iframe = ('<iframe src="about:blank" width="{width}" height="{height}"'
                  'style="border:none !important;" '
                  'data-html={html} onload="{onload}" '
                  '"allowfullscreen" "webkitallowfullscreen" "mozallowfullscreen">'
                  '</iframe>').format
        iframe = iframe(html=html, onload=onload, width=self.width, height=self.height)
    return iframe

folium.branca.element.Figure._repr_html_ = _repr_html_
"""


# In[ ]:


folium.branca.__version__


# ## Inpect code from branca -

# In[ ]:


import inspect
inspect.getsource(folium.branca.element.Figure._repr_html_)


# ## Generating data

# In[ ]:



np.random.seed(3141592)
initial_data = (
    np.random.normal(size=(500, 2)) * np.array([[1, 1]]) +
    np.array([[48, 5]])
)

move_data = np.random.normal(size=(500, 2)) * 0.01

data = [(initial_data + move_data * i).tolist() for i in range(1000)]

weight = 1  # default value
for time_entry in data:
    for row in time_entry:
        row.append(weight)


# In[ ]:


len(data)


# ## Map example - HeatMapWithTime

# In[ ]:


m = folium.Map([48., 5.], tiles='stamentoner', zoom_start=6)

hm = plugins.HeatMapWithTime(data)

hm.add_to(m)

m


# ## Including date for index

# In[ ]:




from datetime import datetime, timedelta


time_index = [
    (datetime.now() + k * timedelta(1)).strftime('%Y-%m-%d') for
    k in range(len(data))
]


# In[ ]:


m = folium.Map([48., 5.], tiles='stamentoner', zoom_start=6)

hm = plugins.HeatMapWithTime(
    data,
    index=time_index,
    auto_play=True,
    max_opacity=0.3
)

hm.add_to(m)

m

