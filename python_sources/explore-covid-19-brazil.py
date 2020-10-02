#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


file_path = "/kaggle/input/corona-virus-brazil/brazil_covid19.csv"

df = (pd
 .read_csv(file_path, parse_dates = [['date', 'hour']])
# .set_index('date_hour')
)


# In[ ]:


from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


# In[ ]:


ALL = 'ALL'
def unique_sorted_values_plus_ALL(array):
    unique = array.unique().tolist()
    unique.sort()
    unique.insert(0, ALL)
    return unique


# ## Plotting Data

# Define Callbacks:

# In[ ]:


output = widgets.Output()
plot_output = widgets.Output()

dd_state = widgets.Dropdown(options=unique_sorted_values_plus_ALL(df.state), description='State')
dd_cases = widgets.Dropdown(options=unique_sorted_values_plus_ALL(df.cases), description='Cases')
bounded_num = widgets.BoundedFloatText(min=0, max=10e5, value=5, step=1, description='Numbers')

def colour_ge_value(value, comparison):
    if value >= comparison:
        return 'color: red'
    else:
        return 'color: black'

def common_filtering(state, cases, num):
    output.clear_output()
    plot_output.clear_output()

    if (state == ALL) & (cases == ALL):
        common_filter = df
    elif (state == ALL):
        common_filter = df[df['cases']>=cases]
    elif (cases == ALL):
        common_filter = df[df['state']=="%s" %state]
    else:
        common_filter = df[(df['cases']>=cases) & (df['state']=="%s" %state)]
        
    with output:
        display(common_filter.style.applymap(lambda x: colour_ge_value(x, num),subset=['suspects']))
        
    with plot_output:
        common_filter.set_index(['date_hour']).plot(style='o')
        
        plt.semilogy()
        plt.xlabel('Date')
        plt.ylabel('Log Number of People')
        plt.title('COVID-19 evolution over time in the state of %s' %dd_state.value)
        
        plt.show()
        
def dd_state_eventhandler(change):
    common_filtering(change.new, dd_cases.value, bounded_num.value)
    
def dd_cases_eventhandler(change):
    common_filtering(dd_state.value, change.new, bounded_num.value)
    
def bounded_num_eventhandler(change):
    common_filtering(dd_state.value, dd_cases.value, change.new)
    
dd_state.observe(dd_state_eventhandler, names='value')
dd_cases.observe(dd_cases_eventhandler, names='value')
bounded_num.observe(bounded_num_eventhandler, names='value')

#display(input_widgets)


# Define Layout:

# In[ ]:


item_layout = widgets.Layout(margin='0 0 50px 0')

input_widgets = widgets.HBox([dd_state, dd_cases, bounded_num], layout = item_layout)

tab = widgets.Tab([output, plot_output], layout = item_layout)
tab.set_title(0, 'Dataset Exploration')
tab.set_title(1, 'Plot Data')

# display(tab)


# Bring it all toghether:

# In[ ]:


dashboard = widgets.VBox([input_widgets, tab])
display(dashboard)


# Step-by-step of the design of the simple dashboard.
# Credits: https://towardsdatascience.com/bring-your-jupyter-notebook-to-life-with-interactive-widgets-bc12e03f0916 
