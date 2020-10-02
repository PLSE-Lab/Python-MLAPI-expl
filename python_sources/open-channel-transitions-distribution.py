#!/usr/bin/env python
# coding: utf-8

# # How open each open channel transition happens
# 
# **If my understand is correct, the open channel follows markovian process which means the previous state is important. I'd like to simply show in this notebook that how frequent each open channel transition (i.e. Open Channel 0 ==> Open Channel 1; denoted as '0-1' in this notebook) happens. Another notebook using Bokeh library for EDA is [here](https://www.kaggle.com/shinsei66/useful-eda-dashboard-by-bokeh). If you find useful, please upvote ;)**

# # Import Libraries

# In[ ]:


import os, time, sys, gc, pickle, warnings
from tqdm.notebook import tqdm as tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#wave analysis
import pywt
import librosa
from statsmodels.robust import mad
#import statsmodels.api as sm
import scipy
from scipy import stats
from scipy.stats.kde import gaussian_kde
from scipy import signal
from scipy.signal import hann, hilbert, convolve, butter, deconvolve
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import gridplot, column, layout, row
from bokeh.io import output_notebook, curdoc, push_notebook
from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter, Select, CustomJS, HoverTool
from bokeh.models import ColumnDataSource, Label, LabelSet, Range1d
import colorcet as cc
#import lightgbm as lgb
#import xgboost as xgb
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')
from tsfresh.feature_extraction import feature_calculators


# In[ ]:


output_notebook()


# # Import Data

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_cell_magic('time', '', "INPUTDIR = '/kaggle/input/data-without-drift/'\nINPUTDIR2 = '/kaggle/input/liverpool-ion-switching/'\nNROWS = None\ndf_train = pd.read_csv(f'{INPUTDIR}/train_clean.csv', nrows=NROWS, dtype={'time':np.float32, 'signal':np.float32})\ndf_test = pd.read_csv(f'{INPUTDIR}/test_clean.csv', nrows=NROWS, dtype={'time':np.float32, 'signal':np.float32})\nsub_df = pd.read_csv(f'{INPUTDIR2}/sample_submission.csv', nrows=NROWS)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_train['open_channels_lag_1'] = df_train['open_channels'].shift(1).fillna(method='bfill')\ndf_train['transition'] =  df_train['open_channels_lag_1'].astype(int).astype(str) + '-'  + df_train['open_channels'].astype(str)\ndf_train.head()")


# In[ ]:


df_train_agg = df_train.groupby('transition')['signal'].count()


# In[ ]:


list(df_train_agg.index)


# # Event Count for Each Transitions

# In[ ]:


df_train.index = ((df_train.time * 10_000) - 1).values
df_test.index = ((df_test.time * 10_000) - 1).values
df_train['GRP'] = 1+(df_train.index // 50_0000)
df_train['GRP'] = df_train['GRP'].astype('int16')
df_test.index = ((df_test.time * 10_000)).values-1
df_test['GRP'] = 1+(df_test.index // 50_0000)
df_test['GRP'] = df_test['GRP'].astype('int16')


# In[ ]:


labels = transitions = df_train.groupby('transition')['signal'].count().index
tm1 = sorted(set([int(i.split('-')[0]) for i in transitions]))
labels = [[ str(x)+'-'+str(y) for y in           sorted(set([int(i.split('-')[1]) for i in transitions if int(i.split('-')[0]) == x]))]            for x in tm1]


# ## All Batches

# In[ ]:


for label in labels:
    counts = [pd.DataFrame(df_train_agg).loc[i,'signal'] for i in label]
    plot_size_and_tools = {'plot_height': 300, 'plot_width': 1800,
                            'tools':['box_zoom', 'reset', 'crosshair','help']}
    p = figure(x_range=label, plot_height=250, title=f"Transition Counts {label}",
               toolbar_location=None, tools="")
    p.vbar(x=label, top=counts, width=0.9)
    p.add_tools(HoverTool())
    p.xgrid.grid_line_color = None
    p.y_range.start = 0

    show(p)


# # Findings
# - The open channel is more likely to stay same state.
# - The transtion distribution histgrams have gaussian shape, but some of them are slightly skewed.
# - The larger the open channel is, the more transition to other states will happen.
# - The maximam open channel transition gap is 6 (i.e. '9-3', '8-2')
# - The weird thing is that there is neither '2-9' transition nor '3-9' transition, anything wrong with me? 

# ## By Batches

# In[ ]:


for b in range(10):
    BATCH = b+1
    transitions = df_train.query(f'GRP=={BATCH}').groupby('transition')['signal'].count().index
    tm1 = sorted(set([int(i.split('-')[0]) for i in transitions]))
    labels = [[ str(x)+'-'+str(y) for y in               sorted(set([int(i.split('-')[1]) for i in transitions if int(i.split('-')[0]) == x]))]                for x in tm1]
    for label in labels:
        counts = [pd.DataFrame(df_train.query(f'GRP=={BATCH}').groupby('transition')['signal'].count()).loc[i,'signal'] for i in label]
        plot_size_and_tools = {'plot_height': 300, 'plot_width': 1800,
                                'tools':['box_zoom', 'reset', 'crosshair','help']}
        p = figure(x_range=label, plot_height=250, title=f"Transition Counts {label} in Batch #{BATCH}",
                   toolbar_location=None, tools="")
        p.vbar(x=label, top=counts, width=0.9)
        p.add_tools(HoverTool())
        p.xgrid.grid_line_color = None
        p.y_range.start = 0

        show(p)


# # Findings
# - There are transitions only between open channels 0 and 1 for batch 1, 2, 3 and 7 except a few outliers 
# - There are not transition from open channel 0 for batch 5 and 10

# # The Difference of open channels of a Transtion

# In[ ]:


df_train['transition_diff'] = df_train['transition'].apply(lambda x: int(x.split('-')[1])-int(x.split('-')[0]))
df_train['signal_diff_1'] = df_train['signal'].diff().fillna(method='bfill')
df_train.head()


# In[ ]:


plot_size_and_tools = {'plot_height': 600, 'plot_width': 600,
                                'tools':['box_zoom', 'reset', 'crosshair','help']}
p = figure(title=f"Signal vs. Channel Transition Difference",
                   toolbar_location='right', **plot_size_and_tools)

p.scatter(df_train.signal.values, df_train.transition_diff.values, 
          fill_color='blue', fill_alpha=0.6,
          line_color=None)
p.xaxis.axis_label = 'Signal'
p.yaxis.axis_label = 'Channel Transition Difference'
#p.add_tools(HoverTool())
show(p)


# In[ ]:


plot_size_and_tools = {'plot_height': 600, 'plot_width': 600,
                                'tools':['box_zoom', 'reset', 'crosshair','help']}
p = figure(title=f"Signal Diff vs. Channel Transition Difference",
                   toolbar_location='right', **plot_size_and_tools)

p.scatter(df_train.signal_diff_1.values, df_train.transition_diff.values, 
          fill_color='purple', fill_alpha=0.6,
          line_color=None)
p.xaxis.axis_label = 'Signal Diff'
p.yaxis.axis_label = 'Channel Transition Difference'
#p.add_tools(HoverTool())
show(p)


# In[ ]:


palette = [cc.rainbow[i*15] for i in range(15)]
x = np.linspace(-11,11, 500)
plot_size_and_tools = {'plot_height': 800, 'plot_width': 700,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}
p1 = figure(y_range=(-7, 9),  x_range=(-11, 11), toolbar_location='above', **plot_size_and_tools)
for i in range(13):
    diff = i-6
    pdf = gaussian_kde(df_train.query(f'transition_diff=={diff}')['signal_diff_1'].values)
    y = pdf(x) - 6 +i
    source = ColumnDataSource(data=dict(x=x, y=y))
    p1.patch('x', 'y', line_width=1, alpha = 0.6, color=palette[i], source=source, line_color='black')
p1.xaxis.ticker = FixedTicker(ticks=list(range(-11, 11, 1)))
p1.xaxis.axis_label = 'Signal Diff Lag1'
p1.yaxis.axis_label = 'Channel Transition Difference'
show(p1)


# In[ ]:


diff_dict = {}
for i in range(15):
    diff = i-6
    diff_dict[diff] = df_train.query(f'transition_diff=={diff}')['signal_diff_1'].describe()

Diff = pd.DataFrame(diff_dict)


# In[ ]:


Diff


# # Signal Diffrence for all batches

# In[ ]:


label = list(pd.DataFrame(df_train.groupby('transition_diff')['signal'].count()).index)
counts = [pd.DataFrame(df_train.groupby('transition_diff')['signal'].count()).loc[i,'signal'] for i in label]
plot_size_and_tools = {'plot_height': 300, 'plot_width': 600,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}
p = figure( title=f"Transition Counts {label}",
           toolbar_location='right',  **plot_size_and_tools)
p.vbar(x=label, top=counts, width=0.5)
p.add_tools(HoverTool())
p.xgrid.grid_line_color = None
p.xaxis.ticker = FixedTicker(ticks=list(range(-8, 8, 1)))
p.y_range.start = 0

show(p)


# # Signal Diffrence for by batches

# In[ ]:


for b in range(10):
    BATCH = b+1
    label = list(pd.DataFrame(df_train.query(f'GRP=={BATCH}').groupby('transition_diff')['signal'].count()).index)
    counts = [pd.DataFrame(df_train.query(f'GRP=={BATCH}').groupby('transition_diff')['signal'].count()).loc[i,'signal'] for i in label]
    plot_size_and_tools = {'plot_height': 300, 'plot_width': 600,
                            'tools':['box_zoom', 'reset', 'crosshair','help']}
    p = figure( title=f"Transition Counts {label} in Batch #{BATCH}",
               toolbar_location='right',  **plot_size_and_tools)
    p.vbar(x=label, top=counts, width=0.5)
    p.add_tools(HoverTool())
    p.xgrid.grid_line_color = None
    p.xaxis.ticker = FixedTicker(ticks=list(range(-8, 8, 1)))
    p.y_range.start = 0

    show(p)

