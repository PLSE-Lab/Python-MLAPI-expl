#!/usr/bin/env python
# coding: utf-8

# ### In order to see detail signal of the data, I'd like to demonstrate a useful dashboard by [Bokeh Library](https://docs.bokeh.org/en/latest/index.html). You can directly panning, zooming in a single graph or make these actions in the graph linked, which enables EDA more efficient. It takes about 2~30 sec to load all the graphs since bokeh graphs keeps data within its graphs so please keep patient.

# # Import Libraries

# In[ ]:


import os, time, sys, gc
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import seaborn as sns
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import gridplot, column, layout, row
from bokeh.io import output_notebook, curdoc, push_notebook
from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter, Select, CustomJS, HoverTool
from scipy.stats.kde import gaussian_kde
import colorcet as cc
#wave analysis
import pywt 
from statsmodels.robust import mad
import statsmodels.api as sm
import scipy
from scipy import stats 
from scipy import signal
from scipy.signal import hann, hilbert, convolve, butter, deconvolve


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
INPUTDIR = '/kaggle/input/liverpool-ion-switching/'
INPUTDIR2 = '/kaggle/input/data-without-drift/'
NROWS=None


# ## Import Data

# In[ ]:


get_ipython().run_cell_magic('time', '', "df_train = pd.read_csv(f'{INPUTDIR}/train.csv', nrows=NROWS, dtype={'time':np.float32, 'signal':np.float32})\ndf_test = pd.read_csv(f'{INPUTDIR}/test.csv', nrows=NROWS, dtype={'time':np.float32, 'signal':np.float32})\ndf_train_clean = pd.read_csv(f'{INPUTDIR2}/train_clean.csv', nrows=NROWS, dtype={'time':np.float32, 'signal':np.float32})\ndf_test_clean = pd.read_csv(f'{INPUTDIR2}/test_clean.csv', nrows=NROWS, dtype={'time':np.float32, 'signal':np.float32})\nsub_df = pd.read_csv(f'{INPUTDIR}/sample_submission.csv', nrows=NROWS)")


# In[ ]:


print(df_train.columns)
print(df_test.columns)
print(df_train_clean.columns)
print(df_test_clean.columns)
print(df_train.shape)
print(df_test.shape)


# # EDA

# In[ ]:


output_notebook()


# # Train Data

# ### Signal and open channels

# #### The Original Data

# In[ ]:


start = 0
end = len(df_train)
res = int(end/1000000)
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    y0=df_train['signal'].values[start:end:res],
                                    y1=df_train['open_channels'].values[start:end:res]))

p1 = figure(title='Train Signal',**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'y0', line_width=1, color = 'red', source=source)
p1.yaxis.axis_label = 'Signal Strength'
p2 = figure(title='Open_Channels',**plot_size_and_tools, x_range=p1.x_range, y_range=p1.y_range)
p2.add_tools(HoverTool())
p2.line('x', 'y1', line_width=1, color = 'black', source=source)
p2.xaxis.axis_label = 'Time'
p2.yaxis.axis_label = 'Opened Channel'
p = gridplot([p1, p2],ncols=1, toolbar_location='right')
show(p)


# In[ ]:


gc.collect()


# #### Data without Drift

# In[ ]:


start = 0
end = len(df_train)
res = int(end/1000000)
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    y0=df_train_clean['signal'].values[start:end:res],
                                    y1=df_train_clean['open_channels'].values[start:end:res]))

p1 = figure(title='Train Signal(w/o drift)',**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'y0', line_width=1, color = 'orange', source=source)
p1.yaxis.axis_label = 'Signal Strength'
p2 = figure(title='Open_Channels',**plot_size_and_tools, x_range=p1.x_range, y_range=p1.y_range)
p2.add_tools(HoverTool())
p2.line('x', 'y1', line_width=1, color = 'black', source=source)
p2.xaxis.axis_label = 'Time'
p2.yaxis.axis_label = 'Opened Channel'
p = gridplot([p1, p2],ncols=1, toolbar_location='right')
show(p)


# In[ ]:


def low_pass_filter(x, high_cutoff, SAMPLE_RATE):
    """
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
    Modified to work with scipy version 1.1.0 which does not have the fs parameter
    """
    
    # nyquist frequency is half the sample rate https://en.wikipedia.org/wiki/Nyquist_frequenc
    nyquist = 0.5 * SAMPLE_RATE
    norm_high_cutoff = high_cutoff / nyquist
    
    # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
    sos = butter(10, Wn=[norm_high_cutoff], btype='lowpass', output='sos')
    filtered_sig = signal.sosfilt(sos, x)

    return filtered_sig


# In[ ]:


start = 3640000
end = 3830000
res = int(len(df_train)/1000000)
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    y0=df_train_clean['signal'].values[start:end:res] - df_train_clean['signal'].shift(1).values[start:end:res],
                                    y1=df_train_clean['open_channels'].values[start:end:res] ))

p1 = figure(title='Train Signal(w/o drift)',**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'y0', line_width=1, color = 'orange', source=source)
p1.yaxis.axis_label = 'Signal Strength'
p2 = figure(title='Open_Channels',**plot_size_and_tools, x_range=p1.x_range, y_range=p1.y_range)
p2.add_tools(HoverTool())
p2.line('x', 'y1', line_width=1, color = 'black', source=source)
p2.xaxis.axis_label = 'Time'
p2.yaxis.axis_label = 'Opened Channel'
p = gridplot([p1, p2],ncols=1, toolbar_location='right')
show(p)


# In[ ]:


start = 3640000
end = 3830000
res = int(len(df_train)/1000000)
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    y0= np.abs(np.fft.fft(df_train_clean['signal'].values[start:end:res] - df_train_clean['signal'].shift(1).values[start:end:res])),
                                    y1=df_train_clean['open_channels'].values[start:end:res] ))

p1 = figure(title='Train Signal(w/o drift)',**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'y0', line_width=1, color = 'orange', source=source)
p2.xaxis.axis_label = 'Freq'
p1.yaxis.axis_label = 'FFT amplitude'
p2 = figure(title='Open_Channels',**plot_size_and_tools, x_range=p1.x_range, )
p2.add_tools(HoverTool())
p2.line('x', 'y1', line_width=1, color = 'black', source=source)
p2.xaxis.axis_label = 'Time'
p2.yaxis.axis_label = 'Opened Channel'
p = gridplot([p1, p2],ncols=1, toolbar_location='right')
show(p)


# In[ ]:


start = 3640000
end = 3830000
res = int(len(df_train)/1000000)
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    y0= low_pass_filter(df_train_clean['signal'].values[start:end:res] - df_train_clean['signal'].shift(1).values[start:end:res],1000 ,40000),
                                    y1=df_train_clean['open_channels'].values[start:end:res] ))

p1 = figure(title='Train Signal(w/o drift)',**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'y0', line_width=1, color = 'orange', source=source)
p2.xaxis.axis_label = 'Freq'
p1.yaxis.axis_label = 'FFT amplitude'
p2 = figure(title='Open_Channels',**plot_size_and_tools, x_range=p1.x_range, )
p2.add_tools(HoverTool())
p2.line('x', 'y1', line_width=1, color = 'black', source=source)
p2.xaxis.axis_label = 'Time'
p2.yaxis.axis_label = 'Opened Channel'
p = gridplot([p1, p2],ncols=1, toolbar_location='right')
show(p)


# In[ ]:


start = 0
end = len(df_train)
res = int(len(df_train)/1000000)
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    y0= low_pass_filter(df_train_clean['signal'].values[start:end:res] - df_train_clean['signal'].shift(1).fillna(0).values[start:end:res],500 ,1000000),
                                    y1=df_train_clean['open_channels'].values[start:end:res] ))

p1 = figure(title='Train Signal(w/o drift)',**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'y0', line_width=1, color = 'orange', source=source)
p2.xaxis.axis_label = 'Freq'
p1.yaxis.axis_label = 'FFT amplitude'
p2 = figure(title='Open_Channels',**plot_size_and_tools, x_range=p1.x_range, )
p2.add_tools(HoverTool())
p2.line('x', 'y1', line_width=1, color = 'black', source=source)
p2.xaxis.axis_label = 'Time'
p2.yaxis.axis_label = 'Opened Channel'
p = gridplot([p1, p2],ncols=1, toolbar_location='right')
show(p)


# ### KDE plots signal vs opened channel

# In[ ]:


df_train.index = ((df_train.time * 10_000)).values-1
df_train['batch'] = 1+(df_train.index // 50_0000)
df_train['batch'] = df_train['batch'].astype('int16')
df_train['sub_batch'] = 1 + (df_train.index  // 1_000).astype('int32')
df_train['batch2'] = df_train['batch'].copy()
df_train.loc[df_train[(df_train.index >= 600000)&(df_train.index < 1000000)].index, 'batch2'] = 11
df_train.tail(2)


# In[ ]:


df_train_clean.index = ((df_train_clean.time * 10_000)).values-1
df_train_clean['batch'] = 1+(df_train_clean.index // 50_0000)
df_train_clean['batch'] = df_train_clean['batch'].astype('int16')
df_train_clean['sub_batch'] = 1 + (df_train_clean.index  // 1_000).astype('int32')
df_train_clean['batch2'] = df_train_clean['batch'].copy()
df_train_clean.loc[df_train_clean[(df_train_clean.index >= 600000)&(df_train_clean.index < 1000000)].index, 'batch2'] = 11
df_train_clean.tail(2)


# In[ ]:


palette = [cc.rainbow[i*15] for i in range(13)]
x = np.linspace(-6,15, 500)
plot_size_and_tools = {'plot_height': 800, 'plot_width': 700,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}
p1 = figure(y_range=(-1, 12),  x_range=(-6, 15), toolbar_location='above', **plot_size_and_tools)
for i in range(11):
    pdf = gaussian_kde(df_train[(df_train['open_channels']==i)]['signal'].values)
    y = pdf(x) + i
    source = ColumnDataSource(data=dict(x=x, y=y))
    p1.patch('x', 'y', line_width=1, alpha = 0.6, color=palette[i], source=source, line_color='black')
p1.xaxis.ticker = FixedTicker(ticks=list(range(-5, 15, 1)))
p1.xaxis.axis_label = 'Signal Strength'
p1.yaxis.axis_label = 'Opened Channel'
show(p1)


# In[ ]:


palette = [cc.rainbow[i*15] for i in range(13)]
x = np.linspace(-6,15, 500)
plot_size_and_tools = {'plot_height': 800, 'plot_width': 700,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}
p1 = figure(y_range=(0, 11),  x_range=(-6, 15), toolbar_location='above', **plot_size_and_tools)
for i in range(10):
    pdf = gaussian_kde(df_train_clean[(df_train_clean['batch']==1+i)]['signal'].values)
    y = pdf(x) + i +1
    source = ColumnDataSource(data=dict(x=x, y=y))
    p1.patch('x', 'y', line_width=1, alpha = 0.6, color=palette[i], source=source, line_color='black')
p1.xaxis.ticker = FixedTicker(ticks=list(range(-5, 15, 1)))
p1.xaxis.axis_label = 'Signal Strength'
p1.yaxis.axis_label = 'Batch'
show(p1)


# The power plot of the signals

# In[ ]:


palette = [cc.rainbow[i*15] for i in range(13)]
x = np.linspace(-3,15, 500)
plot_size_and_tools = {'plot_height': 800, 'plot_width': 700,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}
p1 = figure(y_range=(0, 11),  x_range=(-3, 15), toolbar_location='above', **plot_size_and_tools)
for i in range(10):
    pdf = gaussian_kde(df_train_clean[(df_train_clean['batch']==1+i)]['signal'].values**2)
    y = pdf(x) + i +1
    source = ColumnDataSource(data=dict(x=x, y=y))
    p1.patch('x', 'y', line_width=1, alpha = 0.6, color=palette[i], source=source, line_color='black')
p1.xaxis.ticker = FixedTicker(ticks=list(range(-5, 15, 1)))
p1.xaxis.axis_label = 'Power of Signal Strength'
p1.yaxis.axis_label = 'Batch'
show(p1)


# You can also see the same graph for different batch, the relationship between signal and open channels seems to be different in different batch.

# In[ ]:


def select_batch(batch_val,data):
    
    if (batch_val != 'All'):
        selected = data[data['batch']==int(batch_val)].copy()
    else:
        selected = data.copy()
    return selected

def select_batch2(batch_val, data):
    
    if (batch_val != 'All'):
        selected = df_train[df_train['batch2']==int(batch_val)].copy()
    else:
        selected = df_train.copy()
    return selected

palette = [cc.rainbow[i*20] for i in range(11)]

def batch_plot(batch_val, data):
    x=np.linspace(-6,15, 500)
    # Define Graph
    plot_size_and_tools = {'plot_height': 400, 'plot_width': 350,
                            'tools':['box_zoom', 'reset', 'crosshair','help']}
    p1 = figure(title='Signal vs Channel Batch'+str(batch_val), y_range=(-1, 11),  x_range=(-6, 15), toolbar_location='above', **plot_size_and_tools)
    p1.add_tools(HoverTool())
    p1.xaxis.ticker = FixedTicker(ticks=list(range(-5, 15, 1)))
    p1.xaxis.axis_label = 'Signal Strength'
    p1.yaxis.axis_label = 'Opened Channel'
    df = select_batch(batch_val, data)
    for i in range(11):        
        if len(df[df['open_channels']==i]) !=0:
            pdf = gaussian_kde(df[df['open_channels']==i]['signal'].values)
            y = pdf(x) + i
            source = ColumnDataSource(data=dict(x=x, y=y))
            p1.patch(x,y, line_width=1, alpha = 0.6, color=palette[batch_val],  line_color='black')
        else:
            continue
    return p1

def batch_plot2(batch_val, data):
    x=np.linspace(-6,15, 500)
    # Define Graph
    plot_size_and_tools = {'plot_height': 400, 'plot_width': 350,
                            'tools':['box_zoom', 'reset', 'crosshair','help']}
    p1 = figure(title='Signal vs Channel Batch'+str(batch_val), y_range=(-1, 11),  x_range=(-6, 15), toolbar_location='above', **plot_size_and_tools)
    p1.add_tools(HoverTool())
    p1.xaxis.ticker = FixedTicker(ticks=list(range(-5, 15, 1)))
    p1.xaxis.axis_label = 'Signal Strength'
    p1.yaxis.axis_label = 'Opened Channel'
    df = select_batch2(batch_val, data)
    for i in range(11):        
        if len(df[df['open_channels']==i]) !=0:
            pdf = gaussian_kde(df[df['open_channels']==i]['signal'].values**2)
            y = pdf(x) + i
            source = ColumnDataSource(data=dict(x=x, y=y))
            p1.patch(x,y, line_width=1, alpha = 0.6, color=palette[batch_val],  line_color='black')
        else:
            continue
    return p1


# ### Plots with data drift

# In[ ]:


for i in range(1,6):
    p1 = batch_plot(2*i-1, df_train)
    p2 = batch_plot(2*i, df_train)
    p = gridplot([p1, p2],ncols=2, toolbar_location='right')
    show(p)


# ### Plots without data drift

# Acknowledgment: [Data Without Drift](https://www.kaggle.com/cdeotte/data-without-drift)  
# After removing data drift, we can find that there are two types of relationship between signal and open channels. For batch 1~4, 6~9, they have relatively sharpe peaks. For batch 5 and 10, on the other hand, their peaks are broader namely smaller kurtosis.

# In[ ]:


for i in range(1,6):
    p1 = batch_plot(2*i-1, df_train_clean)
    p2 = batch_plot(2*i, df_train_clean)
    p = gridplot([p1, p2],ncols=2, toolbar_location='right')
    show(p)


# In[ ]:


for i in range(1,6):
    p1 = batch_plot2(2*i-1, df_train_clean)
    p2 = batch_plot2(2*i, df_train_clean)
    p = gridplot([p1, p2],ncols=2, toolbar_location='right')
    show(p)


# Batch 5 and 10 have more complicated data distribution, so lets do some FFT analysis to deep dive in this.

# #### Batch 5

# In[ ]:


s = df_train_clean['signal'].values[2000000:2500000]
y = np.abs(np.fft.fft(s))
start = 10
end = len(s)
res = 1
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    y0=s[start:end:res],
                                    y1=y[start:end:res]))

p1 = figure(title='Train Signal',**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'y0', line_width=1, color = 'red', source=source)
p1.yaxis.axis_label = 'Signal Strength'
p1.xaxis.axis_label = 'Time'
p2 = figure(title='FFT Amplitude',**plot_size_and_tools, x_range=p1.x_range, )
p2.add_tools(HoverTool())
p2.line('x', 'y1', line_width=1, color = 'black', source=source)
p2.xaxis.axis_label = 'Freq'
p2.yaxis.axis_label = 'FFT Amplitude'
p = gridplot([p1, p2],ncols=1, toolbar_location='right')
show(p)


# In[ ]:


s = pd.Series(df_train_clean['signal'].values[2000000:2500000]).rolling(50).mean().fillna(method='bfill')
y = np.abs(np.fft.fft(s))
start = 10
end = len(s)
res = 1
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    y0=s[start:end:res],
                                    y1=y[start:end:res]))

p1 = figure(title='Rolling 50 Train Signal',**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'y0', line_width=1, color = 'red', source=source)
p1.yaxis.axis_label = 'Signal Strength'
p1.xaxis.axis_label = 'Time'
p2 = figure(title='FFT Amplitude',**plot_size_and_tools, x_range=p1.x_range, )
p2.add_tools(HoverTool())
p2.line('x', 'y1', line_width=1, color = 'black', source=source)
p2.xaxis.axis_label = 'Freq'
p2.yaxis.axis_label = 'FFT Amplitude'
p = gridplot([p1, p2],ncols=1, toolbar_location='right')
show(p)


# #### Batch 1

# In[ ]:


s = df_train_clean['signal'].values[0:500000]
y = np.abs(np.fft.fft(s))
start = 10
end = len(s)
res = 1
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    y0=s[start:end:res],
                                    y1=y[start:end:res]))

p1 = figure(title='Train Signal',**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'y0', line_width=1, color = 'red', source=source)
p1.yaxis.axis_label = 'Signal Strength'
p1.xaxis.axis_label = 'Time'
p2 = figure(title='FFT Amplitude',**plot_size_and_tools, x_range=p1.x_range, )
p2.add_tools(HoverTool())
p2.line('x', 'y1', line_width=1, color = 'black', source=source)
p2.xaxis.axis_label = 'Freq'
p2.yaxis.axis_label = 'FFT Amplitude'
p = gridplot([p1, p2],ncols=1, toolbar_location='right')
show(p)


# In[ ]:


s = pd.Series(df_train_clean['signal'].values[0:500000]).rolling(50).mean().fillna(method='bfill')
y = np.abs(np.fft.fft(s))
start = 10
end = len(s)
res = 1
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    y0=s[start:end:res],
                                    y1=y[start:end:res]))

p1 = figure(title='Rolling 50 Train Signal',**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'y0', line_width=1, color = 'red', source=source)
p1.yaxis.axis_label = 'Signal Strength'
p1.xaxis.axis_label = 'Time'
p2 = figure(title='FFT Amplitude',**plot_size_and_tools, x_range=p1.x_range, )
p2.add_tools(HoverTool())
p2.line('x', 'y1', line_width=1, color = 'black', source=source)
p2.xaxis.axis_label = 'Freq'
p2.yaxis.axis_label = 'FFT Amplitude'
p = gridplot([p1, p2],ncols=1, toolbar_location='right')
show(p)


# In[ ]:


s = df_train_clean['signal'].values[0:500000]
y = np.fft.fft(s)
#y[:2480]=0
y[2510:]=0
s_i = np.abs(np.fft.ifft(y))
start = 10
end = len(s)
res = 1
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    y0=s[start:end:res],
                                    y1=s_i[start:end:res]))

p1 = figure(title='Train Signal',**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'y0', line_width=1, color = 'red', source=source)
p1.yaxis.axis_label = 'Signal Strength'
p1.xaxis.axis_label = 'Time'
p2 = figure(title='IFFT signal',**plot_size_and_tools, x_range=p1.x_range, )
p2.add_tools(HoverTool())
p2.line('x', 'y1', line_width=1, color = 'black', source=source)
p2.xaxis.axis_label = 'Freq'
p2.yaxis.axis_label = 'IFFT Amplitude'
p = gridplot([p1, p2],ncols=1, toolbar_location='right')
show(p)


# #### Batch 4

# In[ ]:


s = df_train_clean['signal'].values[1500000:2000000]
y = np.abs(np.fft.fft(s))
start = 10
end = len(s)
res = 1
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    y0=s[start:end:res],
                                    y1=y[start:end:res]))

p1 = figure(title='Train Signal',**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'y0', line_width=1, color = 'red', source=source)
p1.yaxis.axis_label = 'Signal Strength'
p1.xaxis.axis_label = 'Time'
p2 = figure(title='FFT Amplitude',**plot_size_and_tools, x_range=p1.x_range, )
p2.add_tools(HoverTool())
p2.line('x', 'y1', line_width=1, color = 'black', source=source)
p2.xaxis.axis_label = 'Freq'
p2.yaxis.axis_label = 'FFT Amplitude'
p = gridplot([p1, p2],ncols=1, toolbar_location='right')
show(p)


# In[ ]:


s = pd.Series(df_train_clean['signal'].values[1500000:2000000]).rolling(50).mean().fillna(method='bfill')
y = np.abs(np.fft.fft(s))
start = 10
end = len(s)
res = 1
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    y0=s[start:end:res],
                                    y1=y[start:end:res]))

p1 = figure(title='Rolling 50 Train Signal',**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'y0', line_width=1, color = 'red', source=source)
p1.yaxis.axis_label = 'Signal Strength'
p1.xaxis.axis_label = 'Time'
p2 = figure(title='FFT Amplitude',**plot_size_and_tools, x_range=p1.x_range, )
p2.add_tools(HoverTool())
p2.line('x', 'y1', line_width=1, color = 'black', source=source)
p2.xaxis.axis_label = 'Freq'
p2.yaxis.axis_label = 'FFT Amplitude'
p = gridplot([p1, p2],ncols=1, toolbar_location='right')
show(p)


# **The low frequency FFT amplitude for batch 5 seems to be stronger than other batches. Let's check that directly. I guess this is somehow related to the "Ghost signal" mentioned [here](https://www.kaggle.com/miklgr500/ghost-drift-and-outliers/notebook#%22Ghost%22-drift). This maybe another data drift. We need identify what causes this and how to capture the characteristics**

# In[ ]:


s1 = pd.Series(df_train_clean['signal'].values[0:500000]).rolling(50).mean().fillna(method='bfill')
y1 = np.abs(np.fft.fft(s1))
s4 = pd.Series(df_train_clean['signal'].values[1500000:2000000]).rolling(50).mean().fillna(method='bfill')
y4 = np.abs(np.fft.fft(s4))
s5 = pd.Series(df_train_clean['signal'].values[2000000:2500000]).rolling(50).mean().fillna(method='bfill')
y5 = np.abs(np.fft.fft(s5))
start = 10
end = len(s)
res = 1
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    z0=y1[start:end:res],
                                    z1=y4[start:end:res],
                                    z2=y5[start:end:res]))

p1 = figure(title='FFT Amplitude',x_range=(start, end/10),**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'z2', line_width=1, color = 'black', source=source)
p1.line('x', 'z1', line_width=1, color = 'red', source=source)
p1.line('x', 'z0', line_width=1, color = 'yellow', source=source)
p1.yaxis.axis_label = 'FFT Amplitude'
p1.xaxis.axis_label = 'Freq'

show(p1)


# ### Statistical features

# Next, let see if there is any signal characteristics in the sub-batches. The sub-batch is 1,000 time step long which is corresponding to 1/500 of a batch.

# In[ ]:


get_ipython().run_cell_magic('time', '', "aggs = ['max', 'min', 'var', 'mad', 'skew']\n\ndef agg_func(aggs, df):\n    df_agg = pd.DataFrame()\n    for agg in tqdm(aggs):\n        #print(agg)\n        assert agg in ['max', 'min', 'var', 'mad', 'skew'], 'Choose defined aggregation method.'\n        if agg == 'max':\n            df_agg['sub_batch_'+agg] = df.groupby(by='sub_batch')['signal'].max()\n        elif agg == 'min':\n            df_agg['sub_batch_'+agg] = df.groupby(by='sub_batch')['signal'].min()\n        elif agg == 'var':\n            df_agg['sub_batch_'+agg] = df.groupby(by='sub_batch')['signal'].var()\n        elif agg == 'mad':\n            df_agg['sub_batch_'+agg] = df.groupby(by='sub_batch')['signal'].mad()\n        elif agg == 'skew':\n            df_agg['sub_batch_'+agg] = df.groupby(by='sub_batch')['signal'].skew()\n    df_agg['batch'] = df.groupby(by='sub_batch')['batch'].mean().astype('int16').values\n        \n    df_agg = df_agg.reset_index()\n    return df_agg\n\ndf_agg = agg_func(aggs, df_train)\ndf_agg.head(2)")


# In[ ]:


x = df_agg['sub_batch'].values
plot_size_and_tools = {'plot_height': 100,
                       #'plot_width': 500,
                            'tools':['box_zoom', 'reset', 'crosshair','help']}
for i, agg in enumerate(aggs):
    p = figure(title='Statistics for sub batches: '+agg,toolbar_location='above', **plot_size_and_tools, sizing_mode="scale_width")
    p.add_tools(HoverTool())
    y = df_agg['sub_batch_'+agg].values
    p.scatter(x, y, color=palette[i])
    show(p)


# In[ ]:


df_agg_clean = agg_func(aggs, df_train_clean)
x = df_agg_clean['sub_batch'].values
plot_size_and_tools = {'plot_height': 100,
                       #'plot_width': 500,
                            'tools':['box_zoom', 'reset', 'crosshair','help']}
for i, agg in enumerate(aggs):
    p = figure(title='Statistics for sub batches (w/o data drift): '+agg,toolbar_location='above', **plot_size_and_tools, sizing_mode="scale_width")
    p.add_tools(HoverTool())
    y = df_agg_clean['sub_batch_'+agg].values
    p.scatter(x, y, color=palette[i])
    show(p)


# # Test Data

# In[ ]:


start = 0
end = len(df_test)
res = int(end/1000000)
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    y0=df_test['signal'].values[start:end:res],
                                    ))

p1 = figure(title='Test Signal',**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'y0', line_width=1, color = 'blue', source=source)

show(p1)


# In[ ]:


start = 0
end = len(df_test)
res = int(end/1000000)
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    y0=df_test_clean['signal'].values[start:end:res],
                                    ))

p1 = figure(title='Test Signal(w/o drift)',**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'y0', line_width=1, color = 'purple', source=source)

show(p1)


# In[ ]:


start = 0
end = len(df_test)
res = int(end/1000000)
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    y0=df_test_clean['signal'].values[start:end:res]-df_test_clean['signal'].shift(1).fillna(0).values[start:end:res] ,
                                    ))

p1 = figure(title='Test Signal(w/o drift)',**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'y0', line_width=1, color = 'purple', source=source)

show(p1)


# In[ ]:


start = 0
end = len(df_train)
res = int(end/1000000)
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    y0=df_train['signal'].values[start:end:res],
                                    y1=df_train['open_channels'].values[start:end:res]))

p1 = figure(title='Train Signal',**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'y0', line_width=1, color = 'red', source=source)
p1.yaxis.axis_label = 'Signal Strength'
p2 = figure(title='Open_Channels',**plot_size_and_tools, x_range=p1.x_range, y_range=p1.y_range)
p2.add_tools(HoverTool())
p2.line('x', 'y1', line_width=1, color = 'black', source=source)
p2.xaxis.axis_label = 'Time'
p2.yaxis.axis_label = 'Opened Channel'
p = gridplot([p1, p2],ncols=1, toolbar_location='right')
show(p)


# In[ ]:


start = 0
end = len(df_train)
res = int(end/1000000)
plot_size_and_tools = {'plot_height': 200, 'plot_width': 750,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}

source = ColumnDataSource(data=dict(x=list(range(start,end,res)), 
                                    y0=df_train_clean['signal'].values[start:end:res],
                                    y1=df_train_clean['open_channels'].values[start:end:res]))

p1 = figure(title='Train Signal(w/o drift)',**plot_size_and_tools)
p1.add_tools(HoverTool())
p1.line('x', 'y0', line_width=1, color = 'orange', source=source)
p1.yaxis.axis_label = 'Signal Strength'
p2 = figure(title='Open_Channels',**plot_size_and_tools, x_range=p1.x_range, y_range=p1.y_range)
p2.add_tools(HoverTool())
p2.line('x', 'y1', line_width=1, color = 'black', source=source)
p2.xaxis.axis_label = 'Time'
p2.yaxis.axis_label = 'Opened Channel'
p = gridplot([p1, p2],ncols=1, toolbar_location='right')
show(p)


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_test.index = ((df_test.time * 10_000)).values-1\ndf_test['batch'] = 1+(df_test.index // 50_0000)\ndf_test['batch'] = df_test['batch'].astype('int16')\ndf_test['sub_batch'] = 1 + (df_test.index  // 1_000).astype('int32')\ndf_test.tail(2)")


# In[ ]:


df_test_clean.index = ((df_test_clean.time * 10_000)).values-1
df_test_clean['batch'] = 1+(df_test_clean.index // 50_0000)
df_test_clean['batch'] = df_test_clean['batch'].astype('int16')
df_test_clean['sub_batch'] = 1 + (df_test_clean.index  // 1_000).astype('int32')


# In[ ]:


df_test_clean['batch'].unique()


# ### KDE Plots of the test signal by different batches

# In[ ]:


palette = [cc.rainbow[i*15] for i in range(13)]
x = np.linspace(-6,15, 500)
plot_size_and_tools = {'plot_height': 400, 'plot_width': 700,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}
p1 = figure(y_range=(10, 16),  x_range=(-6, 15), toolbar_location='above', **plot_size_and_tools)
for i in range(4):
    pdf = gaussian_kde(df_test[(df_test['batch']==11+i)]['signal'].values)
    y = pdf(x) + i +11
    source = ColumnDataSource(data=dict(x=x, y=y))
    p1.patch('x', 'y', line_width=1, alpha = 0.6, color=palette[i], source=source, line_color='black')
p1.xaxis.ticker = FixedTicker(ticks=list(range(-5, 15, 1)))
p1.xaxis.axis_label = 'Signal Strength'
p1.yaxis.axis_label = 'Batch'
show(p1)


# The batch 11, 13, 14 seem like the distribution of the batch 1~4, 6~9, while the batch 12's distribution resembles that of the batch 5 and 10.

# In[ ]:


palette = [cc.rainbow[i*15] for i in range(13)]
x = np.linspace(-6,15, 500)
plot_size_and_tools = {'plot_height': 400, 'plot_width': 700,
                        'tools':['box_zoom', 'reset', 'crosshair','help']}
p1 = figure(y_range=(10, 16),  x_range=(-6, 15), toolbar_location='above', **plot_size_and_tools)
for i in range(4):
    pdf = gaussian_kde(df_test_clean[(df_test_clean['batch']==11+i)]['signal'].values)
    y = pdf(x) + i +11
    source = ColumnDataSource(data=dict(x=x, y=y))
    p1.patch('x', 'y', line_width=1, alpha = 0.6, color=palette[i], source=source, line_color='black')
p1.xaxis.ticker = FixedTicker(ticks=list(range(-5, 15, 1)))
p1.xaxis.axis_label = 'Signal Strength'
p1.yaxis.axis_label = 'Batch'
show(p1)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_agg_test = agg_func(aggs, df_test)\ndf_agg_test.tail(2)')


# In[ ]:


x = df_agg_test['sub_batch'].values
plot_size_and_tools = {'plot_height': 100,
                       #'plot_width': 500,
                            'tools':['box_zoom', 'reset', 'crosshair','help']}
for i, agg in enumerate(aggs):
    p = figure(title='Statistics for sub batches: '+agg,toolbar_location='above', **plot_size_and_tools, sizing_mode="scale_width")
    p.add_tools(HoverTool())
    y = df_agg_test['sub_batch_'+agg].values
    p.scatter(x, y, color=palette[i])
    show(p)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_agg_test_clean = agg_func(aggs, df_test_clean)\ndf_agg_test_clean.tail(2)')


# In[ ]:


x = df_agg_test_clean['sub_batch'].values
plot_size_and_tools = {'plot_height': 100,
                       #'plot_width': 500,
                            'tools':['box_zoom', 'reset', 'crosshair','help']}
for i, agg in enumerate(aggs):
    p = figure(title='Statistics for sub batches: '+agg,toolbar_location='above', **plot_size_and_tools, sizing_mode="scale_width")
    p.add_tools(HoverTool())
    y = df_agg_test_clean['sub_batch_'+agg].values
    p.scatter(x, y, color=palette[i])
    show(p)


# ## Comparing Train and Test Data

# In[ ]:


start_tr = 0
end_tr = len(df_train)
res_tr = int(end_tr/1000000)
start_te = 0
end_te = len(df_test)
res_te = int(end_te/1000000)
x1 = df_train['time'].values[start_tr:end_tr:res_tr]
x2 = df_test['time'].values[start_te:end_te:res_te] 
y1 = df_train['signal'].values[start_tr:end_tr:res_tr]
y2 = df_test['signal'].values[start_te:end_te:res_te]
plot_size_and_tools = {'plot_height': 200,
                       #'plot_width': 700,
                            'tools':['box_zoom', 'reset', 'crosshair','help']}

p1 = figure(title='Signal ',toolbar_location='above', **plot_size_and_tools, sizing_mode="scale_width")
p1.add_tools(HoverTool())
p1.line(x1, y1, color=palette[1], legend_label='Train')
p1.line(x2, y2, color=palette[6], legend_label='Test')
show(p1)


# In[ ]:


agg = 'mad'
x1 = df_agg['sub_batch'].values
x2 = df_agg_test['sub_batch'].values 
y1 = df_agg['sub_batch_'+agg].values
y2 = df_agg_test['sub_batch_'+agg].values
plot_size_and_tools = {'plot_height': 200,
                       #'plot_width': 700,
                            'tools':['box_zoom', 'reset', 'crosshair','help']}

p1 = figure(title='Statistics for sub batches: '+agg,toolbar_location='above', **plot_size_and_tools, sizing_mode="scale_width")
p1.add_tools(HoverTool())
p1.line(x1, y1, color=palette[1], legend_label='Train')
p1.line(x2, y2, color=palette[6], legend_label='Test')
show(p1)

#show(p2)


# If you find this notebook is useful, please upvote!!
