#!/usr/bin/env python
# coding: utf-8

# # Lets take a look at such features as DisplayResolution and DisplaySizeInInches and see if they might be helpful
# 
# **Content:**
# 
# 1. [Idea description](#1)
# 2. [Loading data](#2)
# 3. [Analyzing Display features](#3)
# 
#     3.1. [New feature - resolution ratio](#4)
#     
#     3.2. [Correlation of new feature with HasDetections](#5)
#     
#     3.3. [Correlation between screen quality and HasDetections](#6)
#     
#     3.4. [Display size correlation with HasDetections](#7)
# 4. [Two features interaction](#8)
# 
#     4.1. [Resolution Rate / Wdft_IsGamer](#9)
#     
#     4.2. [Resolution Rate / Census_IsTouchEnabled](#10)
#     
#     4.3. [Display quality / Processor architecture](#11)
#     
#     4.4. [Display quality / Wdft_IsGamer](#12)
# 
# 
# <a id="1"></a>
# # 1. Idea description
# The main idea is that PC for a home use, most likely, would have medium to big sizes of displays and at least FullHD resolution. Those are some kind of 'gaming PCs' being used by people mostly for surfing and playing. And this is a kind of people who take less precautions regarding malware. Also the most popular aspect ratio amongst gamers is 16:9 (here is a pretty fresh review: https://www.gamingscan.com/best-aspect-ratio-for-gaming/) so we will take a look at aspect ratio as well by creating a new feature.
# 
# On the other had small-sized displays are mostly used either on laptops and (sometimes) servers. We are going to take a look on them as well.

# <a id="2"></a>
# # 2. Loading data

# In[ ]:


import warnings
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import gc
import multiprocessing
import seaborn as sns
pd.set_option('display.max_columns', 83)
pd.set_option('display.max_rows', 83)
plt.style.use('seaborn')
import os
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import cufflinks
import plotly
import matplotlib
init_notebook_mode()
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl', offline=True)
print(os.listdir("../input"))
for package in [pd, np, sns, matplotlib, plotly]:
    print(package.__name__, 'version:', package.__version__)


# In[ ]:


dtypes = {
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Wdft_IsGamer':                                         'float16',
        'Processor':                                            'category',
        'HasDetections':                                        'int8'
        }


# In[ ]:


def load_dataframe(dataset):
    usecols = dtypes.keys()
    if dataset == 'test':
        usecols = [col for col in dtypes.keys() if col != 'HasDetections']
    df = pd.read_csv(f'../input/{dataset}.csv', dtype=dtypes, usecols=usecols)
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'with multiprocessing.Pool() as pool: \n    train, test = pool.map(load_dataframe, ["train", "test"])')


# <a id="3"></a>
# # 3. Analyzing Display features
# 
# First lets take a look at the most popular display resoultion both vertical and horizontal

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,8))
train['Census_InternalPrimaryDisplayResolutionHorizontal'].value_counts().head(10).plot(kind='barh', ax=axes[0], fontsize=14).set_xlabel('Horizontal Resolution', fontsize=18)
train['Census_InternalPrimaryDisplayResolutionVertical'].value_counts().head(10).plot(kind='barh', ax=axes[1], fontsize=14).set_xlabel('Vertical Resolution', fontsize=18)
axes[0].invert_yaxis()
axes[1].invert_yaxis()


# Ok, this doesn't tell us really much.
# 
# Lets create a new feature - ResolutionRation by dividing vertical resolution by horizontal resolution and see what we will have in result.

# <a id="4"></a>
# ## 3.1 New feature - resolution ratio

# In[ ]:


train['ResolutionRatio'] = train['Census_InternalPrimaryDisplayResolutionVertical'] / train['Census_InternalPrimaryDisplayResolutionHorizontal']


# In[ ]:


train['ResolutionRatio'].value_counts().head(10).plot(kind='barh', figsize=(14,8), fontsize=14);
plt.gca().invert_yaxis()


# So 4 most popular rations are:
# 
# * 0.562011 is (mostly) a low-end laptops ratio (1366 by 768)
# * 0.5625 is 16:9 ratio. Most popular amongst gamers
# * 0.625 is 16:10 ratio. Characteristic of the old displays (https://en.wikipedia.org/wiki/Display_aspect_ratio#4:3_and_16:10)
# * 0.75 is 4:3 ratio. Also really old displays

# <a id="5"></a>
# ## 3.2 Correlation of new feature with HasDetections
# 
# Now lets see dependency  between most popular resolution ratios and target value.

# In[ ]:


ratios = train['ResolutionRatio'].value_counts().head(6).index
fig, axes = plt.subplots(nrows=int(len(ratios) / 2), ncols=2, figsize=(16,14))
fig.subplots_adjust(wspace=0.2, hspace=0.4)
for i in range(len(ratios)):
    sns.countplot(x='ResolutionRatio', hue='HasDetections', data=train[train['ResolutionRatio'] == ratios[i]], ax=axes[i // 2,i % 2]);


# So 'gamers' (0.5625, which is 16:9) have more malware detections than others.
# 
# With next step lets divide all displays into 4 categories: low-definition (SD), high-definition (HD), FullHD and 4k and see detections distribution.
# 
# <a id="6"></a>
# ## 3.3 Correlation between screen quality and HasDetections

# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18,10))
train.loc[train['Census_InternalPrimaryDisplayResolutionVertical'] < 720, 'HasDetections'].value_counts().sort_index().plot(kind='bar', rot=0, ax=axes[0,0]).set_xlabel('SD');
train.loc[(train['Census_InternalPrimaryDisplayResolutionVertical'] >= 720) & (train['Census_InternalPrimaryDisplayResolutionVertical'] < 1080), 'HasDetections'].value_counts().sort_index().plot(kind='bar', rot=0, ax=axes[0,1]).set_xlabel('HD');
train.loc[(train['Census_InternalPrimaryDisplayResolutionVertical'] >= 1080) & (train['Census_InternalPrimaryDisplayResolutionVertical'] < 2160), 'HasDetections'].value_counts().sort_index().plot(kind='bar', rot=0, ax=axes[1,0]).set_xlabel('FullHD');
train.loc[train['Census_InternalPrimaryDisplayResolutionVertical'] >= 2160, 'HasDetections'].value_counts().sort_index().plot(kind='bar', rot=0, ax=axes[1,1]).set_xlabel('4k');


# Same with plotly

# In[ ]:


sd_values = train.loc[train['Census_InternalPrimaryDisplayResolutionVertical'] < 720, 'HasDetections'].value_counts().sort_index().values
hd_values = train.loc[(train['Census_InternalPrimaryDisplayResolutionVertical'] >= 720) & (train['Census_InternalPrimaryDisplayResolutionVertical'] < 1080), 'HasDetections'].value_counts().sort_index().values
fullhd_values = train.loc[(train['Census_InternalPrimaryDisplayResolutionVertical'] >= 1080) & (train['Census_InternalPrimaryDisplayResolutionVertical'] < 2160), 'HasDetections'].value_counts().sort_index().values
k_values = train.loc[train['Census_InternalPrimaryDisplayResolutionVertical'] >= 2160, 'HasDetections'].value_counts().sort_index().values
x = ['SD', 'HD', 'FullHD', '4k']
y_0 = [sd_values[0], hd_values[0], fullhd_values[0], k_values[0]]
y_1 = [sd_values[1], hd_values[1], fullhd_values[1], k_values[1]]
trace1 = go.Bar(x=x, y=y_0, name='0 (no detections)')
trace2 = go.Bar(x=x, y=y_1, name='1 (has detections)')
data = [trace1, trace2]
layout = go.Layout(barmode='group')
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# The better display quality is the higher rate of malware detections.

# <a id="7"></a>
# ## 3.4 Display size correlation with HasDetections
# 
# Hope this plot makes sense. We can see here that the bigger display is the higher detection rate is, but also the distribution density is higher for small screens.

# In[ ]:


def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

plot_dict = dict()
for i in train['Census_InternalPrimaryDiagonalDisplaySizeInInches'].value_counts().sort_index().index:
    try:
        plot_dict[i] = train.loc[train['Census_InternalPrimaryDiagonalDisplaySizeInInches'] == i, 'HasDetections'].value_counts(normalize=True)[1]
    except:
        plot_dict[i] = 0.0
fig, ax1 = plt.subplots(figsize=(16,7))
ax1.set_xlabel('Display Size in inches')
ax1.set_ylabel('Count', color='tab:green')
ax1.hist(plot_dict.keys(), color='tab:green', bins=int(len(plot_dict) / 20))
ax1.tick_params(axis='y', labelcolor='tab:green')
ax2 = ax1.twinx()
ax2.set_ylabel('Detection Rate', color='blue')
ax2.plot(plot_dict.keys(), movingaverage(list(plot_dict.values()), int(len(plot_dict) / 20)),color='blue', linewidth=2.0)
ax2.tick_params(axis='y', labelcolor='blue')
plt.show()


# <a id="8"></a>
# # 4. Two features interaction
# <a id="9"></a>
# ## 4.1  Resolution Rate / Wdft_IsGamer
# 
# Next will plot correlation between Display Resolution Rate and feature Wdft_IsGamer also with respect to a detection rate.

# In[ ]:


fig, axes = plt.subplots(nrows=int(len(ratios) / 2), ncols=2, figsize=(18,16))
fig.subplots_adjust(wspace=0.2, hspace=0.4)
for i in range(len(ratios)):
    train.loc[train['ResolutionRatio'] == ratios[i], 'Wdft_IsGamer'].value_counts(True, dropna=False).plot(kind='bar', rot=0, ax=axes[i // 2,i % 2], fontsize=14).set_xlabel('Wdft_IsGamer', fontsize=18)
    axes[i // 2,i % 2].plot(0, train.loc[(train['ResolutionRatio'] == ratios[i]) & (train['Wdft_IsGamer'] == 0.0), 'HasDetections'].value_counts(True, dropna=False)[1], marker='.', color="r", markersize=24)
    axes[i // 2,i % 2].plot(1, train.loc[(train['ResolutionRatio'] == ratios[i]) & (train['Wdft_IsGamer'] == 1.0), 'HasDetections'].value_counts(True, dropna=False)[1], marker='.', color="r", markersize=24)
    axes[i // 2,i % 2].plot(2, train.loc[(train['ResolutionRatio'] == ratios[i]) & (train['Wdft_IsGamer'].isnull()), 'HasDetections'].value_counts(True, dropna=False)[1], marker='.', color="r", markersize=24) 
    axes[i // 2,i % 2].legend(['Detection rate (%)'])
    axes[i // 2,i % 2].set_title('Ratio: ' + str(ratios[i]), fontsize=18)
fig.suptitle('Resolution rate to Wdft_IsGamer interaction', fontsize=18);


# Most popular aspect ratios amongst gamers, according to the data provided, are 0.5625 (16:9) and 0.5649 (which is a 'wrong' [16:9 on laptops](https://en.wikipedia.org/wiki/Graphics_display_resolution#1360_%C3%97_768).)

# <a id="10"></a>
# ## 4.2 Resolution rate / Census_IsTouchEnabled
# Doing the same plot for interaction of Resolution rate and Census_IsTouchEnabled features.

# In[ ]:


fig, axes = plt.subplots(nrows=int(len(ratios) / 2), ncols=2, figsize=(18,16))
fig.subplots_adjust(wspace=0.2, hspace=0.4)
for i in range(len(ratios)):
    train.loc[train['ResolutionRatio'] == ratios[i], 'Census_IsTouchEnabled'].value_counts(True, dropna=False).plot(kind='bar', rot=0, ax=axes[i // 2,i % 2], fontsize=14).set_xlabel('Census_IsTouchEnabled', fontsize=18)
    axes[i // 2,i % 2].plot(0, train.loc[(train['ResolutionRatio'] == ratios[i]) & (train['Census_IsTouchEnabled'] == 0), 'HasDetections'].value_counts(True, dropna=False)[1], marker='.', color="r", markersize=24)
    axes[i // 2,i % 2].plot(1, train.loc[(train['ResolutionRatio'] == ratios[i]) & (train['Census_IsTouchEnabled'] == 1), 'HasDetections'].value_counts(True, dropna=False)[1], marker='.', color="r", markersize=24)
    axes[i // 2,i % 2].legend(['Detection rate (%)'])
    axes[i // 2,i % 2].set_title('Ratio: ' + str(ratios[i]), fontsize=18)
fig.suptitle('Resolution rate to Census_IsTouchEnabled interaction', fontsize=18);


# I personally don't see any affect of those features interaction on HasDetections.

# <a id="11"></a>
# ## 4.3 Display quality / Processor architecture

# In[ ]:


train['SD'] = (train['Census_InternalPrimaryDisplayResolutionVertical'] < 720).astype('uint8')
train['HD'] = (train['Census_InternalPrimaryDisplayResolutionVertical'].isin(range(720,1080))).astype('int8')
train['FullHD'] = (train['Census_InternalPrimaryDisplayResolutionVertical'].isin(range(1080,2160))).astype('int8')
train['4k'] = (train['Census_InternalPrimaryDisplayResolutionVertical'] >= 2160).astype('uint8')

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18,12))
fig.subplots_adjust(wspace=0.2, hspace=0.4)
quals = ['SD', 'HD', 'FullHD', '4k']
axis_to_processor =  ['x86', 'x64', 'arm64']
for i in range(len(quals)):
    train.loc[train[quals[i]] == 1, 'Processor'].value_counts(True).sort_index(ascending=False).plot(kind='bar', rot=0, fontsize=14, ax=axes[i // 2, i % 2]).set_xlabel('Processor', fontsize=18);
    for j in range(len(axis_to_processor)):
        try:
            axes[i // 2,i % 2].plot(j, train.loc[(train[quals[i]] == 1) & (train['Processor'] == axis_to_processor[j]), 'HasDetections'].value_counts(True, dropna=False)[1], marker='.', color="r", markersize=24)
        except:
            pass
    axes[i // 2,i % 2].legend(['Detection rate (%)'])
    axes[i // 2,i % 2].set_title('Display quality: ' + quals[i], fontsize=18)


# No surprise here - higher display quality owners prefer x64 processors.

# <a id="12"></a>
# ## 4.4 Display quality / Wdft_IsGamer

# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18,12))
fig.subplots_adjust(wspace=0.2, hspace=0.4)
for i in range(len(quals)):
    train.loc[train[quals[i]] == 1, 'Wdft_IsGamer'].value_counts(True, dropna=False).sort_index().plot(kind='bar', rot=0, fontsize=14, ax=axes[i // 2, i % 2]).set_xlabel('Wdft_IsGamer', fontsize=18);
    axes[i // 2,i % 2].plot(0, train.loc[(train[quals[i]] == 1) & (train['Wdft_IsGamer'] == 0), 'HasDetections'].value_counts(True, dropna=False)[1], marker='.', color="r", markersize=24)
    axes[i // 2,i % 2].plot(1, train.loc[(train[quals[i]] == 1) & (train['Wdft_IsGamer'] == 1), 'HasDetections'].value_counts(True, dropna=False)[1], marker='.', color="r", markersize=24)
    axes[i // 2,i % 2].plot(2, train.loc[(train[quals[i]] == 1) & (train['Wdft_IsGamer'].isnull()), 'HasDetections'].value_counts(True, dropna=False)[1], marker='.', color="r", markersize=24)
    axes[i // 2,i % 2].legend(['Detection rate (%)'])
    axes[i // 2,i % 2].set_title('Display quality: ' + quals[i], fontsize=18)


# Again no surprise that gamers prefer higher quality displays.
