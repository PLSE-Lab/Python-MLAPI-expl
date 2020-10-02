#!/usr/bin/env python
# coding: utf-8

# ## **Content**
# 
# * [**About the Competition**](#1)
#     * Objective of the Competition
#     * About the kernel
#     * Key TakeAways
# * [**Importing the necessary Packages**](#2)
# * [**Initial Data preparation**](#3)
# * [**Exploratory Data Analysis**](#4)
#     * Distribtuion of Open Channels
#     * Ground truth of each batches
#     * Data distribution across each batches
#     * Exploring the test data
# * [**What are predicting?**](#5)    
# 

# ## About the Competition <a id="1"></a>
# 
# Many diseases, including cancer, are believed to have a contributing factor in common. Ion channels are pore-forming proteins present in animals and plants. They encode learning and memory, help fight infections, enable pain signals, and stimulate muscle contraction. If scientists could better study ion channels, which may be possible with the aid of machine learning, it could have a far-reaching impact.
# 
# 
# 
# When ion channels open, they pass electric currents. Existing methods of detecting these state changes are slow and laborious. Humans must supervise the analysis, which imparts considerable bias, in addition to being tedious. These difficulties limit the volume of ion channel current analysis that can be used in research. Scientists hope that technology could enable rapid automatic detection of ion channel current events in raw data.
# 

# **Objective of the Competition**
# 
# ![](https://www.nature.com/scitable/content/ne0000/ne0000/ne0000/ne0000/14707004/U4CP3-1_IonChannel_ksm.jpg)
# 
# 
# In this competition, contestants are challenged to predict the number of open ion channels using electrophysiological signals from human cells. 

# **About this kernel**
# 
# This kernel will acts as a guide covering the A-Z topics on this data
# 
# If you don't know about the background, then I would highly recommend you to visit this [kernel](https://www.kaggle.com/tarunpaparaju/ion-switching-competition-signal-eda)

# **Note:**
# 
# While the time series appears continuous, the data is from discrete batches of 50 seconds long 10 kHz samples (500,000 rows per batch). In other words, the data from 0.0001 - 50.0000 is a different batch than 50.0001 - 100.0000, and thus discontinuous between 50.0000 and 50.0001.

# **Key Takeaway's**
# 
# * Extensive EDA
# * Understanding the nature of NFL
# * Effective Story Telling
# * Creative Feature Engineering
# * Modelling
# * Ensembling

# **The plots made are interactive one's feel free to hover over**

# ## Importing the necessary Packages <a id="2"></a>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.figure_factory as ff ## For distiribution plot
import plotly.offline as py
import plotly.express as px
import plotly.graph_objects as go


import matplotlib.pylab as plt
import seaborn as sns

py.init_notebook_mode(connected=True)


# ## Importing Data Preparation <a id="3"></a>

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


sample_df  = pd.read_csv("/kaggle/input/liverpool-ion-switching/sample_submission.csv")
test_df = pd.read_csv("/kaggle/input/liverpool-ion-switching/test.csv")
train_df = pd.read_csv("/kaggle/input/liverpool-ion-switching/train.csv")


# In[ ]:


train_df.shape , test_df.shape


# In[ ]:


train_df.isnull().sum()


# **I think,since the data we deal with here are electrophysiological signals from human cells there is no missing data  points.** However it is mentioned that can be chance of biasness since data were monitered by humans. Let's see if there is any...

# In[ ]:


train_df.dtypes


# As I have mentioned earlier that, the data given is of discrete batch types ranging 0.0001 - 50.0000 of each we can create a new column in both train and test which might support our predicition

# In[ ]:


train_df.time.is_monotonic


# The given data is time series, just wanted to make sure the time sorted order before creating the new column

# In[ ]:


print("No of batches in Train Data",train_df.shape[0]/500000)
print("No of batches in Test Data",test_df.shape[0]/500000)


# **Creating the new column Batches**

# In[ ]:


train_df['batch'] = 0
test_df['batch'] = 0
for i in range(0, 10):
    train_df.iloc[i * 500000: 500000 * (i + 1), 3] = i
for i in range(0, 4):
    test_df.iloc[i * 500000: 500000 * (i + 1), 2] = i


# Since, we have done the inital data preparation let's statring exploring the data 

# ## Exploratory Data Analysis <a id ="4"></a>

# **Distribution of the data**

# In[ ]:


train_df.open_channels.unique()


# In[ ]:


from IPython.display import Image
Image("/kaggle/input/ion-channel/ion image.jpg")


# The open Channels are ranged from 1 - 10 in the datasets each step indicating the state changes

# **Distribution of the open Channels**

# In[ ]:


temp_df = train_df.groupby(["open_channels"])["open_channels"].agg(["count"]).reset_index()

fig = px.bar(temp_df, x='open_channels', y='count',
             hover_data=['count'], color='count',
             labels={'pop':'Distribtuion of Open Channels'}, height=400)
fig.show()


# **Observation**
# 
# Distribtuion of Open Channels decreases with increase in the strength of signal

# **Ground truth across each batches**

# In[ ]:


temp_df = train_df.groupby(["open_channels","batch"])["open_channels"].agg(["count"]).reset_index()
temp_df.columns = ["open_channels","batch","count"]
#temp_df.Country = temp_df[temp_df.Country != 'United Kingdom']

fig = px.scatter(temp_df, x="open_channels", y="batch", color="open_channels", size="count")
layout = go.Layout(
    title=go.layout.Title(
        text="Ground Truth across each batches",
        x=0.5
    ),
    font=dict(size=14),
    width=800,
    height=600,
    showlegend=False
)
fig.update_layout(layout)
fig.show()


# **Obeservation:**
# 
# Supporting the above observation
# 
# * As you can, see clearly the distribution of open_channels decreases as the batch number increases
# 
# * Open channels are largely populated in range 1 - 5 range

# **Ion Channels through batches**

# In[ ]:


fig, axs = plt.subplots(5, 2, figsize=(15, 20))
axs = axs.flatten()
i = 0
for b, d in train_df.groupby('batch'):
    sns.violinplot(x='open_channels', y='signal', data=d, ax=axs[i])
    axs[i].set_title(f'Batch {b:0.0f}')
    i += 1
plt.tight_layout()


# **Observation**
# 
# From this plots, we can clearly see that the distribtuion of channels across the batches 

# **Time Vs Signal**

# In[ ]:


fig = px.line(train_df[:200000] , x='time', y='signal')
fig.show()


# **Observation**
# 
# From this graph we are not able understand clearly, so let's go for micro level views for better understanding.

# **Micro level view of Signals Batchwise**

# In[ ]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go


fig = make_subplots(rows=5, cols=2,  subplot_titles=tuple(["Batch No"+str(i) for i in range(0,10)]))

batch_no = 0 
for i in range(0,5):
    for j in range(0,2):
        temp = train_df.loc[train_df["batch"]==batch_no]
        temp = temp[:10000]
        batch_no+=1
        fig.add_trace(
            go.Scatter(
            x=temp['time'],
            y=temp['signal'],
               ),
            row=i+1, col=j+1      
         )
        fig.update_xaxes(title_text="Time", row=i+1, col=j+1)
        fig.update_yaxes(title_text="Signal", row=i+1, col=j+1)

fig.update_layout(height=1000, width=1200, title_text="Signal spread across the batches")

fig.show()


# The above graphs gives a better visualization compared to the pervious one.

# **Exploring the Test Data**

# In[ ]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go


fig = make_subplots(rows=2, cols=2,  subplot_titles=tuple(["Batch No"+str(i) for i in range(0,4)]))

batch_no = 0 
for i in range(0,2):
    for j in range(0,2):
        temp = test_df.loc[train_df["batch"]==batch_no]
        temp = temp[:10000]
        batch_no+=1
        fig.add_trace(
            go.Scatter(
            x=temp['time'],
            y=temp['signal'],
               ),
            row=i+1, col=j+1      
         )
        fig.update_xaxes(title_text="Time", row=i+1, col=j+1)
        fig.update_yaxes(title_text="Signal", row=i+1, col=j+1)

fig.update_layout(height=1000, width=1200, title_text="Signal spread across the batches")


# **Observations:**
# 
# The signals in the test data resembles the train data

# ## **What are we Prediciting?** <a id ="5"></a>

# As I mentioned earlier, we are trying to predict the number of open ion channels using electrophysiological signals from human cells. 
# 
# The evaluation metric used in this competition is macro F1 measure.
# 
# A macro-average will compute the metric independently for each class and then take the average (hence treating all classes equally)

# **References:**
# 
# * https://www.kaggle.com/robikscube/liverpool-ion-switching-data-exploration
# * https://www.kaggle.com/artgor/eda-and-model-qwk-optimization
# * https://www.kaggle.com/tarunpaparaju/ion-switching-competition-signal-eda
# 
# I would like share my thanks to these people for sharing their wonderful work

# **Kernel is under construction please stay tuned for more updates**

# **Please upvote the kernel if you find it useful**
