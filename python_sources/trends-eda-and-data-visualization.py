#!/usr/bin/env python
# coding: utf-8

# # TReNDS Neuroimaging
# 

# <img src="https://i.ibb.co/k2xt7Sy/i-Stock-614832326-800x400.jpg" width="700"> 
# * **Source: https://medicinehealth.leeds.ac.uk/**<br>

# Neuroimaging or brain imaging is the use of various techniques to either directly or indirectly image the structure, function, or pharmacology of the nervous system. It is a relatively new discipline within medicine, neuroscience, and psychology. Physicians who specialize in the performance and interpretation of neuroimaging in the clinical setting are neuroradiologists.
# 
# Neuroimaging falls into two broad categories:
# 
# * Structural imaging, which deals with the structure of the nervous system and the diagnosis of gross (large scale) intracranial disease (such as a tumor) and injury.
#  
# * Functional imaging, which is used to diagnose metabolic diseases and lesions on a finer scale (such as Alzheimer's disease) and also for neurological and cognitive psychology research and building brain-computer interfaces.
# 
# Functional imaging enables, for example, the processing of information by centers in the brain to be visualized directly. Such processing causes the involved area of the brain to increase metabolism and "light up" on the scan. One of the more controversial uses of neuroimaging has been researching "thought identification" or mind-reading. (From Wikipedia)

# ## To be continued... Stay Tuned! If you like the kernel, Please upvote.

# In[ ]:


import pandas as pd
import math
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from wordcloud import WordCloud

init_notebook_mode(connected=True) 

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df_fnc = pd.read_csv("/kaggle/input/trends-assessment-prediction/fnc.csv")
df_loading = pd.read_csv("/kaggle/input/trends-assessment-prediction/loading.csv")
df_train_scores = pd.read_csv("/kaggle/input/trends-assessment-prediction/train_scores.csv")


# In[ ]:


df_fnc.head()


# In[ ]:


print("fnc dataset total row number: {0} \nfnc dataset Total Col Number: {1}".format(df_fnc.shape[0], df_fnc.shape[1]))


# In[ ]:


df_loading.head()


# In[ ]:


print("loading dataset total row number: {0} \nloading dataset Total Col Number: {1}".format(df_loading.shape[0], df_loading.shape[1]))


# More info for dataset:

# In[ ]:


df_fnc.info()


# In[ ]:


df_loading.describe().T


# In[ ]:


f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(df_loading.iloc[:,1:].corr(),annot=True, 
            linewidths=.5, fmt='.1f', ax=ax, cbar=0
           )

plt.show()


# In[ ]:


df_train_scores.head()


# There are NaN values in Top 5 record. To learn the number of NaN values:

# In[ ]:


print(df_train_scores.loc[:, df_train_scores.isnull().any()].isnull().sum())


# In[ ]:


df_train_scores.describe().T


# In[ ]:


fig = px.histogram(x=df_train_scores["age"],
                   title='Distribution of Age',
                   opacity=0.8,
                   color_discrete_sequence=['darkorange'],
                   nbins=30 )

fig.update_layout(
    yaxis_title_text='',
    xaxis_title_text='',
    height=450, width=600)

fig.update_traces(marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.8
                 )

fig.show()


# In[ ]:


fig = make_subplots(rows=2, cols=2, subplot_titles=('domain1_var1', 'domain1_var2','domain2_var1', 'domain2_var2'))

fig.add_trace(go.Histogram(x=df_train_scores["domain1_var1"],
                      marker_color='#FF9999',
                      opacity=0.2,
                      nbinsx=50),
    row=1, col=1)

fig.add_trace(go.Histogram(x=df_train_scores["domain1_var2"],
                      marker_color='#FF9999',
                      opacity=0.2,
                      nbinsx=40),
    row=1, col=2)

fig.add_trace(go.Histogram(x=df_train_scores["domain2_var1"],
                      marker_color='#FF9999',
                      opacity=0.2,
                      nbinsx=40),
    row=2, col=1)

fig.add_trace(go.Histogram(x=df_train_scores["domain2_var2"],
                      marker_color='#FF9999',
                      opacity=0.2,
                      nbinsx=40),
    row=2, col=2)


fig.update_layout(
    height=500, width=800, 
    showlegend=False,
    title="Distribution of domain_vars"
)


fig.update_traces(marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.8
                 )

fig.show()


# In[ ]:


loading_cols = df_loading.columns[1:]

fig = make_subplots(rows=13, cols=2, 
                    subplot_titles=loading_cols
                   )

for i, col in enumerate(loading_cols):
    fig.add_trace(go.Histogram(x=df_loading[col],
                      marker_color='rosybrown',
                      opacity=0.2,
                      nbinsx=50),
    row=math.ceil((i+1)/2), col=(i%2)+1)

fig.update_layout(
    height=2500, width=800, showlegend=False,
    title="Distribution of ICs"
)


fig.update_traces(marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.8
                 )

fig.show()    

