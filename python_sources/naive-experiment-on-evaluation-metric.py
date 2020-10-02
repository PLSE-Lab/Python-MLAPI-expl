#!/usr/bin/env python
# coding: utf-8

# <font color=lightseagreen size=6><b> Naive Experiment on evaluation metric </b></font>
# <br>
# <font color=lightseagreen size=5><b> 2 sigma : Using News to Predict Stock Movements </b></font>

# ![hedge](https://cdn-ak.f.st-hatena.com/images/fotolife/g/greenwind120170/20181002/20181002214157.jpg)

# <font size=4>
# In this competition, we evaluate our models with following metric.  
# </font>  
#   
# $$x_t = \Sigma_{i} \; \hat{y}_{ti} r_{ti} u_{ti}$$  
#   
# $$\mathrm{score} = \frac{\bar{x_t}}{\sigma(x_t)}$$  
#   
# $r_{ti}$ is the 10-day market-adjusted leading return for day t for instrument i,  
# so this metric is a kind of [Information Ratio](https://www.investopedia.com/terms/i/informationratio.asp) , I think.  
#   
# I did a naive experiment on this metric and got some insights about validation strategy in this competition.  

# # Load packages

# In[ ]:


from kaggle.competitions import twosigmanews
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import gc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(font_scale=1)

import warnings
import missingno as msno

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None
# dir(pd.options.display)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

plt.style.use('ggplot')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# # make 2sigma kaggle env

# In[ ]:


env = twosigmanews.make_env()


# We will concentrate on market data.   
# Let's delete news data.

# In[ ]:


(market_train, news_train) = env.get_training_data()
del news_train
gc.enable()
gc.collect()


# In @jannesklaas awesome [kernel](https://www.kaggle.com/jannesklaas/lb-0-63-xgboost-baseline),   
# > Stocks can only go up or down, if the stock is not going up, it must go down (at least a little bit). So if we know our model confidence in the stock going up, then our new confidence is:
# > $$\hat{y}=up-(1-up)=2*up-1$$
# > 
# > We are left with a "simple" binary classification problem, for which there are a number of good tool, here we use XGBoost, but pick your poison.  
#   
# I followed his setting.  
# Addition to it, I assumed that we managed to get the perfect model.  
# It means that when stocks go up, our prediction is always 1, vice versa.

# In[ ]:


ret = market_train.returnsOpenNextMktres10
univ = market_train.universe
label = (ret > 0).astype(int)


# Define the function to calculate evalution metric.  
# This function has `window` arg to limit the range of time.  
# Within this function, we calculate the perfect confidence value.  

# In[ ]:


def ir(label, window):
    global market_train, ret, univ
    time_idx = market_train.time.factorize()[0]
    # (label * 2 - 1) : perfect confidence value
    x_t = (label * 2 - 1) * ret * univ
    x_t_sum = x_t.groupby(time_idx).sum()
    x_t_sum = x_t_sum[window:]
    score = x_t_sum.mean() / x_t_sum.std()
    return score


# Move by 10 operational days ( ~ 252days / year ), calculate scores.  

# In[ ]:


ir_l = [ir(label, t) for t in range(0, market_train.time.nunique(), 10)]


# In[ ]:


trace = go.Scatter(
    x = np.arange(0, market_train.time.nunique(), 10),
    y = ir_l,
    mode = 'lines+markers',
    marker = dict(
        size = 4,
        color = 'lightblue'
    ),
    line = dict(
        width = 1
    )
)
data = [trace]
layout = go.Layout(dict(
    title = 'Eval Metric trend',
    xaxis = dict(title = 'operational days passed ( window start point )'),
    yaxis = dict(title = 'Evaluation metric'),
    height = 400,
    width = 750
))
py.iplot(dict(data=data, layout=layout), filename='IR trend')


# From above pitcure,   
# it is obvious that in the early stage of this data period ( 2007 ~ 2008 ), score is too low **due to its high volatility** .  
# Below picture shows the standard deviation of `returnsOpenPrevRaw1` with time.  

# In[ ]:


op = ['mean', 'std']
df = market_train[['time', 'returnsOpenPrevRaw1']].groupby('time').agg({
    'returnsOpenPrevRaw1' : op,
}).reset_index()
df.columns = ['time'] + [o + '_returnsOpenPrevRaw1' for o in op]


# In[ ]:


trace = go.Scatter(
    x = df.time,
    y = df.std_returnsOpenPrevRaw1,
    mode = 'lines+markers',
    marker = dict(
        size = 4,
        color = 'pink'
    ),
    line = dict(
        width = 1
    )
)
data = [trace]
layout = go.Layout(dict(
    title = 'std of returnsOpenPrevRaw1',
    xaxis = dict(title = 'date'),
    yaxis = dict(title = 'std of returnsOpenPrevRaw1'),
    height = 400,
    width = 750
))
py.iplot(dict(data=data, layout=layout), filename='.')


# <font size=4 color=deeppink>
# My conclusion is,  
# </font>
# <br>
# - Including data within 2007 ~ 2008 to bulid or evaluate model will not be good due to its high volatility.  
# <br>
# - The current financial market is similar to 2009 ~ 2017 rather than 2007 ~ 2008.  
# <br>
# - The possibility of the shock like Lehman or Pariba occured will be very low with the current situation.  
#     ( some rigid law like Basel III and Solvency II will protect the world to some extent. )
