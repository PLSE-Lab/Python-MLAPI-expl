#!/usr/bin/env python
# coding: utf-8

# Lets have a look over the data first and then perform the basic EDA.

# In[ ]:


import pandas as pd
calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
calendar.fillna("missing")

sell_prices  =pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
sales_train_validation = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")


# In[ ]:


calendar


# In[ ]:


sell_prices


# In[ ]:


sales_train_validation


# In[ ]:


sales_train_validation['updated_ID'] = sales_train_validation.id.apply(lambda x:"_".join(x.split("_")[:-1]))


# In[ ]:


sales_train_validation[['dept_id',"cat_id","store_id","item_id"]].nunique()


#  Note: This notebook is almost a duplicate of [this kernel](https://www.kaggle.com/tarunpaparaju/m5-competition-eda-models). I have just created this learning purpose . You can have look at the original kernel too.

# In[ ]:


# reference from https://www.kaggle.com/tarunpaparaju/m5-competition-eda-models
import os
import gc
import time
import math
import datetime
from math import log, floor
from sklearn.neighbors import KDTree

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle
from tqdm.notebook import tqdm as tqdm

import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import pywt
from statsmodels.robust import mad

import scipy
import statsmodels
from scipy import signal
import statsmodels.api as sm
from fbprophet import Prophet
from scipy.signal import butter, deconvolve
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


ids = sorted(list(set(sales_train_validation['id'])))
d_cols = [c for c in sales_train_validation.columns if 'd_' in c]
x_1 = sales_train_validation.loc[sales_train_validation['id'] == ids[2]].set_index('id')[d_cols].values[0]
x_2 = sales_train_validation.loc[sales_train_validation['id'] == ids[1]].set_index('id')[d_cols].values[0]
x_3 = sales_train_validation.loc[sales_train_validation['id'] == ids[17]].set_index('id')[d_cols].values[0]

fig = make_subplots(rows=3, cols=1)

fig.add_trace(go.Scatter(x=np.arange(len(x_1)), y=x_1, showlegend=False,
                    mode='lines', name="First sample",
                         marker=dict(color="red")),
             row=1, col=1)

fig.add_trace(go.Scatter(x=np.arange(len(x_2)), y=x_2, showlegend=False,
                    mode='lines', name="Second sample",
                         marker=dict(color="violet")),
             row=2, col=1)

fig.add_trace(go.Scatter(x=np.arange(len(x_3)), y=x_3, showlegend=False,
                    mode='lines', name="Third sample",
                         marker=dict(color="dodgerblue")),
             row=3, col=1)

fig.update_layout(height=1200, width=800, title_text="Sample sales")
fig.show()


# Above plot represents sales of 3 randomly choosen stores. We can see that these sales are so inconsistent and hence we can infer that sales of any store depends upon various other features as we see on some days these sales have gone down to 0 , may be there was no stock or some kind of problem in that area the store didn't open or even the customers liked the product so much that they have stocked it in there house as they want to go out.

# In[ ]:


ids = sorted(list(set(sales_train_validation['id'])))
d_cols = [c for c in sales_train_validation.columns if 'd_' in c]
x_1 = sales_train_validation.loc[sales_train_validation['id'] == ids[2]].set_index('id')[d_cols].values[0][:90]
x_2 = sales_train_validation.loc[sales_train_validation['id'] == ids[1]].set_index('id')[d_cols].values[0][0:90]
x_3 = sales_train_validation.loc[sales_train_validation['id'] == ids[17]].set_index('id')[d_cols].values[0][0:90]
fig = make_subplots(rows=3, cols=1)

fig.add_trace(go.Scatter(x=np.arange(len(x_1)), y=x_1, showlegend=False,
                    mode='lines+markers', name="First sample",
                         marker=dict(color="mediumseagreen")),
             row=1, col=1)

fig.add_trace(go.Scatter(x=np.arange(len(x_2)), y=x_2, showlegend=False,
                    mode='lines+markers', name="Second sample",
                         marker=dict(color="violet")),
             row=2, col=1)

fig.add_trace(go.Scatter(x=np.arange(len(x_3)), y=x_3, showlegend=False,
                    mode='lines+markers', name="Third sample",
                         marker=dict(color="dodgerblue")),
             row=3, col=1)

fig.update_layout(height=1200, width=800, title_text="Sample sales snippets")
fig.show()


# In the above plots i have just plotted graph of sales for first 90 days of same store which i had used in the previous graph. As you can see , the sales are very volatile as stated before too.

# # Denoising Data

# Denoising is a technique to find trend in time-series data , although it may loose some useful information in the data but it helps us to understand the trend more better.

# ## Wavelet Denoising

# Wavelet denoising (usually used with electric signals) is a way to remove the unnecessary noise from a time series. This method calculates coefficients called the "wavelet coefficients". These coefficients decide which pieces of information to keep (signal) and which ones to discard (noise).
# 
# We make use of the MAD (mean absolute deviation) value to understand the randomness in the sales and accordingly decide the minimum threshold for the wavelet coefficients in the time series. We filter out the low coefficients from the wavelets and reconstruct the sales data from the remaining coefficients and that's it; we have successfully removed noise from the sales data.

# In[ ]:


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    return pywt.waverec(coeff, wavelet, mode='per')


# In[ ]:


y_w1 = denoise_signal(x_1)
y_w2 = denoise_signal(x_2)
y_w3 = denoise_signal(x_3)


fig = make_subplots(rows=3, cols=1)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_1)), mode='lines+markers', y=x_1, marker=dict(color="mediumaquamarine"), showlegend=False,
               name="Original signal"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_1)), y=y_w1, mode='lines', marker=dict(color="darkgreen"), showlegend=False,
               name="Denoised signal"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_2)), mode='lines+markers', y=x_2, marker=dict(color="thistle"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_2)), y=y_w2, mode='lines', marker=dict(color="purple"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_3)), mode='lines+markers', y=x_3, marker=dict(color="lightskyblue"), showlegend=False),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_3)), y=y_w3, mode='lines', marker=dict(color="navy"), showlegend=False),
    row=3, col=1
)

fig.update_layout(height=1200, width=800, title_text="Original (pale) vs. Denoised (dark) sales")
fig.show()


# In[ ]:


fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(30, 20))

ax[0, 0].plot(x_1, color='seagreen', marker='o') 
ax[0, 0].set_title('Original Sales', fontsize=24)
ax[0, 1].plot(y_w1, color='red', marker='.') 
ax[0, 1].set_title('After Wavelet Denoising', fontsize=24)

ax[1, 0].plot(x_2, color='seagreen', marker='o') 
ax[1, 0].set_title('Original Sales', fontsize=24)
ax[1, 1].plot(y_w2, color='red', marker='.') 
ax[1, 1].set_title('After Wavelet Denoising', fontsize=24)

ax[2, 0].plot(x_3, color='seagreen', marker='o') 
ax[2, 0].set_title('Original Sales', fontsize=24)
ax[2, 1].plot(y_w3, color='red', marker='.') 
ax[2, 1].set_title('After Wavelet Denoising', fontsize=24)

plt.show()


# The above graph shows the actual time series graph on the left side and on right side we have shown wavelet denoised time series data, just used to extract trends from the original data.

# ## Average Smoothing

# A different technique to extract trends from our data.Both the Wavelet and Avg. smoothing techniques provides different trends in our data , but to use them as data to train model on , is not a good choice as they may lack some important details about the data which we might require for better performance.

# In[ ]:


def average_smoothing(signal, kernel_size=5, stride=1):
    sample = []
    start = 0
    end = kernel_size
    while end <= len(signal):
        start = start + stride
        end = end + stride
        sample.extend(np.ones(end - start)*np.mean(signal[start:end]))
    return np.array(sample)


# In[ ]:


y_a1 = average_smoothing(x_1)
y_a2 = average_smoothing(x_2)
y_a3 = average_smoothing(x_3)

fig = make_subplots(rows=3, cols=1)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_1)), mode='lines+markers', y=x_1, marker=dict(color="lightskyblue"), showlegend=False,
               name="Original sales"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_1)), y=y_a1, mode='lines', marker=dict(color="navy"), showlegend=False,
               name="Denoised sales"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_2)), mode='lines+markers', y=x_2, marker=dict(color="thistle"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_2)), y=y_a2, mode='lines', marker=dict(color="indigo"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_3)), mode='lines+markers', y=x_3, marker=dict(color="mediumaquamarine"), showlegend=False),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_3)), y=y_a3, mode='lines', marker=dict(color="darkgreen"), showlegend=False),
    row=3, col=1
)

fig.update_layout(height=1200, width=800, title_text="Original (pale) vs. Denoised (dark) signals")
fig.show()


# In[ ]:


fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(30, 20))

ax[0, 0].plot(x_1, color='seagreen', marker='o') 
ax[0, 0].set_title('Original Sales', fontsize=24)
ax[0, 1].plot(y_w1, color='red', marker='.') 
ax[0, 1].set_title('After Wavelet Denoising', fontsize=24)
ax[0, 2].plot(y_a1, color='red', marker='.') 
ax[0, 2].set_title('After Average_smoothing', fontsize=24)


ax[1, 0].plot(x_2, color='seagreen', marker='o') 
ax[1, 0].set_title('Original Sales', fontsize=24)
ax[1, 1].plot(y_w2, color='red', marker='.') 
ax[1, 1].set_title('After Wavelet Denoising', fontsize=24)
ax[1, 2].plot(y_a2, color='red', marker='.') 
ax[1, 2].set_title('After Average Smoothing', fontsize=24)

ax[2, 0].plot(x_3, color='seagreen', marker='o') 
ax[2, 0].set_title('Original Sales', fontsize=24)
ax[2, 1].plot(y_w3, color='red', marker='.') 
ax[2, 1].set_title('After Wavelet Denoising', fontsize=24)
ax[2, 2].plot(y_a3, color='red', marker='.') 
ax[2, 2].set_title('After Average Smoothing', fontsize=24)

plt.show()


# here , i have plotted the actual data, wavelet denoised trend data, and average smoothed trend data. Just to show there representations with each other. 

# In[ ]:


def max_val_smoothing(signal, kernel_size=5, stride=1):
    sample = []
    start = 0
    end = kernel_size
    while end <= len(signal):
        start = start + stride
        end = end + stride
        sample.extend(np.ones(end - start)*np.ma(signal[start:end]))
    return np.array(sample)


# In[ ]:


y_a1 = average_smoothing(x_1)
y_a2 = average_smoothing(x_2)
y_a3 = average_smoothing(x_3)

fig = make_subplots(rows=3, cols=1)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_1)), mode='lines+markers', y=x_1, marker=dict(color="lightskyblue"), showlegend=False,
               name="Original sales"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_1)), y=y_a1, mode='lines', marker=dict(color="navy"), showlegend=False,
               name="Denoised sales"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_2)), mode='lines+markers', y=x_2, marker=dict(color="thistle"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_2)), y=y_a2, mode='lines', marker=dict(color="indigo"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_3)), mode='lines+markers', y=x_3, marker=dict(color="mediumaquamarine"), showlegend=False),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(x_3)), y=y_a3, mode='lines', marker=dict(color="darkgreen"), showlegend=False),
    row=3, col=1
)

fig.update_layout(height=1200, width=800, title_text="Original (pale) vs. Denoised (dark) signals")
fig.show()


# The above plot shows details the same way  as we get from  average smoothing , but just a small difference is that we have used max value of the window instead of the mean value to plot the graph , showing a bit of different trend front the averge one. 

# In[ ]:


past_sales = sales_train_validation.set_index('id')[d_cols]     .T     .merge(calendar.set_index('d')['date'],
           left_index=True,
           right_index=True,
            validate='1:1') \
    .set_index('date')

store_list = sell_prices['store_id'].unique()


# In[ ]:


means = []
fig = go.Figure()
for s in store_list:
  store_items = [c for c in past_sales.columns if s in c]
  data = past_sales[store_items].sum(axis=1).rolling(90).mean()
  means.append(np.mean(past_sales[store_items].sum(axis=1)))
  fig.add_trace(go.Scatter(x=np.arange(len(data)), y=data, name=s))
    
fig.update_layout(yaxis_title="Sales", xaxis_title="Time", title="Rolling Average Sales vs. Time (per store)")


# Here we have plotted rolling means of 90 days of the sales based of different store present in clifornia , texas and winsonica locations.
# Showing various trends in there sales. 
# 
# We cans see that CA-3 has maintained high sales over the given period of time , but it looks like a kind of period graph which after a time gets reduced in sales but then again it heads up to overcome the previous top mark. 
# 
# An all time increase and maintained sales have been marked in WI_2. there is no such time they have got there sales reduced to much.

# In[ ]:


fig = go.Figure()

for i, s in enumerate(store_list):
  store_items = [c for c in past_sales.columns if s in c]
  data = past_sales[store_items].sum(axis=1).rolling(90).mean()
  fig.add_trace(go.Box(x=[s]*len(data), y=data, name=s))
    
fig.update_layout(yaxis_title="Sales", xaxis_title="Time", title="Rolling Average Sales vs. Store name ")


# The box plot shows the sales ditribution in different Stores provided in the dataset. We can infer that california sales vs stores have the highest variance and california has also maintained the highest overall mean sale. Where other Texas and Winsonica have maintained there sales with no so much varation and also each store in Texas have similar sales ,whereas in Winsonica there is high varaince in sales of each store

# In[ ]:


df = pd.DataFrame(np.transpose([means, store_list]))
df.columns = ["Mean sales", "Store name"]
px.bar(df, y="Mean sales", x="Store name", color="Store name", title="Mean sales vs. Store name")


# The baar plot shows all sales of different stores over a period of time. 

# In[ ]:


greens = ["darkgreen", "mediumseagreen", "seagreen", "green"]
store_list = sell_prices['store_id'].unique()
fig = go.Figure()
means = []
stores = []
for i, s in enumerate(store_list):
    if "tx" in s or "TX" in s:
        store_items = [c for c in past_sales.columns if s in c]
        data = past_sales[store_items].sum(axis=1).rolling(90).mean()
        means.append(np.mean(past_sales[store_items].sum(axis=1)))
        stores.append(s)
        fig.add_trace(go.Scatter(x=np.arange(len(data)), y=data, name=s, marker=dict(color=greens[i%len(greens)])))
        
fig.update_layout(yaxis_title="Sales", xaxis_title="Time", title="Rolling Average Sales vs. Time (Texas)")


# In[ ]:


fig = go.Figure()

for i, s in enumerate(store_list):
    if "tx" in s or "TX" in s:
        store_items = [c for c in past_sales.columns if s in c]
        data = past_sales[store_items].sum(axis=1).rolling(90).mean()
        fig.add_trace(go.Box(x=[s]*len(data), y=data, name=s, marker=dict(color=greens[i%len(greens)])))
    
fig.update_layout(yaxis_title="Sales", xaxis_title="Time", title="Rolling Average Sales vs. Store name (Texas)")


# In[ ]:


df = pd.DataFrame(np.transpose([means, stores]))
df.columns = ["Mean sales", "Store name"]
px.bar(df, y="Mean sales", x="Store name", color="Store name", title="Mean sales vs. Store name", color_continuous_scale=greens)


fig = go.Figure(data=[
    go.Bar(name='', x=stores, y=means, marker={'color' : greens})])

fig.update_layout(title="Mean sales vs. Store name (Texas)", yaxis=dict(title="Mean sales"), xaxis=dict(title="Store name"))
fig.update_layout(barmode='group')
fig.show()


# In the above plots ,  we can see how the sales in 3 different store in texas variate over time , there distributions in overall texas area. We can see that in the some initial time period when the sales of one store were increasing ,the other were side by side to each other .It is like , the crowd in region of store 1 liked the product when the people near other stores didn't. And after some time it got reversed. THe sales of store 1 went down to be almost same as the other two.
# 
# We conclude that store 2 TX_2 has the maximum sales and TX_1 has the minimum sales.

# Note : The same can be also be done for other cities too and details analytics and there trends can be extracted.

# In[ ]:




