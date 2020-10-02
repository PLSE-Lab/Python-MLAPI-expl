#!/usr/bin/env python
# coding: utf-8

# To look at and get an insight from the price movements of stocks, I simply draw these graphs.
# Assumed that the target 'y' sequence is a differenced time series data, the (scaled) price can be calculated by cumulative sum. 
# If I want to cluster assets using the other features to improve a prediction model, after that, I can simply check the clusters' practical validity by looking at these time series.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

with pd.HDFStore("../input/train.h5", "r") as train:
    df = train.get("train")


# In[ ]:


timemax = max(df["timestamp"])
timemin = min(df["timestamp"])
xlim = [timemin, timemax]

for asset in df["id"].unique() :
    #print(df["id"=asset])
    x = df[df["id"]==asset]["timestamp"]
    diffy = df[df["id"]==asset]["y"]
    y = np.cumsum(diffy)
    
    plt.figure(figsize=(9,1))
    plt.plot(x, y, 'k-')
    plt.plot(x, diffy, 'b-')
    plt.xlim(xlim)
    plt.title("ID # %s" %(asset),size=10)
    
    tmax = max(x)
    ax = plt.subplot()
    ax.axvline(tmax, color='r', linestyle='--')

    if asset > 50 :
        break;

