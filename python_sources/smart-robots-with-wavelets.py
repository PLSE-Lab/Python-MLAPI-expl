#!/usr/bin/env python
# coding: utf-8

# # A starter view of robots moves with continuous wavelet transform 
# 
# Vibration caused by motors, ooching, woobling from the soil surface, and so one, are prone to frequency analysis.
# 
# Continuous wavelet transform is an excellent tool for booth visual analysis and production of features for the next block in the chain.
# 
# The next few examples show instructive patterns. The input signal needs some more processing and spectrograms may be combined to produce usefull cross-spectra, which I won't put here otherwise it would be too easy ;-)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scaleogram as scg
import matplotlib.pyplot as plt

Xtr = pd.read_csv("../input/X_train.csv") # row_id series_id measurement_number orientation_X orientation_Y orientation_Z orientation_W 
ytr = pd.read_csv("../input/y_train.csv") #series_id group_id surface
Xtr.head(1)
ytr.head(1)


# In[ ]:


def plot_serie(df, id):
    surface = ytr[ytr.series_id==id].surface.values[0]
    fig = plt.figure(figsize=(16, 4))
    ax = plt.subplot(1, 3, 1)
    for orient in [ 'X', 'Y', 'Z', 'W']:
        X = df[df.series_id==id]['orientation_'+orient].values
        plt.plot(X - np.mean(X), label='o'+orient)
    plt.legend(); plt.title("Orientation (%s)" %(surface)); 
    
    plt.subplot(1, 3, 2)
    for orient in ['X', 'Y', 'Z']:
        X = df[df.series_id==id]['angular_velocity_'+orient].values
        plt.plot(X,label=orient)
    plt.legend(); plt.title("Angular velocity (%s)" %(surface))

    plt.subplot(1, 3, 3)
    for orient in ['X', 'Y', 'Z']:
        X = df[df.series_id==id]['linear_acceleration_'+orient].values
        plt.plot(X,label=orient)
    plt.legend(); plt.title("Linear acceleration (%s)" %(surface))

    fig = plt.figure(figsize=(18,9))
    for i, orient in enumerate(['X', 'Y', 'Z', 'W']):
        ax=plt.subplot(4, 3, 1+3*i)
        X = df[df.series_id==id]['orientation_'+orient].values
        scales = scg.periods2scales(10**np.linspace(np.log10(1), np.log10(60), 50))
        scg.cws(X-X.mean(), scales=scales, ax=ax, yscale='log', cscale='log', clim=(1e-6, 1e-4))
        i == 0 and ax.set_title('Orientation (%s)'%(surface))
    for i, orient in enumerate(['X', 'Y', 'Z']):
        ax=plt.subplot(4, 3, 2+3*i)
        X = df[df.series_id==id]['angular_velocity_'+orient].values
        scales = scg.periods2scales(10**np.linspace(np.log10(1), np.log10(60), 50))
        scg.cws(X-X.mean(), scales=scales, ax=ax, yscale='log', clim=(0, 0.1))
        i == 0 and ax.set_title('Angular Velocity (%s)'%(surface))
    for i, orient in enumerate(['X', 'Y', 'Z']):
        ax=plt.subplot(4, 3, 3+3*i)
        scales = scg.periods2scales(10**np.linspace(np.log10(1), np.log10(60), 50))
        X = df[df.series_id==id]['linear_acceleration_'+orient].values
        scg.cws(X-X.mean(), scales=scales, ax=ax, yscale='log', clim=(0,3))
        i ==0 and ax.set_title('Linear Acceleration (%s)' %(surface))
    
plot_serie(Xtr,0)
    


# In[ ]:


plot_serie(Xtr,1)


# In[ ]:


plot_serie(Xtr,2)


# In[ ]:


plot_serie(Xtr,3)


# In[ ]:


plot_serie(Xtr,4)


# In[ ]:


plot_serie(Xtr,5)


# In[ ]:


plot_serie(Xtr,6)


# In[ ]:


plot_serie(Xtr,7)


# In[ ]:


plot_serie(Xtr,8)


# In[ ]:


plot_serie(Xtr,9)


# In[ ]:


plot_serie(Xtr,10)

