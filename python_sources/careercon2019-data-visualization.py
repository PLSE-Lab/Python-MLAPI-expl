#!/usr/bin/env python
# coding: utf-8

# # import libs

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from scipy.signal import periodogram

import dask.dataframe as dd


# # utils

# In[ ]:


def quaternion_to_euler_X(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = np.math.atan2(t0, t1)
    return X


def quaternion_to_euler_Y(x, y, z, w):
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = np.math.asin(t2)
    return Y


def quaternion_to_euler_Z(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = np.math.atan2(t3, t4)
    return Z


def get_series(df, series_id):
    return df[df.series_id == series_id].set_index("measurement_number")


def plot_single_series_element(pd_dataseries, ax):
    pd_dataseries.plot(ax = ax)
    
    
def plot_periodogram(pd_dataseries, ax):
    f, Pxx_den = periodogram(pd_dataseries)
    ax.semilogy(f[1:], Pxx_den[1:]) # drop first point (0,0)
    plt.xlabel('frequency')
    plt.ylabel('PSD')   
    
    
def plot_series(df, l_s_id, plot_PSD=False):
    l_feat_groups = ["euler_","angular_velocity_","linear_acceleration_"]
    l_feats = list(map(lambda s:[s+"X", s+"Y", s+"Z"], l_feat_groups))
    l_feats = l_feats[0] + l_feats[1] + l_feats[2] # i know i don't like it either but it works
    l_surf = []
    fig, l_ax_np = plt.subplots(3,3,figsize=(20,9))
    l_ax = l_ax_np.flatten().tolist()
    for s_id in l_s_id:
        i=0
        df_series = get_series(df, s_id)
        l_surf.append(df_series.surface.iloc[0])
        for feat in l_feats:
            if plot_PSD:
                plot_periodogram(df_series[feat], l_ax[i])
            else:
                plot_single_series_element(df_series[feat],l_ax[i])
            l_ax[i].set_title(feat)
            i+=1
    plt.legend(l_surf)
    plt.tight_layout()
    plt.show()


# # Load Data

# In[ ]:


df_train = pd.read_csv("../input/X_train.csv")
df_target_train = pd.read_csv("../input/y_train.csv")

df_train = df_train.join(df_target_train,on='series_id',rsuffix='_').drop("series_id_",1)
df_train.head()


# ### Converting quaternions to euler angles

# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_train["euler_X"] = df_train.apply(lambda r:quaternion_to_euler_X(r.orientation_X,r.orientation_Y,r.orientation_Z,r.orientation_W),1)\ndf_train["euler_Y"] = df_train.apply(lambda r:quaternion_to_euler_Y(r.orientation_X,r.orientation_Y,r.orientation_Z,r.orientation_W),1)\ndf_train["euler_Z"] = df_train.apply(lambda r:quaternion_to_euler_Z(r.orientation_X,r.orientation_Y,r.orientation_Z,r.orientation_W),1)')


# > ![small](https://upload.wikimedia.org/wikipedia/en/thumb/3/30/Plane_with_ENU_embedded_axes.svg/425px-Plane_with_ENU_embedded_axes.svg.png)
# from [wikipedia](https://en.wikipedia.org/wiki/Euler_angles#Conventions_2)

# ---
# # Data Viz

# In[ ]:


df_target_train.sort_values(by='group_id').head()


# In[ ]:


#concrete only
plot_series(df_train, [151,152])


# In[ ]:


df_target_train.sort_values(by='group_id').iloc[300:].head()


# In[ ]:


# carpet only
plot_series(df_train, [520,528])


# In[ ]:


# carpet vs concrete
plot_series(df_train, [151,528])


# # Visualizing in Frequency Space

# In[ ]:


#concrete only
plot_series(df_train, [151,152], plot_PSD=True)


# In[ ]:


# carpet only
plot_series(df_train, [520,528], plot_PSD=True)


# In[ ]:


# carpet vs concrete
plot_series(df_train, [151,528], plot_PSD=True)

