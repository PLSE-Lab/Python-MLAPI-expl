#!/usr/bin/env python
# coding: utf-8

# This is an uncommented notebook showing kill/death heatmaps of both erangel and miramar in 5 minute slices. red is 2+ kills/death, and purple is near-0. 

# In[1]:


# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.misc.pilutil import imread


df = pd.read_csv('../input/PUBG_MatchData_Flattened.tsv', sep='\t')

edf = df.loc[df['map_id'] == 'ERANGEL']
mdf = df.loc[df['map_id'] == 'MIRAMAR']

# print(edf.head())
# print(mdf.head())

def killer_victim_time_df_maker(df):
    victim_x_df = df.filter(regex='victim_position_x')
    victim_y_df = df.filter(regex='victim_position_y')
    killer_x_df = df.filter(regex='killer_position_x')
    killer_y_df = df.filter(regex='killer_position_y')
    time_df = df.filter(regex = 'time_event')

    victim_x_s = pd.Series(victim_x_df.values.ravel('F'))
    victim_y_s = pd.Series(victim_y_df.values.ravel('F'))
    killer_x_s = pd.Series(killer_x_df.values.ravel('F'))
    killer_y_s = pd.Series(killer_y_df.values.ravel('F'))
    time_s = pd.Series(time_df.values.ravel('F'))

    vdata={'x': victim_x_s, 'y':victim_y_s, 't':time_s}
    kdata={'x': killer_x_s, 'y':killer_y_s, 't':time_s}


    victim_df = pd.DataFrame(data = vdata).dropna(how='any')
    victim_df = victim_df[victim_df['x']>0]
    killer_df = pd.DataFrame(data = kdata).dropna(how='any')
    killer_df = killer_df[killer_df['x']>0]
    return killer_df,victim_df

ekdf,evdf = killer_victim_time_df_maker(edf)
mkdf,mvdf = killer_victim_time_df_maker(mdf)


# In[2]:


ekdf_0_5 = ekdf[ekdf['t']<=300]
ekdf_5_10 = ekdf[(300 <=  ekdf['t']) &  (ekdf['t'] <= 600)]
ekdf_10_15 = ekdf[(600<=ekdf['t']) & (ekdf['t']<=900)]
ekdf_15_20 = ekdf[(900<=ekdf['t']) & (ekdf['t']<=1200)]
ekdf_20_25 = ekdf[(1200<=ekdf['t']) & (ekdf['t']<=1500)]
ekdf_25_30 = ekdf[(1500<=ekdf['t']) & (ekdf['t']<=1800)]
ekdf_30_inf = ekdf[1800<=ekdf['t']]

evdf_0_5 = evdf[evdf['t']<=300]
evdf_5_10 = evdf[(300<=evdf['t']) & (evdf['t']<=600)]
evdf_10_15 = evdf[(600<=evdf['t']) & (evdf['t']<=900)]
evdf_15_20 = evdf[(900<=evdf['t']) & (evdf['t']<=1200)]
evdf_20_25 = evdf[(1200<=evdf['t']) & (evdf['t']<=1500)]
evdf_25_30 = evdf[(1500<=evdf['t']) & (evdf['t']<=1800)]
evdf_30_inf = evdf[1800<=evdf['t']]

mkdf_0_5 = mkdf[mkdf['t']<=300]
mkdf_5_10 = mkdf[(300<=mkdf['t']) & (mkdf['t']<=600)]
mkdf_10_15 = mkdf[(600<=mkdf['t']) & (mkdf['t']<=900)]
mkdf_15_20 = mkdf[(900<=mkdf['t']) & (mkdf['t']<=1200)]
mkdf_20_25 = mkdf[(1200<=mkdf['t']) & (mkdf['t']<=1500)]
mkdf_25_30 = mkdf[(1500<=mkdf['t']) & (mkdf['t']<=1800)]
mkdf_30_inf = mkdf[1800<=mkdf['t']]

mvdf_0_5 = mvdf[mvdf['t']<=300]
mvdf_5_10 = mvdf[(300<=mvdf['t']) & (mvdf['t']<=600)]
mvdf_10_15 = mvdf[(600<=mvdf['t']) & (mvdf['t']<=900)]
mvdf_15_20 = mvdf[(900<=mvdf['t']) & (mvdf['t']<=1200)]
mvdf_20_25 = mvdf[(1200<=mvdf['t']) & (mvdf['t']<=1500)]
mvdf_25_30 = mvdf[(1500<=mvdf['t']) & (mvdf['t']<=1800)]
mvdf_30_inf = mvdf[1800<=mvdf['t']]


# In[3]:


plot_data_ek_0_5 = ekdf_0_5[['x', 'y']].values*4040/800000
plot_data_ek_5_10 = ekdf_5_10[['x', 'y']].values*4040/800000
plot_data_ek_10_15 = ekdf_10_15[['x', 'y']].values*4040/800000
plot_data_ek_15_20 = ekdf_15_20[['x', 'y']].values*4040/800000
plot_data_ek_20_25 = ekdf_20_25[['x', 'y']].values*4040/800000
plot_data_ek_25_30 = ekdf_25_30[['x', 'y']].values*4040/800000
plot_data_ek_30_inf = ekdf_30_inf[['x', 'y']].values*4040/800000

plot_data_ev_0_5 = evdf_0_5[['x', 'y']].values*4040/800000
plot_data_ev_5_10 = evdf_5_10[['x', 'y']].values*4040/800000
plot_data_ev_10_15 = evdf_10_15[['x', 'y']].values*4040/800000
plot_data_ev_15_20 = evdf_15_20[['x', 'y']].values*4040/800000
plot_data_ev_20_25 = evdf_20_25[['x', 'y']].values*4040/800000
plot_data_ev_25_30 = evdf_25_30[['x', 'y']].values*4040/800000
plot_data_ev_30_inf = evdf_30_inf[['x', 'y']].values*4040/800000

plot_data_mk_0_5 = mkdf_0_5[['x', 'y']].values*976/800000
plot_data_mk_5_10 = mkdf_5_10[['x', 'y']].values*976/800000
plot_data_mk_10_15 = mkdf_10_15[['x', 'y']].values*976/800000
plot_data_mk_15_20 = mkdf_15_20[['x', 'y']].values*976/800000
plot_data_mk_20_25 = mkdf_20_25[['x', 'y']].values*976/800000
plot_data_mk_25_30 = mkdf_25_30[['x', 'y']].values*976/800000
plot_data_mk_30_inf = mkdf_30_inf[['x', 'y']].values*976/800000

plot_data_mv_0_5 = mvdf_0_5[['x', 'y']].values*976/800000
plot_data_mv_5_10 = mvdf_5_10[['x', 'y']].values*976/800000
plot_data_mv_10_15 = mvdf_10_15[['x', 'y']].values*976/800000
plot_data_mv_15_20 = mvdf_15_20[['x', 'y']].values*976/800000
plot_data_mv_20_25 = mvdf_20_25[['x', 'y']].values*976/800000
plot_data_mv_25_30 = mvdf_25_30[['x', 'y']].values*976/800000
plot_data_mv_30_inf = mvdf_30_inf[['x', 'y']].values*976/800000


# In[4]:


from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def heatmap(x, y, s, bins=100):
   heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
   heatmap = gaussian_filter(heatmap, sigma=s)

   extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
   return heatmap.T, extent


# In[5]:


def divbutnotbyzero(a,b):
    c = np.zeros(a.shape)
    for i, row in enumerate(b):
        for j, el in enumerate(row):
            if el==0:
                c[i][j] = a[i][j]
            else:
                c[i][j] = a[i][j]/el
    return c


# In[6]:


bg = imread('../input/erangel.jpg')
hmap, extent = heatmap(plot_data_ev_0_5[:,0], plot_data_ev_0_5[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_ek_0_5[:,0], plot_data_ek_0_5[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, 2, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(0, 100, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()

# plt.scatter(plot_data_ev_0_5[:,0], plot_data_ev_0_5[:,1])


# In[7]:


bg = imread('../input/erangel.jpg')
hmap, extent = heatmap(plot_data_ev_5_10[:,0], plot_data_ev_5_10[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_ek_5_10[:,0], plot_data_ek_5_10[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, 2, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(0, 100, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()


# In[8]:


bg = imread('../input/erangel.jpg')
hmap, extent = heatmap(plot_data_ev_10_15[:,0], plot_data_ev_10_15[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_ek_10_15[:,0], plot_data_ek_10_15[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, 2, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(0, 100, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()


# In[9]:


bg = imread('../input/erangel.jpg')
hmap, extent = heatmap(plot_data_ev_15_20[:,0], plot_data_ev_15_20[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_ek_15_20[:,0], plot_data_ek_15_20[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, 2, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(0, 100, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()


# In[10]:


bg = imread('../input/erangel.jpg')
hmap, extent = heatmap(plot_data_ev_20_25[:,0], plot_data_ev_20_25[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_ek_20_25[:,0], plot_data_ek_20_25[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, 2, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(0, 100, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()


# In[ ]:


bg = imread('../input/erangel.jpg')
hmap, extent = heatmap(plot_data_ev_25_30[:,0], plot_data_ev_25_30[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_ek_25_30[:,0], plot_data_ek_25_30[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, 2, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(0, 100, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()


# In[ ]:


bg = imread('../input/erangel.jpg')
hmap, extent = heatmap(plot_data_ev_30_inf[:,0], plot_data_ev_30_inf[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_ek_30_inf[:,0], plot_data_ek_30_inf[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, 2, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(0, 100, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()


# In[ ]:


bg = imread('../input/miramar.jpg')
hmap, extent = heatmap(plot_data_mv_0_5[:,0], plot_data_mv_0_5[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_mk_0_5[:,0], plot_data_mk_0_5[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, 2, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(0, 100, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 1000); ax.set_ylim(0, 1000)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()


# In[ ]:


bg = imread('../input/miramar.jpg')
hmap, extent = heatmap(plot_data_mv_5_10[:,0], plot_data_mv_5_10[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_mk_5_10[:,0], plot_data_mk_5_10[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, 2, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(0, 100, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 1000); ax.set_ylim(0, 1000)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()


# In[ ]:


bg = imread('../input/miramar.jpg')
hmap, extent = heatmap(plot_data_mv_10_15[:,0], plot_data_mv_10_15[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_mk_10_15[:,0], plot_data_mk_10_15[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, 2, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(0, 100, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 1000); ax.set_ylim(0, 1000)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()


# In[ ]:


bg = imread('../input/miramar.jpg')
hmap, extent = heatmap(plot_data_mv_15_20[:,0], plot_data_mv_15_20[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_mk_15_20[:,0], plot_data_mk_15_20[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, 2, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(0, 100, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 1000); ax.set_ylim(0, 1000)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()


# In[ ]:


bg = imread('../input/miramar.jpg')
hmap, extent = heatmap(plot_data_mv_20_25[:,0], plot_data_mv_20_25[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_mk_20_25[:,0], plot_data_mk_20_25[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, 2, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(0, 100, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 1000); ax.set_ylim(0, 1000)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()


# In[ ]:


bg = imread('../input/miramar.jpg')
hmap, extent = heatmap(plot_data_mv_25_30[:,0], plot_data_mv_25_30[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_mk_25_30[:,0], plot_data_mk_25_30[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, 2, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(0, 100, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 1000); ax.set_ylim(0, 1000)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()


# In[ ]:


bg = imread('../input/miramar.jpg')
hmap, extent = heatmap(plot_data_mv_30_inf[:,0], plot_data_mv_30_inf[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_mk_30_inf[:,0], plot_data_mk_30_inf[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, 2, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(0, 100, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 1000); ax.set_ylim(0, 1000)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()

