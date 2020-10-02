#!/usr/bin/env python
# coding: utf-8

# The purpose of this kernel is to explore where people die and get kills in PUBG and see if we can tease out good places to play.
# First, since we don't care about the internal label of player number, lets combine all of the Erangel kills together and deaths together, and the same for Miramar.
# lets also drop any death that happened on the far left side of the map at coordinate (x=0) since those represent bad data, stored incorrectly on Bluehole's end. 

# In[69]:


# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.misc.pilutil import imread


df = pd.read_csv('../input/PUBG_MatchData_Flattened.tsv', sep='\t')

edf = df.loc[df['map_id'] == 'ERANGEL']
mdf = df.loc[df['map_id'] == 'MIRAMAR']

# print(edf.head())
# print(mdf.head())

def killer_victim_df_maker(df):
    victim_x_df = df.filter(regex='victim_position_x')
    victim_y_df = df.filter(regex='victim_position_y')
    killer_x_df = df.filter(regex='killer_position_x')
    killer_y_df = df.filter(regex='killer_position_y')

    victim_x_s = pd.Series(victim_x_df.values.ravel('F'))
    victim_y_s = pd.Series(victim_y_df.values.ravel('F'))
    killer_x_s = pd.Series(killer_x_df.values.ravel('F'))
    killer_y_s = pd.Series(killer_y_df.values.ravel('F'))

    vdata={'x': victim_x_s, 'y':victim_y_s}
    kdata={'x': killer_x_s, 'y':killer_y_s}

    victim_df = pd.DataFrame(data = vdata).dropna(how='any')
    victim_df = victim_df[victim_df['x']>0]
    killer_df = pd.DataFrame(data = kdata).dropna(how='any')
    killer_df = killer_df[killer_df['x']>0]
    return killer_df,victim_df

ekdf,evdf = killer_victim_df_maker(edf)
mkdf,mvdf = killer_victim_df_maker(mdf)

# print(ekdf.head())
# print(evdf.head())
# print(mkdf.head())
# print(mvdf.head())
# print(len(ekdf), len(evdf), len(mkdf), len(mvdf))



# Next we rescale to the image. Normally the scalers would be 4096 and 1000, but it seems that the images were compressed somewhere and the scalers of 4040 and 976 better fit the images.

# In[70]:


plot_data_ev = evdf[['x', 'y']].values
plot_data_ek = ekdf[['x', 'y']].values
plot_data_mv = mvdf[['x', 'y']].values
plot_data_mk = mkdf[['x', 'y']].values

plot_data_ev = plot_data_ev*4040/800000
plot_data_ek = plot_data_ek*4040/800000
plot_data_mv = plot_data_mv*976/800000
plot_data_mk = plot_data_mk*976/800000


# Let's shamelessly take the heatmap function from https://www.kaggle.com/skihikingkevin/final-circle-heatmap

# In[71]:


from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def heatmap(x, y, s, bins=100):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


# Time to load in the image and play around with the alphas and the colors. A lot of time went into picking the correct places to clip the alphas and color so that you dont just see one spike of color at school/pochinki and nowhere else, but also dont see a white fog covering the whole map. First lets do it for the victims' death locations:

# In[72]:


bg = imread('../input/erangel.jpg')
hmap, extent = heatmap(plot_data_ev[:,0], plot_data_ev[:,1], 1.5, bins=800)
alphas = np.clip(Normalize(0, hmap.max()/100, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(hmap.max()/100, hmap.max()/20, clip=True)(hmap)
colors = cm.bwr(colors)
colors[..., -1] = alphas


# Then the killers' locations:

# In[73]:


hmap2, extent2 = heatmap(plot_data_ek[:,0], plot_data_ek[:,1], 1.5, bins=800)
alphas2 = np.clip(Normalize(0, hmap2.max()/100, clip=True)(hmap2)*1.5, 0.0, 1.)
colors2 = Normalize(hmap2.max()/100, hmap2.max()/20, clip=True)(hmap2)
colors2 = cm.RdBu(colors2)
colors2[..., -1] = alphas2


# ...and draw it all to the image. Let's first see just the deaths:

# In[74]:


fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.bwr, alpha=0.5)
# ax.imshow(colors2, extent=extent2, origin='lower', cmap=cm.RdBu, alpha=0.5)
plt.gca().invert_yaxis()


# ...now just where the killers were standing, with reversed color scheme:

# In[75]:


fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
# ax.imshow(colors, extent=extent, origin='lower', cmap=cm.bwr, alpha=0.5)
ax.imshow(colors2, extent=extent2, origin='lower', cmap=cm.RdBu, alpha=0.5)
plt.gca().invert_yaxis()


# ...and finally both together (if something has more red hue, its bad to stand at, if it has more blue, its good to stand at).

# In[76]:


fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.bwr, alpha=0.5)
ax.imshow(colors2, extent=extent2, origin='lower', cmap=cm.RdBu, alpha=0.5)
plt.gca().invert_yaxis()


# Well, that is remarkably uninformative. We should take a look at that in slices of time, or perhaps try to make a heatmap of kills divided by deaths inside of 100 square meter bins. But first, lets look at the overall activity on Miramar, because it looks pretty.
# Since the above showed us that kills and deaths are basically the same maps in PUBG when looking at all games and the entirety of them, we may as well just show them with the same colorscheme as an "activity" heatmap. Here I use the rainbow color scheme.

# In[77]:


bg = imread('../input/miramar.jpg')
hmap, extent = heatmap(plot_data_mv[:,0], plot_data_mv[:,1], 1.5, bins=800)
alphas = np.clip(Normalize(0, hmap.max()/200, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(hmap.max()/100, hmap.max()/20, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas


# In[78]:


hmap2, extent2 = heatmap(plot_data_mk[:,0], plot_data_mk[:,1], 1.5, bins=800)
alphas2 = np.clip(Normalize(0, hmap2.max()/200, clip=True)(hmap2)*1.5, 0.0, 1.)
colors2 = Normalize(hmap2.max()/100, hmap2.max()/20, clip=True)(hmap2)
colors2 = cm.rainbow(colors2)
colors2[..., -1] = alphas2


# In[79]:


fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 1000); ax.set_ylim(0, 1000)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
ax.imshow(colors2, extent=extent2, origin='lower', cmap=cm.rainbow, alpha=0.5)
#plt.scatter(plot_data_er[:,0], plot_data_er[:,1])
plt.gca().invert_yaxis()


# When looking at the heatmap above, it's important to remember the effects of the blue circle on overall deaths. A lot of the purple splotches in the central area of the map may be due to the zone pushing players inwards! Again, this can be mitigated a bit by looking at time slices. I will make another kernel to do so, since it requires a change in the first code block of this kernel. Before that, let's try to view kill-death-ratio in each bin. First, let's define a division function so that we don't divide by 0.

# In[80]:


def divbutnotbyzero(a,b):
    c = np.zeros(a.shape)
    for i, row in enumerate(b):
        for j, el in enumerate(row):
            if el==0:
                c[i][j] = a[i][j]
            else:
                c[i][j] = a[i][j]/el
    return c


# Next, we do the division and plot it, for Erangel.

# In[95]:


bg = imread('../input/erangel.jpg')
hmap, extent = heatmap(plot_data_ev[:,0], plot_data_ev[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_ek[:,0], plot_data_ek[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, hmap3.max()/100, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(hmap3.max()/100, hmap3.max()/20, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()


# Pretty cool! Notably, the typical "hot zones" arent the only places for getting a good kill/death ratio. Anywhere that you are seeing red is a pretty good spot to land. Let's print the k/d mean:

# In[96]:


print(hmap3.mean())


# This k/d is pretty low, especially since we seeded our data on a high elo player in making the dataset. This suggests that we skewed our results when we threw away the (0,0) coordinate data and things should be taken with a grain of salt. Of course, it is possible that the mean is actually close to 0.55 since 1.0 is the max and would not happen due to bluezone/ redzone/ crashes/ suicides.
# 
# To see more, we will need to do time-slices, which I will save for another notebook. Let's end with the same style of heatmap for Miramar, and print the k/d mean there too.

# In[97]:


bg = imread('../input/miramar.jpg')
hmap, extent = heatmap(plot_data_mv[:,0], plot_data_mv[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_mk[:,0], plot_data_mk[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, hmap3.max()/100, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(hmap3.max()/100, hmap3.max()/20, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas


fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 1000); ax.set_ylim(0, 1000)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()


# In[98]:


print(hmap3.mean())


# Wow! That is much closer to 1! Impossibly close! Something is certainly going wrong with one or the other sets. Please comment if you spot an error or can figure out how to get a more accurate kill/death graph. 
