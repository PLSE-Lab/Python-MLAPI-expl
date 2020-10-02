#!/usr/bin/env python
# coding: utf-8

# # PUBG Analytics: Final Circle

# In this series of notebooks we'll explore a few different analytics to visualize potential patterns. First up is the final circle inference from second placed players in solo matches that didn't die to bluezone.  Credit to [/u/n23_](https://www.reddit.com/user/n23_) for the idea

# In[1]:


# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.misc.pilutil import imread

deaths = pd.concat([pd.read_csv("../input/deaths/kill_match_stats_final_0.csv"),
                   pd.read_csv("../input/deaths/kill_match_stats_final_1.csv")], ignore_index=True)
meta = pd.concat([pd.read_csv("../input/aggregate/agg_match_stats_0.csv"),
                 pd.read_csv("../input/aggregate/agg_match_stats_1.csv")], ignore_index=True)


# ---
# 
# ## 1. Final Circle
# 
# The Final Circle is usually the most important, anyone that's outside of the circle will usually die to bluezone within seconds.  Looking at deaths by second place would usually be a good proxy for where the final safezone circle is.  For now, let's restrict it to solo matches to parse out some variance

# In[2]:


match_list = meta.loc[meta['party_size'] == 1, 'match_id'].drop_duplicates()
print('Number of solo queue matches: %i' % len(match_list))
deaths_solo = deaths[deaths['match_id'].isin(match_list.values)]
deaths_solo_er = deaths_solo[deaths_solo['map'] == 'ERANGEL']
deaths_solo_mr = deaths_solo[deaths_solo['map'] == 'MIRAMAR']
print('  Number of Erangel solo matches: %i' % len(deaths_solo_er.groupby('match_id').first()))
print('  Number of Miramar solo matches: %i' % len(deaths_solo_mr.groupby('match_id').first()))


# In[3]:


df_second_er = deaths_solo_er[(deaths_solo_er['victim_placement'] == 2)].dropna()
df_second_mr = deaths_solo_mr[(deaths_solo_mr['victim_placement'] == 2)].dropna()
print('%i Erangel matches where 2nd place didn''t die to bluezone' % len(df_second_er))
print('%i Miramar matches where 2nd place didn''t die to bluezone' % len(df_second_mr))


# Add in the 1st player killer's position too because there's a good chance he/she was in the zone as well

# In[4]:


plot_data_er = np.vstack([df_second_er[['victim_position_x', 'victim_position_y']].values, 
                          df_second_er[['killer_position_x', 'killer_position_y']].values])
plot_data_mr = np.vstack([df_second_mr[['victim_position_x', 'victim_position_y']].values, 
                          df_second_mr[['killer_position_x', 'killer_position_y']].values])

# transcribe location data to image size
plot_data_er = plot_data_er*4096/800000
plot_data_mr = plot_data_mr*1000/800000


# Let's plot it now as a heatmap: histogram2d + gaussian smoothing

# In[11]:


from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def heatmap(x, y, s, bins=100):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

bg = imread('../input/erangel.jpg')
hmap, extent = heatmap(plot_data_er[:,0], plot_data_er[:,1], 1.5)
alphas = np.clip(Normalize(0, hmap.max(), clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(0, hmap.max(), clip=True)(hmap)
colors = cm.Reds(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.Reds, alpha=0.9)
#plt.scatter(plot_data_er[:,0], plot_data_er[:,1])
plt.gca().invert_yaxis()


# In[12]:


bg = imread('../input/miramar.jpg')
hmap, extent = heatmap(plot_data_mr[:,0], plot_data_mr[:,1], 1.5)
alphas = np.clip(Normalize(0, hmap.max(), clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(0, hmap.max(), clip=True)(hmap)
colors = cm.Blues(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 1000); ax.set_ylim(0, 1000)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.Blues, alpha=0.9)
#plt.scatter(plot_data_mr[:,0], plot_data_mr[:,1])
plt.gca().invert_yaxis()


# # Conclusion
# 
# - Erangel:  Although a lot more spread out, there are a few pockets of land that is more contested for final circle, primarily near very open areas between pochinki and mylta.  It is worth noting that a lot of close river areas north are avoided so players based off of seeing the first circle, should be able to make an educated guess where the next circle will be.
# - Miramar:  A lot different and more clustered. Notable clusters are between bendita and Leones, south of San Martin and west of Pecado.  It is interesting to see that a large portion on the  of the map is often never in the final circle
