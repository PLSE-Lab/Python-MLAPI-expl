#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from IPython.display import Image, display
from skimage import io
from skimage.transform import rotate as sk_rotate
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


PATH = '../input/mlcourse-dota2-win-prediction/'
train = pd.read_csv(PATH + 'train_features.csv', index_col='match_id_hash')
target = pd.read_csv(PATH + 'train_targets.csv', index_col='match_id_hash')
image = io.imread('https://i.imgur.com/a7KCt5J.jpg')


# <center>
# 
# ### "Always pay respect to EDA". --[@utility](https://www.kaggle.com/utility)
# 
# 

# # Dota 2 EDA on Coordinates (Animated!)
# 
# by [@marketneutral](https://www.kaggle.com/marketneutral)
# 
# This notebook contains some fragments of EDA (Exploratory Data Analysis) which I used profitably in this competition. It also shows how to animate `matplotlib` plots for your future medal winning Kaggle kernels. I do not give you final features themselves in this kernel...but you can figure them out from the EDA.
# 
# Note that I am not a Dota player -- I've never played! Thanks to [@koshiu](https://www.kaggle.com/koshiu) for help finding the right map image for this kernel! :-)
# 

# <center>
# 
# ![image.png](attachment:image.png)
# 
# ### **Dota 2 Board Map 6.86 Patch**

# # Hero Location Over Time
# 
# Location of heroes proved to be a fruitful area of feature engineering. Before engineering, it's EDA time... Let's look at an animation to answer the question: "how does the location of Radiant Hero 1 differ in cases of **Radiant Win** vs **Radiant Loss** as a function of `game_time`"? Then we can extrapolate and perhaps make team-wise features...
# 

# In[ ]:


# I am adding jitter because x and y are integers.
# This is just an aid for visualization in the plot.
# In the competition we would not do this with the real data!

train['r1_x'] = train['r1_x']*(1+np.random.normal(loc=0, scale=0.01, size=len(train)))
train['r1_y'] = train['r1_y']*(1+np.random.normal(loc=0, scale=0.01, size=len(train)))


# In[ ]:


# set up the main figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# initialize the subplots as blank scatterplots
scat1, = ax1.plot([], [], 'o', color='blue', alpha=0.1)
scat2, = ax2.plot([], [], 'o', color='red', alpha=0.1)
scat = [scat1, scat2]

# turn axis lines and ticks off; add the background image of the map
# the "extent" tells matplotlib how to stretch the image
ax1.axis('off')
ax2.axis('off')
ax1.imshow(image, extent=[65, 190, 65, 185], alpha=0.35);
ax2.imshow(image, extent=[65, 190, 65, 185], alpha=0.35);

# set the subplot axis limits and title font
for ax in [ax1, ax2]:
    ax.set_ylim(70, 190)
    ax.set_xlim(70, 190)
    ax.set_title('', fontweight="bold", size=20)
    ax.grid()

# the matplotlib "FuncAnimation" class requires an init and animate function
def init():
    scat[0].set_data([], [])
    scat[1].set_data([], [])
    return scat

def animate(i):
    # i is frames incrementing
    minutes = i*2
    train_cut = train.query('game_time < 60*@minutes')
    target_cut = target.loc[train_cut.index]['radiant_win']
    
    # update Radiant win plot
    x = train_cut[target_cut==1]['r1_x']
    y = train_cut[target_cut==1]['r1_y']
    scat[0].set_data(x, y)
    ax1.title.set_text(f'r1 loc for Radiant WIN at time < {minutes} minutes')
    
    # update Radiant loss plot
    x = train_cut[target_cut==0]['r1_x']
    y = train_cut[target_cut==0]['r1_y']
    scat[1].set_data(x, y)
    ax2.title.set_text(f'r1 loc for Radiant LOSS at time < {minutes} minutes')

    return scat

anim = FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=21,
    interval=500,   # time in milliseconds between each frame
    blit=True
);

plt.close(fig)  # don't show fig yet... suspense!!!

anim.save('map.gif', writer='imagemagick');


# In[ ]:


# in order to diplay in Jupyter notebook, we need to wrap it as
display(Image(url='map.gif'))


# Watch it loop a few times. You can see some clear pattern in the case of Radiant wins which makes sense ...
# 
# - **less** times where R1 is near his or her home
# - **more** times when R1 is near Dire home
# 
# Note also that there is a lot of presence in the **middle lane**.

# # Map Orientation
# 
# When you make features there are a couple philosophies... *heavy* vs *light* engineering. Light means you make some small transformation to make the feature simply more compatible with the model. Heavy means...well...you make some significant transformations, combine features explicitly, and explicitly specify interactions. Let's think about light engineering here. We posit above that target (Radiant Win) is proportional to a strong presenence of Radiant Heroes near Dire home. In the case of GBDT (e.g., `LightGBM`), the model should find this on it's own if it splits on `r1_x`, `r1_y`, etc. But...there is a problem. One possible issue is with the *map orientation*. A decision tree can only make splits **parallel to an axis.** We saw in `mlcourse.ai` materials:
# 
# Source: https://mlcourse.ai/articles/topic3-dt-knn/
# ![image.png](attachment:image.png)
# 
# In this case, to split on the diagonal you need many splits and a deep tree. That's silly and too complex, especially in the case of GBDT where we will want a large ensemble of shallow trees. To capture a separation on the diagonal uses up too many splits. So what can we do?

# In[ ]:


minutes = 30
train_cut = train.query('game_time < 60*@minutes')
target_cut = target.loc[train_cut.index]['radiant_win']
plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.imshow(image, extent=[65, 190, 65, 185], alpha=0.6);

plt.scatter(train_cut[target_cut==1]['r1_x'], train_cut[target_cut==1]['r1_y'], alpha=0.1, c='blue');
plt.title(f'r1 loc for Radiant WIN at time < {minutes} minutes', fontweight="bold", size=20);

# annotation lines
l1 = [(83, 95), (170, 175)]
l2 = [(83, 77), (170, 157)]
l3 = [(100, 180), (180, 100)]
l4 = [(70, 145), (145, 70)]

lc = LineCollection([l1, l2, l3, l4], color = ['k', 'k'], lw=2) # linestyle='dashed', 
plt.gca().add_collection(lc)
plt.axis('off');

plt.subplot(1, 2, 2)
plt.imshow(image, extent=[65, 190, 65, 185], alpha=0.6);
plt.scatter(train_cut[target_cut==0]['r1_x'], train_cut[target_cut==0]['r1_y'], alpha=0.1, c='red');
plt.title(f'r1 loc for Radiant LOSS at time < {minutes} minutes', fontweight="bold", size=20);

lc = LineCollection([l1, l2, l3, l4], color = ['k', 'k'], lw=2) # linestyle='dashed', 
plt.gca().add_collection(lc);
plt.axis('off');


# We want to capture regions **separated by diagonals**. With the addition of the annotation lines, we can see clearly that certain areas of the map are important to distinguish between Radiant win or loss. To make it easy for GBDT to separate on these diagonals with a single (or just two) splits, we can **rotate** the board by -45 (need in radians, $45^{\circ} \pi 180 =  0.785398$ radians).

# In[ ]:


map_center = (127., 127.)

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


# We make new features with each hero's location rotated 45 degrees counterclockwise.

# In[ ]:


train['r1_x_rot'] = train.apply(lambda row: rotate(map_center, (row.r1_x, row.r1_y), 0.785398)[0], axis=1)
train['r1_y_rot'] = train.apply(lambda row: rotate(map_center, (row.r1_x, row.r1_y), 0.785398)[1], axis=1)


# In[ ]:


# and we have to rotate the map image as well :-)
image_rot = sk_rotate(image, angle=45, resize=True)


# In[ ]:


# rotate our annotation lines

l1_rot = []
l2_rot = []
l3_rot = []
l4_rot = []

for pt in l1:
    l1_rot.append(rotate(map_center, pt,  0.785398))
for pt in l2:
    l2_rot.append(rotate(map_center, pt,  0.785398))
for pt in l3:
    l3_rot.append(rotate(map_center, pt,  0.785398))
for pt in l4:
    l4_rot.append(rotate(map_center, pt,  0.785398))    


# In[ ]:


minutes = 30
train_cut = train.query('game_time < 60*@minutes')
target_cut = target.loc[train_cut.index]['radiant_win']
plt.figure(figsize=(22,10))
plt.subplot(1, 2, 1)
plt.imshow(image_rot, extent=[54, 200, 47, 203], alpha=0.6);
plt.scatter(train_cut[target_cut==1]['r1_x_rot'], train_cut[target_cut==1]['r1_y_rot'], alpha=0.1, c='blue');
plt.title(f'r1 loc for Radiant WIN at time < {minutes} minutes', fontweight="bold", size=20);

lc = LineCollection([l1_rot, l2_rot, l3_rot, l4_rot], color = ['k', 'k'], lw=2) # linestyle='dashed', 
plt.gca().add_collection(lc)

plt.axis('off');
plt.subplot(1, 2, 2)
plt.imshow(image_rot, extent=[54, 200, 47, 203], alpha=0.6);
plt.scatter(train_cut[target_cut==0]['r1_x_rot'], train_cut[target_cut==0]['r1_y_rot'], alpha=0.1, c='red');
plt.title(f'r1 loc for Radiant LOSS at time < {minutes} minutes', fontweight="bold", size=20);

lc = LineCollection([l1_rot, l2_rot, l3_rot, l4_rot], color = ['k', 'k'], lw=2) # linestyle='dashed', 
plt.gca().add_collection(lc)
plt.axis('off');


# With the addition of the **rotation points**, now GBDT can split parallel to a board axis and capture important hero location in a **single split** (or just two in the case of finding heroes in the middle lane). We will leave both the raw `r1_x`, `r2_x`, etc. and add the rotated points.
# 
# I got the idea to rotate the map from a watching a talk on YouTube by [@cpmpml](https://www.kaggle.com/cpmpml) at [Kaggle Days Paris](https://www.youtube.com/watch?v=VC8Jc9_lNoY). He talks about rotating apartment location coordinates for New York City so that XGBoost can split easily on the avenues and streets (NYC is not naturally longitude and lattitude parallel).
# 
# Thanks for looking at my kernel!
