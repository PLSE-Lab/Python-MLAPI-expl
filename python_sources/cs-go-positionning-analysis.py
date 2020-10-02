#!/usr/bin/env python
# coding: utf-8

# # CS:GO positionning analysis
# For my first Kernel I decided to work on a game that I like which is Counter Strike Global Offensive.<br>
# As Counter Strike Global Offensive relies on defending bomb sites, it is a useful information to know where the bullets might be coming from when attacking a bomb site or where to position yourself to defend.<br>
# The goal of this Kernel is to determine what are the most common and efficient positions played on the Counter Terrorist (CT) side before the bomb is planted, and the positions played by the Terrorists (T) side after the bomb is planted.
# 
# I'll be using data from [skihikingkevin's CS:GO Dataset](https://www.kaggle.com/skihikingkevin/csgo-matchmaking-damage#mm_master_demos.csv). And I'll limit my analysis to de_mirage which is my favourite map.
# <br>
# I borrowed some code from [billfreeman44's T side smokes on mirage kernel](https://www.kaggle.com/billfreeman44/finding-classic-smokes-by-t-side-on-mirage) don't hesitate to give it a look it's pretty interesting espescially as smokes are a big part of mirage's gameplay !
# 
# 1. The data<br>
# 2. CT Side defense analysis<br>
#   2.1. Positions regardless of weapons<br>
#   2.2. Specific case : AWP<br>
# 3. T Side defense analysis<br>
#   3.1. Bombsite A<br>
#   3.2. Bombsite B<br>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

data = pd.read_csv('../input/mm_master_demos.csv')
analyzed_map = 'de_mirage' # It's the best map after all !!!

# Filter by map & type of rounds, we don't want eco rounds as those tend to be more agressive
data = data[(data.map == analyzed_map) & ((data.round_type == 'PISTOL_ROUND') | (data.round_type == 'NORMAL'))]


# In[ ]:


# Code from billfreeman44 | See his kernel : https://www.kaggle.com/billfreeman44/finding-classic-smokes-by-t-side-on-mirage 
# Used to convert Dataset positions to positions on the radar
# Keep in mind that the startX, endX resX, startY, endY, resY are specific to mirage, coordinates for each map are available in the dataset

def pointx_to_resolutionx(xinput,startX=-3217,endX=1912,resX=1024):
    sizeX=endX-startX
    if startX < 0:
        xinput += startX *(-1.0)
    else:
        xinput += startX
    xoutput = float((xinput / abs(sizeX)) * resX);
    return xoutput

def pointy_to_resolutiony(yinput,startY=-3401,endY=1682,resY=1024):
    sizeY=endY-startY
    if startY < 0:
        yinput += startY *(-1.0)
    else:
        yinput += startY
    youtput = float((yinput / abs(sizeY)) * resY);
    return resY-youtput

# Convert the data to radar positions
data['attacker_mapX'] = data['att_pos_x'].apply(pointx_to_resolutionx)
data['attacker_mapY'] = data['att_pos_y'].apply(pointy_to_resolutiony)


# ## 1. The data
# This dataset consists of more than 30k rounds where each line of data is a damage entry.<br>
# We will focus on attacker position, and bomb state.<br>
# Here you can see how it looks :

# In[ ]:


data.head()


# ## 2. CT Side defense analysis
# To understand CT Side positionning, first we'll filter actions from CT, before the bomb is planted<br>
# Also we'll filter only when the CT side won as it's the only way we can measure efficiency (using points density) for a position by seeing if it is frequently used when the round is won.
# 
# ### 2.1. Regardless of weapon :

# In[ ]:


ct_data = data[(data.is_bomb_planted == False) & (data.att_side == 'CounterTerrorist') & (data.winner_side == 'CounterTerrorist')]

# Code from billfreeman44 | See his kernel : https://www.kaggle.com/billfreeman44/finding-classic-smokes-by-t-side-on-mirage 
im = plt.imread('../input/de_mirage.png')
plt.figure(figsize=(20,20))
t = plt.imshow(im)
t = plt.scatter(ct_data['attacker_mapX'], ct_data['attacker_mapY'],alpha=0.005,c='blue')


# We can see that the most efficient positions on A tend to be behind the boxes, upper stairs, window, jungle, short and truck. So positions that are a bit agressive.
# 
# ### 2.2. AWPs :

# In[ ]:


ct_data_awp = data[(data.is_bomb_planted == False) & (data.att_side == 'CounterTerrorist') & (data.wp == 'AWP') & (data.winner_side == 'CounterTerrorist')]

# Code from billfreeman44 | See his kernel : https://www.kaggle.com/billfreeman44/finding-classic-smokes-by-t-side-on-mirage 
im = plt.imread('../input/de_mirage.png')
plt.figure(figsize=(20,20))
t = plt.imshow(im)
t = plt.scatter(ct_data_awp['attacker_mapX'], ct_data_awp['attacker_mapY'],alpha=0.025,c='blue')


# AWPs are playing more defensive positions to benefit from the range offered by the sniper, however playing AWP on B doesn't seem really efficient.
# 
# ## 3. T Side defense analysis
# First let's remind that from the dataset actions when the bomb is planted only represents 16% of all the actions registered in the dataset.<br>
# So when the bomb is planted, terrorists should take an optimal positionning to defend the bomb. Unfortunately the dataset doesn't provide the bomb coordinates, only the site it is planted on. So we don't know if they have the bomb in sight for example.<br>
# Errors might come from the fact that Terrorists economy encourages to hunt remaining CTs when stomping a site.
# 
# ### 3.1. Bombsite A

# In[ ]:


t_data_siteA = data[(data.is_bomb_planted == True) & (data.att_side == 'Terrorist') & (data.bomb_site == 'A') & (data.winner_side == 'Terrorist')]

# Code from billfreeman44 | See his kernel : https://www.kaggle.com/billfreeman44/finding-classic-smokes-by-t-side-on-mirage 
im = plt.imread('../input/de_mirage.png')
plt.figure(figsize=(20,20))
t = plt.imshow(im)
t = plt.scatter(t_data_siteA['attacker_mapX'], t_data_siteA['attacker_mapY'],alpha=0.025,c='orange')


# Terrorists positionning seems a bit chaotic, but we can still see that palace/under palace, CT, connector seem like good positions. However the further from bombsite the less effective the position is.<br>
# Also the position called "bitch" in the right corner behind the boxes doesn't look really effective yet people play it a lot.
# 
# ### 3.2. Bombsite B

# In[ ]:


t_data_siteB = data[(data.is_bomb_planted == True) & (data.att_side == 'Terrorist') & (data.bomb_site == 'B') & (data.winner_side == 'Terrorist')]

# Code from billfreeman44 | See his kernel : https://www.kaggle.com/billfreeman44/finding-classic-smokes-by-t-side-on-mirage 
im = plt.imread('../input/de_mirage.png')
plt.figure(figsize=(20,20))
t = plt.imshow(im)
t = plt.scatter(t_data_siteB['attacker_mapX'], t_data_siteB['attacker_mapY'],alpha=0.025,c='orange')


# Again the positionning on B is a bit chaotic, we can still see that market/kitchen, truck, appartments, bench seem like good positions as they provide a lot of cover.
# 
# ## Conclusion
# From this analysis it seems that the most efficient tactic as a T is to play really deep into the site with a lot of cover, whereas for CTs it's to play semi-agressive positions near contact points or to hold long lines using an AWP.<br>
# Also a lot of the defense is focused on mid & bombsite A. Which leads to CTs playing retake scenario on bombsite B.
# 
# 
# Thanks for taking a look at my work, I highly recommend you to check [billfreeman44's T side smokes on mirage kernel](https://www.kaggle.com/billfreeman44/finding-classic-smokes-by-t-side-on-mirage) which has done a pretty good work on covering T side smokes on mirage, his kernel will give you an understanding of why T side can really take advantage of a bombsite easily with some teamwork.
