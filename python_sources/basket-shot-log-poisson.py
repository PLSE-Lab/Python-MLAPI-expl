#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import scipy as sp

# Graph drawing
import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

from matplotlib import animation as ani
from IPython.display import Image

plt.rcParams["patch.force_edgecolor"] = True
#rc('text', usetex=True)
from IPython.display import display # Allows the use of display() for DataFrames
import seaborn as sns
sns.set(style="whitegrid", palette="muted", color_codes=True)
sns.set_style("whitegrid", {'grid.linestyle': '--'})
red = sns.xkcd_rgb["light red"]
green = sns.xkcd_rgb["medium green"]
blue = sns.xkcd_rgb["denim blue"]

# pandas formatting
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.options.display.float_format = '{:,.5f}'.format

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")


# In[ ]:


df = pd.read_csv("../input/nba-shot-logs/shot_logs.csv")


# # basic check

# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.player_name.nunique()


# In[ ]:


df.player_name.value_counts()


# In[ ]:


player_name = df.player_name.value_counts().index[0]
player_name


# # Poisson distribution

# In[ ]:


shot_count_on_game = df[df.player_name==player_name].groupby("GAME_ID").size().sort_values(ascending=False)
shot_count_on_game.iloc[:5]


# In[ ]:


plt.figure(figsize=(10,7))
shot_count_on_game.plot.hist(bins=10, density=True)
plt.title(f"{player_name}'s shot history.")


# In[ ]:


mean = shot_count_on_game.mean()
mean


# In[ ]:


xx = np.arange(10, 35)
p = sp.stats.poisson.pmf(xx, mu=mean)

plt.figure(figsize=(10,7))
shot_count_on_game.plot.hist(bins=12, density=True)
plt.plot(xx, p)
plt.title(f"{player_name}'s shot history & poisson distribution.");


# In[ ]:


for player_name in df.player_name.value_counts().index.tolist()[1:11]:
    shot_count_on_game = df[df.player_name==player_name].groupby("GAME_ID").size().sort_values()

    xx = np.arange(shot_count_on_game.min()-1, shot_count_on_game.max()+1)
    mean = shot_count_on_game.mean()
    p = sp.stats.poisson.pmf(xx, mu=mean)

    plt.figure(figsize=(10,7))
    shot_count_on_game.plot.hist(bins=12, density=True)
    plt.plot(xx, p)
    plt.title(f"{player_name}'s shot history & poisson distribution. mu:{mean:0.4f}");
    plt.show()

