#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_palette('husl')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Read Data

# In[ ]:


df = pd.read_csv('/kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv')


# # Initial Analysis

# In[ ]:


df.head()


# In[ ]:


df.columns


# # Best Developers

# In[ ]:


get_ipython().run_cell_magic('time', '', "fig, ax = plt.subplots(1,2,figsize=(16,32))\nwordcloud = WordCloud(background_color='black',width=1500,height=1500).generate(' '.join(df['Developer']))\nwordcloud_sub = WordCloud(background_color='white',width=1500,height=1500).generate(' '.join(df['Name'].dropna().astype(str)))\nax[0].imshow(wordcloud)\nax[0].axis('off')\nax[0].set_title('Developer')\nax[1].imshow(wordcloud_sub)\nax[1].axis('off')\nax[1].set_title('Name')\nplt.show()")


# # Best Language and Geners

# In[ ]:


get_ipython().run_cell_magic('time', '', "fig, ax = plt.subplots(1,2,figsize=(16,32))\nwordcloud = WordCloud(background_color='black',width=1500,height=1500).generate(' '.join(df['Genres']))\nwordcloud_sub = WordCloud(background_color='white',width=1500,height=1500).generate(' '.join(df['Languages'].dropna().astype(str)))\nax[0].imshow(wordcloud)\nax[0].axis('off')\nax[0].set_title('Geners')\nax[1].imshow(wordcloud_sub)\nax[1].axis('off')\nax[1].set_title('Languages')\nplt.show()")


# # Summery

# ** With the Help of wordcloud we can get some intuition about:**
# 1. Best Gaming Company
# 2. Tranding Geners
# 3. Language Market of games
# 

# In[ ]:




