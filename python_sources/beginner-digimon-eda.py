#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Digimon Exploratory Data Analysis
# ### Intro
# Digimon stands for Digital Monster, an anime TV show that tells a story about an entirely different world inside the computer created with data. This notebook is for knowing the information of Digimon, their moves, and support to achieve their maximum potential

# ### Inputting the Data

# In[ ]:


digimon = pd.read_csv('../input/digidb/DigiDB_digimonlist.csv')
move = pd.read_csv('../input/digidb/DigiDB_movelist.csv')
support = pd.read_csv('../input/digidb/DigiDB_supportlist.csv')


# ### Peeking at the Datasets
# As we can see here the datasets is clean and ready to use, no missing values and such.

# In[ ]:


digimon.head()


# In[ ]:


move.head()


# In[ ]:


support.head()


# In[ ]:


digimon.info()


# In[ ]:


move.info()


# In[ ]:


support.info()


# ## Stats
# Stats, because Power is Everything!! Unles you're that friend that like a Digimon just because it looks cute.

# In[ ]:


sns.pairplot(data=digimon[['Lv 50 HP', 'Lv50 SP', 'Lv50 Atk', 'Lv50 Def', 'Lv50 Int', 'Lv50 Spd']])
plt.show()


# From the pairplot above we can see that, the stats of the Digimons are pretty diverse and reasonable, where it's because this plot contains all Digimon Stages (In-Training to Mega), that's why there are low overall stats and high overall stats. Most notable is on SP vs Int and makes sense because more INT = more SP.

# ### Digimon Data

# In[ ]:


sns.countplot(x=digimon['Attribute'])
plt.show()


# From the above countplot, it seems that Digimon's Attribute are dominantly Dark, followed by Fire and Light, and the rest is pretty much the same.

# In[ ]:


sns.countplot(x=digimon['Type'])
plt.show()


# The plot above shows that Digimon's Type is dominantly by Virus.

# ## Answering Questions

# ### Which set of moves will get the best ratio of attack power to SP spent?

# In[ ]:


maxratio = move.copy()
maxratio['Max Ratio'] = move.Power / move['SP Cost']


# In[ ]:


maxratio['Max Ratio'].idxmax(axis=0)
print(move.iloc[80])


# It seems the Digimon that have the move 'Heavy Strike I' is good because it has a really efficient damage with low SP cost and somewhat good Power.

# ### Which team of 3 digimon have the highest attack? Defense?

# In[ ]:


digimon.sort_values(by=['Lv50 Atk'], ascending=False).head(3)


# In[ ]:


digimon.sort_values(by=['Lv50 Def'], ascending=False).head(3)


# ### Are some types over- or under-represented?

# In[ ]:


sns.countplot(x=digimon['Type'])
plt.show()


# This question got answered above when plotting the Digimon's Types, as seen in the plot the Virus-type Digimon is over-represented and the Free-type Digimon is under-represented. Also please give more love to Data-type Digimon :(

# ### Are different types and attributes evenly represented across stages?

# In[ ]:


plt.figure(figsize=(12,4))
sns.countplot(hue=digimon['Type'], x=digimon['Stage'])
plt.show()


# From the chart above, Digimon Types are not really evenly represented across stages. Digimon's type started as Free-type in Baby and In-Training stages, then as they reached their Rookie stage, they branches evenly towards the other types, with only some Digimon retaining their Free-type. As they keep growing to other higher stages (besides Ultra and Armor stage, as they are special stage), the Digimon's type tends to be more of a Virus-type.

# In[ ]:


plt.figure(figsize=(15,8))
sns.countplot(hue=digimon['Attribute'], x=digimon['Stage'])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()


# Yikes, that's a heavy chart. The Rookie and Champion stage are Fire-type dominated, and Ultimate and Mega stage are Dark-type dominated. It's not evenly represented
