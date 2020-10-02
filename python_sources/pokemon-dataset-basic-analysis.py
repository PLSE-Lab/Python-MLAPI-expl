#!/usr/bin/env python
# coding: utf-8

# # Pokemon dataset basic analysis
# 
# Beginning with Kaggle and Data Visualization

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt
sns.set()

print('Modules loaded')


# # Pokemon data
# Import and load [Pokemon with Stats](https://www.kaggle.com/abcsds/pokemon) dataset.

# In[ ]:


pokemon_filepath = '../input/pokemon/Pokemon.csv'
pokemon = pd.read_csv(pokemon_filepath)
pokemon.head()


# ## Subplotting

# In the cell below we create a subplot with 2 rows and 3 columns, and we plot 6 charts on it. We also define a figure size of(18, 10). So, we plot all charts with Generation columns as x-axis and a different power stat as y-axis for each chart. We use seaborn boxplot categorical plot.

# In[ ]:


fig, axes = plt.subplots(2, 3, figsize=(18, 10))

fig.suptitle('Pokemon Stats by Generation')

sns.boxplot(ax=axes[0, 0], data=pokemon, x='Generation', y='Attack')
sns.boxplot(ax=axes[0, 1], data=pokemon, x='Generation', y='Defense')
sns.boxplot(ax=axes[0, 2], data=pokemon, x='Generation', y='Speed')
sns.boxplot(ax=axes[1, 0], data=pokemon, x='Generation', y='Sp. Atk')
sns.boxplot(ax=axes[1, 1], data=pokemon, x='Generation', y='Sp. Def')
sns.boxplot(ax=axes[1, 2], data=pokemon, x='Generation', y='HP')


# From the data visualization above we can figure out a slightly superiority of the 4th Generation based on the median value. 4h Generation group has the higher median value in all charts.
