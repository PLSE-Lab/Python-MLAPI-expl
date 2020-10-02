#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import umap
import numpy as np
import pandas as pd
import requests
import os
import datashader as ds
import datashader.utils as utils
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="paper", style="white")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
train = pd.read_csv("../input/train.csv")
data = train.iloc[:, 2:].values.astype(np.float32)
target = train['activity'].values
activ = ["Walking", "Walking Upstairs", "Walking Downstairs", "Sitting", "Standing", "Laying"]
activities = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
pal = [
 'green',
 'red',
 'yellow',
 'grey',
 'brown',
 'black'
]
color_key = {activities[i]:pal[i] for i in range(len(pal))}

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(data)

df = pd.DataFrame(embedding, columns=('x', 'y'))
df['class'] = pd.Series([str(x) for x in target], dtype="category")

fig, ax = plt.subplots()
cmap = plt.cm.get_cmap('jet')

for i, cluster in df.groupby('class'):
    _ = ax.scatter(cluster['x'], cluster['y'], c=color_key[i], label=activ[activities.index(i)])
ax.legend()
plt.setp(ax, xticks=[], yticks=[])
plt.title("Human activity with Smartphone data embedded\n"
          "into two dimensions by UMAP",
          fontsize=12)

plt.show()


# In[ ]:




