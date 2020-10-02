#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.colors import LogNorm

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


print(df_train[:5])
df_train.describe()


# In[ ]:



# Sample them for quicker visualisations
df_train_sample = df_train.sample(n=1000000)
df_test_sample = df_test.sample(n=1000000)

# Check if accuracy of signal corresponds with time
plt.figure(4, figsize=(12,10))

plt.subplot(211)
plt.scatter(df_train_sample["time"], df_train_sample["accuracy"], s=1, c='k', lw=0, alpha=0.1)
plt.xlabel("Time")
plt.ylabel("Accuracy")
plt.xlim(df_train_sample["time"].min(), df_train_sample["time"].max())
plt.ylim(df_train_sample["accuracy"].min(), df_train_sample["accuracy"].max())
plt.title("Train")

plt.subplot(212)
plt.scatter(df_test_sample["time"], df_test_sample["accuracy"], s=1, c='k', lw=0, alpha=0.1)
plt.xlabel("Time")
plt.ylabel("Accuracy")
plt.xlim(df_test_sample["time"].min(), df_test_sample["time"].max())
plt.ylim(df_test_sample["accuracy"].min(), df_test_sample["accuracy"].max())
plt.title("Test")

plt.show()


# In[ ]:


# See the distribution of checkins in x and y.
plt.figure(4, figsize=(12,12))

plt.subplot(211)
plt.scatter(df_train_sample["time"], df_train_sample["accuracy"], s=1, c='k', lw=0, alpha=0.1)
plt.xlabel("Time")
plt.ylabel("Accuracy")
plt.xlim(df_train_sample["time"].min(), df_train_sample["time"].max())
plt.ylim(df_train_sample["accuracy"].min(), df_train_sample["accuracy"].max())
plt.title("Train")

plt.subplot(212)
plt.scatter(df_test_sample["time"], df_test_sample["accuracy"], s=1, c='k', lw=0, alpha=0.1)
plt.xlabel("Time")
plt.ylabel("Accuracy")
plt.xlim(df_test_sample["time"].min(), df_test_sample["time"].max())
plt.ylim(df_test_sample["accuracy"].min(), df_test_sample["accuracy"].max())
plt.title("Test")

plt.show()


# In[ ]:


stats = pd.DataFrame({
    'checkins': [0]*10000,
    'place_cnt': 0,
    'place_max': 0,
    })

for i in range(100):
    for j in range(100):
        cell = df_train[df_train.x < 0.1 * (i+1)]
        cell = cell [cell.x > 0.1*i]
        cell = cell [cell.y > 0.1*j]
        cell = cell [cell.y < 0.1 * (j+1)]
        
        stats.at[i*100+j, 'checkins'] = cell.x.count()

        places = cell['place_id'].value_counts()
        stats.at[i*100+j, 'place_cnt'] = places.count()
        stats.at[i*100+j, 'place_max'] = places.max()
        
print(stats)


# In[ ]:


place_hist = df_train['place_id'].value_counts()
print(place_hist.describe())
print

for place in place_hist[:3]:
    print(place)


# In[ ]:


place_hist = df_train['place_id'].value_counts()
place_hist.describe()
print

for place in place_hist[:3].keys():
    print(place)


# In[ ]:


for place in place_hist[:3].keys():
  hotspot = df_train[df_train.place_id == place]

  x = hotspot['x']
  y = hotspot['y']
  bins = 20
  while bins <=40:
      plt.hist2d(x, y, bins=bins, norm=LogNorm())
      plt.colorbar()
      plt.title('x and y location histogram - ' + str(bins) + ' bins')
      plt.xlabel('x')
      plt.ylabel('y')
      plt.show()
      bins = bins * 2


# In[ ]:


hotspot = df_train[df_train.place_id == 1308450003]

x = hotspot['x']
y = hotspot['y']
bins = 20
plt.hist2d(x, y, bins=bins, norm=LogNorm())
plt.colorbar()
plt.title('x and y location histogram - ' + str(bins) + ' bins')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
bins = bins * 2


# In[ ]:




