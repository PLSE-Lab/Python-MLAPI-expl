#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn # data visualization
import matplotlib.pyplot as ml #data visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
olym = pd.read_csv("../input/athlete_events.csv", sep=",", header=0)
#print(wine.head)
print(olym.shape)
# Any results you write to the current directory are saved as output.


# In[ ]:


print(olym.describe())


# In[ ]:


olym.plot(kind="box", subplots = True, layout=(4,6), sharex=False, sharey = False)


# In[ ]:


cor = olym.corr()
sn.heatmap(cor, annot = True, linewidths=1.0)


# # excluding very yound athletes most seem to have the same median weight across all sports

# In[ ]:


sn.violinplot(x="Age", y="Weight", data=olym, size=10)

