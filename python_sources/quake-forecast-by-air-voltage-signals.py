#!/usr/bin/env python
# coding: utf-8

# ![2019-11-07%20%E6%96%B0%E7%AB%B9%E7%B6%A0%E5%85%89%E7%AB%99%20Hsinchu%20Green%20Light%20Station%20%28NI%20USB-6210%29%201%20-%20Analysis.png](attachment:2019-11-07%20%E6%96%B0%E7%AB%B9%E7%B6%A0%E5%85%89%E7%AB%99%20Hsinchu%20Green%20Light%20Station%20%28NI%20USB-6210%29%201%20-%20Analysis.png)

# # Quake Forecast by Air Voltage Signals
# ### Dyson Lin, Founder & CEO of Taiwan Quake Forecast Institute
# ### 2020-02-12 05:59 UTC+8
# I measure air voltage signals to predict quakes.
# 
# I also interpret IRIS signals to predict quakes.
# 
# I have 30+ quake forecast stations around the world.
# 
# I accurately predicted a lot of big quakes around the world.
# 
# Recently, I accurately predicted two big quakes in Turkey.
# 
# That made me famous in Turkey within a few days.
# 
# I will develop some AI models to predict quakes automatically.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:




