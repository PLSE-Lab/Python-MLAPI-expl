#!/usr/bin/env python
# coding: utf-8

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


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_csv("../input/youtube-new/USvideos.csv")
f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True, linewidths=.1, fmt= '.4f',ax=ax)
plt.show()


# In[ ]:


data.plot(kind = "scatter", x = "views", y = "likes", color = 'red', label = "speed", alpha=0.2, figsize = (15,15))
plt.title("views")
plt.show


# In[ ]:



data['prop_viewlike'] = (data['likes'] / data['views'])*100
propvtol = data['prop_viewlike']>10
data[propvtol]


# In[ ]:


def makescatter():
   
    for i in data.iterrows():
    for j in data.iteritems():

    data.plot(kind = "scatter", x = i, y = j, color = 'red', label = "speed", alpha=0.2, figsize = (15,15))
    plt.title("views")
    plt.show
    


# In[ ]:




