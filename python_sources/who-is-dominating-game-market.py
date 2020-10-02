#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for ploting
import seaborn as sns # for heat map

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")
data.head()


# In[ ]:


data.info()


# In this part we can say that we ahve 16598 games and we are going to consider them in 11 parts.

# In[ ]:


f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt=".1f",ax=ax)


# From this heat map we can say that EU sales and NA sales have right proportion with global sales.

# In[ ]:


data.plot(kind="line",x="Rank",y="Global_Sales",alpha=1,color="b",lw=4)
plt.xlabel("Rank")
plt.ylabel("Global_Sales")
plt.title("Ranl Golabal Sales Plot")


# Firstly I made linewidth to 4 to observe the graph easily. Then from this chart we can say that most of the games don't have great numbers at global sales. We can say that from this knowledge there are some games are damonating.
# 

# In[ ]:



print(data[data["Global_Sales"]>20])


# After all we can say that in the game market from the beginning to now Nintendo has really big part and big effect on the game market. Nintendo has 16 games in global sales are greater than 20.
