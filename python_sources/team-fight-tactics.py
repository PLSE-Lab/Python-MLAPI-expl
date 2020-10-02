#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


read_data = pd.read_csv("/kaggle/input/league-of-legends-tftteamfight-tacticschampion/TFT_Champion_CurrentVersion.csv")
data = pd.DataFrame(read_data)
data.head(10)


# In[ ]:


dropped_data = data.drop(["cost","skill_name", "skill_cost", "origin", "class"], axis = 1)

columns = ["health","defense","attack","attack_range","speed_of_attack","dps"]

def sorting():
    for i in columns:
        print("Sorting by {}".format(i).title())
        print(dropped_data.sort_values(by = i, ascending = False))
        print()
    
sorting()


# In[ ]:


origin = data.groupby(["origin"]).mean()
origin


# In[ ]:


sns.catplot(x="origin", y="health", hue="class", kind="swarm", data=data, height = 9, s = 15);


# In[ ]:


# Brawler Blaster Comp Table

blaster = data[data["class"].isin(["['Blaster']"])]
brawler = data[data["class"].isin(["['Brawler']"])]
brawler_blaster = pd.concat([brawler,blaster]).sort_values("cost")
brawler_blaster

