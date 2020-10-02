#!/usr/bin/env python
# coding: utf-8

# # Introductroy Story-line
# 
# * Please upvote and visit again.
# 
# Tasked to investigate the relationship between the playing surface and the injury and performance of National Football League (NFL) athletes and to examine factors that may contribute to lower extremity injuries.

# In[ ]:


import numpy as np
import pandas as pd
import os as os
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


print(os.listdir("../input/nfl-playing-surface-analytics/"))


# In[ ]:


playList= pd.read_csv("../input/nfl-playing-surface-analytics/PlayList.csv")
playerTrackData= pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv")
InjuryRecord = pd.read_csv("../input/nfl-playing-surface-analytics/InjuryRecord.csv")


# In[ ]:


playList.columns


# In[ ]:


playerTrackData.columns


# In[ ]:


InjuryRecord.columns


# In[ ]:


playList.head(4)


# In[ ]:


playerTrackData.head(4)


# In[ ]:


InjuryRecord.head(4)


# In[ ]:


print(playList.shape)
print(playerTrackData.shape)
print(InjuryRecord.shape)


# In[ ]:


playList.isna().sum()


# In[ ]:


playerTrackData.isna().sum()


# In[ ]:


InjuryRecord.isna().sum()


# In[ ]:


pp = sns.pairplot(playList, hue="StadiumType")


# In[ ]:


sns.pairplot(playList, hue="RosterPosition")


# In[ ]:


sns.pairplot(playList, hue="Weather")


# In[ ]:


sns.pairplot(playList, hue="Position")


# In[ ]:


pp3 = sns.pairplot(InjuryRecord)

