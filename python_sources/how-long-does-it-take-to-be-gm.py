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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


user = pd.read_csv('../input/Users.csv')
user_ac = pd.read_csv('../input/UserAchievements.csv')


# In[ ]:


user[user.UserName=='onodera']


# In[ ]:


user_ac[user_ac.UserId==317344]


# In[ ]:


user_ac = user_ac[(user_ac.AchievementType=='Competitions') & (user_ac.Tier>=2)]


# In[ ]:


user.rename(columns={'Id': 'UserId'}, inplace=True)
user_ac = pd.merge(user_ac, user, on='UserId', how='left')


# In[ ]:


user_ac.TierAchievementDate = user_ac.TierAchievementDate.map(pd.to_datetime)
user_ac.RegisterDate = user_ac.RegisterDate.map(pd.to_datetime)


# In[ ]:


user_ac_ = user_ac[user_ac.TierAchievementDate != pd.to_datetime('2016-07-15')]


# In[ ]:


user_gm = user_ac_[(user_ac_.Tier==4)]
user_ms = user_ac_[(user_ac_.Tier==3)]
user_ex = user_ac_[(user_ac_.Tier==2)]


# In[ ]:


(user_gm.TierAchievementDate - user_gm.RegisterDate).dt.days.hist(bins=20)
plt.title('How long does it take to be Grandmaster?(days)')


# In[ ]:


(user_ms.TierAchievementDate - user_ms.RegisterDate).dt.days.hist(bins=20)
plt.title('How long does it take to be Master?(days)')


# In[ ]:


(user_ex.TierAchievementDate - user_ex.RegisterDate).dt.days.hist(bins=20)
plt.title('How long does it take to be Expert?(days)')


# In[ ]:


(user_gm.TierAchievementDate - user_gm.RegisterDate).dt.days.describe()


# In[ ]:


(user_ms.TierAchievementDate - user_ms.RegisterDate).dt.days.describe()


# In[ ]:


(user_ex.TierAchievementDate - user_ex.RegisterDate).dt.days.describe()

