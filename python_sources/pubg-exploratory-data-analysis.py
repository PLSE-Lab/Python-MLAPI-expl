#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))


# In[ ]:


training_set = pd.read_csv('../input/train_V2.csv')
training_set.head()


# In[ ]:


training_set.shape


# In[ ]:


training_set.columns


# In[ ]:


training_set.describe()


# In[ ]:


training_set.isna().sum()


# There is no missing data. 

# In[ ]:


training_set.plot(x = "kills", y = "damageDealt", kind="scatter", figsize = (15,10))


# Clearly, there is a positive correlation between the number of kills and damage dealt.

# In[ ]:


import seaborn as sns
headshots = training_set[training_set['headshotKills'] > 0]
plt.figure(figsize = (15, 5))
sns.countplot(headshots['headshotKills'])


# In[ ]:


dbno = training_set[training_set['DBNOs'] > 0]
plt.figure(figsize = (15, 5))
sns.countplot(dbno['DBNOs'])


# In[ ]:


training_set.plot(x = 'kills', y = 'DBNOs', kind = 'scatter', figsize = (15, 10))


# There is a positive correlation between no. of enemies down but not out (DBNO) and the number of kills.

# In[ ]:


walk0 = training_set["walkDistance"] == 0
ride0 = training_set["rideDistance"] == 0
swim0 = training_set["swimDistance"] == 0
print("{} of players didn't walk at all, {} players didn't drive and {} didn't swim." .format(walk0.sum(),ride0.sum(),swim0.sum()))


# In[ ]:


walk0_data = training_set[walk0]
print("Average place for non walkers is {:.3f}, minimum is {}, and best is {}, 95% players have a score below {}."
     .format(walk0_data['winPlacePerc'].mean(), walk0_data['winPlacePerc'].min(), walk0_data['winPlacePerc'].max(), walk0_data['winPlacePerc'].quantile(0.95)))
walk0_data.hist('winPlacePerc',bins = 50, figsize = (15, 5))


# Most non walkers tend to be on the lower side of the scoreboard but some of them have the best scores. These could be suspicious players. Following are the players that did not walk at all but have the best score.

# In[ ]:


suspicious = training_set.query('walkDistance == 0 & winPlacePerc == 1')
suspicious.head()


# In[ ]:


print("Maximum ride distance for suspected entries is {:.3f} meters, and swim distance is {:.1f} meters." .format(suspicious["rideDistance"].max(), suspicious["swimDistance"].max()))


# Non walker- winners are non-rider winners as well becsause their ride distance is 0.

# In[ ]:


plt.plot(suspicious['swimDistance'])


# In[ ]:


suspicious_non_swimmer = suspicious[suspicious['swimDistance'] == 0]
suspicious_non_swimmer.shape


# So there are 162 non swimmers, non walkers and non riders who won. They clearly cheated.

# In[ ]:


ride = training_set.query('rideDistance >0 & rideDistance <10000')
walk = training_set.query('walkDistance >0 & walkDistance <4000')
ride.hist('rideDistance', bins=40, figsize = (15,10))
walk.hist('walkDistance', bins=40, figsize = (15,10))


# This shows that players mostly walk.
