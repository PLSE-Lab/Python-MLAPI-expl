#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author: malai_guhan(ram1510).This is released under the Apache 2.0 open source license.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train=pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
test=pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')


# **1. Does healing improve the Chance of winning the game?**

# In[ ]:


x=train.corr()
plt.figure(figsize=(20,20))
sns.heatmap(x,annot=True)


# **Healing improves the chance of winning the game as the pearson coefficient is found to be 0.42 **

# **Does Knocking (DBNO), Assisting or Reviving effect on Winning Percentage ?**

# In[ ]:


sns.jointplot(data=train,y='DBNOs',x='winPlacePerc',height=10,ratio=3)


# In[ ]:


sns.jointplot(x='winPlacePerc',y='assists',data=train,height=10,ratio=3)


# In[ ]:


sns.jointplot(x='winPlacePerc',y='revives',data=train,height=10,ratio=3)


# Assists, revives and DBNOs have very little effect on Winning percentage

# In[ ]:


mod_train=train[0:100000]


# In[ ]:


np.unique(train['matchType'])


# In[ ]:


mio=lambda x:'Solo' if 'solo' in x else 'duo' if ('duo' or 'crash') in x else 'squad'


# In[ ]:


mod_train['matchtype']=mod_train['matchType'].apply(mio)


# In[ ]:


train_data=mod_train.drop(columns=['winPlacePerc','matchType','Id','groupId','matchId'])


# In[ ]:


train_data.info()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *


# In[ ]:


train_data=pd.get_dummies(train_data,drop_first=True)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(train_data,mod_train['winPlacePerc'],random_state=100,test_size=0.2)


# In[ ]:


classifier=RandomForestRegressor()


# In[ ]:


classifier.fit(x_train,y_train)


# In[ ]:


y_pred=classifier.predict(x_test)


# In[ ]:


from sklearn.metrics import mean_absolute_error


# In[ ]:


mae=mean_absolute_error(y_test,y_pred)
print(f'The mean absolute error is {mae}')


# The MAE value is higher when all the attributes are considered.

# In[ ]:


mod_train.columns


# In[ ]:


mod_train['totaldistancetravelled']=mod_train['rideDistance']+mod_train['walkDistance']


# In[ ]:


len(mod_train)


# In[ ]:


#To find the number of players in a match
no_of_players=train.groupby('matchId')['Id'].size().to_frame('no.of.players')


# In[ ]:


mod1_data=pd.merge(mod_train,no_of_players,on='matchId',how='inner')


# In[ ]:


mod1_data['no.of.players'].value_counts()


# A new variable is created by combining ride distance and walk distance to form the total distance travelled. This has the correlation of 0.67

# In[ ]:


len(mod1_data)


# In[ ]:


plt.figure(figsize=(25,25))
sns.heatmap(mod1_data.corr(),annot=True)
#mod1_data.corr()


# In[ ]:


mod1_data.columns


# In[ ]:


train_data=mod1_data.drop(columns=['winPlacePerc','matchType','Id','groupId','matchId','killPoints','matchDuration','maxPlace','numGroups','rankPoints','roadKills','teamKills','vehicleDestroys','winPoints'])


# In[ ]:


train_data=pd.get_dummies(train_data)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(train_data,mod1_data['winPlacePerc'],random_state=100,test_size=0.2)


# In[ ]:


classifier.fit(x_train,y_train)


# In[ ]:


x_pred=classifier.predict(x_test)


# In[ ]:


accuracy=r2_score(y_test,x_pred)
print(f'The accuracy is {accuracy}')


# With highly correlated variables the accuracy is 91%

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


classifier1=GradientBoostingRegressor(n_estimators=100)


# In[ ]:


classifier1.fit(x_train,y_train)


# In[ ]:


x_pred=classifier1.predict(x_test)


# In[ ]:


accuracy=r2_score(y_test,x_pred)
print(f'The accuracy is {accuracy}')

