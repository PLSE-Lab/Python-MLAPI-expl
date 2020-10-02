#!/usr/bin/env python
# coding: utf-8

# Hi Guys, Thanks for opening my kernel. There are some question which I have solved in this kernel are described below.

# Those Players can be cheaters:
# 1. Is it possible to kill or knock out players without even moving ?
# 2. Is it possible to get on killplace equal to 1 without moving or killing anyone ?
# 3. Is it possible to kill someone, If the player distance travelled only by swimming ?
# 4. Is it possible a player killed more than 10 players and 90 percent kills are head shot ?
# 5. Is it possible without walking the player accuquired weapons(more than 2) ?
# 6. Is it psiible to kill like more than 10 players, Only by walkDistance less than 100 metres ?
# 7. Is it possible Damage Dealt is 0 but kills or vehicle destroyed are more than 1 ?
# 8. Is is possible to increase the game play speed ?

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


Original_Train_df = pd.read_csv('../input/train_V2.csv')
Original_Test_df = pd.read_csv('../input/test_V2.csv')
Train_df = Original_Train_df
Test_df = Original_Test_df


# In[ ]:


Train_df.head(5)


# In[ ]:


Train_df.isna().sum()


# Train_df with missing value in winPlacePerc

# In[ ]:


Train_df[Train_df['winPlacePerc'].isna()]


# As we can see all column have 0 as value so winning percentage will definately be 0

# In[ ]:


Train_df = Train_df.fillna(0)


# In[ ]:


Train_df = Train_df.drop(['Id','groupId','matchId'],axis=1)


# Dropping unnecessary columns which no use in the predictions 

# Question : **Is it possible to kill or knock out players without even moving ?**
# 
# Answer : I think That is not possible

# In[ ]:


Train_df[((Train_df['DBNOs']>0)&(Train_df['walkDistance']==0)&(Train_df['rideDistance']==0)&(Train_df['swimDistance']==0))|((Train_df['kills']>0)&(Train_df['walkDistance']==0)&(Train_df['rideDistance']==0)&(Train_df['swimDistance']==0))]


# We can say that this can be hackers who have travelled 0 distance but still DBNOs and kills more then 0, That is not possible **To Knock out or kill someone player need to move or travel(Increase Distance)**

# In[ ]:


Train_df = Train_df.drop(Train_df[((Train_df['DBNOs']>0)&(Train_df['walkDistance']==0)&(Train_df['rideDistance']==0)&(Train_df['swimDistance']==0))|((Train_df['kills']>0)&(Train_df['walkDistance']==0)&(Train_df['rideDistance']==0)&(Train_df['swimDistance']==0))].index,axis = 0)


# Question : **Is it possible to get on killPlace equals to 1 without moving or killing anyone ?**

# In[ ]:


Train_df[(Train_df['killPlace']==1)&(Train_df['kills']==0)&(Train_df['walkDistance']==0)&(Train_df['rideDistance']==0)&(Train_df['swimDistance']==0)]


# The above players are also cheater Because they have not travelled a little and not even killed a single player but still they have kill Place = 1, That mean they are hacks.

# In[ ]:


Train_df = Train_df.drop(Train_df[(Train_df['killPlace']==1)&(Train_df['kills']==0)&(Train_df['walkDistance']==0)&(Train_df['rideDistance']==0)&(Train_df['swimDistance']==0)].index,axis = 0)


# Question : **Is it possible to kill someone, If the player distance travelled only by swimming ?**

# In[ ]:


Train_df[(Train_df['swimDistance']>0)&(Train_df['kills']>0)&(Train_df['walkDistance']==0)&(Train_df['rideDistance']==0)]


# A Player cannot kill some one only by swimmiing

# In[ ]:


Train_df = Train_df.drop(Train_df[(Train_df['swimDistance']>0)&(Train_df['kills']>0)&(Train_df['walkDistance']==0)&(Train_df['rideDistance']==0)].index, axis = 0)


# Question : **Is it possible a player killed more than 10 players and 90 percent kills are head shot ?**

# In[ ]:


Train_df['headshotRate'] = Train_df['headshotKills']/Train_df['kills']
Train_df[(Train_df['kills']>10)&(Train_df['headshotRate']>=.9)]


# The attribute created head shot rate higher than .9 and when kill are more than 10 most probably those players are using Hack

# In[ ]:


Train_df = Train_df.drop(Train_df[(Train_df['kills']>10)&(Train_df['headshotRate']>=.9)].index,axis = 0)


# Question : **Is it possible without walking the player accuquired weapons(more than 2)?**

# In[ ]:


Train_df[(Train_df['walkDistance']==0)&(Train_df['weaponsAcquired']>2)]


# Acquiring weapons without moving
# I think player accquiring weapons without walking is not possible, Only one Condition is possible that where the player landed got weapons over there only, I also think getting more than 2 weapons like this is not possible.

# In[ ]:


Train_df = Train_df.drop(Train_df[(Train_df['walkDistance']==0)&(Train_df['weaponsAcquired']>2)].index, axis = 0)


# In[ ]:



Train_df.kills.describe().astype(int)


# Question : **Is it psiible to kill like more than 10 players, Only by walkDistance less than 100 metres ?**

# In[ ]:


Train_df[(Train_df['kills']>10)&(Train_df['walkDistance']<100)]


# Walking Distance is less 100 meters but killings are more than 10, I think that looks like an outlier or hackers

# In[ ]:


Train_df = Train_df.drop(Train_df[(Train_df['kills']>10)&(Train_df['walkDistance']<100)].index, axis = 0)


# Question : **Is it possible Damage Dealt is 0 but kills or vehicle destroyed are more than 1 ?**

# In[ ]:


Train_df[(((Train_df['vehicleDestroys']>1)|(Train_df['kills']>1))&(Train_df['damageDealt']==0))]


# When a player killed an enemy or destroyed a vehicle, Then definetely the damage dealt will be greater than 0. But in the above tuples the damage dealt is 0.

# In[ ]:


Train_df = Train_df.drop(Train_df[(((Train_df['vehicleDestroys']>1)|(Train_df['kills']>1))&(Train_df['damageDealt']==0))].index,axis = 0)


# In[ ]:


Train_df.walkDistance.describe().astype(int)                                      


# In[ ]:


Train_df['Player_speed_m/s'] = Train_df['walkDistance']/Train_df['matchDuration']


# In[ ]:


Train_df['Player_speed_m/s'].describe().astype(int)


# In[ ]:


Train_df.corr()['Player_speed_m/s']


# In[ ]:


Train_df.corr().winPlacePerc


# Question : **Is is possible to increase the game play speed ?**

# **Some stats from https://pubg.gamepedia.com/Movement_Speed**
# Variable	    Speed    	Comparison
# Standing Sprint	6.3 m/s	    Baseline
# Crouch Sprint	4.8 m/s	    24% Slower
# Standing Run	4.7 m/s	    25% Slower
# Crouch Run	    3.4 m/s	    46% Slower
# Standing Walk	1.7 m/s	    72% Slower
# Crouch Walk	    1.3 m/s	    79% Slower
# Crawling	    1.2 m/s	    81% Slower
# Swimming	    2.9 m/s	    54% Slower
# 
# This means player cannot run more than 6.3
# 

# In[ ]:


Train_df[((Train_df['Player_speed_m/s']>5.5)&(Train_df['walkDistance']>2000)&(Train_df['kills']>1))]


# According to the stats players cannot run more than 6.3 meter per second, It is not possible to run for 2000 meter continuously when players land on the ground after killing. I think player will run like 5.5 m/s if landed and travelled alot.

# In[ ]:


Train_df = Train_df.drop(Train_df[((Train_df['Player_speed_m/s']>5.5)&(Train_df['walkDistance']>2000)&(Train_df['kills']>1))].index, axis = 0)


# In[ ]:


Train_df.corr().winPlacePerc


# In[ ]:


Final_Train_df =Train_df.drop(['killPoints','matchDuration','maxPlace','numGroups','rankPoints','roadKills','teamKills','vehicleDestroys','walkDistance','winPoints','headshotRate','swimDistance'], axis = 1)


# In[ ]:


Final_Train_df.corr().winPlacePerc


# In[ ]:


Final_Train_df.info()


# In[ ]:


Final_Train_df['matchType'].nunique()


# In[ ]:


# Importing the dataset
y = Final_Train_df.loc[:, ['winPlacePerc']].values
Final_Train_df = Final_Train_df.drop(['winPlacePerc'],axis = 1)
Final_Train_df = pd.get_dummies(Final_Train_df['matchType'])
#X = Final_Train_df.loc[:, ['assists','boosts','damageDealth','DBNOs','headshotKills','heals', 'killPlace','kills','killsStreaks','matchType','longestKill','revives','rideDistane','weaponAcquired','Player_speed_m/s']].values
X = Final_Train_df.loc[:,:].values


# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# **ANN**

# In[ ]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:



# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(256, activation = 'relu', input_dim = X_train.shape[1]))

# Adding the second hidden layer
model.add(Dense(units = 256, activation = 'relu'))

# Adding the third hidden layer
model.add(Dense(units = 512, activation = 'relu'))

# Adding the fourth hidden layer
model.add(Dense(units = 256, activation = 'relu'))

# Adding the fifth hidden layer
model.add(Dense(units = 256, activation = 'relu'))

# Adding the output layer
model.add(Dense(units = 1))


# In[ ]:


model.compile(optimizer = 'adam',loss = 'mean_squared_error')


# In[ ]:


model.fit(X_train, y_train, batch_size = 200, epochs = 10)


# In[ ]:


# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = model.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error

mae = mean_squared_error(y_test, y_pred)
mae


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_pred))
rms

