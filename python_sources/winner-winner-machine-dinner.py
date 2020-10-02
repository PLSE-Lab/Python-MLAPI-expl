#!/usr/bin/env python
# coding: utf-8

# ## PUBG analisys
# **Lets make some data magic!**

# In[ ]:


#misc.
import warnings

#libraries for analysis.
import numpy as np
import pandas as pd

#libraries for visualization.
import matplotlib.pyplot as plt
import seaborn as sns 

#libraries for machine learning.
from sklearn import linear_model
from sklearn import svm
import lightgbm as lgb

#reading train.csv and test.csv with pandas
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

warnings.filterwarnings("ignore")


# **Getting some basic info about the train dataset.**

# In[ ]:


train.info()
train.head()
train.count()
train.describe()


# **Identifiyng solos, duos and squads by "numGroups" attribute.** Review

# In[ ]:


solos = train[train['numGroups']>50]
duos = train[(train['numGroups']>25) & (train['numGroups']<=50)]
squads = train[train['numGroups']<=25]
print("There are {} ({:.2f}%) solo games, {} ({:.2f}%) duo games and {} ({:.2f}%) squad games.".format(len(solos), 100*len(solos)/len(train), len(duos), 100*len(duos)/len(train), len(squads), 100*len(squads)/len(train),))


# **Correlation heatmap for solos, duos and squad.** Review

# In[ ]:


f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)
#sns.heatmap(solos.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)
#sns.heatmap(duos.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)
#sns.heatmap(squads.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)
plt.show()


# **Removing features that not going to influence in prediction, like id's and others identified througt correlation heatmap.** Test

# In[ ]:


train03 = train.drop([
    'Id', 'groupId', 'matchId', 'killPoints', 'maxPlace', 'roadKills', 
    'swimDistance', 'teamKills', 'vehicleDestroys', 'winPoints', 'numGroups'
], axis=1).copy()

train04 = train.drop([
    'Id', 'groupId', 'matchId', 'killPoints', 'maxPlace', 'roadKills', 
    'swimDistance', 'teamKills', 'vehicleDestroys', 'winPoints', 'numGroups', 
    'assists', 'DBNOs', 'headshotKills', 'revives', 'rideDistance'
], axis=1).copy()

train05 = train.drop([
    'Id', 'groupId', 'matchId', 'killPoints', 'maxPlace', 'roadKills', 
    'swimDistance', 'teamKills', 'vehicleDestroys', 'winPoints', 'numGroups', 
    'assists', 'DBNOs', 'headshotKills', 'revives', 'rideDistance', 'killPlace'
], axis=1).copy()

X_test = test.drop([
    'Id', 'groupId', 'matchId', 'killPoints', 'maxPlace', 'roadKills', 
    'swimDistance', 'teamKills', 'vehicleDestroys', 'winPoints', 'numGroups'
], axis=1).copy()


# **Preparing our datasets to prediction.**

# In[ ]:


X_train03 = train03.drop("winPlacePerc", axis=1)
X_train04 = train04.drop("winPlacePerc", axis=1)
X_train05 = train05.drop("winPlacePerc", axis=1)

Y_train03 = train03["winPlacePerc"]
Y_train04 = train04["winPlacePerc"]
Y_train05 = train05["winPlacePerc"]


# **Predicting with several models and scoring them.**

# In[ ]:


models = [
    #linear_model.BayesianRidge(),           # train03=79.71 | train04=79.48 | train05=70.56
    #linear_model.LinearRegression(),        # train03=79.71 | train04=79.48 | train05=70.56
    lgb.LGBMRegressor(n_estimators=550, num_leaves=55)                     # train03=87.17 | train04=86.69 | train05=78.66
]

for model in models:
    print(model)
    model.fit(X_train03, Y_train03)
    print(round(model.score(X_train03, Y_train03) * 100, 2), '\n')
    
#for model in models:
#    print(model)
#    model.fit(X_train04, Y_train04)
#    print(round(model.score(X_train04, Y_train04) * 100, 2), '\n')
    
#for model in models:
#    print(model)
#    model.fit(X_train05, Y_train05)
#    print(round(model.score(X_train05, Y_train05) * 100, 2), '\n')
    
prediction = model.predict(X_test)


# **Sending prediction.**

# In[ ]:


submission = pd.DataFrame({
        "Id": test["Id"],
        "winPlacePerc": prediction
})

submission.to_csv('submission.csv', index=False)


# **Still working on it...**
