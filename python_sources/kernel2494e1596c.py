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


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import copy
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLS
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV 
import re
import os

get_ipython().run_line_magic('matplotlib', 'inline')

print(os.listdir("../input"))


# In[4]:


train_df = pd.read_csv("../input/train_V2.csv")
test_df = pd.read_csv("../input/test_V2.csv")
print(train_df.shape, test_df.shape)


# In[5]:


train_df.head()


# In[6]:


train_df.columns


# In[7]:


groupsize = train_df.groupby("matchType").size().reset_index()
groupsize.columns = ["matchType","percentage"]
groupsize.percentage = groupsize.percentage / groupsize.percentage.sum() * 100


# In[8]:


train_df.groupby("matchType").size()


# In[9]:


test_df.groupby("matchType").size()


# In[10]:


groupsize_test = test_df.groupby("matchType").size().reset_index()
groupsize_test.columns = ["matchType","percentage_test"]
groupsize_test.percentage_test = groupsize_test.percentage_test / groupsize_test.percentage_test.sum() * 100


# In[11]:


groupsize = pd.merge(groupsize, groupsize_test)


# In[12]:


standard_mode = ["solo", "duo", "squad", "solo-fpp","duo-fpp", "squad-fpp"]


# In[13]:


groupsize[groupsize.matchType.isin(standard_mode)][["percentage", "percentage_test"]].sum()


# Treat only standard mode as solo,duo,squad(seems like tpp) and solo-fpp,duo-fpp, squad-fpp

# In[15]:


train_df = train_df[train_df.matchType.isin(standard_mode)]


# In[16]:


train_df.isnull().sum(axis=0)


# In[17]:


test_df.isnull().sum(axis=0)


# Missing value exist in winPlacePerc

# In[18]:


train_df[np.isnan(train_df.winPlacePerc)]


# I think that it doesn't have any meaningful data, so drop this object in train_df

# In[19]:


nan_index = train_df[np.isnan(train_df.winPlacePerc)].index
train_df=train_df.drop(nan_index, axis =0)


# In[20]:


fig = plt.figure(figsize = (16,6))
sns.distplot(train_df["winPlacePerc"], kde = False, color = 'green', bins=120)


# In[21]:


train_df.info()


# In[22]:


numeric_features = train_df.select_dtypes(exclude = ["object"]).columns.tolist()


# In[23]:


train_df.select_dtypes(include = ["object"]).head()


# In[24]:


train_df[["Id"]].duplicated().unique()


# There is no duplicated ID

# In[25]:


train_df[numeric_features].nunique()


# In[26]:


corr = train_df[numeric_features].corr()
f, ax = plt.subplots(figsize = (20,20))

sns.heatmap(corr, vmax=.8, cmap = 'YlGnBu', annot = True, square=True)


# Since match duration is same value for same matchId,and it has no relation with target value  
# So, drop this feature in train_df

# In[27]:


train_df = train_df.drop(["matchDuration"], axis=1)


# maxPlace and numGroups have same definition, so the correlation calculated as 1, but some cases, maxPlace is larger than numGroups
# Also, winPlacePerc is calculated by maxPlace, not numGroups. So I'll drop numGroups in train_df

# In[28]:


train_df = train_df.drop(["numGroups"], axis =1)


# In[29]:


numeric_features.remove("numGroups")
numeric_features.remove("matchDuration")


# In[30]:


train_df.shape


# In[34]:


train_df[(train_df.kills == 0 ) & (train_df.winPlacePerc == 1) & (train_df.matchType.isin(["solo","solo-fpp"]))][["matchType"]].groupby("matchType").size()


# In[35]:


zero_kill_winner = train_df[(train_df.kills == 0 ) & (train_df.winPlacePerc == 1) & (train_df.matchType.isin(["solo","solo-fpp"]))].index


# In[36]:


train_df = train_df.drop(zero_kill_winner, axis=0)


# 0-kill winners in Solo or Solo-fpp are created by error or very very rare case  
# So drop them in train_df

# In[ ]:


groupsize_test = test_df.groupby("matchType").size().reset_index()
groupsize_test.columns = ["matchType","percentage_test"]
groupsize_test.percentage_test = groupsize_test.percentage_test / groupsize_test.percentage_test.sum() * 100


# In[ ]:


kills_perc = train_df[["kills"]].groupby("kills").size().reset_index()
kills_perc.columns = ["kills", "kill_perc"]
kills_perc.kill_perc = kills_perc.kill_perc / kills_perc.kill_perc.sum() * 100


# In[ ]:


kills_perc["cumulative_perc"] = kills_perc.kill_perc.cumsum()


# In[37]:


kills_perc


# In[38]:


group_kills = train_df[["groupId","matchId", "kills"]].groupby(["groupId","matchId"]).sum().reset_index()


# In[39]:


group_kills.columns=["groupId","matchId","groupKills"]


# In[40]:


train_df = pd.DataFrame.merge(train_df, group_kills, on=["groupId","matchId"])


# In[41]:


gc.collect()


# In[42]:


duplicate_groupid = train_df[(train_df.matchType.isin(["solo","solo-fpp"]))&(train_df.kills != train_df.groupKills)].groupId.unique().tolist()


# In[43]:


train_df[train_df.groupId=="a533f21e0d9e98"][["groupId","matchId","kills","matchType"]]


# How they have same groupId and matchId in solo type match???

# In[44]:


plt.figure(figsize=(20,8))
sns.boxplot(x="groupKills", y="winPlacePerc", data=train_df)


# Make some features by 'kills' in train_df

# In[45]:


train_solo = train_df[train_df.matchType.isin(["solo","solo-fpp"])]
train_not_solo = train_df[~train_df.matchType.isin(["solo","solo-fpp"])]


# In[46]:


test_solo_index = test_df[test_df.matchType.apply(lambda matchtype : bool(re.search("solo", matchtype)))].index
test_not_solo_index = test_df[~test_df.matchType.apply(lambda matchtype : bool(re.search("solo", matchtype)))].index


# In[47]:


test_solo = test_df.iloc[test_solo_index,]
test_not_solo = test_df.iloc[test_not_solo_index,]


# In[48]:


min_kills = train_not_solo[["groupId", "matchId","kills"]].groupby(["groupId","matchId"]).min().reset_index()
max_kills = train_not_solo[["groupId", "matchId","kills"]].groupby(["groupId","matchId"]).max().reset_index()
mean_kills = train_not_solo[["groupId", "matchId","kills"]].groupby(["groupId","matchId"]).mean().reset_index()


# In[49]:


test_mean_kills = test_not_solo[["groupId", "matchId","kills"]].groupby(["groupId","matchId"]).mean().reset_index()


# In[50]:


min_kills.columns = ["groupId","matchId","min_kills"]
max_kills.columns = ["groupId","matchId","max_kills"]
mean_kills.columns = ["groupId","matchId","mean_kills"]


# In[51]:


test_mean_kills.columns = ["groupId","matchId","mean_kills"]


# In[52]:


tmp_df = copy.deepcopy(train_solo)
test_tmp_df = copy.deepcopy(test_solo)


# In[53]:


tmp_df["mean_kills"] = train_solo.kills
tmp_df["min_kills"] = train_solo.kills
tmp_df["max_kills"] = train_solo.kills


# In[54]:


test_tmp_df["mean_kills"] =  test_solo.kills


# In[55]:


kills_df = pd.DataFrame.merge(min_kills,max_kills,on=min_kills.columns[:-1].tolist())
kills_df = pd.DataFrame.merge(kills_df, mean_kills, on=min_kills.columns[:-1].tolist())


# In[56]:


kills_df.head()


# In[57]:


train_not_solo = pd.merge(train_not_solo, kills_df, on = ["groupId","matchId"])


# In[58]:


test_not_solo = pd.merge(test_not_solo, test_mean_kills, on = ["groupId","matchId"])


# In[59]:


train_df = train_not_solo.append(tmp_df, sort=True)
test_df = test_not_solo.append(test_tmp_df, sort=True)


# In[60]:


print(train_df.shape, test_df.shape)


# In[61]:


del [[train_not_solo, train_solo, tmp_df, kills_df, test_not_solo, test_solo, test_tmp_df, test_mean_kills]]
gc.collect()


# In[62]:


train_df[["kills","killStreaks","min_kills","mean_kills","max_kills","groupKills","winPlacePerc"]].corr()


# mean_kills is mostly related with target variable and they have strong muiticolinearlit. So, drop other features about kill

# In[63]:


train_df = train_df.drop(["kills","killStreaks","min_kills","max_kills","groupKills"], axis=1)


# In[64]:


plt.figure(figsize=(20,8))
sns.boxplot(x=np.round(train_df.mean_kills), y=train_df.winPlacePerc)


# In[65]:


column_order = ['DBNOs', 'Id', 'assists', 'boosts', 'damageDealt', 'groupId',
       'headshotKills', 'heals', 'killPlace', 'killPoints',
       'longestKill', 'matchId', 'matchType', 'maxPlace',
       'mean_kills', 'rankPoints', 'revives', 'rideDistance', 'roadKills',
       'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance',
       'weaponsAcquired', 'winPoints', 'winPlacePerc']
train_df = train_df[column_order]


# In[66]:


numeric_features = train_df.select_dtypes(exclude = ["object"]).columns.tolist()


# In[67]:


corr = train_df[numeric_features].corr()
f, ax = plt.subplots(figsize = (20,20))

sns.heatmap(corr, vmax=.8, cmap = 'YlGnBu', annot = True, square=True)


# In[68]:


high_corr_features = {}
for col in corr.columns:
    high_corr_features.setdefault(col,[])
    for row in corr.index:
        if row != col:
            if abs(corr.loc[row,col]) > 0.7:
                high_corr_features[col].append(row)
                   
corr.winPlacePerc.sort_values(ascending = False)


# In[69]:


high_corr_features


# In[70]:


not_correlated = corr[["winPlacePerc"]][abs(corr.winPlacePerc) <= 0.2].index.tolist()
not_correlated


# In[71]:


train_df = train_df.drop(not_correlated, axis=1)


# In[72]:


numeric_features = train_df.select_dtypes(exclude = ["object"]).columns.tolist()


# In[73]:


corr = train_df[numeric_features].corr()
corr.shape


# In[74]:


corr = corr.drop("winPlacePerc", axis=1)
corr = corr.drop("winPlacePerc", axis=0)
corr.shape


# In[75]:


vif = pd.DataFrame()  
vif["VIF_Factor"] = [variance_inflation_factor(corr.values,i) for i in range(corr.shape[1])]  
vif["features"] = corr.columns  
vif = vif.sort_values(by = "VIF_Factor", ascending = False)
vif


# In[76]:


while True:
    features = corr.columns.tolist()
    vif = pd.DataFrame()
    vif["VIF_factor"] = [variance_inflation_factor(corr.values, i) for i in range(len(features))]
    vif["features"] = features
    vif = vif.sort_values(by="VIF_factor", ascending = False)
    features = vif.features.tolist()
    print(features)
    if vif.iloc[0,0] >= 15:
        exceped = features.pop(0)
        corr = train_df[features].corr()
    else:
        break
vif


# In[77]:


train_df_X = train_df[features]
test_df_X = test_df[features]
train_df_label = train_df.winPlacePerc
print("train_X_shape : {}, train_label_shape : {} \n test_X_shape : {}".format(train_df_X.shape, train_df_label.shape,test_df_X.shape))


# In[78]:


get_ipython().run_cell_magic('time', '', 'base_Model = OLS(train_df_label, train_df_X).fit()\npredictions = base_Model.predict(train_df_X)')


# In[79]:


base_Model.summary()


# In[80]:


plt.figure(figsize=(5,9))
sns.distplot(predictions, kde = False, bins = 120)


# In[81]:


mean_squared_error(predictions, train_df_label)


# In[82]:


predictions[predictions >= 1] = 1
mean_squared_error(predictions, train_df_label)


# In[95]:


params = {
    "n_estimators" : [100],
    "max_depth" : [3],
    "learning_rate" : [0.1],
    "sub_sample" : [0.7],
    "nthreads" : [-1]
}


# In[96]:


xgbmodel = XGBRegressor()
RSCxgb = RandomizedSearchCV(xgbmodel, params, cv=5)


# In[97]:


get_ipython().run_cell_magic('time', '', 'model = RSCxgb.fit(train_df_X, train_df_label)')


# In[98]:


pred_train = model.predict(train_df_X)
mean_squared_error(pred_train, train_df_label)


# In[99]:


predict = model.predict(test_df_X)
sns.distplot(predict, kde = True, color="green", bins=120)


# In[100]:


predict[predict >= 1] = 1


# In[101]:


submission = pd.DataFrame({"Id" : test_df.Id,
             "winPlacePerc" : predict})


# In[102]:


submission.shape[0] == test_df.shape[0]


# In[104]:


submission.to_csv("submission.csv", index=False)


# In[ ]:




