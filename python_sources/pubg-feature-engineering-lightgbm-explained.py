#!/usr/bin/env python
# coding: utf-8

# # This is the second part of my Kernel containing only the Feature Engineering and LightGBM algorithm. The Kernel is divided into two sections due to memory and time constraints of kaggle kernel. For Exploratory Data Analysis and Base Model of my kernel, Visit the first part of the model <a href='https://www.kaggle.com/iamarjunchandra/part-1-pubg-eda-base-model'>Here!</a>

# In[ ]:


#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import gc

#Figures Inline and Visualization style
get_ipython().run_line_magic('matplotlib', 'inline')
sb.set()


# In[ ]:


train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
train.dropna(inplace=True)


# # **3. FEATURE ENGINEERING**

# Let's Inspect the categorical colmn match type. 

# In[ ]:


train['matchType'].value_counts()


# 'groupId' and 'matchId' are available in the data.  From these, no. of players in the team and total players entered in the match can be extracted.

# In[ ]:


train['teamPlayers']=train.groupId.map(train.groupId.value_counts())
test['teamPlayers']=test.groupId.map(test.groupId.value_counts())
train['gamePlayers']=train.matchId.map(train.matchId.value_counts())
test['gamePlayers']=test.matchId.map(test.matchId.value_counts())


# Let's create a new column with total enemy players . The players remaining other than the player's squad. 

# In[ ]:


train['enemyPlayers']=train['gamePlayers']-train['teamPlayers']
test['enemyPlayers']=test['gamePlayers']-test['teamPlayers']


# Let's create a new column representing the total distance(ride+swim+walk) covered by the player in the game. 

# In[ ]:


train['totalDistance']=train['rideDistance']+train['swimDistance']+train['walkDistance']
test['totalDistance']=test['rideDistance']+test['swimDistance']+test['walkDistance']


# New column which is the sum of assists and kills.

# In[ ]:


train['enemyDamage']=train['assists']+train['kills']
test['enemyDamage']=test['assists']+test['kills']


# New column containing total kills by the team. For this, rows are grouped based on 'matchId', 'groupId' and the sum of matching row 'kills' are taken.

# In[ ]:


totalKills = train.groupby(['matchId','groupId']).agg({'kills': lambda x: x.sum()})
totalKills.rename(columns={"kills": "squadKills"}, inplace=True)
train = train.join(other=totalKills, on=['matchId', 'groupId'])
totalKills = test.groupby(['matchId','groupId']).agg({'kills': lambda x: x.sum()})
totalKills.rename(columns={"kills": "squadKills"}, inplace=True)
test = test.join(other=totalKills, on=['matchId', 'groupId'])


# Lets create  new columns and find if any of them helps improve model prediction.

# In[ ]:


train['medicKits']=train['heals']+train['boosts']
test['medicKits']=test['heals']+test['boosts']


# In[ ]:


train['medicPerKill'] = train['medicKits']/train['enemyDamage']
test['medicPerKill'] = test['medicKits']/test['enemyDamage']


# In[ ]:


train['distancePerHeals'] = train['totalDistance']/train['heals']
test['distancePerHeals'] = test['totalDistance']/test['heals']


# In[ ]:


train['headShotKillRatio']=train['headshotKills']/train['kills']
test['headShotKillRatio']=test['headshotKills']/test['kills']


# In[ ]:


train['headshotKillRate'] = train['headshotKills'] / train['kills']
test['headshotKillRate'] = test['headshotKills'] / test['kills']


# In[ ]:


train['killPlaceOverMaxPlace'] = train['killPlace'] / train['maxPlace']
test['killPlaceOverMaxPlace'] = test['killPlace'] / test['maxPlace']


# In[ ]:


train['kills/distance']=train['kills']/train['totalDistance']
test['kills/distance']=test['kills']/test['totalDistance']


# In[ ]:


train['kills/walkDistance']=train['kills']/train['walkDistance']
test['kills/walkDistance']=test['kills']/test['walkDistance']


# In[ ]:


train['avgKills'] = train['squadKills']/train['teamPlayers']
test['avgKills'] = test['squadKills']/test['teamPlayers']


# In[ ]:


train['damageRatio'] = train['damageDealt']/train['enemyDamage']
test['damageRatio'] = test['damageDealt']/test['enemyDamage']


# In[ ]:


train['distTravelledPerGame'] = train['totalDistance']/train['matchDuration']
test['distTravelledPerGame'] = test['totalDistance']/test['matchDuration']


# In[ ]:


train['killPlacePerc'] = train['killPlace']/train['gamePlayers']
test['killPlacePerc'] = test['killPlace']/test['gamePlayers']


# In[ ]:


train["playerSkill"] = train["headshotKills"]+ train["roadKills"]+train["assists"]-(5*train['teamKills']) 
test["playerSkill"] = test["headshotKills"]+ test["roadKills"]+test["assists"]-(5*test['teamKills'])


# In[ ]:


train['gamePlacePerc'] = train['killPlace']/train['maxPlace']
test['gamePlacePerc'] = test['killPlace']/test['maxPlace']


# The newly created features contains missing values and Infinity values in it. Let's replace these with 0.

# In[ ]:


train.fillna(0,inplace=True)
train.replace(np.inf, 0, inplace=True)
test.fillna(0,inplace=True)
test.replace(np.inf, 0, inplace=True)


# In[ ]:


train.count()


# From the heat map, killPoints, rankPoints, winPoints, maxPlace are found to be not having any significance in determining winPlacePerc. So let's remove these features from the data set. 

# In[ ]:


train.drop(columns=['killPoints','rankPoints','winPoints','maxPlace'],inplace=True)
test.drop(columns=['killPoints','rankPoints','winPoints','maxPlace'],inplace=True)


# In[ ]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage, took from Kaggle.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                    
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


# In Pubg, if a player wins, his team mates are also winners. So instead on finding winPlacePerc for individual payers, let's find the winPlacePerc for each group in a match.  Let's write a function that will create new columns that are the match wise and group wise mean, max, min of all the current features and also rank them.

# In[ ]:


def feature(df):
    features = list(df.columns)
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchType")
    condition='False'
    
    if 'winPlacePerc' in df.columns:
        y = np.array(df.groupby(['matchId','groupId'])['winPlacePerc'].agg('mean'), dtype=np.float64)
        features.remove("winPlacePerc")
        condition='True'
        
    print("get group mean feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = agg.reset_index()[['matchId','groupId']]
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])
        
    print("get group max feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    
    print("get group min feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
    
    print("get match mean feature")
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    df_id=df_out[["matchId", "groupId"]].copy()
    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)
    
    del df, agg, agg_rank
    gc.collect()
    if condition=='True':
        return df_out,pd.DataFrame(y),df_id
    else:
        return df_out,df_id


# In[ ]:


x,y,id_train=feature(reduce_mem_usage(train))


# In[ ]:


x_test,id_test=feature(reduce_mem_usage(test))


# In[ ]:


del train,test
gc.collect()


# # **4. GRADIENT BOOSTING MODEL**

# Split the data into train and validation set.

# In[ ]:


x['matchId']=id_train['matchId']
x['groupId']=id_train['groupId']
# Train test split
x_train,x_val,y_train,y_val=train_test_split(reduce_mem_usage(x),y,test_size=.1)
x_test=reduce_mem_usage(x_test)
id_val=x_val[['matchId','groupId']]
x_val.drop(['matchId','groupId'],axis=1,inplace=True)
x_train.drop(['matchId','groupId'],axis=1,inplace=True)
x.drop(['matchId','groupId'],axis=1,inplace=True)
del y
gc.collect()


# In[ ]:


params = {
        "objective" : "regression", 
        "metric" : "mae", 
        "num_leaves" : 149, 
        "learning_rate" : 0.03, 
        "bagging_fraction" : 0.9,
        "bagging_seed" : 0, 
        "num_threads" : 4,
        "colsample_bytree" : 0.5,
        'min_data_in_leaf':1900, 
        'min_split_gain':0.00011,
        'lambda_l2':9
}


# In[ ]:


# create dataset for lightgbm
lgb_train = lgb.Dataset(x_train, y_train,
                       free_raw_data=False)
lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train,
                      free_raw_data=False)


# In[ ]:


model = lgb.train(params,
                lgb_train,
                num_boost_round=22000,
                valid_sets=lgb_eval,
                early_stopping_rounds=10,
                verbose_eval=1000)


# # 6.Post Processing

# Now that we have trained the model, let' have a look if we can make some tweaks in the predicted data so that the predicted value can be improved. First let's merge the predicted value with appropriate gamer Id in the train data.

# In[ ]:


y_pred_val = model.predict(x, num_iteration=model.best_iteration)
id_train['win_pred']=y_pred_val
id_train.set_index(['matchId','groupId'])
train = reduce_mem_usage(pd.read_csv("../input/train_V2.csv"))

df=pd.merge(train,id_train,on=['matchId','groupId'],how='right')
df


# In[ ]:


print('The mae score is {}'.format(mean_absolute_error(df['winPlacePerc'],df['win_pred'])))
df = df[["Id", "matchId", "groupId", "maxPlace", "numGroups",'winPlacePerc', 'win_pred']]


# Let's take only one row from each groupby matchId and groupId since the winPlacePerc is almost same for each player in a team. Now sort and rank each group in a match. Rank is directly proportional to winPlacePerc.

# In[ ]:


df_grouped = df.groupby(["matchId", "groupId"]).first().reset_index()
df_grouped["team_place"] = df_grouped.groupby(["matchId"])["win_pred"].rank()
df_grouped


# It has been found out that rank of team/team_place is proportional to winPlacePerc. So team_place can be used as the most important factor judging winplacePerc. Let's try to explain winPlacePerc as the ratio of team_place to numGroups. team_place will never be equal to zero. However winPlacePerc can also be zero. So let's subtract 1 from team_place as that will return zero in cases where team_place=1.

# In[ ]:


df_grouped["win_perc"] = (df_grouped["team_place"] - 1) / (df_grouped["numGroups"]-1)
df = df.merge(df_grouped[["win_perc","matchId", "groupId"]], on=["matchId", "groupId"], how="left")


# Let's post process the new win_perc similar. winPlacePerc shoul not exceed 1 and should not drop below 0. It should be between 1 and 0. Also maxPlace=0 is impossible in a game and maxPlace=0 means their is no team. So winPerc=0. Similarly maxPlace=0 means only one team. 

# In[ ]:


df.loc[df['maxPlace'] == 0, "win_perc"] = 0
df.loc[df['maxPlace'] == 1, "win_perc"] = 1
df.loc[(df['maxPlace'] > 1) & (df['numGroups'] == 1), "win_perc"] = 0
df.loc[df['win_perc'] < 0,"win_perc"] = 0
df.loc[df['win_perc'] > 1,"win_perc"] = 1
df['win_perc'].fillna(df['win_pred'],inplace=True)


# In[ ]:


df_grouped[df_grouped['maxPlace']>1][['winPlacePerc','win_perc','maxPlace','numGroups','team_place']]


# # This idea I got while referring similar kernels published publicly during the competion time and the credit goes for <a href='https://www.kaggle.com/anycode/simple-nn-baseline-3'>Kernel Here</a>. This helps to change the predicted win by few decimal points and improve the mae score. 

# In[ ]:


subset = df.loc[df['maxPlace'] > 1]
gap = 1 / (subset['maxPlace'].values-1)
new_perc = np.around(subset['win_perc'].values / gap) * gap
df.loc[df.maxPlace > 1, "win_perc"] = new_perc


# In[ ]:


print('The new mae score is {}'.format(mean_absolute_error(df['winPlacePerc'],df['win_perc'])))


# # Woahh!!! The Score has improved a lot. 

# In[ ]:


del x,train,df
gc.collect()


# # SUBMISSION

# In[ ]:


y_pred = model.predict(x_test, num_iteration=model.best_iteration)
id_test['win_pred']=y_pred
id_test.set_index(['matchId','groupId'])
del x_train,x_val,y_train,y_val,x_test
gc.collect()

test = reduce_mem_usage(pd.read_csv("../input/test_V2.csv"))
df=pd.merge(test,id_test,on=['matchId','groupId'],how='right')
del id_test,test
gc.collect()
df


# In[ ]:


df = df[["Id", "matchId", "groupId", "maxPlace", "numGroups",'win_pred']]


# Let's take only one row from each groupby matchId and groupId since the winPlacePerc is almost same for each player in a team. Now sort and rank each group in a match. Rank is directly proportional to predicted winPerc.

# In[ ]:


df_grouped = df.groupby(["matchId", "groupId"]).first().reset_index()
df_grouped["team_place"] = df_grouped.groupby(["matchId"])["win_pred"].rank()
df_grouped


# In[ ]:


df_grouped["win_perc"] = (df_grouped["team_place"] - 1) / (df_grouped["numGroups"]-1)
df = df.merge(df_grouped[["win_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")


# In[ ]:


df.loc[df.maxPlace == 0, "win_perc"] = 0
df.loc[df.maxPlace == 1, "win_perc"] = 1
df.loc[(df.maxPlace > 1) & (df.numGroups == 1), "win_perc"] = 0
df.loc[df['win_perc'] < 0,"win_perc"] = 0
df.loc[df['win_perc'] > 1,"win_perc"] = 1
df['win_perc'].fillna(df['win_pred'],inplace=True)


# In[ ]:


subset = df.loc[df['maxPlace'] > 1]
gap = 1 / (subset['maxPlace'].values-1)
new_perc = np.around(subset['win_perc'].values / gap) * gap
df.loc[df.maxPlace > 1, "win_perc"] = new_perc
df['winPlacePerc']=df['win_perc']


# In[ ]:


df=df[['Id','winPlacePerc']]
df.to_csv("submission_final.csv", index=False)


# # If you liked the kernel, DO upvote! 
