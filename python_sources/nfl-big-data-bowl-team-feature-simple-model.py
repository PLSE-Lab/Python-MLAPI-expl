#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


train = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv")


# In[ ]:


def fix_play_direction(df):
    df.loc[df['PlayDirection'] == 'left', 'X'] = 120 - df.loc[df['PlayDirection'] == 'left', 'X']
    df.loc[df['PlayDirection'] == 'left', 'Y'] = (160 / 3) - df.loc[df['PlayDirection'] == 'left', 'Y']
    df.loc[df['PlayDirection'] == 'left', 'Orientation'] = np.mod(180 + df.loc[df['PlayDirection'] == 'left', 'Orientation'], 360)
    df.loc[df['PlayDirection'] == 'left', 'Dir'] = np.mod(180 + df.loc[df['PlayDirection'] == 'left', 'Dir'], 360)
    df['FieldPosition'].fillna('', inplace=True)
    df.loc[df['PossessionTeam'] != df['FieldPosition'], 'YardLine'] = 100 - df.loc[df['PossessionTeam'] != df['FieldPosition'], 'YardLine']
    return df

train = fix_play_direction(train)


# In[ ]:


#we can go much more granular with player level features but going to stick with team features for now for simplicity
player_features = ["X", "Y", "S", "A", "Dis", "Orientation", "Dir", "Distance"]
team_features = ['YardLine', 'Quarter', 'PossessionTeam', 'Down', 
                 'Distance', 'FieldPosition', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 
                 'OffenseFormation', 'OffensePersonnel', 'DefendersInTheBox', 'DefensePersonnel',
                 'PlayDirection','Week', 'Stadium', 'StadiumType', 'Turf',
                 'GameWeather', 'Temperature', 'Humidity', 'WindSpeed', 'WindDirection'
                ]


# In[ ]:


from tqdm import tqdm


# In[ ]:


closest_defender = []
closest_offender = []
distant_counts = []
play_ids = train["PlayId"].unique()
for play in tqdm(play_ids):
    play_df = train[train["PlayId"] == play]
    rb_xy = play_df[play_df["NflId"] == play_df["NflIdRusher"]][["X", "Y", "Team"]]
    play_df["player_dist"] = (play_df[['X', 'Y']] - np.array(rb_xy[["X", "Y"]].values)).pow(2).sum(1).pow(0.5)
    defender_df = play_df[play_df["Team"] != rb_xy["Team"].values[0]]["player_dist"]
    distant_counts.append([(defender_df < 2).sum(),
                            (defender_df < 3).sum(),
                           (defender_df < 5).sum(),
                           (defender_df < 10).sum()])
    closest_defender.append(defender_df)
    closest_offender.append(play_df[play_df["Team"] == rb_xy["Team"].values[0]]["player_dist"])


# In[ ]:


closest_offender[0].sort_values()


# In[ ]:


play_df = train[team_features + ["PlayId", "Yards"]].drop_duplicates(subset="PlayId")


# In[ ]:


closest_offender[0].sort_values().values


# In[ ]:


player_dists = np.zeros((len(play_df), 22))


# In[ ]:


for i in tqdm(range(len(closest_offender))):
    player_dists[i, :len(closest_offender[i])] = closest_offender[i].sort_values().values
    player_dists[i, len(closest_offender[i]):] = closest_defender[i].sort_values().values


# In[ ]:


play_df


# In[ ]:


player_dists_df = pd.DataFrame(player_dists, columns = [str(i) + "_offender" for i in range(11)] + [str(i) + "_defender" for i in range(11)])


# In[ ]:


play_df = pd.concat([play_df.reset_index(drop = True), player_dists_df.reset_index(drop = True)], axis = 1)


# In[ ]:


distant_counts[0]


# In[ ]:


distant_counts = np.array(distant_counts)


# In[ ]:


play_df["under2"] = distant_counts[:, 0]
play_df["under3"] = distant_counts[:, 1]
play_df["under5"] = distant_counts[:, 2]
play_df["under10"] = distant_counts[:, 3]


# In[ ]:


# rb_count = []
# te_count = []
# wr_count = []
# ol_count = []
# dl = []

# for row in play_df["OffensePersonnel"].str.split(","):
#     print(row)


# In[ ]:


from tqdm import tqdm


# In[ ]:


from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(play_df, shuffle = True, test_size = .2)


# In[ ]:


categoricals = "PossessionTeam, FieldPosition, OffenseFormation, OffensePersonnel, DefensePersonnel, PlayDirection, Stadium, StadiumType, Turf, GameWeather, WindSpeed, WindDirection".split(", ") 


# In[ ]:


train_df[categoricals] = train_df[categoricals].astype("category")
val_df[categoricals] = val_df[categoricals].astype("category")


# In[ ]:


team_features = team_features + [str(i) + "_offender" for i in range(11)] + [str(i) + "_defender" for i in range(11)] + ["under2", "under3", "under5", "under10"]


# In[ ]:


import lightgbm as lgb
lgb_train = lgb.Dataset(train_df[team_features], train_df["Yards"])
lgb_eval = lgb.Dataset(val_df[team_features], val_df["Yards"])
params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'mae'},
        'eta': .28,
        'learning_rate': 0.01,
        'feature_fraction': 0.2,
        'bagging_fraction': 0.2,
        'bagging_freq': 5,
        'verbose': 0
        }
gbm = lgb.train(params,
            lgb_train,
            num_boost_round=2000,
            valid_sets=lgb_eval,
           early_stopping_rounds=20,
           verbose_eval = 1)

y_pred = gbm.predict(val_df[team_features], num_iteration=gbm.best_iteration)


# In[ ]:


list(sorted(zip(gbm.feature_importance(), gbm.feature_name())))


# In[ ]:


from kaggle.competitions import nflrush


# In[ ]:


def make_my_predictions(test_df, sample_prediction_df):
    test_df = test_df[team_features + ["PlayId"]].drop_duplicates(subset="PlayId")
    test_df[categoricals] = test_df[categoricals].astype("category")
    pred = gbm.predict(test_df[team_features], num_iteration=gbm.best_iteration)
    sample_prediction_df.iloc[:, 0:int(round(pred[0]))+ 100] = 0
    sample_prediction_df.iloc[:, int(round(pred[0])+ 100):-1] = 1
    sample_prediction_df.iloc[:, -1] = 1
    sample_prediction_df.iloc[:, int(round(pred[0]) + 100)] = .95
    sample_prediction_df= sample_prediction_df.T
    sample_prediction_df = sample_prediction_df.interpolate(axis = 0, method = 'linear').T
    return sample_prediction_df
# test_pred = make_my_predictions(test_df, sample_prediction_df)


# In[ ]:


# test_pred


# In[ ]:


env = nflrush.make_env()
for (test_df, sample_prediction_df) in env.iter_test():
    predictions_df = make_my_predictions(test_df, sample_prediction_df)
    env.predict(predictions_df)

env.write_submission_file()


# In[ ]:





# In[ ]:




