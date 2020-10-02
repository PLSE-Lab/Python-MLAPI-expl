#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import forest, RandomForestRegressor
import lightgbm as lgb
import gc
import feather
import warnings; 
warnings.filterwarnings('ignore')
INPUT_DIR = "../input/"


# In[ ]:


#Function to display all
def display_all(df):
    with pd.option_context("display.max_rows", 500, "display.max_columns", 500):
        display(df)

#Function to reduce memory usage
def reduce_mem_usage(df):
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
    return df


# In[ ]:


#Dataframe pre-processing
def dataproc1(df):
    df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')
    df['headshotrate'] = df['kills']/df['headshotKills']
    df['killStreakrate'] = df['killStreaks']/df['kills']
    df['DBNOs_over_kills'] = df['DBNOs']/df['kills']
    df['healthitems'] = df['heals'] + df['boosts']
    df['heals_over_boosts'] = df['heals'] / df['boosts']
    df['healthitems_norm'] = df['heals']/1.37 + df['boosts']/1.1
    df['healthitems_over_kills'] = df['healthitems'] / df['kills']
    df['healthitems_norm_over_kills'] = df['healthitems_norm'] / df['kills']
    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']
    df['walkDistance_x_heals'] = df['walkDistance'] * df['heals']
    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']
    df['walkDistance_x_kills'] = df['walkDistance'] * df['kills']
    df['walkDistance_over_healthitems'] = df['walkDistance'] / df['healthitems']
    df['walkDistance_x_healthitems'] = df['walkDistance'] * df['healthitems']
    df["skill"] = df["headshotKills"] + df["roadKills"]
    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']
    df['killPlace_over_numGroups'] = df['killPlace'] / df['numGroups']
    df['rideDistance'] = (df['rideDistance']/500)
    df['walkDistance'] = (df['walkDistance']/500)
    df['swimDistance'] = (df['swimDistance']/500)
    df["total_time_by_distance"] = (df["rideDistance"]/4.5+df["walkDistance"]+df["swimDistance"]*1.5)
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    df['distance_over_weapons'] = df['totalDistance'] / df['weaponsAcquired']
    df['distance_x_weapons'] = df['totalDistance'] * df['weaponsAcquired']
    df['total_time_by_distance_over_weapons'] = df['total_time_by_distance'] / df['weaponsAcquired']
    df['total_time_by_distance_x_weapons'] = df['total_time_by_distance'] * df['weaponsAcquired']
    df['killPlace_over_total_time_by_distance'] = df['killPlace'] / df['total_time_by_distance']
    df['killPlace_x_total_time_by_distance'] = df['killPlace'] * df['total_time_by_distance']
    df['killPlace_over_totalDistance'] = df['killPlace'] / df['totalDistance']
    df['killPlace_x_totalDistance'] = df['killPlace'] * df['totalDistance']    
    df['boosts_over_total_time_by_distance'] = df['boosts'] / df['total_time_by_distance']
    df['boosts_x_total_time_by_distance'] = df['boosts'] * df['total_time_by_distance']
    df['boosts_over_totalDistance'] = df['boosts'] / df['totalDistance']
    df['boosts_x_totalDistance'] = df['boosts'] * df['totalDistance']    
    df['teamwork'] = df['assists'] + df['revives'] - df['teamKills']
    df['total_items_acquired'] = (df["boosts"] + df["heals"] + df["weaponsAcquired"])
    df['total_items_acquired_norm'] = (df["boosts"]/1.1 + df["heals"]/1.37 + df["weaponsAcquired"]/3.66)
    df['total_items_acquired_over_total_time_by_distance'] = df['total_items_acquired'] / df['total_time_by_distance']
    df['total_items_acquired_x_total_time_by_distance'] = df['total_items_acquired'] * df['total_time_by_distance']
    df['total_items_acquired_norm_over_total_time_by_distance'] = df['total_items_acquired_norm'] / df['total_time_by_distance']
    df['total_items_acquired_norm_x_total_time_by_distance'] = df['total_items_acquired_norm'] * df['total_time_by_distance']
    df['heals_over_total_time_by_distance'] = df['heals'] / df['total_time_by_distance']
    df['heals_x_total_time_by_distance'] = df['heals'] * df['total_time_by_distance']    
    df['heals_over_totalDistance'] = df['heals'] / df['totalDistance']
    df['heals_x_totalDistance'] = df['heals'] * df['totalDistance']    
    df['kills_over_total_time_by_distance'] = df['kills'] / df['total_time_by_distance']
    df['kills_x_total_time_by_distance'] = df['kills'] * df['total_time_by_distance']
    df['kills_over_totalDistance'] = df['kills'] / df['totalDistance']
    df['kills_x_totalDistance'] = df['kills'] * df['totalDistance']    
    df['killsNorm'] = df['kills']*((100-df['playersJoined'])/100 + 1)
    df['damageDealtNorm'] = df['damageDealt']*((100-df['playersJoined'])/100 + 1)
    df['maxPlaceNorm'] = df['maxPlace']*((100-df['playersJoined'])/100 + 1)
    df['killPlace_over_maxPlaceNorm'] = df['killPlace'] / df['maxPlaceNorm']
    df['killPlace_over_playersJoined'] = df['killPlace'] / df['playersJoined'] 
    df['matchDurationNorm'] = df['matchDuration']*((100-df['playersJoined'])/100 + 1)
    df['killPlace_over_matchDuration'] = df['killPlace'] / df['matchDuration']
    df['killPlace_over_matchDurationnorm'] = df['killPlace'] / df['matchDurationNorm']    
    df['killPlacePerc'] = (df['playersJoined'] - df['killPlace']) / (df['playersJoined'] - 1)
    df['L1'] = df['roadKills'] + df['vehicleDestroys'] + df['teamKills']
    df['L2'] = df['revives'] + df['headshotKills'] + df['assists']
    df['L3'] = df['killStreaks'] + df['DBNOs'] + df['kills'] + df['boosts'] + df['heals']
    df['points'] = df['killPoints']+df['rankPoints'] + df['winPoints']
    df['L1_over_total_time_by_distance'] = df['L1'] / df['total_time_by_distance']
    df['L1_x_total_time_by_distance'] = df['L1'] * df['total_time_by_distance']
    df['L1_over_totalDistance'] = df['L1'] / df['totalDistance']
    df['L1_x_totalDistance'] = df['L1'] * df['totalDistance']    
    df['L2_over_total_time_by_distance'] = df['L2'] / df['total_time_by_distance']
    df['L2_x_total_time_by_distance'] = df['L2'] * df['total_time_by_distance']
    df['L2_over_totalDistance'] = df['L2'] / df['totalDistance']
    df['L2_x_totalDistance'] = df['L2'] * df['totalDistance']    
    df['L3_over_total_time_by_distance'] = df['L3'] / df['total_time_by_distance']
    df['L3_x_total_time_by_distance'] = df['L3'] * df['total_time_by_distance']
    df['L3_over_totalDistance'] = df['L3'] / df['totalDistance']
    df['L3_x_totalDistance'] = df['L3'] * df['totalDistance']
    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    df.fillna(0, inplace=True)  
    reduce_mem_usage(df)
    return df

#List of features
def dataproc2(df, is_train='TRUE'):
    print("Starting dataproc2")
    features = list(df.columns)
    if is_train=='TRUE':
        features.remove(target)
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchType")
    features.remove("numGroups")
    features.remove("playersJoined")
    features.remove("roadKills")
    features.remove("vehicleDestroys")
    imp_cols=['matchId', 'groupId']
    print("Step 1")
    tmp = df.groupby(['matchId','groupId'])[features].agg('mean')
    df_out = tmp.reset_index()[['matchId','groupId']]
    if is_train=='TRUE':
        imp_cols.extend([target])
        tmp3 = df.groupby(['matchId','groupId'])[target].agg('mean')
        df_out = df_out.merge(tmp3.reset_index(), how='left', on=['matchId','groupId'])
        del tmp3
        gc.collect()
    print("Step 2")
    features.remove("maxPlace")
    features.remove("matchDuration")
    tmp2 = tmp.groupby('matchId')[features].rank(pct=True)
    df_out = df_out.merge(tmp.add_suffix('_mean').reset_index(), how='left', on=['matchId', 'groupId'])
    imp_cols.extend(['killPlace_mean', 'matchDuration_mean', 'killPlace_over_maxPlace_mean', 'killPlace_over_numGroups_mean', 'boosts_x_total_time_by_distance_mean', 'killsNorm_mean', 'killPlace_over_playersJoined_mean', 'killPlacePerc_mean'])
    df_out = df_out[imp_cols]
    df_out = df_out.merge(tmp2.add_suffix('_mean_rank').reset_index(), how='left', on=['matchId', 'groupId'])
    imp_cols.extend(['DBNOs_mean_rank', 'kills_mean_rank', 'killStreaks_mean_rank', 'longestKill_mean_rank', 'walkDistance_mean_rank', 'killStreakrate_mean_rank', 'walkDistance_x_kills_mean_rank', 'total_time_by_distance_mean_rank', 'total_time_by_distance_x_weapons_mean_rank', 'killPlace_over_total_time_by_distance_mean_rank', 'killPlace_x_total_time_by_distance_mean_rank', 'boosts_x_total_time_by_distance_mean_rank', 'total_items_acquired_x_total_time_by_distance_mean_rank', 'total_items_acquired_norm_x_total_time_by_distance_mean_rank', 'kills_over_total_time_by_distance_mean_rank', 'kills_x_total_time_by_distance_mean_rank', 'kills_x_totalDistance_mean_rank', 'killsNorm_mean_rank', 'killPlacePerc_mean_rank'])
    df_out = df_out[imp_cols]
    del tmp
    del tmp2
    gc.collect()
    print("Step 3")
    tmp = df.groupby(['matchId','groupId'])[features].agg('median')
    tmp2 = tmp.groupby('matchId')[features].rank(pct=True)
    df_out = df_out.merge(tmp.add_suffix('_median').reset_index(), how='left', on=['matchId', 'groupId'])
    imp_cols.extend(['killPlace_over_numGroups_median', 'killsNorm_median', 'killPlace_over_playersJoined_median', 'killPlacePerc_median'])
    df_out = df_out[imp_cols]
    df_out = df_out.merge(tmp2.add_suffix('_median_rank').reset_index(), how='left', on=['matchId', 'groupId'])
    imp_cols.extend(['assists_median_rank', 'killPlace_median_rank', 'kills_median_rank', 'killStreaks_median_rank', 'longestKill_median_rank', 'revives_median_rank', 'walkDistance_median_rank', 'killStreakrate_median_rank', 'DBNOs_over_kills_median_rank', 'walkDistance_over_kills_median_rank', 'walkDistance_x_kills_median_rank', 'killPlace_over_numGroups_median_rank', 'total_time_by_distance_median_rank', 'total_time_by_distance_x_weapons_median_rank', 'killPlace_over_total_time_by_distance_median_rank', 'killPlace_x_total_time_by_distance_median_rank', 'killPlace_over_totalDistance_median_rank', 'total_items_acquired_x_total_time_by_distance_median_rank', 'total_items_acquired_norm_x_total_time_by_distance_median_rank', 'kills_over_total_time_by_distance_median_rank', 'kills_x_total_time_by_distance_median_rank', 'kills_over_totalDistance_median_rank', 'kills_x_totalDistance_median_rank', 'killsNorm_median_rank', 'killPlace_over_matchDuration_median_rank', 'killPlacePerc_median_rank', 'L3_over_total_time_by_distance_median_rank'])
    df_out = df_out[imp_cols]
    del tmp
    del tmp2
    gc.collect()
    print("Step 4")
    tmp = df.groupby(['matchId','groupId'])[features].agg('max')
    tmp2 = tmp.groupby('matchId')[features].rank(pct=True)
    df_out = df_out.merge(tmp.add_suffix('_max').reset_index(), how='left', on=['matchId', 'groupId'])
    imp_cols.extend(['damageDealt_max', 'killPlace_max', 'kills_max', 'walkDistance_max', 'walkDistance_over_kills_max', 'killPlace_over_maxPlace_max', 'killPlace_over_numGroups_max', 'boosts_x_total_time_by_distance_max', 'killsNorm_max', 'killPlace_over_maxPlaceNorm_max', 'killPlace_over_playersJoined_max', 'killPlacePerc_max'])
    df_out = df_out[imp_cols]
    df_out = df_out.merge(tmp2.add_suffix('_max_rank').reset_index(), how='left', on=['matchId', 'groupId'])
    imp_cols.extend(['boosts_max_rank', 'DBNOs_max_rank', 'killPlace_max_rank', 'kills_max_rank', 'killStreaks_max_rank', 'longestKill_max_rank', 'walkDistance_max_rank', 'weaponsAcquired_max_rank', 'killStreakrate_max_rank', 'walkDistance_over_kills_max_rank', 'killPlace_over_maxPlace_max_rank', 'killPlace_over_numGroups_max_rank', 'total_time_by_distance_max_rank', 'totalDistance_max_rank', 'distance_x_weapons_max_rank', 'total_time_by_distance_x_weapons_max_rank', 'killPlace_over_total_time_by_distance_max_rank', 'killPlace_x_total_time_by_distance_max_rank', 'boosts_x_total_time_by_distance_max_rank', 'boosts_x_totalDistance_max_rank', 'total_items_acquired_x_total_time_by_distance_max_rank', 'total_items_acquired_norm_x_total_time_by_distance_max_rank', 'kills_x_total_time_by_distance_max_rank', 'kills_x_totalDistance_max_rank', 'killsNorm_max_rank', 'killPlace_over_maxPlaceNorm_max_rank', 'killPlace_over_playersJoined_max_rank', 'killPlace_over_matchDuration_max_rank', 'killPlace_over_matchDurationnorm_max_rank', 'killPlacePerc_max_rank', 'points_max_rank', 'L3_x_totalDistance_max_rank'])
    df_out = df_out[imp_cols]
    del tmp
    del tmp2
    gc.collect()
    print("Step 5")
    tmp = df.groupby(['matchId','groupId'])[features].agg('min')
    tmp2 = tmp.groupby('matchId')[features].rank(pct=True)
    df_out = df_out.merge(tmp.add_suffix('_min').reset_index(), how='left', on=['matchId', 'groupId'])
    imp_cols.extend(['kills_min', 'longestKill_min', 'killStreakrate_min', 'walkDistance_over_kills_min', 'killPlace_over_maxPlace_min', 'killPlace_over_numGroups_min', 'killPlace_over_total_time_by_distance_min', 'kills_over_total_time_by_distance_min', 'killsNorm_min', 'killPlace_over_maxPlaceNorm_min', 'killPlace_over_playersJoined_min', 'killPlace_over_matchDuration_min', 'killPlace_over_matchDurationnorm_min', 'killPlacePerc_min'])
    df_out = df_out[imp_cols]
    df_out = df_out.merge(tmp2.add_suffix('_min_rank').reset_index(), how='left', on=['matchId', 'groupId'])
    imp_cols.extend(['assists_min_rank', 'DBNOs_min_rank', 'kills_min_rank', 'killStreaks_min_rank', 'longestKill_min_rank', 'rankPoints_min_rank', 'revives_min_rank', 'swimDistance_min_rank', 'walkDistance_min_rank', 'weaponsAcquired_min_rank', 'killStreakrate_min_rank', 'DBNOs_over_kills_min_rank', 'heals_over_boosts_min_rank', 'healthitems_over_kills_min_rank', 'healthitems_norm_over_kills_min_rank', 'walkDistance_over_kills_min_rank', 'walkDistance_x_kills_min_rank', 'killPlace_over_maxPlace_min_rank', 'total_time_by_distance_min_rank', 'total_time_by_distance_x_weapons_min_rank', 'killPlace_over_total_time_by_distance_min_rank', 'killPlace_x_total_time_by_distance_min_rank', 'killPlace_over_totalDistance_min_rank', 'teamwork_min_rank', 'total_items_acquired_x_total_time_by_distance_min_rank', 'kills_over_total_time_by_distance_min_rank', 'kills_x_total_time_by_distance_min_rank', 'kills_over_totalDistance_min_rank', 'kills_x_totalDistance_min_rank', 'killsNorm_min_rank', 'damageDealtNorm_min_rank', 'killPlace_over_maxPlaceNorm_min_rank', 'killPlace_over_matchDuration_min_rank', 'killPlacePerc_min_rank', 'L3_min_rank', 'points_min_rank', 'L3_over_total_time_by_distance_min_rank'])
    df_out = df_out[imp_cols]
    del tmp
    del tmp2
    gc.collect()
    print("Step 6")
    tmp = df.groupby(['matchId'])[features].agg('mean')
    df_out = df_out.merge(tmp.add_suffix('_match_mean').reset_index(), how='left', on=['matchId'])   
    imp_cols.extend(['assists_match_mean', 'boosts_match_mean', 'damageDealt_match_mean', 'headshotKills_match_mean', 'heals_match_mean', 'kills_match_mean', 'killStreaks_match_mean', 'longestKill_match_mean', 'swimDistance_match_mean', 'weaponsAcquired_match_mean', 'headshotrate_match_mean', 'killStreakrate_match_mean', 'DBNOs_over_kills_match_mean', 'heals_over_boosts_match_mean', 'healthitems_over_kills_match_mean', 'healthitems_norm_over_kills_match_mean', 'walkDistance_over_heals_match_mean', 'walkDistance_over_kills_match_mean', 'walkDistance_x_kills_match_mean', 'walkDistance_over_healthitems_match_mean', 'skill_match_mean', 'distance_over_weapons_match_mean', 'total_time_by_distance_over_weapons_match_mean', 'killPlace_over_total_time_by_distance_match_mean', 'killPlace_x_total_time_by_distance_match_mean', 'killPlace_over_totalDistance_match_mean', 'killPlace_x_totalDistance_match_mean', 'boosts_over_total_time_by_distance_match_mean', 'boosts_over_totalDistance_match_mean', 'teamwork_match_mean', 'total_items_acquired_norm_match_mean', 'total_items_acquired_over_total_time_by_distance_match_mean', 'total_items_acquired_norm_over_total_time_by_distance_match_mean', 'heals_over_total_time_by_distance_match_mean', 'heals_over_totalDistance_match_mean', 'kills_over_total_time_by_distance_match_mean', 'kills_x_total_time_by_distance_match_mean', 'kills_over_totalDistance_match_mean', 'kills_x_totalDistance_match_mean', 'killsNorm_match_mean', 'damageDealtNorm_match_mean', 'L2_match_mean', 'points_match_mean', 'L1_over_totalDistance_match_mean', 'L2_over_total_time_by_distance_match_mean', 'L2_x_total_time_by_distance_match_mean', 'L2_over_totalDistance_match_mean', 'L2_x_totalDistance_match_mean', 'L3_over_total_time_by_distance_match_mean', 'L3_over_totalDistance_match_mean'])
    df_out = df_out[imp_cols]
    del tmp
    gc.collect()
    tmp = df.groupby(['matchId'])[features].agg('median')
    df_out = df_out.merge(tmp.add_suffix('_match_median').reset_index(), how='left', on=['matchId'])   
    imp_cols.extend(['damageDealt_match_median', 'killStreaks_match_median', 'walkDistance_match_median', 'killStreakrate_match_median', 'distance_over_weapons_match_median', 'total_time_by_distance_over_weapons_match_median', 'killPlace_over_total_time_by_distance_match_median', 'killPlace_over_totalDistance_match_median', 'killPlace_x_totalDistance_match_median', 'total_items_acquired_over_total_time_by_distance_match_median', 'total_items_acquired_norm_over_total_time_by_distance_match_median', 'kills_over_total_time_by_distance_match_median', 'kills_x_totalDistance_match_median', 'damageDealtNorm_match_median', 'L3_match_median', 'L3_over_total_time_by_distance_match_median', 'L3_x_total_time_by_distance_match_median', 'L3_over_totalDistance_match_median', 'L3_x_totalDistance_match_median'])
    df_out = df_out[imp_cols]
    del tmp
    gc.collect()
    tmp = df.groupby(['matchId'])[features].agg('max')
    df_out = df_out.merge(tmp.add_suffix('_match_max').reset_index(), how='left', on=['matchId'])   
    imp_cols.extend(['damageDealt_match_max', 'longestKill_match_max', 'rankPoints_match_max', 'rideDistance_match_max', 'swimDistance_match_max', 'walkDistance_match_max', 'heals_over_boosts_match_max', 'healthitems_norm_over_kills_match_max', 'walkDistance_over_heals_match_max', 'walkDistance_x_heals_match_max', 'walkDistance_over_kills_match_max', 'walkDistance_x_kills_match_max', 'walkDistance_over_healthitems_match_max', 'walkDistance_x_healthitems_match_max', 'total_time_by_distance_match_max', 'totalDistance_match_max', 'distance_over_weapons_match_max', 'distance_x_weapons_match_max', 'total_time_by_distance_over_weapons_match_max', 'total_time_by_distance_x_weapons_match_max', 'killPlace_over_total_time_by_distance_match_max', 'killPlace_x_total_time_by_distance_match_max', 'killPlace_over_totalDistance_match_max', 'killPlace_x_totalDistance_match_max', 'boosts_over_total_time_by_distance_match_max', 'boosts_x_total_time_by_distance_match_max', 'boosts_over_totalDistance_match_max', 'boosts_x_totalDistance_match_max', 'total_items_acquired_norm_match_max', 'total_items_acquired_over_total_time_by_distance_match_max', 'total_items_acquired_x_total_time_by_distance_match_max', 'total_items_acquired_norm_over_total_time_by_distance_match_max', 'heals_over_total_time_by_distance_match_max', 'heals_x_total_time_by_distance_match_max', 'heals_over_totalDistance_match_max', 'heals_x_totalDistance_match_max', 'kills_over_total_time_by_distance_match_max', 'kills_x_total_time_by_distance_match_max', 'kills_over_totalDistance_match_max', 'kills_x_totalDistance_match_max', 'killsNorm_match_max', 'damageDealtNorm_match_max', 'L3_match_max', 'points_match_max', 'L1_x_total_time_by_distance_match_max', 'L1_over_totalDistance_match_max', 'L2_over_total_time_by_distance_match_max', 'L2_x_total_time_by_distance_match_max', 'L2_over_totalDistance_match_max', 'L2_x_totalDistance_match_max', 'L3_over_total_time_by_distance_match_max', 'L3_x_total_time_by_distance_match_max', 'L3_over_totalDistance_match_max', 'L3_x_totalDistance_match_max'])
    df_out = df_out[imp_cols]
    del tmp
    gc.collect()
    print("Step 7")
    tmp = df.groupby(['matchId','groupId'])[features].agg('sum')
    tmp2 = tmp.groupby('matchId')[features].rank(pct=True)
    df_out = df_out.merge(tmp.add_suffix('_sum').reset_index(), how='left', on=['matchId', 'groupId'])
    imp_cols.extend(['longestKill_sum', 'walkDistance_over_kills_sum', 'killPlace_over_maxPlace_sum', 'killPlace_over_numGroups_sum', 'damageDealtNorm_sum', 'maxPlaceNorm_sum', 'killPlace_over_playersJoined_sum', 'killPlacePerc_sum'])
    df_out = df_out[imp_cols]
    df_out = df_out.merge(tmp2.add_suffix('_sum_rank').reset_index(), how='left', on=['matchId', 'groupId'])
    imp_cols.extend(['kills_sum_rank', 'killStreaks_sum_rank', 'longestKill_sum_rank', 'rankPoints_sum_rank', 'walkDistance_sum_rank', 'winPoints_sum_rank', 'killStreakrate_sum_rank', 'killPlace_over_maxPlace_sum_rank', 'total_time_by_distance_sum_rank', 'totalDistance_sum_rank', 'distance_x_weapons_sum_rank', 'total_time_by_distance_x_weapons_sum_rank', 'killPlace_over_total_time_by_distance_sum_rank', 'killPlace_x_total_time_by_distance_sum_rank', 'killPlace_over_totalDistance_sum_rank', 'boosts_x_total_time_by_distance_sum_rank', 'boosts_x_totalDistance_sum_rank', 'total_items_acquired_x_total_time_by_distance_sum_rank', 'total_items_acquired_norm_x_total_time_by_distance_sum_rank', 'kills_x_total_time_by_distance_sum_rank', 'killsNorm_sum_rank', 'killPlacePerc_sum_rank', 'L3_sum_rank', 'points_sum_rank'])
    df_out = df_out[imp_cols]
    del tmp
    del tmp2
    gc.collect()
    print("Step 8")
    df_out = df_out.assign(agg_group_size=df.groupby('groupId').groupId.transform('count'))
    df_out = df_out.assign(agg_match_size=df.groupby('matchId').Id.transform('nunique'))
    print("Step 9")
    del df_out["matchId"]
    del df_out["groupId"]
    reduce_mem_usage(df_out)
    gc.collect()
    return df_out


# In[ ]:


get_ipython().run_cell_magic('time', '', "df = pd.read_csv(INPUT_DIR + 'train_V2.csv')\ndf = df[df['maxPlace'] > 1]\nreduce_mem_usage(df)\ntarget = 'winPlacePerc'\ndf = dataproc1(df)\n#df.reset_index().to_feather('tmp/20181217_df_308_features')\ndf_out = dataproc2(df)\ndel df\ngc.collect()\n#df_out.to_feather('tmp/20181217_df_out_308_features')\ny_train = df_out[target]\nx_train = df_out.drop(target, axis=1)\ndel df_out\ngc.collect()")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'def run_lgb(train_X, train_y):\n    params = {"objective" : "regression", "metric" : "mae", \'n_estimators\':50000,\n              "num_leaves" : 31, "learning_rate" : 0.05, "bagging_fraction" : 0.7,\n               "bagging_seed" : 0, "num_threads" : -1,"colsample_bytree" : 0.7\n             }\n    \n    lgtrain = lgb.Dataset(train_X, label=train_y)\n    model = lgb.train(params, lgtrain, verbose_eval=1)\n    \n    #pred_test_y = model.predict(x_test, num_iteration=model.best_iteration)\n    return model\n\n# Training the model #\nm = run_lgb(x_train, y_train)')


# In[ ]:


df = pd.read_csv(INPUT_DIR + 'test_V2.csv')
reduce_mem_usage(df)
df = dataproc1(df)
df_out = dataproc2(df, is_train='FALSE')
y_pred = m.predict(df_out).clip(0, 1)


# In[ ]:


print("Preparing output file")
long_file = df[['Id', 'matchId', 'groupId']]
short_file = df.groupby(['matchId','groupId'])['DBNOs'].agg('mean').reset_index()
short_file['winPlacePerc'] = y_pred
short_file = short_file[['matchId', 'groupId', 'winPlacePerc']]
merge_file = long_file.merge(short_file, on='groupId')
output = merge_file.drop(['matchId_x', 'groupId', 'matchId_y'], axis=1)
output = output.set_index('Id')


# In[ ]:


df_test = df
df_sub = output


# In[ ]:


df_sub = df_sub.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")
df_sub_group = df_sub.groupby(["matchId", "groupId"]).first().reset_index()
df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()
df_sub_group = df_sub_group.merge(
    df_sub_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(), 
    on="matchId", how="left")
df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)

df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
df_sub["winPlacePerc"] = df_sub["adjusted_perc"]

# Deal with edge cases
df_sub.loc[df_sub.maxPlace == 0, "winPlacePerc"] = 0
df_sub.loc[df_sub.maxPlace == 1, "winPlacePerc"] = 1

# Align with maxPlace
# Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4
subset = df_sub.loc[df_sub.maxPlace > 1]
gap = 1.0 / (subset.maxPlace.values - 1)
new_perc = np.around(subset.winPlacePerc.values / gap) * gap
df_sub.loc[df_sub.maxPlace > 1, "winPlacePerc"] = new_perc

# Edge case
df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 1), "winPlacePerc"] = 0
assert df_sub["winPlacePerc"].isnull().sum() == 0


# In[ ]:


df_sub[["Id", "winPlacePerc"]].to_csv("submission.csv", index=False)

