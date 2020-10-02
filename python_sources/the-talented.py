#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import integrate
import itertools 
import pylab as plt
import datetime
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plotly import tools
import string
import plotly.offline as py
# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import math
import re
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch
from string import punctuation
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
pd.options.display.max_rows = 4000



import os
import numpy as np
import pandas as pd
import sklearn.metrics as mtr
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import time
import lightgbm as lgb

from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda,BatchNormalization,LeakyReLU,PReLU,ELU,ThresholdedReLU,Concatenate
from keras.models import Model
import keras.backend as K
from  keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
import re
from keras.losses import binary_crossentropy
from  keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
import codecs
from keras.utils import to_categorical
from sklearn.metrics import f1_score

import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')



pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 150)


# In[ ]:


def clean_WindDirection(txt):
    if isinstance(txt, int):
        return ('NA')
    elif pd.isna(txt):
        return ('NA')
    else:
        if pd.isna(txt):
            return np.nan
        txt = txt.lower()
        txt = ''.join([c for c in txt if c not in punctuation])
        txt = txt.replace('from', '')
        txt = txt.replace(' ', '')
        txt = txt.replace('north', 'n')
        txt = txt.replace('south', 's')
        txt = txt.replace('west', 'w')
        txt = txt.replace('east', 'e')
        return txt
def transform_WindDirection(txt):
    if pd.isna(txt):
        return ('-1')
    
    if txt=='n':
        return 0
    elif txt=='nne' or txt=='nen':
        return 1/8
    elif txt=='ne':
        return 2/8
    elif txt=='ene' or txt=='nee':
        return 3/8
    elif txt=='e':
        return 4/8
    elif txt=='ese' or txt=='see':
        return 5/8
    elif txt=='se':
        return 6/8
    elif txt=='ses' or txt=='sse':
        return 7/8
    elif txt=='s':
        return 8/8
    elif txt=='ssw' or txt=='sws':
        return 9/8
    elif txt=='sw':
        return 10/8
    elif txt=='sww' or txt=='wsw':
        return 11/8
    elif txt=='w':
        return 12/8
    elif txt=='wnw' or txt=='nww':
        return 13/8
    elif txt=='nw':
        return 14/8
    elif txt=='nwn' or txt=='nnw':
        return 15/8 
    else: return -1
    return np.nan
def clean_StadiumType(txt):
    if pd.isna(txt):
        return ('NA')
    if isinstance(txt, int):
        return ('NA')
    else:
        txt = txt.lower()
        txt = ''.join([c for c in txt if c not in punctuation])
        txt = re.sub(' +', ' ', txt)
        txt = txt.strip()
        txt = txt.replace('outside', 'outdoor')
        txt = txt.replace('outdor', 'outdoor')
        txt = txt.replace('outddors', 'outdoor')
        txt = txt.replace('outdoors', 'outdoor')
        txt = txt.replace('oudoor', 'outdoor')
        txt = txt.replace('indoors', 'indoor')
        txt = txt.replace('ourdoor', 'outdoor')
        txt = txt.replace('retractable', 'rtr.')
        return txt
def set_Turf (df):
    Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 
            'UBU Sports Speed S5-M':'Artificial', 'Artificial':'Artificial', 
            'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 'UBU Speed Series-S5-M':'Artificial', 
            'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural',
            'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 
            'Field turf':'Artificial', 'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 
            'natural grass':'Natural'} 
    df['Turf_type']=df['Turf'].map(Turf)
    return df

def GameWeather(df):
    if isinstance( df['GameWeather'].iloc[0], np.integer):
        df['GameWeather']='NA'
        return (df)      
    else:
        df['GameWeather'] = df['GameWeather'].str.lower()
        indoor = "indoor"
        df['GameWeather'] = df['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
        df['GameWeather'] = df['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
        df['GameWeather'] = df['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
        df['GameWeather'] = df['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
    return df


# In[ ]:


def clean_TeamAbbreviations(train_df):
    # Fix Team Names
    train_df.loc[train_df.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
    train_df.loc[train_df.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

    train_df.loc[train_df.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
    train_df.loc[train_df.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

    train_df.loc[train_df.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
    train_df.loc[train_df.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

    train_df.loc[train_df.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
    train_df.loc[train_df.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"
    
    return train_df


# In[ ]:


def tag_rusher(train_df):
    train_df['IsBallCarrier'] = train_df.NflId == train_df.NflIdRusher
    return train_df


# In[ ]:


def get_direction(train_df):
    train_df['ToLeft'] = train_df.PlayDirection == "left"
    return train_df


# In[ ]:


def feet_inch_to_cm(height_fi):
    foot = 30.48
    inch = 2.54
    return int(height_fi.split('-')[0]) * foot + int(height_fi.split('-')[1]) * inch

def imperial_to_metric(train_df):
    train_df['PlayerHeight_cm']= train_df['PlayerHeight'].apply(feet_inch_to_cm)
    train_df['PlayerWeight_kg']= train_df['PlayerWeight']*0.453592
    return train_df


# In[ ]:


def get_player_age(train_df):
    train_df['Age']=2019 - pd.DatetimeIndex(train_df['PlayerBirthDate']).year
    return train_df


# In[ ]:


def possession_team(possessionteam, hometeam, visitorteam):
    if possessionteam == hometeam:
        return 'home'
    else:
        return 'visitor'
def leading_team_home_visitor(homescore, visitorscore):
    if homescore == visitorscore:
        return 'draw'
    else:
        if homescore > visitorscore:
            return 'home'
        else:
            return 'visitor'
        
def leading_team_def_off(team, posessionteam, hometeam, visitorteam, homescore, visitorscore):
    if homescore == visitorscore:
        return 'draw'
    else:
        if posessionteam == hometeam:
            if homescore > visitorscore:
                return 'offense'
            else:
                return 'defense'
        else:
            if visitorscore > homescore:
                return 'offense'
            else:
                return 'defense'

def get_leading_team_features(train_df):
    # Leading team - home/away
    train_df['possession_team_home_visitor'] = train_df.apply(lambda row: possession_team(row.PossessionTeam,row.HomeTeamAbbr,row.VisitorTeamAbbr), axis = 1)
    train_df['leading_by'] = abs(train_df['HomeScoreBeforePlay'] - train_df['VisitorScoreBeforePlay'])
    train_df['leading_team_defense_offense'] = train_df.apply(lambda row: leading_team_def_off(row.Team,
                                                                                                row.PossessionTeam,
                                                                                                row.HomeTeamAbbr,
                                                                                                row.VisitorTeamAbbr,
                                                                                                row.HomeScoreBeforePlay,
                                                                                                row.VisitorScoreBeforePlay), axis = 1)
    train_df['leading_team_home_visitor'] = train_df.apply(lambda row: leading_team_home_visitor(row.HomeScoreBeforePlay,row.VisitorScoreBeforePlay), axis = 1)
    train_df['Offense_score']=np.where(train_df['leading_team_defense_offense']=='offense', train_df['leading_by'], -train_df['leading_by'])
    train_df['Deffense_score']=np.where(train_df['leading_team_defense_offense']=='defense', train_df['leading_by'], -train_df['leading_by'])
    train_df['Offense_Team'] = train_df['PossessionTeam']
    train_df['Defense_Team'] = train_df.apply(lambda row: row['HomeTeamAbbr'] if row['PossessionTeam'] != row['HomeTeamAbbr'] else row['VisitorTeamAbbr'],axis=1)
    return train_df


# In[ ]:


def get_adjusted_yardline(train_df):
    train_df['Home_team_play']=np.where(train_df['PossessionTeam']==train_df['HomeTeamAbbr'], 1, 0)
    train_df['Team_side']=np.where(train_df['PossessionTeam']==train_df['FieldPosition'], 1, 0)
    train_df['YardLine_adj']=np.where(train_df['Team_side']==1,train_df['YardLine'],100-train_df['YardLine'])
    
    return train_df


# In[ ]:


def get_GameClock_features(train_df):
    train_df['snap_to_handoff']=(pd.to_datetime(train_df['TimeHandoff'])-pd.to_datetime(train_df['TimeSnap'])).dt.total_seconds()
    # Convert Q,Clocks to bins
    sec=pd.to_timedelta(train_df['GameClock'])
    sec=sec.dt.total_seconds()/60
    train_df['GameClock_sec']=sec
    train_df['clock_bin']=np.where(train_df['Quarter']==1,1,np.where(train_df['Quarter']==2,2,np.where(train_df['Quarter']==3,3,
    np.where(train_df['Quarter']==4,np.where(train_df['GameClock_sec']<=564,4,5),np.where(train_df['GameClock_sec']<=333,6,7)))))
    return train_df


# In[ ]:


# Break Personal Defense/Ofense
def get_personnel_types(dataset):
    setups = list(dataset['OffensePersonnel'].unique())
    setups = ','.join(setups)
    setups = re.sub(r'\d+', '', setups).replace(' ','')
    mylist = list(set(setups.split(',')))
    offense_column_names = ['Offense_' + s for s in mylist]

    setups = list(dataset['DefensePersonnel'].unique())
    setups = ','.join(setups)
    setups = re.sub(r'\d+', '', setups).replace(' ','')
    mylist = list(set(setups.split(',')))
    defense_column_names = ['Defense_' + s for s in mylist]
    return offense_column_names + defense_column_names

def extract_personnel_count(whole_personnel, specific_personnel):
    specific_personnel = specific_personnel.split('_')[1]
    if specific_personnel in whole_personnel:
        whole_personnel = whole_personnel.split(',')
        types = [re.sub(r'\d+', '', x).replace(' ','') for x in whole_personnel]
        counts = [[int(s) for s in x.split() if s.isdigit()][0] for x in whole_personnel]
        dictionary = dict(zip(types, counts))
        return dictionary[specific_personnel]
    else:
        return 0

def get_split_personnel_types(columns_to_add, train_df):
    for col in columns_to_add:
        if col.split('_')[0] == 'Defense':
            train_df[col] = train_df['DefensePersonnel'].apply(lambda x: extract_personnel_count(x, col))
        else:
            train_df[col] = train_df['OffensePersonnel'].apply(lambda x: extract_personnel_count(x, col))
    return train_df


# In[ ]:


def add_bins_and_generate_stats(train_df):
    bins=[-99,-0.01,0.01,1.72,4.21,6.7,99]
    train_df['Yards_Bin'] = pd.cut(train_df['Yards'], bins)

    stats = {}
    for my_type in ['DisplayName', 'Offense_Team', 'Defense_Team']:
        per_my_type = train_df[[my_type,'Yards','Yards_Bin']].groupby([my_type,'Yards_Bin']).count().fillna(0).reset_index()
        per_my_type = per_my_type.merge(per_my_type.groupby(my_type)['Yards'].sum().reset_index(), on=my_type, how='left')
        per_my_type['Yards_Bin_Rate'] = per_my_type['Yards_x'] / per_my_type['Yards_y']
        per_my_type.rename(columns={'Yards_x':'Bin_Count', 'Yards_y': 'Total_Plays'}, inplace=True)
        per_my_type = pd.pivot_table(per_my_type, values=['Bin_Count', 'Yards_Bin_Rate'], index=[my_type], columns=['Yards_Bin'])
        per_my_type.columns = per_my_type.columns.map('{0[0]}_{0[1]}'.format)
        per_my_type = per_my_type.reset_index()
        avg_yards = train_df.groupby(my_type)['Yards'].mean().reset_index()
        std_yards = train_df.groupby(my_type)['Yards'].std().reset_index().fillna(0)
        counts = train_df.groupby(my_type)['Yards'].count().reset_index()
        per_type = avg_yards.merge(counts, on=my_type, how='inner').merge(std_yards, on=my_type, how='inner')
        per_type.rename(columns={'Yards_x':'Avg_Yards', 'Yards_y':'Count', 'Yards':'Std_Yards'}, inplace= True)
        per_my_type= per_my_type.merge(per_type, on=my_type, how='left')
        if my_type == 'DisplayName':
            per_player_enough_plays = per_type[per_type['Count'] >= 10]
            per_player_not_enough_plays = per_type[per_type['Count'] < 10]
            per_player_not_enough_plays['rusher_performance_bin'] = 'few_plays'
            per_player_enough_plays['rusher_performance_bin'] = pd.qcut(per_player_enough_plays['Avg_Yards'], q=[0, 0.2,0.5,0.9,1], labels= ['bottom_20','20-50','50-90','top_10'])
            per_player = pd.concat([per_player_enough_plays, per_player_not_enough_plays])
            per_my_type= per_my_type.merge(per_player[[my_type,'rusher_performance_bin']], on=my_type, how='left')

        per_my_type.columns = [per_my_type.columns[0]] + [my_type + '_' + str(col) for col in per_my_type.columns[1:]]
        stats[my_type] = per_my_type
        train_df = train_df.merge(per_my_type, on=my_type, how='left')
    return stats, train_df


# In[ ]:


def convert_to_cat(df):
    for col in ['DefensePersonnel','Home_team_play','Team_side','Down','Defense_Team','OffenseFormation','OffensePersonnel',
                'leading_team_defense_offense','PlayDirection','clock_bin','Defense_DB','Defense_LB','Defense_DL','Offense_WR']:#,'DisplayName_rusher_performance_bin']:
        df[col] = df[col].astype('category')
    return df


# In[ ]:


def get_models_columns():
    player_model_columns = [#'DisplayName_Bin_Count_(-99.0, -0.01]','DisplayName_Bin_Count_(-0.01, 0.01]','DisplayName_Bin_Count_(0.01, 1.72]','DisplayName_Bin_Count_(1.72, 4.21]','DisplayName_Bin_Count_(4.21, 6.7]',
                            'DisplayName_Bin_Count_(6.7, 99.0]',
                            'DisplayName_Yards_Bin_Rate_(-99.0, -0.01]',
                            #'DisplayName_Yards_Bin_Rate_(-0.01, 0.01]',
                            #'DisplayName_Yards_Bin_Rate_(0.01, 1.72]',
                            'DisplayName_Yards_Bin_Rate_(1.72, 4.21]',
                            #'DisplayName_Yards_Bin_Rate_(4.21, 6.7]',
                            'DisplayName_Yards_Bin_Rate_(6.7, 99.0]',
                            'DisplayName_Avg_Yards',
                            #'DisplayName_Count',
                            'DisplayName_Std_Yards',
                            'DisplayName_rusher_performance_bin',#'JerseyNumber',
                            #'Position',
                            'PlayerCollegeName',
                            #'PlayerHeight_cm','PlayerWeight_kg','Age','PlayerWeight','PlayerHeight'
                            ]
    offense_team_model_columns =['PossessionTeam', 'OffenseFormation','OffensePersonnel',
                                 #'Offense_Team_Bin_Count_(-99.0, -0.01]','Offense_Team_Bin_Count_(-0.01, 0.01]','Offense_Team_Bin_Count_(0.01, 1.72]','Offense_Team_Bin_Count_(1.72, 4.21]',
                                 #'Offense_Team_Bin_Count_(4.21, 6.7]','Offense_Team_Bin_Count_(6.7, 99.0]','Offense_Team_Yards_Bin_Rate_(-99.0, -0.01]','Offense_Team_Yards_Bin_Rate_(-0.01, 0.01]','Offense_Team_Yards_Bin_Rate_(0.01, 1.72]','Offense_Team_Yards_Bin_Rate_(1.72, 4.21]','Offense_Team_Yards_Bin_Rate_(4.21, 6.7]',
                                 'Offense_Team_Yards_Bin_Rate_(6.7, 99.0]',
                                'Team_side','Offense_Team_Avg_Yards',
                                'YardLine_adj','Distance','Offense_score',
                                'clock_bin',
                                'Offense_WR']
    defense_team_model_columns = [# 'Defense_Team_Bin_Count_(-99.0, -0.01]','Defense_Team_Bin_Count_(-0.01, 0.01]','Defense_Team_Bin_Count_(0.01, 1.72]',
                                 #'Defense_Team_Bin_Count_(1.72, 4.21]','Defense_Team_Bin_Count_(4.21, 6.7]','Defense_Team_Bin_Count_(6.7, 99.0]',
                                 #'Defense_Team_Yards_Bin_Rate_(-99.0, -0.01]','Defense_Team_Yards_Bin_Rate_(-0.01, 0.01]','Defense_Team_Yards_Bin_Rate_(0.01, 1.72]','Defense_Team_Yards_Bin_Rate_(1.72, 4.21]',
                                 #'Defense_Team_Yards_Bin_Rate_(4.21, 6.7]', 'Defense_Team_Yards_Bin_Rate_(6.7, 99.0]',
                                'Defense_Team_Avg_Yards',#'Defense_Team_Count','Defense_Team_Std_Yards',
                                'Defense_DB',#'Defense_LB',#'Defense_OL',
                                 #'Defense_DL',
                                'Defense_Team','DefendersInTheBox','DefensePersonnel','YardLine_adj','Distance','Deffense_score',
                               ]
#     final_model_columns = ['A',
#                             'DisplayName',
#                             'predict_defense',
#                             'predict_offense',
#                             'YardLine_adj',
#                             'Location',
#                             'DisplayName_Avg_Yards',
#                             'PlayerCollegeName',
#                             'S',
#                             'Stadium',
#                             #'Season',
#                             'YardLine',
#                             'Defense_Team',
#                             'WindDirection',
#                             'VisitorTeamAbbr',
#                             'DisplayName_Yards_Bin_Rate_(6.7, 99.0]',
#                             'FieldPosition',
#                             'GameWeather',
#                             'HomeTeamAbbr',
#                             'OffensePersonnel',
#                             'predict_player',
#                             'DefendersInTheBox',
#                             'DisplayName_rusher_performance_bin',
#                             'Dis',
#                             'DisplayName_Std_Yards',
#                             'PossessionTeam',
#                             'Offense_Team',
#                             'DefensePersonnel'
#                           ]
    return {'player': player_model_columns, 'offense':offense_team_model_columns, 'defense':defense_team_model_columns}


# In[ ]:


# def eucl_distance(x1,x2,y1,y2):
#     dis=np.sqrt((x1-x2).pow(2)+ (y1-y2).pow(2))
#     return dis

def euclidean_distance(x1,x2,y1,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def norm_x(df):
    df['Player_o_d']=np.where((df["Home_team_play"] ==1) & (df["Team"] == 'home')|(df["Home_team_play"] ==0) & (df["Team"] == 'away'),'Offense_Player','Defense_Player')
    df['YardLine_Direction']=np.where(df["PlayDirection"] =='right',df['YardLine_adj'],100-df['YardLine_adj'])
    df['X_adj']=np.where(df["Player_o_d"] =='Offense_Player',abs(df['YardLine_Direction']+10-df['X']),-abs(df['YardLine_Direction']+10-df['X']))
    return(df)

def qb_and_rusher_to_closest_defender(df):
    play_ids = list(df.PlayId.unique())
    rows = []
    for play_id in play_ids:
        play = df[df['PlayId'] == play_id]
        rusher = play[play['IsBallCarrier']==True][['X', 'Y']]
        QB_offense=play[play['Position']=='QB'][['X','Y']]
        defenders = play[play['Player_o_d']=='Defense_Player'][['X', 'Y']]
        defenders['rusher_X'] = rusher['X'].iloc[0]
        defenders['rusher_Y'] = rusher['Y'].iloc[0]
        try:
            defenders['qb_X'] = QB_offense['X'].iloc[0]
            defenders['qb_Y'] = QB_offense['Y'].iloc[0]
        except:
            defenders['qb_X'] = 50
            defenders['qb_Y'] = 50
        
        defenders['dist_rusher_to_defender'] = defenders.apply(lambda row: euclidean_distance(row.X,row.rusher_X,row.Y,row.rusher_Y), axis = 1)
        defenders['dist_qb_to_defender'] = defenders.apply(lambda row: euclidean_distance(row.X,row.qb_X,row.Y,row.qb_Y), axis = 1)
        #mean_dist_closest_3_defenders = defenders.sort_values(by=dist_rusher_to_defender)['dist_rusher_to_defender'].iloc[:3].mean()
        rows.append([play_id, defenders['dist_rusher_to_defender'].min(), defenders['dist_qb_to_defender'].min(), euclidean_distance(rusher['X'].iloc[0], defenders['qb_X'].iloc[0], rusher['Y'].iloc[0], defenders['qb_Y'].iloc[0])])
    rows = pd.DataFrame(rows, columns=['PlayId', 'min_dist_rusher_to_defender', 'min_dist_qb_to_defender', 'dist_rusher_QB'])
    df = df.merge(rows, on='PlayId', how='left')
    return df
        
# def ruser_qb_dis(df):
#     QB_offense=df[(df['Position']=='QB') &  (df['Player_o_d']=='Offense_Player')]
#     rushers=df[df['IsBallCarrier']==True]
#     rushers=rushers[['PlayId','IsBallCarrier','X_adj','Y',]].merge(QB_offense[['PlayId','X_adj','Y']],on='PlayId', how='left')
#     rushers.rename(columns={ 'X_adj_x':'X_adj_rusher','Y_x':'Y_rusher','X_adj_y':'X_adj_QB_offense', 'Y_y':'Y_QB_offense'}, inplace= True)
#     rushers['dis_from_qb_offense']=euclidean_distance(rushers['X_adj_rusher'],rushers['X_adj_QB_offense'],rushers['Y_rusher'],rushers['Y_QB_offense'])
#     df=df.merge(rushers[['dis_from_qb_offense','PlayId','IsBallCarrier']],on=['PlayId','IsBallCarrier'], how='left')
#     return(df)

def Direction_orientation_adj(df):
    df['dir_adj']=np.where(df["Player_o_d"] =='Offense_Player',np.where(df["PlayDirection"] =='right',df["Dir"],360-df["Dir"]),np.where(df["PlayDirection"] =='right',360-df["Dir"],df["Dir"]))
    df['Orientation_adj']=np.where(df["Player_o_d"] =='Offense_Player',np.where(df["PlayDirection"] =='right',df["Orientation"],360-df["Orientation"]),np.where(df["PlayDirection"] =='right',360-df["Orientation"],df["Orientation"]))
    df['dir_to_Orientation']=180 - abs(abs(df['dir_adj'] - df['Orientation_adj']) - 180) 
    return(df)
def X_Y_Velocity(df):
    df['v_x']=np.sin(df['dir_adj'] * np.pi/180)*df['S']
    df['v_y']=np.cos(df['dir_adj'] * np.pi/180)*df['S']
    #df['v_x_y']=np.sqrt(df['v_x'].pow(2)+df['v_y'].pow(2))
    df['arc_tan']=np.arctan((df['v_y']/df['v_x']))*180/np.pi
    df['arc_tan_cat']=pd.cut(df['arc_tan'],bins=[-90, -45, 0, 45, 90],labels=False)
    df['t1_distance']=df['Dis']+df['S']*1+0.5*df['A']*1
    df['t2_distance']=df['Dis']+df['S']*2+0.5*df['A']*4
    df['t3_distance']=df['Dis']+df['S']*3+0.5*df['A']*9
    df['v_x_a1']=np.sin(df['dir_adj'] * np.pi/180)*(df['S']+df['A']*1)
    df['v_y_a1']=np.cos(df['dir_adj'] * np.pi/180)*(df['S']+df['A']*1)
    df['v_x_a2']=np.sin(df['dir_adj'] * np.pi/180)*(df['S']+df['A']*2)
    df['v_y_a2']=np.cos(df['dir_adj'] * np.pi/180)*(df['S']+df['A']*2)
    df['v_x_a3']=np.sin(df['dir_adj'] * np.pi/180)*(df['S']+df['A']*3)
    df['v_y_a3']=np.cos(df['dir_adj'] * np.pi/180)*(df['S']+df['A']*3)
    df['a_x']=np.sin(df['dir_adj'] * np.pi/180)*df['A']
    df['a_y']=np.cos(df['dir_adj'] * np.pi/180)*df['A']
    return(df)


# In[ ]:


h2o.init(min_mem_size='16G')


# In[ ]:


def build_and_return_model(train_df, training_columns, name, ntrees, max_depth, min_rows):
    train = h2o.H2OFrame(train_df)
    model = H2ORandomForestEstimator(ntrees=ntrees, max_depth=max_depth, min_rows=min_rows, keep_cross_validation_predictions=True, nfolds=10, seed=1)
    model.train(x=training_columns, y='Yards', training_frame=train)
    cv_predictions = model.cross_validation_holdout_predictions()
    train_df = train_df.join(cv_predictions.as_data_frame())
    train_df.rename(columns={'predict':('predict_' + name)}, inplace=True)
    return model, train_df


# In[ ]:


train_df_all = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
train_df_all = clean_TeamAbbreviations(train_df_all)
train_df_all = imperial_to_metric(train_df_all)
train_df_all = tag_rusher(train_df_all)
train_df_all = get_player_age(train_df_all)
train_df_all = get_adjusted_yardline(train_df_all)
train_df_all=norm_x(train_df_all)
train_df_all['dis_from_center']= abs(train_df_all['X_adj'])  #eucl_distance(train_df_all['X_adj'],0,train_df_all['Y'],0)
#train_df_all=ruser_qb_dis(train_df_all)
train_df_all=Direction_orientation_adj(train_df_all)
train_df_all=X_Y_Velocity(train_df_all)
train_df_all = qb_and_rusher_to_closest_defender(train_df_all)
train_df=train_df_all[train_df_all['NflId']==train_df_all['NflIdRusher']]
train_df.FieldPosition = train_df.FieldPosition.fillna('NA')
train_df = get_direction(train_df)
train_df = get_leading_team_features(train_df)
train_df = get_GameClock_features(train_df)
train_df['StadiumType'] = train_df['StadiumType'].apply(clean_StadiumType)
train_df['WindDirection']=train_df['WindDirection'].apply(clean_WindDirection)
train_df['WindDirection']=train_df['WindDirection'].apply(transform_WindDirection)
train_df=set_Turf(train_df)
train_df=GameWeather(train_df)
personnel_columns_to_add = get_personnel_types(train_df) 
train_df = get_split_personnel_types(personnel_columns_to_add, train_df)
player_and_team_stats, train_df = add_bins_and_generate_stats(train_df)


models_columns=get_models_columns()
model_player,train_df = build_and_return_model(train_df, models_columns['player'], 'player', 500, 5, 27)
model_offense,train_df = build_and_return_model(train_df, models_columns['offense'], 'offense', 500, 5, 15)
model_defense,train_df = build_and_return_model(train_df, models_columns['defense'], 'defense', 500, 6, 25)
models = {'player': model_player, 'offense': model_offense,'defense': model_defense}


# In[ ]:


training_columns =[
    'Down',
    'Age',
     'Offense_DL',
    'Offense_QB',
    'Offense_RB',
    'Offense_LB',
    'Offense_DB',
    'Offense_TE',
    'Defense_DL',
    'Defense_OL',
    'min_dist_rusher_to_defender',
    'min_dist_qb_to_defender',
    'dist_rusher_QB',
   'A',
   'predict_defense',
   'predict_offense',
   'predict_player',
   'YardLine_adj',
   'S',
   'DefendersInTheBox',
   'Dis',
   'Distance',
   'X_adj',
   'DisplayName_Count',
   'Defense_DB',
   'DisplayName_Bin_Count_(1.72, 4.21]',
   'DisplayName_Bin_Count_(6.7, 99.0]',
   'DisplayName_Yards_Bin_Rate_(-99.0, -0.01]',
   'DisplayName_Bin_Count_(0.01, 1.72]',
   'VisitorScoreBeforePlay',
   'DisplayName_Bin_Count_(4.21, 6.7]',
   'clock_bin',
   'Defense_Team_Yards_Bin_Rate_(6.7, 99.0]',
   'Offense_WR',
   'Team_side',
   'Offense_score',
   'Deffense_score',
   'Defense_LB',
   'DisplayName_Yards_Bin_Rate_(0.01, 1.72]',
   'Defense_Team_Yards_Bin_Rate_(-0.01, 0.01]',
   'Defense_Team_Bin_Count_(1.72, 4.21]',
   'Defense_Team_Avg_Yards',
   'Offense_OL',
   'DisplayName_Yards_Bin_Rate_(-0.01, 0.01]',
   'Defense_Team_Yards_Bin_Rate_(-99.0, -0.01]',
   'Defense_Team_Bin_Count_(-99.0, -0.01]',
   'dis_from_center',
   'dir_adj',
   'Orientation_adj',
   'dir_to_Orientation',
   'v_x',
   'v_y',
   'arc_tan',
   'arc_tan_cat',
   't1_distance',
   't2_distance',
   't3_distance',
   'v_x_a1',
   'v_y_a1',
   'v_x_a2',
   'v_y_a2',
   'v_x_a3',
   'v_y_a3'
]


# In[ ]:


categorical_columns = ['DisplayName_rusher_performance_bin', 'Position', 'Stadium','GameWeather','OffensePersonnel','Turf_type','OffenseFormation']
one_hot = pd.get_dummies(train_df[categorical_columns])
train_df = train_df.join(one_hot)


# In[ ]:


training_columns = training_columns + list(one_hot.columns)


# In[ ]:


train_df_temp=train_df[training_columns].fillna(0)
X = train_df_temp
yards = train_df.Yards

y = np.zeros((yards.shape[0], 37))
for idx, target in enumerate(list(yards)):
    if target > 22:
        y[idx][36] = 1
    elif target < -14:
        y[idx][0] = 1
    else:
        y[idx][14 + target] = 1

X = np.array(X)


# In[ ]:


# train_df_temp=train_df[training_columns].fillna(0)
# X = train_df_temp
# yards = train_df.Yards

# y = np.zeros((yards.shape[0], 199))
# for idx, target in enumerate(list(yards)):
#     y[idx][99 + target] = 1

# X = np.array(X)


# In[ ]:


# # Calculate CRPS score
# def crps_score(y_prediction, y_valid, shape=X.shape[0]):
#     y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
#     y_pred = np.clip(np.cumsum(y_prediction, axis=1), 0, 1)
#     val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * shape)
#     crps = np.round(val_s, 6)
    
#     return crps


# In[ ]:


# def get_rf(x_tr, y_tr, x_val, y_val, max_features, min_samples_leaf, min_samples_split, n_estimators, shape):
#     model = RandomForestRegressor(bootstrap=False, max_features=max_features, min_samples_leaf=min_samples_leaf,
#                                   min_samples_split=min_samples_split, n_estimators=n_estimators, n_jobs=-1, random_state=42)
#     model.fit(x_tr, y_tr)
    
#     y_pred = model.predict(x_val)
#     y_valid = y_val
#     crps = crps_score(y_pred, y_valid, shape=shape)
    
#     return model, crps


# In[ ]:


# max_features_list = [0.3]
# min_samples_leaf_list = [10]
# min_samples_split_list = [20]
# n_estimators_list = [200]


# rows = []
# fold = 5
# #models_rf = []
# i = 0
# max_i = len(max_features_list)*len(min_samples_leaf_list)*len(min_samples_split_list)*len(n_estimators_list)
# s_time = time.time()
# kfold = KFold(fold, random_state = 42, shuffle = True)
# for max_features in max_features_list:
#     for min_samples_leaf in min_samples_leaf_list:
#         for min_samples_split in min_samples_split_list:
#             for n_estimators in n_estimators_list:
#                 i = i+1
#                 print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#                 print(f'Grid {i}/{max_i}')
#                 print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#                 crps_csv_rf = []
#                 feature_importance = np.zeros([fold, X.shape[1]])
#                 for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(yards)):
                    
#                     print(f' Fold {k_fold+1}/{fold}')
#                     print("-----------")
#                     tr_x, tr_y = X[tr_inds], y[tr_inds]
#                     val_x, val_y = X[val_inds], y[val_inds]

#                     # Train RF
#                     rf, crps_rf = get_rf(tr_x, tr_y, val_x, val_y, max_features, min_samples_leaf, min_samples_split, n_estimators, shape=val_x.shape[0])
#                     #models_rf.append(rf)
#                     print("the %d fold crps (RF) is %f"%((k_fold+1), crps_rf))
#                     crps_csv_rf.append(crps_rf)

                    

#                     # Feature Importance
#                     feature_importance[k_fold, :] = rf.feature_importances_
#                 rows.append([
#                     max_features,
#                     min_samples_leaf,
#                     min_samples_split,
#                     n_estimators,
#                     sum(crps_csv_rf) / len(crps_csv_rf),
#                     np.mean(feature_importance, axis=0)
#                 ])

# gridsearch = pd.DataFrame(rows, columns = ['max_features', 'min_samples_leaf', 'min_samples_split', 'n_estimators', 'mean_CRPS', 'feature_importance'])


# In[ ]:


# gridsearch.sort_values(by = 'mean_CRPS')


# In[ ]:


# feature_imp = list(zip(training_columns, list(gridsearch['feature_importance'].iloc[0]))) # Index of best model
# feature_imp = [list(elem) for elem in feature_imp]
# feature_imp = pd.DataFrame(feature_imp, columns=['feature_name', 'importance'])
# feature_imp.sort_values(by='importance', ascending= False)


# In[ ]:


final_model = RandomForestRegressor(bootstrap=False, max_features=0.3, min_samples_leaf=10, 
                              min_samples_split=20, n_estimators=200, n_jobs=-1, random_state=42)
final_model.fit(X, y)


# In[ ]:


def make_my_predictions(test_df_all, sample_prediction_df, models, player_and_team_stats, personnel_columns_to_add, final_model, training_columns, categorical_columns, one_hot_columns):
    test_df_all = clean_TeamAbbreviations(test_df_all)
    test_df_all = imperial_to_metric(test_df_all)
    test_df_all = tag_rusher(test_df_all)
    test_df_all = get_player_age(test_df_all)
    test_df_all = get_adjusted_yardline(test_df_all)
    test_df_all=norm_x(test_df_all)
    test_df_all['dis_from_center']=abs(train_df_all['X_adj']) #eucl_distance(test_df_all['X_adj'],0,test_df_all['Y'],0)
    #test_df_all=ruser_qb_dis(test_df_all)
    test_df_all=Direction_orientation_adj(test_df_all)
    test_df_all=X_Y_Velocity(test_df_all)
    test_df_all = qb_and_rusher_to_closest_defender(test_df_all)

    test_df=test_df_all[test_df_all['NflId']==test_df_all['NflIdRusher']]
    test_df = test_df.fillna(0)

    test_df = get_direction(test_df)
    test_df = get_leading_team_features(test_df)
    test_df = get_GameClock_features(test_df)
    test_df = get_split_personnel_types(personnel_columns_to_add, test_df)
    for key,value in player_and_team_stats.items():
        test_df = test_df.merge(value, on=key, how='left').fillna(0)
        if key == 'DisplayName' and test_df['DisplayName_rusher_performance_bin'].iloc[0] == 0:
            test_df['DisplayName_rusher_performance_bin'] = 'few_plays'
#    test_df=convert_to_cat(test_df)
    test_df=set_Turf(test_df)
    test_df['StadiumType'] = test_df['StadiumType'].apply(clean_StadiumType)
    test_df['WindDirection']=test_df['WindDirection'].apply(clean_WindDirection)
    test_df['WindDirection']=test_df['WindDirection'].apply(transform_WindDirection)
    test_df=GameWeather(test_df)

    for model_type,model in models.items():
        test_df_curr = h2o.H2OFrame(test_df)
        pred=model.predict(test_df_curr)
        test_df = test_df.join(pred.as_data_frame())
        test_df.rename(columns={'predict':'predict_' + model_type}, inplace=True)

    one_hot_test = pd.get_dummies(test_df[categorical_columns])
    one_hot_test_cols = [x for x in one_hot_test if x in one_hot_columns]
    one_hot_test = one_hot_test[one_hot_test_cols]
    test_df = test_df.join(one_hot_test)
    more_to_add = [x for x in one_hot_columns if x not in list(one_hot_test.columns)]
    for column_to_add in more_to_add:
        test_df[column_to_add] = 0

    test_df=test_df[training_columns]#.fillna(0)    

    test_X = np.array(test_df)
    pred = final_model.predict(test_X)
    pred = np.array([np.concatenate((np.zeros(85), pred[0], np.zeros(77)), axis=0)]) #comment out this line
    pred = np.clip(np.cumsum(pred, axis=1), 0, 1)
    pred = pd.DataFrame(data=[list(pred[0])], columns=sample_prediction_df.columns)
    return pred


# In[ ]:


from kaggle.competitions import nflrush
env = nflrush.make_env()
#h2o.init(min_mem_size='16G')
# Training data is in the competition dataset as usual
#train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

#player_and_team_stats, personnel_columns_to_add, models = train_my_model(train_df, get_models_columns())

for (test_df, sample_prediction_df) in env.iter_test():
    predictions_df = make_my_predictions(test_df, sample_prediction_df, models, player_and_team_stats, personnel_columns_to_add, final_model, training_columns, categorical_columns, list(one_hot.columns))
    env.predict(predictions_df)
env.write_submission_file()


# In[ ]:


# test_df_all = test_df2.copy()


# In[ ]:


# test_df_all = imperial_to_metric(test_df_all)
# test_df_all = tag_rusher(test_df_all)
# test_df_all = get_player_age(test_df_all)
# test_df_all = get_adjusted_yardline(test_df_all)
# test_df_all=norm_x(test_df_all)
# test_df_all['euc_dis_from_center']=eucl_distance(test_df_all['X_adj'],0,test_df_all['Y'],0)
# test_df_all=ruser_qb_dis(test_df_all)
# test_df_all=Direction_orientation_adj(test_df_all)
# test_df_all=X_Y_Velocity(test_df_all)

# test_df=test_df_all[test_df_all['NflId']==test_df_all['NflIdRusher']]
# test_df = test_df.fillna(0)

# test_df = clean_TeamAbbreviations(test_df)
# test_df = get_direction(test_df)
# test_df = get_leading_team_features(test_df)
# test_df = get_GameClock_features(test_df)
# test_df = get_split_personnel_types(personnel_columns_to_add, test_df)
# for key,value in player_and_team_stats.items():
#     test_df = test_df.merge(value, on=key, how='left').fillna(0)
#     if key == 'DisplayName' and test_df['DisplayName_rusher_performance_bin'].iloc[0] == 0:
#         test_df['DisplayName_rusher_performance_bin'] = 'few_plays'
# test_df=convert_to_cat(test_df)
# test_df=set_Turf(test_df)
# test_df['StadiumType'] = test_df['StadiumType'].apply(clean_StadiumType)
# test_df['WindDirection']=test_df['WindDirection'].apply(clean_WindDirection)
# test_df['WindDirection']=test_df['WindDirection'].apply(transform_WindDirection)
# test_df=GameWeather(test_df)

# for model_type,model in models.items():
#     test_df_curr = h2o.H2OFrame(test_df)
#     pred=model.predict(test_df_curr)
#     test_df = test_df.join(pred.as_data_frame())
#     test_df.rename(columns={'predict':'predict_' + model_type}, inplace=True)



# one_hot_test = pd.get_dummies(test_df[categorical_columns])
# test_df = test_df.join(one_hot_test)
# more_to_add = [x for x in one_hot_columns if x not in list(one_hot_test.columns)]
# for column_to_add in more_to_add:
#     test_df[column_to_add] = 0

# test_df=test_df[training_columns]#.fillna(0)    

# test_X = np.array(test_df)
# pred = final_model.predict(test_X)
# pred = np.clip(np.cumsum(pred, axis=1), 0, 1)
# pred = pd.DataFrame(data=[list(pred[0])], columns=sample_prediction_df.columns)


# In[ ]:





# In[ ]:





# In[ ]:





# # Test submodels here

# In[ ]:


# train_df, test_df, portion = train_test_split_random(0.2, train_df, 3)

# for key,value in player_and_team_stats.items():
#     test_df = test_df.merge(value, on=key, how='left').fillna(0)
# if key == 'DisplayName' and test_df['DisplayName_rusher_performance_bin'].iloc[0] == 0:
#     test_df['DisplayName_rusher_performance_bin'] = 'few_plays'

# for model_type,model in models.items():
#     test_df_curr = h2o.H2OFrame(test_df)
#     pred=model.predict(test_df_curr)
#     test_df = test_df.join(pred.as_data_frame())
#     test_df.rename(columns={'predict':'predict_' + model_type}, inplace=True)


# In[ ]:


# train = h2o.H2OFrame(train_df)
# validation = h2o.H2OFrame(test_df)
# model = H2ORandomForestEstimator(ntrees=300, max_depth=5,min_rows=20,seed=55#,keep_cross_validation_predictions=True, nfolds=10
#                                 )
# model.train(x=training_columns, y='Yards', training_frame=train, validation_frame = validation )
# #cv_predictions = model.cross_validation_holdout_predictions()
# # train=train.as_data_frame().join(cv_predictions.as_data_frame())
# # train.rename(columns={'predict':'predict_final'}, inplace=True)
# performance=model.model_performance 
# performance


# In[ ]:


# model.varimp(True)


# In[ ]:


# # Evalute_CRPS 
# validation = h2o.H2OFrame(test_df)
# pred=model.predict(validation);pred=pred.as_data_frame()
# pred['stdv']=model.model_performance().mae()
# validation=validation.as_data_frame()
# CRPS=Evalute_CRPS(validation.iloc[:,:],pred.iloc[:,:])
# CRPS


# In[ ]:


#  training_columns =[#'Team',
# #                             'X',
# #                             'Y',
# #                             'S',
# #                             'A',
# #                             'Dis',
# #                             'Orientation',
# #                             'Dir',
# #                             'NflId',
# #                             'DisplayName',
# #                             'JerseyNumber',
# #                             'Season',
# #                             'YardLine',
# #                             'Quarter',
# #                             'GameClock',
# #                             'PossessionTeam',
# #                             'Down',
# #                             'Distance',
# #                             'FieldPosition',
# #                             'HomeScoreBeforePlay',
# #                             'VisitorScoreBeforePlay',
# #                             'NflIdRusher',
# #                             'OffenseFormation',
# #                             'OffensePersonnel',
# #                             'DefendersInTheBox',
# #                             'DefensePersonnel',
# #                             'PlayDirection',
# #                             'TimeHandoff',
# #                             'TimeSnap',
# #                             'PlayerHeight',
# #                             'PlayerWeight',
# #                             'PlayerBirthDate',
# #                             'PlayerCollegeName',
# #                             'Position',
# #                             'HomeTeamAbbr',
# #                             'VisitorTeamAbbr',
# #                             'Week',
# #                             'Stadium',
# #                             'Location',
# #                             'StadiumType',
# #                             'Turf',
# #                             'GameWeather',
# #                             'Temperature',
# #                             'Humidity',
# #                             'WindSpeed', #SAW A CASE WHERE VALUE WAS '6mph'
# #                             'WindDirection',
# #                             'PlayerHeight_cm',
# #                             'PlayerWeight_kg',
# #                             'IsBallCarrier',
# #                             'Age',
# #                             'ToLeft',
# #                             'possession_team_home_visitor',
# #                             'leading_by',
# #                             'leading_team_defense_offense',
# #                             'leading_team_home_visitor',
# #                             'Offense_score',
# #                             'Deffense_score',
# #                             'Offense_Team',
# #                             'Defense_Team',
# #                             'Home_team_play',
# #                             'Team_side',
# #                             'YardLine_adj',
# #                             'snap_to_handoff',
# #                             'GameClock_sec',
# #                             'clock_bin',
# #                             'Offense_DB',
# #                             'Offense_LB',
# #                             'Offense_TE',
# #                             'Offense_QB',
# #                             'Offense_DL',
# #                             'Offense_OL',
# #                             'Offense_WR',
# #                             'Offense_RB',
# #                             'Defense_DB',
# #                             'Defense_LB',
# #                             'Defense_OL',
# #                             'Defense_DL',
# #                             'DisplayName_Bin_Count_(-99.0, -0.01]',
# #                             'DisplayName_Bin_Count_(-0.01, 0.01]',
# #                             'DisplayName_Bin_Count_(0.01, 1.72]',
# #                             'DisplayName_Bin_Count_(1.72, 4.21]',
# #                             'DisplayName_Bin_Count_(4.21, 6.7]',
# #                             'DisplayName_Bin_Count_(6.7, 99.0]',
# #                             'DisplayName_Yards_Bin_Rate_(-99.0, -0.01]',
# #                             'DisplayName_Yards_Bin_Rate_(-0.01, 0.01]',
# #                             'DisplayName_Yards_Bin_Rate_(0.01, 1.72]',
# #                             'DisplayName_Yards_Bin_Rate_(1.72, 4.21]',
# #                             'DisplayName_Yards_Bin_Rate_(4.21, 6.7]',
# #                             'DisplayName_Yards_Bin_Rate_(6.7, 99.0]',
# #                             'DisplayName_Avg_Yards',
# #                             'DisplayName_Count',
# #                             'DisplayName_Std_Yards',
# #                             'DisplayName_rusher_performance_bin',
# #                             'Offense_Team_Bin_Count_(-99.0, -0.01]',
# #                             'Offense_Team_Bin_Count_(-0.01, 0.01]',
# #                             'Offense_Team_Bin_Count_(0.01, 1.72]',
# #                             'Offense_Team_Bin_Count_(1.72, 4.21]',
# #                             'Offense_Team_Bin_Count_(4.21, 6.7]',
# #                             'Offense_Team_Bin_Count_(6.7, 99.0]',
# #                             'Offense_Team_Yards_Bin_Rate_(-99.0, -0.01]',
# #                             'Offense_Team_Yards_Bin_Rate_(-0.01, 0.01]',
# #                             'Offense_Team_Yards_Bin_Rate_(0.01, 1.72]',
# #                             'Offense_Team_Yards_Bin_Rate_(1.72, 4.21]',
# #                             'Offense_Team_Yards_Bin_Rate_(4.21, 6.7]',
# #                             'Offense_Team_Yards_Bin_Rate_(6.7, 99.0]',
# #                             'Offense_Team_Avg_Yards',
# #                             'Offense_Team_Count',
# #                             'Offense_Team_Std_Yards',
# #                             'Defense_Team_Bin_Count_(-99.0, -0.01]',
# #                             'Defense_Team_Bin_Count_(-0.01, 0.01]',
# #                             'Defense_Team_Bin_Count_(0.01, 1.72]',
# #                             'Defense_Team_Bin_Count_(1.72, 4.21]',
# #                             'Defense_Team_Bin_Count_(4.21, 6.7]',
# #                             'Defense_Team_Bin_Count_(6.7, 99.0]',
# #                             'Defense_Team_Yards_Bin_Rate_(-99.0, -0.01]',
# #                             'Defense_Team_Yards_Bin_Rate_(-0.01, 0.01]',
# #                             'Defense_Team_Yards_Bin_Rate_(0.01, 1.72]',
# #                             'Defense_Team_Yards_Bin_Rate_(1.72, 4.21]',
# #                             'Defense_Team_Yards_Bin_Rate_(4.21, 6.7]',
# #                             'Defense_Team_Yards_Bin_Rate_(6.7, 99.0]',
# #                             'Defense_Team_Avg_Yards',
# #                             'Defense_Team_Count',
# #                             'Defense_Team_Std_Yards',
# #                            'predict_player',
# #                            'predict_offense',
# #                            'predict_defense'
# 'A',
# #'DisplayName',
# 'predict_defense',
# 'predict_offense',
# 'YardLine_adj',
# 'Location',
# #'DisplayName_Avg_Yards',
# #'PlayerCollegeName',
# 'S',
# 'Stadium',
# #'Season',
# #'YardLine',
# 'Defense_Team',
# 'WindDirection',
# #'VisitorTeamAbbr',
# #'DisplayName_Yards_Bin_Rate_(6.7, 99.0]',
# 'FieldPosition',
# 'GameWeather',
# 'HomeTeamAbbr',
# 'OffensePersonnel',
# 'predict_player',
# 'DefendersInTheBox',
# 'DisplayName_rusher_performance_bin',
# 'Dis',
# 'DisplayName_Std_Yards',
# 'PossessionTeam',
# 'Offense_Team',
# 'DefensePersonnel',
# 'Distance',
# 'Orientation',
# 'X',
# 'DisplayName_Count',
# 'DisplayName_Yards_Bin_Rate_(1.72, 4.21]',
# 'Defense_DB',
# 'OffenseFormation',
# 'DisplayName_Bin_Count_(1.72, 4.21]',
# 'DisplayName_Bin_Count_(6.7, 99.0]',
# 'DisplayName_Yards_Bin_Rate_(-99.0, -0.01]',
# 'Deffense_score',
# 'DisplayName_Bin_Count_(0.01, 1.72]',
# 'Dir',
# 'Position',
# 'VisitorScoreBeforePlay',
# 'Turf',
# 'StadiumType',
# 'DisplayName_Bin_Count_(4.21, 6.7]',
# 'clock_bin',
# 'DisplayName_Yards_Bin_Rate_(0.01, 1.72]',
# 'GameClock',
# 'Defense_Team_Count',
# 'Defense_Team_Yards_Bin_Rate_(6.7, 99.0]',
# 'GameClock_sec',
# 'Offense_WR',
# 'Down',
# 'Team_side',
# 'Defense_Team_Yards_Bin_Rate_(-0.01, 0.01]',
# 'DisplayName_Yards_Bin_Rate_(-0.01, 0.01]',
# 'Defense_Team_Bin_Count_(-99.0, -0.01]',
# 'PlayerHeight',
# 'Defense_LB',
# 'Offense_OL',
# 'Defense_Team_Yards_Bin_Rate_(-99.0, -0.01]',
# 'Offense_score',
# 'Y',
# 'DisplayName_Yards_Bin_Rate_(4.21, 6.7]',
# 'Defense_Team_Bin_Count_(1.72, 4.21]',
# 'leading_team_defense_offense',
# 'Defense_Team_Avg_Yards',
# 'leading_by',
# 'WindSpeed'


#  ]



# In[ ]:


# train = h2o.H2OFrame(train_df)
# # GBM hyperparameters
# gbm_params1 = {'ntrees': [300,400],
#                 'max_depth': [5,6],
#                 'min_rows': [10,15,20]}

# # Train and validate a cartesian grid of GBMs
# gbm_grid1 = H2OGridSearch(model=H2ORandomForestEstimator,
#                           grid_id='gbm_grid99',
#                           hyper_params=gbm_params1)
# gbm_grid1.train(x=training_columns, y='Yards',
#                 training_frame=train,
#                 seed=1)
# gbm_gridperf1 = gbm_grid1.get_grid(sort_by='mae')
# gbm_gridperf1


# In[ ]:





# In[ ]:





# ### !!! TEST ZONE BELOW !!!

# In[ ]:


# def pre_process_train(train_df_all):
#     train_df_all = imperial_to_metric(train_df_all)
#     train_df_all = tag_rusher(train_df_all)
#     train_df_all = get_player_age(train_df_all)
#     #train_df_all, per_play = get_movement_features(train_df_all)
#     train_df=train_df_all[train_df_all['NflId']==train_df_all['NflIdRusher']]

#     train_df = clean_TeamAbbreviations(train_df)
#     train_df = get_direction(train_df)
#     train_df = get_leading_team_features(train_df)
#     train_df = get_adjusted_yardline(train_df)
#     train_df = get_GameClock_features(train_df)
#     personnel_columns_to_add = get_personnel_types(train_df) #KEEP THIS VARIABLE FOR FUTURE USE
#     train_df = get_split_personnel_types(personnel_columns_to_add, train_df)
#     player_and_team_stats, train_df = add_bins_and_generate_stats(train_df)
#     train_df=convert_to_cat(train_df)
#     train_df=set_Turf(train_df)
#     train_df['StadiumType'] = train_df['StadiumType'].apply(clean_StadiumType)
#     train_df['WindDirection']=train_df['WindDirection'].apply(clean_WindDirection)
#     train_df['WindDirection']=train_df['WindDirection'].apply(transform_WindDirection)
#     train_df=GameWeather(train_df)
#     #train_df = g.cat.add_categories("D").fillna("D")
#     #train_df=train_df.fillna(0)

#     return player_and_team_stats, personnel_columns_to_add, train_df


# In[ ]:


# def build_and_return_model(train_df, training_columns, name, ntrees, max_depth, min_rows):
#     train = h2o.H2OFrame(train_df)
#     model = H2ORandomForestEstimator(ntrees=ntrees, max_depth=max_depth, min_rows=min_rows, keep_cross_validation_predictions=True, nfolds=10, seed=1)
#     model.train(x=training_columns, y='Yards', training_frame=train)
#     cv_predictions = model.cross_validation_holdout_predictions()
#     train_df = train_df.join(cv_predictions.as_data_frame())
#     train_df.rename(columns={'predict':('predict_' + name)}, inplace=True)
#     return model, train_df


# In[ ]:


# def train_my_model(train_df, models_columns):
#     player_and_team_stats, personnel_columns_to_add, train_df = pre_process_train(train_df)
#     model_player,train_df = build_and_return_model(train_df, models_columns['player'], 'player', 300, 5, 27)
#     model_offense,train_df = build_and_return_model(train_df, models_columns['offense'], 'offense', 300, 5, 15)
#     model_defense,train_df = build_and_return_model(train_df, models_columns['defense'], 'defense', 300, 6, 25)
#     model_final,train_df = build_and_return_model(train_df, models_columns['final'], 'final', 300, 6, 20)
#     return player_and_team_stats, personnel_columns_to_add, {'player':model_player, 'offense':model_offense,'defense':model_defense,'final':model_final}


# In[ ]:


# def make_my_predictions(test_df_all, sample_prediction_df, models, player_and_team_stats, personnel_columns_to_add):
#     test_df_all = imperial_to_metric(test_df_all)
#     test_df_all = tag_rusher(test_df_all)
#     test_df_all = get_player_age(test_df_all)
#     test_df_all = get_adjusted_yardline(test_df_all)
#     test_df_all=norm_x(test_df_all)
#     test_df_all['euc_dis_from_center']=eucl_distance(test_df_all['X_adj'],0,test_df_all['Y'],0)
#     test_df_all=ruser_qb_dis(test_df_all)
#     test_df_all=Direction_orientation_adj(test_df_all)
#     test_df_all=X_Y_Velocity(test_df_all)
#     #train_df_all, per_play = get_movement_features(train_df_all)

#     test_df=test_df_all[test_df_all['NflId']==test_df_all['NflIdRusher']]
#     test_df = test_df.fillna(0)

#     test_df = clean_TeamAbbreviations(test_df)
#     test_df = get_direction(test_df)
#     test_df = get_leading_team_features(test_df)
#     test_df = get_GameClock_features(test_df)
#     test_df = get_split_personnel_types(personnel_columns_to_add, test_df)
#     for key,value in player_and_team_stats.items():
#         test_df = test_df.merge(value, on=key, how='left').fillna(0)
#         if key == 'DisplayName' and test_df['DisplayName_rusher_performance_bin'].iloc[0] == 0:
#             test_df['DisplayName_rusher_performance_bin'] = 'few_plays'
#     test_df=convert_to_cat(test_df)
#     test_df=set_Turf(test_df)
#     test_df['StadiumType'] = test_df['StadiumType'].apply(clean_StadiumType)
#     test_df['WindDirection']=test_df['WindDirection'].apply(clean_WindDirection)
#     test_df['WindDirection']=test_df['WindDirection'].apply(transform_WindDirection)
#     test_df=GameWeather(test_df)
#     #test_df=test_df.fillna(0)
    
#     for model_type,model in models.items():
#         test_df_curr = h2o.H2OFrame(test_df)
#         try:
#             pred=model.predict(test_df_curr)
#             test_df = test_df.join(pred.as_data_frame())
       
#         except Exception as e: 
#             print(e)
#             print(test_df.iloc[0])
#             test_df['predict'] = 3
#             pred = pd.DataFrame([[3]], columns=['predict'])
#         test_df.rename(columns={'predict':'predict_' + model_type}, inplace=True)
    
#     #m=models['final'].predict_contributions(h2o.H2OFrame(test_df))
# #     st=models['final'].model_performance().mae()

# #     #st=1.634490746
# #     ssx=1.17829300428312*st*4086
# #     se=[];n=4086
# #     pred = pd.DataFrame(columns = sample_prediction_df.columns)
# #     pred.loc[len(pred)] = 0
# #     x0=test_df['predict_final'].iloc[0]
# #     x=4.059960841
# #     std=st*math.sqrt(1+1/n+(np.float_power(x0-x, 2))/ssx)
# #     for j in range(0, sample_prediction_df.shape[1]):  
# #         pred.iloc[0,j]=norm.cdf(j-99,test_df['predict_final'].iloc[0], std)
#     return pred


# In[ ]:


# #train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
# player_and_team_stats, personnel_columns_to_add, train_df = pre_process_train(train_df)


# ### get_models_columns in hidden cell below

# In[ ]:


# def get_models_columns():
#     player_model_columns = [#'DisplayName_Bin_Count_(-99.0, -0.01]','DisplayName_Bin_Count_(-0.01, 0.01]','DisplayName_Bin_Count_(0.01, 1.72]','DisplayName_Bin_Count_(1.72, 4.21]','DisplayName_Bin_Count_(4.21, 6.7]',
#                             'DisplayName_Bin_Count_(6.7, 99.0]',
#                             'DisplayName_Yards_Bin_Rate_(-99.0, -0.01]',
#                             #'DisplayName_Yards_Bin_Rate_(-0.01, 0.01]',
#                             #'DisplayName_Yards_Bin_Rate_(0.01, 1.72]',
#                             'DisplayName_Yards_Bin_Rate_(1.72, 4.21]',
#                             #'DisplayName_Yards_Bin_Rate_(4.21, 6.7]',
#                             'DisplayName_Yards_Bin_Rate_(6.7, 99.0]',
#                             'DisplayName_Avg_Yards',
#                             #'DisplayName_Count',
#                             'DisplayName_Std_Yards',
#                             'DisplayName_rusher_performance_bin',#'JerseyNumber',
#                             #'Position',
#                             'PlayerCollegeName',
#                             #'PlayerHeight_cm','PlayerWeight_kg','Age','PlayerWeight','PlayerHeight'
#                             ]
#     offense_team_model_columns =['PossessionTeam', 'OffenseFormation','OffensePersonnel',
#                                  #'Offense_Team_Bin_Count_(-99.0, -0.01]','Offense_Team_Bin_Count_(-0.01, 0.01]','Offense_Team_Bin_Count_(0.01, 1.72]','Offense_Team_Bin_Count_(1.72, 4.21]',
#                                  #'Offense_Team_Bin_Count_(4.21, 6.7]','Offense_Team_Bin_Count_(6.7, 99.0]','Offense_Team_Yards_Bin_Rate_(-99.0, -0.01]','Offense_Team_Yards_Bin_Rate_(-0.01, 0.01]','Offense_Team_Yards_Bin_Rate_(0.01, 1.72]','Offense_Team_Yards_Bin_Rate_(1.72, 4.21]','Offense_Team_Yards_Bin_Rate_(4.21, 6.7]',
#                                  'Offense_Team_Yards_Bin_Rate_(6.7, 99.0]',
#                                 'Team_side','Offense_Team_Avg_Yards',
#                                 'YardLine_adj','Distance','Offense_score',
#                                 'clock_bin',
#                                 'Offense_WR']
#     defense_team_model_columns = [# 'Defense_Team_Bin_Count_(-99.0, -0.01]','Defense_Team_Bin_Count_(-0.01, 0.01]','Defense_Team_Bin_Count_(0.01, 1.72]',
#                                  #'Defense_Team_Bin_Count_(1.72, 4.21]','Defense_Team_Bin_Count_(4.21, 6.7]','Defense_Team_Bin_Count_(6.7, 99.0]',
#                                  #'Defense_Team_Yards_Bin_Rate_(-99.0, -0.01]','Defense_Team_Yards_Bin_Rate_(-0.01, 0.01]','Defense_Team_Yards_Bin_Rate_(0.01, 1.72]','Defense_Team_Yards_Bin_Rate_(1.72, 4.21]',
#                                  #'Defense_Team_Yards_Bin_Rate_(4.21, 6.7]', 'Defense_Team_Yards_Bin_Rate_(6.7, 99.0]',
#                                 'Defense_Team_Avg_Yards',#'Defense_Team_Count','Defense_Team_Std_Yards',
#                                 'Defense_DB',#'Defense_LB',#'Defense_OL',
#                                  #'Defense_DL',
#                                 'Defense_Team','DefendersInTheBox','DefensePersonnel','YardLine_adj','Distance','Deffense_score',
#                                ]
#     final_model_columns = ['A',
#                             'DisplayName',
#                             'predict_defense',
#                             'predict_offense',
#                             'YardLine_adj',
#                             'Location',
#                             'DisplayName_Avg_Yards',
#                             'PlayerCollegeName',
#                             'S',
#                             'Stadium',
#                             #'Season',
#                             'YardLine',
#                             'Defense_Team',
#                             'WindDirection',
#                             'VisitorTeamAbbr',
#                             'DisplayName_Yards_Bin_Rate_(6.7, 99.0]',
#                             'FieldPosition',
#                             'GameWeather',
#                             'HomeTeamAbbr',
#                             'OffensePersonnel',
#                             'predict_player',
#                             'DefendersInTheBox',
#                             'DisplayName_rusher_performance_bin',
#                             'Dis',
#                             'DisplayName_Std_Yards',
#                             'PossessionTeam',
#                             'Offense_Team',
#                             'DefensePersonnel'
#                           ]
#     return {'player': player_model_columns, 'offense':offense_team_model_columns, 'defense':defense_team_model_columns, 'final':final_model_columns}


# In[ ]:


# from kaggle.competitions import nflrush
# env = nflrush.make_env()
# h2o.init(min_mem_size='16G')
# # Training data is in the competition dataset as usual
# train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

# models_columns = get_models_columns()

# player_and_team_stats, personnel_columns_to_add, models = train_my_model(train_df, models_columns)

# for (test_df, sample_prediction_df) in env.iter_test():
#     predictions_df = make_my_predictions(test_df, sample_prediction_df, models, player_and_team_stats, personnel_columns_to_add)
#     env.predict(predictions_df)
# env.write_submission_file()


# In[ ]:


#df=train_df


# In[ ]:


# # Split train/test functions
# def train_test_split_latest(test_percent, dataset, random_state):
#     last_games_sample = list(games.iloc[round(len(games) * (1-test_percent)):])
#     test = dataset[dataset.GameId.isin(last_games_sample)]
#     train = dataset[~dataset.GameId.isin(last_games_sample)]
#     portion = test.PlayId.nunique() / (dataset.PlayId.nunique())
#     return train, test, portion
# def train_test_split_random(test_percent, dataset, random_state):
#     # Random_state = 3 works
#     games = dataset.GameId.drop_duplicates()
#     random_games_sample = list(games.sample(frac=test_percent, random_state=random_state))
#     test = dataset[dataset.GameId.isin(random_games_sample)]
#     train = dataset[~dataset.GameId.isin(random_games_sample)]
#     portion = test.PlayId.nunique() / (dataset.PlayId.nunique())
#     return train, test, portion


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




