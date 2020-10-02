#!/usr/bin/env python
# coding: utf-8

# # DSB Team Peggy Peg Notebook
# 
# 

# # Get environment ready
# * Import Packages
# * Create data variables
# * Display kaggle directory

# In[ ]:


# Import packages

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.stats import mode
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import shap
from time import time
from tqdm import tqdm_notebook as tqdm
from collections import Counter
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold, StratifiedKFold
import gc
import json
from IPython.display import Image



# So we are all on the same page, use the walk method to generate files within this kaggle directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
#Create data variables



sample_submission = pd.read_csv("../input/data-science-bowl-2019/sample_submission.csv")
specs = pd.read_csv("../input/data-science-bowl-2019/specs.csv")
test = pd.read_csv("../input/data-science-bowl-2019/test.csv")
train = pd.read_csv("../input/data-science-bowl-2019/train.csv")
train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")
pd.set_option('display.max_columns', 1000)


# # List our data files

# In[ ]:


os.listdir('../input/data-science-bowl-2019')


# In[ ]:


# Display our hero, Peggy Peg, and her cat

Image("../input/peggypeg/peggypeg.jpeg")


# # Data Preperation
# * Find installation id's contained in train.csv which are not in train_labels.csv.  Remove these rows(train_labels.csv contains our target variable, accuracy group)

# In[ ]:


df_train=pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
df_test=pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
spec=pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')


# In[ ]:


#In the Spec file there are 387 unique event_ids but unique info is only 167
# Merge redundant event_ids

#spec=None
spec['info']=spec['info'].str.upper()
spec['hashed_info']=spec['info'].transform(hash)
spec_unique=pd.DataFrame(spec[['hashed_info']].drop_duplicates())
spec_unique['deduped_event_id']=np.arange(len(spec_unique))
spec=pd.merge(spec,spec_unique,on='hashed_info',how='left')
z=dict(zip(spec.event_id,spec.deduped_event_id))
df_train['event_id']=df_train['event_id'].map(z)
df_test['event_id']=df_test['event_id'].map(z)
    #df_train=df_train[df_train['event_id'].isin(df_test['event_id'])]
df_train=df_train[df_train['event_id']!=137]  # this particular event id only has 2 records in train and none in test....
df_event_id_train=pd.pivot_table(df_train.loc[:,['installation_id','game_session','event_id']],aggfunc=len,columns=['event_id'],index=['installation_id','game_session']).add_prefix('event_id_').rename_axis(None,axis=1).reset_index()
df_event_id_test=pd.pivot_table(df_test.loc[:,['installation_id','game_session','event_id']],aggfunc=len,columns=['event_id'],index=['installation_id','game_session']).add_prefix('event_id_').rename_axis(None,axis=1).reset_index()
df_event_id_train=df_event_id_train.fillna(0)
df_event_id_train=df_event_id_train.fillna(0)
df_event_id_test=df_event_id_test.fillna(0)


# # Features, baby

# # Time, World, Correct/Incorrect, Accuracy
# 
# * Weekend
# * Phase of Day
# 
# 
# * Extract hour_of_day from timestamp and drop timestamp
# * on_hot encoding on event_code.  Group df by installation_id and game_session
# * Define agg dictionary to define the aggregate functions to be performed after grouping df
# * Take first value, as it's unique for installation_id and game_session pair
# * Join this together and return df

# In[ ]:


def create_features(df):
    df['timestamp']=pd.to_datetime(df['timestamp'])
    df['Incorrect_Game_Attempt']=np.where((df['event_data'].str.contains('"correct":false')&(df['type']=='Game')),1,0)
    df['Correct_Game_Attempt']=np.where((df['event_data'].str.contains('"correct":true')&(df['type']=='Game')),1,0)
    df['Is_Weekend']=np.where((df['timestamp'].dt.day_name()=='Sunday')|(df['timestamp'].dt.day_name()=='Saturday'),1,0)
    df['Phase_Of_Day']=np.where(df['timestamp'].dt.hour.isin(range(6,12)),'Morning',np.where(df['timestamp'].dt.hour.isin(range(13,19)),'Evening','Night'))
    df_world=pd.pivot_table(df.loc[df['world']!='NONE',['installation_id','game_session','world']].drop_duplicates(),index=['installation_id','game_session'],columns=['world'],aggfunc=len).add_prefix('rolling_').rename_axis(None, axis=1).reset_index()
    
    df_type_world=pd.merge(df_world,pd.pivot_table(df.loc[:,['installation_id','game_session','type']].drop_duplicates(),index=['installation_id','game_session'],columns=['type'],fill_value=0,aggfunc=len).rename_axis(None, axis=1).reset_index(),on=['installation_id','game_session'],how='right')
    df_type_world_title=pd.merge(df_type_world,pd.pivot_table(df.loc[:,['installation_id','game_session','title']].drop_duplicates(),index=['installation_id','game_session'],columns=['title'],fill_value=0,aggfunc=len).add_prefix('rolling_').rename_axis(None, axis=1).reset_index(),on=['installation_id','game_session'],how='right')

    df_activity_weekend=pd.merge(df_type_world_title,pd.DataFrame(pd.pivot_table(df.loc[:,['installation_id','game_session','Is_Weekend']].drop_duplicates(),index=['installation_id','game_session'],columns=['Is_Weekend'],fill_value=0,aggfunc=len)).add_prefix('Weekend_').rename_axis(None, axis=1).reset_index(),on=['installation_id','game_session'],how='right')
    df_activity_weekend_phase_of_day=pd.merge(pd.DataFrame(pd.pivot_table(df.loc[:,['installation_id','game_session','Phase_Of_Day']].drop_duplicates(),index=['installation_id','game_session'],columns=['Phase_Of_Day'],fill_value=0,aggfunc=len)).rename_axis(None, axis=1).reset_index(),df_activity_weekend,on=['installation_id','game_session'],how='left')
    df_train_Assessed=df.copy()
    df_train_Assessed['Incorrect_Attempt']=np.where((df['event_data'].str.contains('"correct":false'))&(((df['title'] != "Bird Measurer (Assessment)")&(df['event_code']==4100))|((df['title'] == "Bird Measurer (Assessment)")&(df['event_code']==4110))),1,0)
    df_train_Assessed['Correct_Attempt']=np.where((df['event_data'].str.contains('"correct":true'))&(((df['title'] != "Bird Measurer (Assessment)")&(df['event_code']==4100))|((df['title'] == "Bird Measurer (Assessment)")&(df['event_code']==4110))),1,0)
    df_train_acc=df_train_Assessed[df_train_Assessed['title'].isin(['Bird Measurer (Assessment)','Mushroom Sorter (Assessment)','Cauldron Filler (Assessment)','Chest Sorter (Assessment)','Cart Balancer (Assessment)'])].groupby(['installation_id','title','game_session'])['Incorrect_Attempt','Correct_Attempt'].sum().rename_axis(None, axis=1).reset_index()
    df_train_acc['Total_Attempts']=df_train_acc.apply(lambda x: x['Incorrect_Attempt'] + x['Correct_Attempt'], axis=1)

    
    df_train_acc['accuracy']=np.where(df_train_acc['Total_Attempts']>0,df_train_acc['Correct_Attempt']/ df_train_acc['Total_Attempts'],0)
    df_train_acc['accuracy_group']=np.where(df_train_acc['accuracy']==1,3,np.where(df_train_acc['accuracy']==.5,2,np.where(df_train_acc['accuracy']==0,0,1)))
    df_game_attempt=df.groupby(['installation_id','game_session'])['Incorrect_Game_Attempt','Correct_Game_Attempt'].sum().rename_axis(None, axis=1).reset_index()

    df_event_codes=pd.pivot_table(df_train_Assessed.loc[:,['installation_id','game_session','event_code']],index=['installation_id','game_session'],columns=['event_code'],fill_value=0,aggfunc=len).add_prefix('event_code_').rename_axis(None, axis=1).reset_index()
    df_final=pd.merge(pd.merge(df_train_acc,df_activity_weekend_phase_of_day,on=['installation_id','game_session'],how='right'),df_event_codes,on=['installation_id','game_session'],how='right')
    df_gametime=df.groupby(['installation_id','game_session'])['game_time','timestamp','event_count'].max().reset_index()
    df_final=pd.merge(df_final,df_gametime,on=['installation_id','game_session'],how='left')
    df_final=df_final.fillna(value=0)

    df_final=pd.merge(df_final,df.loc[df['world']!="NONE",['installation_id','game_session','world']].drop_duplicates(),on=['installation_id','game_session'],how='left')
    df_final=pd.merge(df_final,df_game_attempt,on=['installation_id','game_session'],how='left')
    df_final=df_final.fillna(value=0)
    return(df_final)


# In[ ]:


#Event Code 4020 Feature

train[(train.type=="Assessment") & (train.event_code==4020) ]


# In[ ]:


specs[specs.event_id=="5f0eb72c"]


# This event occurs when the player places a mushroom on one of the three stumps. It contains information about the mushroom that was placed, the correctness of the action, and where the placement occurred. This event is used to calculate accuracy and to diagnose player strategies and understanding.

# In[ ]:


def get_4020_acc(df):
     
    counter_dict = {'Cauldron Filler (Assessment)_4020_accuracy':0,
                    'Mushroom Sorter (Assessment)_4020_accuracy':0,
                    'Bird Measurer (Assessment)_4020_accuracy':0,
                    'Chest Sorter (Assessment)_4020_accuracy':0 }
        
    for e in ['Cauldron Filler (Assessment)','Bird Measurer (Assessment)','Mushroom Sorter (Assessment)','Chest Sorter (Assessment)']:
        
        Assess_4020 = df[(df.event_code == 4020) & (df.title==activities_map[e])]   
        true_attempts_ = Assess_4020['event_data'].str.contains('true').sum()
        false_attempts_ = Assess_4020['event_data'].str.contains('false').sum()

        measure_assess_accuracy_ = true_attempts_/(true_attempts_+false_attempts_) if (true_attempts_+false_attempts_) != 0 else 0
        counter_dict[e+"_4020_accuracy"] += (counter_dict[e+"_4020_accuracy"] + measure_assess_accuracy_) / 2.0
    
    return counter_dict


# In[ ]:


# Event_code 4025 Feature at Cauldron Filler Assesment

train[ (train.event_code==4025) & (train.title == 'Cauldron Filler (Assessment)')]


# In[ ]:


specs[specs.event_id=="91561152"]


# This event occurs when the player clicks on a filled bucket when prompted to select a bucket. It contains information about the bucket that was clicked and the correctness of the action. This event is used to calculate accuracy and to diagnose player strategies and understanding.

# In[ ]:


def calculate_accuracy(session):
    Assess_4025 = session[(session.event_code == 4025) & (session.title=='Cauldron Filler (Assessment)')]   
    true_attempts_ = Assess_4025['event_data'].str.contains('true').sum()
    false_attempts_ = Assess_4025['event_data'].str.contains('false').sum()

    accuracy_ = true_attempts_/(true_attempts_+false_attempts_) if (true_attempts_+false_attempts_) != 0 else 0


# In[ ]:




