#!/usr/bin/env python
# coding: utf-8

# ## Object
# - In this note book, we treat this task as regression task.
# - We use accuracy instead of accuracy_group.
# - We determine thresholds by cross validation.
# - We do not use accumulate feature.
# 
# ## Reference
# - https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-657027

# ## Imports

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split, GroupKFold
from sklearn.metrics import confusion_matrix
import os
import gc
from sklearn.utils import shuffle


# In[ ]:


titles = ['12 Monkeys', 'Treasure Map', 'Lifting Heavy Things', 'Crystal Caves - Level 2', 'Dino Drink', 'Air Show', 'Honey Cake', 'Scrub-A-Dub',           'Welcome to Lost Lagoon!', 'Heavy, Heavier, Heaviest', 'Rulers', 'Bottle Filler (Activity)', 'Fireworks (Activity)', 'Magma Peak - Level 1',           'Mushroom Sorter (Assessment)', 'Magma Peak - Level 2', 'Bubble Bath', 'Ordering Spheres', 'Leaf Leader', 'Sandcastle Builder (Activity)',           'Costume Box', 'Chicken Balancer (Activity)', 'Crystals Rule', 'Tree Top City - Level 1', 'Bug Measurer (Activity)', 'Chow Time', 'Bird Measurer (Assessment)',           'Watering Hole (Activity)', 'Happy Camel', 'Pan Balance', 'Crystal Caves - Level 1', 'Egg Dropper (Activity)', "Pirate's Tale", 'Cauldron Filler (Assessment)',           'Crystal Caves - Level 3', 'Tree Top City - Level 2', 'Balancing Act', 'Slop Problem', 'Dino Dive', 'All Star Sorting', 'Tree Top City - Level 3',           'Chest Sorter (Assessment)', 'Flower Waterer (Activity)', 'Cart Balancer (Assessment)']

events = [['27253bdc'], ['27253bdc'], ['27253bdc'], ['27253bdc'],           ['7f0836bf', '792530f8', '51311d7a', '6c517a88', 'a29c5338', '6f8106d9', '9ed8f6da', '77ead60d', '4d6737eb', '16dffff1', '4d911100', '1996c610', 'f806dc10', '5be391b5', 'ab4ec3a4', 'c6971acf', '74e5f8a7', 'e5734469', '89aace00'],          ['65abac75', '6f4bd64e', 'bcceccc6', '9b4001e4', 'd2659ab4', '58a0de5c', '06372577', 'f5b8c21a', 'e04fb33d', 'f28c589a', '28f975ea', '14de4c5d', '15ba1109', 'd88ca108', 'dcb1663e', 'dcb55a27', '7423acbc', '1575e76c', 'a1bbe385'],          ['27253bdc'], ['f7e47413', '4a09ace1', 'dcaede90', 'c1cac9a2', '6d90d394', '08fd73f3', 'ac92046e', '92687c59', '2b9272f4', '5a848010', '5c3d2b2f', '7040c096', '37c53127', '26fd2d99', 'd88e8f25', 'cf82af56', 'f71c4741', '73757a5e'],          ['27253bdc'], ['27253bdc'], ['27253bdc'], ['15a43e5b', '67439901', '47efca07', '5f5b2617', 'e9c52111', 'd3f1e122', 'bb3e370b', 'd2278a3b', '90efca10', 'df4940d3', 'b7530680'],          ['02a42007', 'beb0a7b9', '4901243f', '884228c8', 'b88f38da', 'e694a35b', 'f54238ee', '611485c5'], ['27253bdc'],           ['9d29771f', '7da34a02', 'c7128948', '83c6c409', 'fbaf3456', '5f0eb72c', '88d4a5be', '160654fd', '3dfd4aa4', 'db02c830', 'eb2c19cd', '28ed704e', '0d18d96c', 'a1e4395d', 'c74f40cd', '6c930e6e', '13f56524', '3bfd1a65', 'a52b92d5', 'a5be6304', '25fa8af4'],          ['27253bdc'], ['8f094001', '29a42aea', '0413e89d', '1beb320a', 'ecc36b7f', 'ad148f58', '8d84fa81', '3bb91dda', '99abe2bb', '6aeafed4', 'a0faea5d', '85de926c', '55115cbd', '5859dfb6', '857f21c0', 'c54cf6c5', '15eb4a7d', 'd06f75b5', '1340b8d7', '90ea0bac', '6f4adc4b', '895865f3', '1cf54632', '99ea62f3'],          ['27253bdc'], ['e5c9df6f', '53c6e11a', 'f32856e4', 'e57dd7af', '8ac7cce4', '3afde5dd', '3b2048ee', '01ca3a3c', '7dfe6d8a', '262136f4', 'fd20ea40', '763fc34e', 'b012cd7f', '33505eae', '2a512369', '86ba578b', '67aa2ada', '29f54413'],          ['84538528', '37937459', '5e812b27', '30df3273', 'c58186bf', '77261ab5', '9ee1c98c', 'b2dba42b', '1325467d', '1bb5fbdb'], ['27253bdc'],           ['cdd22e43', '56bcd38d', '4bb2f698', 'ea321fb1', '756e5507', '499edb7c', '46cd75b4', '84b0e0c8', '16667cc5', '85d1b0de'],           ['3ddc79c3', 'e720d930', 'a1192f43', '5154fc30', 'cc5087a3', '93edfe2e', '3babcb9b', '86c924c4', '3323d7e9', '7cf1bc53', '44cb4907', '8b757ab8', '5e3ea25a', '48349b14'],          ['27253bdc'], ['363c86c9', '565a3990', '8d748b58', '022b4259', '0a08139c', 'e79f3763', '2ec694de', 'c7f7f0e1', '71fe8f75'],          ['63f13dd7', '2230fab4', '47026d5f', '9e6b7fb5', 'cb6010f8', '6f445b57', 'cfbd47c8', 'd185d3ea', '0330ab6a', '0d1da71f', '7372e1a5', '4ef8cdd3', '19967db1', '7ec0c298', '56817e2b', 'f93fc684', '7d093bf9'],          ['a76029ee', '7525289a', '3393b68b', 'e37a2b78', '1375ccb7', 'ad2fc29c', '45d01abe', 'a16a373e', 'bdf49a58', 'd38c2fd7', '4a4c3d21', '070a5291', '6077cc36', 'ec138c1c', 'f56e0afc', '8fee50e2', '51102b85', 'f6947f54', '17113b36', '731c0cbe'],           ['71e712d8', '1b54d27f', 'e7e44842', '49ed92e9', 'c952eb01', '2fb91ec1', 'a6d66e51', 'd2e9262e', 'f50fc6c1', 'bd701df8', 'e64e2cfd'],          ['d51b1749', '69fdac0a', '46b50ba8', 'a2df0760', '05ad839b', '37db1c2f', 'a8a78786', '3bf1cf26', 'a7640a16', '6bf9e3e1', '8d7e386c', '8af75982', '3d8c61b0', '1af8be29', 'c7fe2a55', 'abc5811c', 'd9c005dd', 'c2baf0bd', 'c189aaf2', '3bb91ced', '0ce40006', '36fa3ebe'],           ['0086365d', 'a5e9da97', 'e4d32835', 'f3cd5473', '907a054b', 'c51d8688', 'cf7638f3', '9c5ef70c', '804ee27f', '2a444e03', 'a592d54e', '6cf7d25c', 'e7561dd2', 'bc8f2793', '1c178d24', '250513af', 'e080a381', '15f99afc'],           ['27253bdc'], ['7fd1ac25', '736f9581', '9b23e8ee', '461eace6', '9e34ea74', '08ff79ad', '4c2ec19f', '7ab78247', 'b80e5e84'], ['27253bdc'],          ['3ee399c3', '2dcad279', 'b5053438', '532a2afb', '5290eab1', '91561152', '90d848e0', '923afab1', '37ee8496', '392e14df', '28520915', '77c76bc5', '9554a50b', '5348fd84', '04df9b66', 'd3268efa', '3edf6747', '30614231', '2b058fe3'],          ['27253bdc'], ['27253bdc'], ['27253bdc'], ['27253bdc'],           ['ab3136ba', 'd3640339', '9de5e594', '709b1251', '7d5c30a2', 'e3ff61fb', '832735e1', 'c0415e5c', '00c73085', '87d743c1', '119b5b02', '28a4eb9a', '29bdd9ba', '6088b756', '7961e599', '76babcde'],          ['2c4e6db0', '587b5989', '6043a2b4', 'ca11f653', '1cc7cfca', 'daac11b0', '26a5a3dd', '9e4c8c7b', 'b7dc8128', '4b5efe37', 'd45ed6a1', '1f19558b', '363d3849', 'b1d5101d', 'c277e121', 'b120f2ac', 'd02b7a8e', '2dc29e21'],           ['27253bdc'], ['3afb49e6', '38074c54', '155f62a4', '9ce586dd', 'bfc77bd6', 'bd612267', '93b353f2', '222660ff', '3ccd3f02', 'a8efe47b', '5b49460a', 'cb1178ad', 'df4fe8b6', 'ea296733', '3d0b9317', '0db6d71d', 'e4f1efe6', '562cec5f', '3dcdda7f'],          ['47f43a44', '3a4be871', 'a44b10dc', '598f4598', 'de26c3a6', '9b01374f', '56cd3b43', 'fcfdffb6', 'bbfe0445', '5d042115'],           ['7ad3efc6', '5e109ec3', 'b74258a0', '31973d56', '65a38bf7', 'd122731b', '5de79a6a', 'ecc6157f', '795e4a37', 'b2e5b0f1', '5c2f29ca', 'acf5c23f', '3d63345e', '4e5fc6f5', '828e68f9', '9d4e7b25', 'a8876db3', 'ecaab346']]


# In[ ]:


title_title = ['Cart Balancer (Assessment)', 'Chest Sorter (Assessment)', 'Cauldron Filler (Assessment)', 'Bird Measurer (Assessment)', 'Mushroom Sorter (Assessment)', 'Chicken Balancer (Activity)', 'Egg Dropper (Activity)', 'Sandcastle Builder (Activity)', 'Bottle Filler (Activity)', 'Watering Hole (Activity)', 'Bug Measurer (Activity)', 'Fireworks (Activity)', 'Flower Waterer (Activity)', 'Crystal Caves - Level 3', 'Honey Cake', 'Lifting Heavy Things', 'Crystal Caves - Level 2', 'Heavy, Heavier, Heaviest', 'Balancing Act', 'Crystal Caves - Level 1', 'Magma Peak - Level 1', 'Slop Problem', 'Magma Peak - Level 2', 'Welcome to Lost Lagoon!', 'Costume Box', "Pirate's Tale", 'Tree Top City - Level 2', 'Tree Top City - Level 3', 'Treasure Map', '12 Monkeys', 'Tree Top City - Level 1', 'Ordering Spheres', 'Rulers', 'Happy Camel', 'Leaf Leader', 'Chow Time', 'Pan Balance', 'Scrub-A-Dub', 'Bubble Bath', 'Dino Dive', 'Dino Drink', 'Air Show', 'All Star Sorting', 'Crystals Rule']
title_type = ['Assessment', 'Assessment', 'Assessment', 'Assessment', 'Assessment', 'Activity', 'Activity', 'Activity', 'Activity', 'Activity', 'Activity', 'Activity', 'Activity', 'Clip', 'Clip', 'Clip', 'Clip', 'Clip', 'Clip', 'Clip', 'Clip', 'Clip', 'Clip', 'Clip', 'Clip', 'Clip', 'Clip', 'Clip', 'Clip', 'Clip', 'Clip', 'Clip', 'Clip', 'Game', 'Game', 'Game', 'Game', 'Game', 'Game', 'Game', 'Game', 'Game', 'Game', 'Game']
title_world = ['CRYSTALCAVES', 'CRYSTALCAVES', 'MAGMAPEAK', 'TREETOPCITY', 'TREETOPCITY', 'CRYSTALCAVES', 'CRYSTALCAVES', 'MAGMAPEAK', 'MAGMAPEAK', 'MAGMAPEAK', 'TREETOPCITY', 'TREETOPCITY', 'TREETOPCITY', 'CRYSTALCAVES', 'CRYSTALCAVES', 'CRYSTALCAVES', 'CRYSTALCAVES', 'CRYSTALCAVES', 'CRYSTALCAVES', 'CRYSTALCAVES', 'MAGMAPEAK', 'MAGMAPEAK', 'MAGMAPEAK', 'NONE', 'TREETOPCITY', 'TREETOPCITY', 'TREETOPCITY', 'TREETOPCITY', 'TREETOPCITY', 'TREETOPCITY', 'TREETOPCITY', 'TREETOPCITY', 'TREETOPCITY', 'CRYSTALCAVES', 'CRYSTALCAVES', 'CRYSTALCAVES', 'CRYSTALCAVES', 'MAGMAPEAK', 'MAGMAPEAK', 'MAGMAPEAK', 'MAGMAPEAK', 'TREETOPCITY', 'TREETOPCITY', 'TREETOPCITY']
title_title_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
title_type_id = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
title_world_id = [0, 0, 1, 2, 2, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
title = pd.DataFrame(index=range(len(title_title)))
title['title'] = title_title
title['type'] = title_type
title['world'] = title_world
title['title_id'] = title_title_id
title['type_id'] = title_type_id
title['world_id'] = title_world_id


# # Make Feature

# In[ ]:


def make_test_label(submit_input, test):
    submit = submit_input.copy()
    submit['game_session'] = 'game_session'
    for i in range(len(submit)):
        idx = submit.loc[i,'installation_id']
        _test = test.loc[test['installation_id']==idx,:].copy()
        _test = (_test.sort_values('timestamp')).reset_index(drop=True)
        submit.loc[i,'game_session'] = _test.loc[_test.index[len(_test.index)-1],'game_session']
    return submit

def make_test_label2(submit_input, test):
    _test = test.loc[test['event_count']==1,:].copy()
    _test = _test.loc[~_test['installation_id'].duplicated(keep='last'),['installation_id','game_session','timestamp']]
    submit = submit_input.copy()
    submit = pd.merge(submit,_test,on='installation_id',how='left')
    return submit

def previous_index_col(df_input,col):
    df = df_input.copy()
    for c in col:
        df['previous_'+c] = df[c]
        df.loc[df.index[1:],'previous_'+c] = np.array(df.loc[df.index[:len(df.index)-1],c])
    df = df.loc[df.index[1:],:]
    return df


# # Feature 1: Previous Game Event
# - Number of event_id in previous game session

# In[ ]:


def feature_previous_game_event(df_, title, spec, label,Train=True):
    df = df_.copy()
    title_dic = dict(zip(title['title'],title['title_id']))
    type_dic = dict(zip(title['type'],title['type_id']))
    world_dic = dict(zip(title['world'],title['world_id']))
    df['title'] = df['title'].apply(lambda x:title_dic[x] if x in title_dic.keys() else -1)
    df['type'] = df['type'].apply(lambda x:type_dic[x] if x in type_dic.keys() else -1)
    df['world'] = df['world'].apply(lambda x:world_dic[x] if x in world_dic.keys() else -1)
    event_dic = dict(zip(spec['event_id'],spec.index))
    df['event_id'] = df['event_id'].map(lambda x:event_dic[x])
    play = df[df['event_count']==1].copy()
    for e in tqdm(range(len(spec))):
        _df = df[df['event_id']==e]
        gr = _df.groupby('game_session').agg({'event_id':'count'})
        gr = gr.rename(columns={'event_id':'e{}'.format(str(e).zfill(3))})
        play = pd.merge(play,gr,on='game_session',how='left')
    df = df[df['event_count']==1].reset_index(drop=True)
    gttw = df.loc[:,['game_session','title','type','world','installation_id']].copy()
    df = previous_index_col(df,['installation_id','game_session'])
    df = df.loc[df['installation_id']==df['previous_installation_id'],:].reset_index(drop=True)
    df = df.drop(['event_id','timestamp','event_data','game_time','event_count','event_code'],axis=1)
    if Train:
        label = label.drop(['installation_id','title'],axis=1)
    else:
        label = label.drop(['installation_id'],axis=1)
    play = play.drop(['event_id','timestamp','event_data','event_count','event_code','game_time','installation_id'],axis=1)
    play = play.rename(columns={'title':'previous_title','type':'previous_type','world':'previous_world'})
    play = play.rename(columns={'game_session':'previous_game_session'})
    feature = pd.merge(label,df,on='game_session',how='left')
    feature = pd.merge(feature,play,on='previous_game_session',how='left')
    if Train:
        feature = feature.drop(['num_correct','num_incorrect','accuracy','accuracy_group','previous_installation_id','title','world','type','installation_id','timestamp'],axis=1)
    else:
        feature = feature.drop(['previous_installation_id','title','world','type','installation_id','accuracy_group','timestamp'],axis=1)
    feature = pd.merge(label,feature,on='game_session',how='left')
    feature = pd.merge(feature,gttw,on='game_session',how='left')
    for i in range(386):
        feature['e{}'.format(str(i).zfill(3))] = feature['e{}'.format(str(i).zfill(3))].fillna(0)
    feature['eall'] = 0
    for i in range(386):
        feature['eall']+=feature['e{}'.format(str(i).zfill(3))]
    for i in range(386):
        feature['e{}'.format(str(i).zfill(3))]/=feature['eall']+10
    return feature


# # Feature 2: Previous Title Event
# - Number of event_id in previous game session of each title

# In[ ]:


def feature_previous_title_event(df_,label_,titles,events):
    df = df_.copy()
    label = label_.copy()
    insta_v = label['installation_id'].unique()
    df_insta = df.loc[df['installation_id'].isin(insta_v),:].copy()
    title_df_v = []
    for idx,title in tqdm(enumerate(titles)):
        df_insta_title = df_insta.loc[df_insta['title']==title,:]
        title_event_id = events[idx]
        v = []
        title_df = pd.DataFrame(columns=[title+'_'+x+'_nm' for x in title_event_id])
        for idx,(game,df) in enumerate(df_insta_title.groupby('game_session')):
            feature = []
            for e_id in title_event_id:
                feature.append((df['event_id']==e_id).sum())
            title_df.loc[game,:]=feature
        title_df_v.append(title_df)
    df_event_cnt1 = df_insta.loc[df_insta['event_count']==1,:]
    for idx,title in tqdm(enumerate(titles)):
        title_df = title_df_v[idx]
        df_event_cnt1_title = df_event_cnt1.loc[df_event_cnt1['title']==title,:]
        label[title] = None
        cnt=0
        for jdx in label.index:
            insta_id, time = label.loc[jdx,['installation_id','timestamp']]
            df_event_cnt1_title_insta = df_event_cnt1_title.loc[df_event_cnt1_title['installation_id']==insta_id,:]
            df_event_cnt1_title_insta_before = df_event_cnt1_title_insta.loc[df_event_cnt1_title_insta['timestamp']<time,:]
            if len(df_event_cnt1_title_insta_before)>0:
                previous_index = df_event_cnt1_title_insta_before.index[len(df_event_cnt1_title_insta_before)-1]
                previous_game = df_event_cnt1_title_insta_before.loc[previous_index,'game_session']
                label.loc[jdx,title]=previous_game
    feature = pd.merge(label,df_event_cnt1.drop(['installation_id','title','timestamp'],axis=1),on='game_session',how='left').copy()
    for idx,title in enumerate(titles):
        title_df = title_df_v[idx]
        feature = pd.merge(feature,title_df,left_on=title,right_index=True,how='left')
    return feature


# # Make Train Feature

# In[ ]:


train = pd.read_csv('../input/data-science-bowl-2019/train.csv')
train['timestamp'] = pd.to_datetime(train['timestamp'])
train_start = train.loc[train['event_count']==1,['game_session','timestamp']]
train_label = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
train_label = pd.merge(train_label, train_start, on='game_session', how='left')
del train_start
gc.collect()
spec = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
print('Train Data Load Done')
print('Start Make Train feature_previous_game_event')
feature_previous_game_event_train = feature_previous_game_event(train, title, spec, train_label,Train=True)
print('Start Make Train feature_previous_title_event')
feature_previous_title_event_train = feature_previous_title_event(train,train_label,titles,events)
feature_train = pd.merge(feature_previous_game_event_train,feature_previous_title_event_train.drop(['installation_id','title','num_correct','num_incorrect','accuracy','accuracy_group',
                         'event_id','timestamp','event_data','event_count', 'event_code', 'game_time', 'type', 'world']+titles,axis=1),on='game_session',how='left')
del train
del feature_previous_title_event_train
del feature_previous_geme_event_train
gc.collect()
feature_train.head()


# # Make Test Feature

# In[ ]:


test = pd.read_csv('../input/data-science-bowl-2019/test.csv')
test['timestamp'] = pd.to_datetime(test['timestamp'])
submit = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
print('Test Data Load Done')
print('Start Make Test Label')
test_label = make_test_label2(submit,test)
print('Start Make Test feature_previous_game_event')
feature_previous_game_event_test = feature_previous_game_event(test,title,spec,test_label,False)
print('Start Make Test feature_previous_title_event')
feature_previous_title_event_test = feature_previous_title_event(test, test_label, titles, events)
feature_test = pd.merge(feature_previous_game_event_test,feature_previous_title_event_test.drop(['installation_id','accuracy_group','event_id',                        'timestamp','event_data','event_count', 'event_code', 'game_time', 'type', 'world']+titles,axis=1),on='game_session',how='left')
gc.collect()
feature_test = feature_test.reset_index(drop=True)
del test
del feature_previous_game_event_test
del feature_previous_title_event_test
gc.collect()
feature_test.head()


# # Traing

# In[ ]:


dataset = feature_train.drop(['game_session','num_correct','num_incorrect','previous_game_session','previous_title','previous_type','previous_world','timestamp'],axis=1)
float_col = [x for x in dataset.columns[:len(dataset.columns)-1] if not x in ['installation_id','accuracy_group']]
dataset[float_col] = dataset[float_col].astype(np.float)
dataset['accuracy'] = dataset['accuracy']*3
dataset = dataset.rename(columns={'accuracy_group':'y'})
dataset['y'] = dataset['accuracy']
dataset = dataset.reset_index(drop=True)
W = np.array([[0,1,4,9],[1,0,1,4],[4,1,0,1],[9,4,1,0]])/9
print('dataset size:',len(dataset))
para = {'objective':'regression','boosting':'gbdt','metric':{'l2'},'fraction_rate':0.7}
gkf1 = GroupKFold(n_splits=5)
gr1 = dataset['installation_id']
dataset['predict'] = 0
models = []
for c in dataset.columns:
    _c = c.replace(',','')
    if _c != c:
        dataset = dataset.rename(columns={c:_c})
dataset, gr1 = shuffle(dataset,gr1,random_state=42)
for idx,(tr1_idx,test_idx) in enumerate(gkf1.split(dataset,groups=gr1)):
    tr1_X = (dataset.loc[dataset.index[tr1_idx],:]).drop(['y','predict','installation_id','accuracy'],axis=1).copy()
    test_X = (dataset.loc[dataset.index[test_idx],:]).drop(['y','predict','installation_id','accuracy'],axis=1).copy()
    tr1_y = dataset.loc[dataset.index[tr1_idx],'accuracy'].copy()
    test_y = dataset.loc[dataset.index[test_idx],'accuracy'].copy()
    gkf2 = GroupKFold(n_splits=5)
    gr2 = gr1[gr1.index[tr1_idx]]
    for jdx,(tr2_idx,val_idx) in enumerate(gkf2.split(tr1_X,tr1_y,gr2)):
        tr2_X, val_X = tr1_X.loc[tr1_X.index[tr2_idx],:],tr1_X.loc[tr1_X.index[val_idx],:]
        tr2_y, val_y = tr1_y[tr1_y.index[tr2_idx]],tr1_y[tr1_y.index[val_idx]]
        train_lgb = lgb.Dataset(tr2_X, tr2_y)
        val_lgb = lgb.Dataset(val_X, val_y, reference=train_lgb)
        model = lgb.train(para,train_lgb,num_boost_round=50,valid_sets=[train_lgb,val_lgb],early_stopping_rounds=10)
        predict = model.predict(test_X)
        dataset.loc[dataset.index[test_idx],'predict'] += predict/5
        models.append(model)
        if jdx==0:
            models.append(model)


# # Determine Threshold
# - Thresholds are determined by CV.

# In[ ]:


def qwk3(a1, a2, max_rat=3):
    assert(len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e

def estimate(dataset):
    q1s = np.linspace(0.3,1.2,10)
    q2s = np.linspace(0.5,2.2,18)
    q3s = np.linspace(1.0,2.2,13)
    m = 0
    q1r,q2r,q3r = -1,-1,-1
    for q1 in q1s:
        for q2 in q2s:
            for q3 in q3s:
                if q2 > q1 and q3 > q2:
                    dataset['predict2'] = 0
                    dataset['predict2'] += (dataset['predict']>q1).astype(np.int)
                    dataset['predict2'] += (dataset['predict']>q2).astype(np.int)
                    dataset['predict2'] += (dataset['predict']>q3).astype(np.int)
                    dataset['predict2'] = dataset['predict2'].astype(np.int)
                    score = qwk3(dataset['y'],dataset['predict2'])
                    if m < score:
                        print(q1,q2,q3,qwk3(dataset['y'],dataset['predict2']))
                        m = score
                        q1r,q2r,q3r = q1,q2,q3
    return q1r,q2r,q3r

q1,q2,q3 = estimate(dataset)


# # Prediction

# In[ ]:


test_df = feature_test.copy()
test_df = test_df.drop(['game_session','installation_id','accuracy_group','previous_game_session',                        'previous_title','previous_type','previous_world','timestamp'],axis=1)
float_col = [x for x in dataset.columns[:len(dataset.columns)-1] if not x in ['installation_id','accuracy_group']]

test_df = test_df.astype(np.float)
for c in test_df.columns:
    _c = c.replace(',','')
    if _c != c:
        test_df = test_df.rename(columns={c:_c})
for k, model in enumerate(models):
    if k==0:
        predict = model.predict(test_df)/len(models)
    else:
        predict += model.predict(test_df)/len(models)
test_df.head()


# In[ ]:


test_df['predict2'] = 0
test_df['predict2'] += (predict>q1).astype(np.int)
test_df['predict2'] += (predict>q2).astype(np.int)
test_df['predict2'] += (predict>q3).astype(np.int)
test_df['predict2'] = test_df['predict2'].astype(np.int)
test_df.head()


# In[ ]:


import matplotlib.pyplot as plt
plt.hist(test_df['predict2'],bins=20)
plt.show()


# In[ ]:


submit = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
submit['accuracy_group'] = test_df['predict2']
submit.head()


# In[ ]:


submit.to_csv('submission.csv', index=False)

