#!/usr/bin/env python
# coding: utf-8

# ## LSTM RNN with feature engineering
# Simple LSTM RNN with cumulative feature aggregation for each game session. Each training sample represents a sequence of 250 latest game sessions for specific installation_id. 
# 
# **Features include:** 
# * 	Count game sessions for each world:
# 	*'activities_by_world_{world}'*, *'{world}_Clip'*, *'{world}_Activity'*, *'{world}_Game'*, *'{world}_Assessment'*
# * 	Count game sessions for each session type (Clip, Activity, Game, Assessment): *'activities_by_type_{type}'*
# * 	Time since first game session, time since previous game session
# * 	Mean, median, std for deltas between *'game_time'* within session: *'game_time_mean'*, *'game_time_median'*, *'game_time_std'*, 
# *   Session day of week and hour: *'time_start_dow'*, *'time_start_hour'*
# * 	Evaluation event stats for each world:
# 	*'{world}_eval_invalid'*, *'{world}_eval_valid'*, *'{world}_eval_sum'*, *'{world}_eval_ratio'*
# * 	Total evaluation events stats across all worlds:
# 	*'Total_eval_invalid'*, *'Total_eval_valid'*, *'Total_eval_sum'*, *'Total_eval_ratio'*
# 	
# The model was inspired by the following kernel: [bowl-lstm-prediction](https://www.kaggle.com/nikitagrec/bowl-lstm-prediction)

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, Masking, Bidirectional, Dropout
import datetime
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import gc
from datetime import datetime
from keras.callbacks import *
import keras.backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy as sp
from functools import partial
from sklearn import metrics
from collections import Counter


# In[ ]:


input_dir = '../input/data-science-bowl-2019/'

def create_features(df, world_names, type_names, event_code_idx_lookup, event_codes, win_codes):
    world_by_idx = {k:v for k,v in zip(world_names, range(len(world_names)))}
    type_by_idx = {k:v for k,v in zip(type_names, range(len(type_names)))}

    new_col_names = ['activities_by_world_NONE', 'NONE_Clip', 'NONE_Activity', 'NONE_Game', 'NONE_Assessment', 
         'activities_by_world_MAGMAPEAK', 'MAGMAPEAK_Clip', 'MAGMAPEAK_Activity', 'MAGMAPEAK_Game', 'MAGMAPEAK_Assessment', 
         'activities_by_world_CRYSTALCAVES', 'CRYSTALCAVES_Clip', 'CRYSTALCAVES_Activity', 'CRYSTALCAVES_Game', 'CRYSTALCAVES_Assessment', 
         'activities_by_world_TREETOPCITY', 'TREETOPCITY_Clip', 'TREETOPCITY_Activity', 'TREETOPCITY_Game', 'TREETOPCITY_Assessment', 
         'activities_by_type_Clip', 'activities_by_type_Activity', 'activities_by_type_Game', 'activities_by_type_Assessment', 
         'first_install_event_time', 'time_since_previous_game']
    new_col_names += ['time_start', 'game_time_mean', 'game_time_median', 'game_time_std', 'time_start_dow', 'time_start_hour', 'distinct_event_codes']    
    new_col_names += event_codes
    new_col_names += ['NONE_eval_invalid', 'NONE_eval_valid', 'NONE_eval_sum', 'NONE_eval_ratio',
          'MAGMAPEAK_eval_invalid', 'MAGMAPEAK_eval_valid', 'MAGMAPEAK_eval_sum', 'MAGMAPEAK_eval_ratio',
          'TREETOPCITY_eval_invalid', 'TREETOPCITY_eval_valid', 'TREETOPCITY_eval_sum', 'TREETOPCITY_eval_ratio',
          'CRYSTALCAVES_eval_invalid', 'CRYSTALCAVES_eval_valid', 'CRYSTALCAVES_eval_sum', 'CRYSTALCAVES_eval_ratio',
          'Total_eval_invalid', 'Total_eval_valid', 'Total_eval_sum', 'Total_eval_ratio']
    
    df_new = df.reindex(columns=[*df.columns.tolist(), *new_col_names], fill_value=0, copy=True)
    
    #table to aggregate activities stats per world/type
    sess_agg = np.zeros((len(world_names), len(type_names)), dtype=np.int)
    
    #table to aggregate evaluation performance stats per world
    eval_agg = np.zeros((len(world_names), 2), dtype=np.int)
    
    idx_agg = list()
    new_values_agg = list()
    second = np.timedelta64(1, 's')
    
    for install_id, install in df.groupby('installation_id', sort=False):
        first_install_event_time = install['timestamp'].values[0] 
        groupby_session = install.groupby('game_session', sort=False)
        previous_sess_time = first_install_event_time
        
        # reset stats for each new install_id
        sess_agg.fill(0)
        eval_agg.fill(0)
        
        for sess_id, sess in groupby_session:            
            new_values = list()
            
            # extract activities stats per world
            for w in world_names:
                world_stats = sess_agg[world_by_idx[w], :]
                #activities_by_world_{w}
                new_values.append(np.sum(world_stats))   
                #['{w}_Clip', '{w}_Activity', '{w}_Game', '{w}_Assessment']
                new_values.extend(world_stats)                
            
            # extract activities stats per activity type
            for t in type_names:
                # 'activities_by_type_{t}'
                new_values.append(np.sum(sess_agg[:, type_by_idx[t]]))                                  
                
            sess_world = sess['world'].iloc[0]        
            sess_agg[world_by_idx[sess_world], type_by_idx[sess['type'].iloc[0]]] += 1            
            
            new_values.append((sess['timestamp'].values[0] - first_install_event_time)/second)            
            new_values.append((sess['timestamp'].values[0] - previous_sess_time)/second)

            previous_sess_time = sess['timestamp'].values[-1]            
            
            time_start = sess['timestamp'].values[0]
            new_values.append(time_start)
            
            game_time_stats = sess['game_time'].diff().agg(['mean','median', 'std']).fillna(0)
            new_values.append(game_time_stats.loc['mean'])
            new_values.append(game_time_stats.loc['median'])
            new_values.append(game_time_stats.loc['std'])
            dt_start = datetime.utcfromtimestamp(time_start.astype(int) * 1e-9)
            new_values.append(dt_start.weekday())
            new_values.append(dt_start.hour)
            
            # count event codes per session
            event_counts = Counter(sess['event_code'])
            ec = np.repeat(0, len(event_codes))
            for key,val in event_counts.items():
                ec[event_code_idx_lookup[key]] = val
            new_values.append(len(event_counts.keys()))
            new_values.extend(ec)
            
            # calculate evaluations
            # ['{w}_eval_invalid', '{w}_eval_valid', '{w}_eval_sum', '{w}_eval_ratio']
            for w in world_names:                
                evals_world = eval_agg[world_by_idx[w], :]
                new_values.extend(evals_world)
                evals_sum = np.sum(evals_world)
                new_values.append(evals_sum)
                new_values.append(np.nan_to_num(evals_world[1]/evals_sum))
                        
            # total evaluations across all worlds
            # 'Total_eval_invalid', 'Total_eval_valid', 'Total_eval_sum', 'Total_eval_ratio'
            evals_total = np.sum(eval_agg, axis=0)
            new_values.extend(evals_total)
            evals_total_sum = np.sum(evals_total)
            new_values.append(evals_total_sum)
            new_values.append(np.nan_to_num(evals_total[1]/evals_total_sum))
            
            all_evaluations = sess[sess['event_code'] == win_codes[sess['title'].iloc[0]]]            
            eval_agg[world_by_idx[sess_world], 0] += all_evaluations['event_data'].str.contains('false').sum()
            eval_agg[world_by_idx[sess_world], 1] += all_evaluations['event_data'].str.contains('true').sum()
            
            idx_agg.append(sess.index[-1])
            new_values_agg.append(new_values)     
        
    df_new.set_value(idx_agg, new_col_names, new_values_agg)
    return df_new


# In[ ]:


test_df_event_codes = pd.read_csv(input_dir+'test.csv', low_memory=True, usecols=['event_code'], dtype={'event_code': str})
train_df_temp = pd.read_csv(input_dir+'train.csv', low_memory=True, usecols=['event_code', 'type', 'world', 'title'], dtype={'event_code': str})
event_codes = [str(i) for i in sorted(set(test_df_event_codes['event_code'].unique()).union(set(train_df_temp['event_code'].unique())))]
event_code_idx_lookup = {v:i for v,i in zip(event_codes, range(len(event_codes)))}

# map title to evaluation event code
win_code = dict(zip(train_df_temp['title'].unique(), np.repeat('4100', len(train_df_temp['title'].unique()))))
win_code['Bird Measurer (Assessment)'] = '4110'

world_names = train_df_temp['world'].unique()
type_names = train_df_temp['type'].unique()
del  train_df_temp, test_df_event_codes


# In[ ]:


def process_in_batches(filename, installation_ids, event_codes, win_codes):
    merges = []
    
    for b in np.array_split(installation_ids, 5):
        df = pd.read_csv(input_dir+filename, parse_dates=['timestamp'], low_memory=True, 
                              dtype={'event_count': int, 'game_time': float, 'event_code': str}, 
                              usecols=['game_session','timestamp', 'installation_id','event_count','event_code',
                                       'game_time','title','type','world', 'event_data'])
        df_batch = df[df['installation_id'].isin(b)]
        df_batch.sort_values('timestamp', inplace=True)
        df_batch_agg = create_features(df_batch, world_names, type_names, event_code_idx_lookup, event_codes, win_codes)
        df_batch_agg_tail = df_batch_agg.groupby(['installation_id', 'game_session']).tail(1)
        merges.append(df_batch_agg_tail)
        
    agg_df = merges[0]
    for i in merges[1:]:    
        agg_df = agg_df.append(i)
    return agg_df

installation_ids_test = pd.read_csv(input_dir+'test.csv', usecols=['installation_id'])['installation_id'].unique()
test_sess_agg_df = process_in_batches('test.csv', installation_ids_test, event_codes, win_code)


# In[ ]:


train_df = pd.read_csv(input_dir+ 'train.csv', parse_dates=['timestamp'], low_memory=True, 
                       dtype={'event_count': int, 'game_time': float, 'event_code': str}, 
                       usecols=['game_session','timestamp', 'installation_id','event_count','event_code','game_time',
                                                                             'title','type','world'])
train_df.sort_values('timestamp', inplace=True)

labels = pd.read_csv(input_dir+'train_labels.csv', low_memory=True, usecols=['installation_id', 'game_session', 'accuracy_group'])


# In[ ]:


def filter_outliers(df):
    vc = df.groupby(['installation_id']).apply(lambda x: len(x))
    print('installation ids: ' + str(len(vc)))
    installation_ids_inclue = vc[vc<15000].index.values
    print('installation ids to inclue: ' + str(len(installation_ids_inclue)))
    return df[df['installation_id'].isin(installation_ids_inclue)]

train_df = filter_outliers(train_df)

print(len(train_df))
i_ids = train_df.groupby('installation_id').tail(1).merge(labels, how='inner', on='installation_id')['installation_id'].unique()
train_df = train_df[train_df['installation_id'].isin(i_ids)]
print(len(train_df))


# In[ ]:


installation_ids_train = train_df['installation_id'].unique()
del train_df
train_sess_agg_df = process_in_batches('train.csv', installation_ids_train, event_codes, win_code)
game_sessions_with_label = set(train_sess_agg_df
                               .merge(labels, how='inner', on=['installation_id', 'game_session'])['game_session']
                               .unique()
                               )


# In[ ]:


# trim installation_id so that it ends with assessment with label
def trim_train_sequences(x, game_sessions_with_label):
    x.reset_index(inplace=True, drop=True)    
    assessments = x[(x['type'] == 'Assessment') & (x['game_session'].isin(game_sessions_with_label))]    
    if len(assessments) == 0:
        return pd.DataFrame()
    else:
        idx = np.random.choice(len(assessments), 1)[0]  
        return x.iloc[:assessments.index[idx] + 1]
    
game_stats_trimmed_train = train_sess_agg_df    .groupby('installation_id')    .apply(lambda x: trim_train_sequences(x, game_sessions_with_label))    .reset_index(drop=True)

del train_sess_agg_df


# In[ ]:


def ohe(train, test, target, features):
    if not train.columns.isin([target]).any():
        return train    
    
    train_objs_num = len(train)
    dataset = pd.concat(objs=[train[target], test[target]], axis=0, copy=False)
    dataset_preprocessed = pd.get_dummies(dataset, prefix= '{}_'.format(target))
    train_preprocessed = dataset_preprocessed[:train_objs_num]
    test_preprocessed = dataset_preprocessed[train_objs_num:]    
    
    merge = train.merge(train_preprocessed, left_index=True, right_index=True)
    train_result = merge[merge.columns[~merge.columns.isin([target])]]
    
    merge = test.merge(test_preprocessed, left_index=True, right_index=True)
    test_result = merge[merge.columns[~merge.columns.isin([target])]]

    features.extend(train_preprocessed.columns.values)
    return train_result, test_result

def mean_encoding(train_df, test_df, target):
    train_df[target] = (train_df[target]-train_df[target].mean())/train_df[target].std()
    test_df[target] = (test_df[target]-test_df[target].mean())/test_df[target].std()      


# In[ ]:


mask = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
mask_cols = ['event_count', 'game_time', 'game_time_mean', 'game_time_median', 'game_time_std', '2000', '2010', '2020', '2025', '2030', '2035', '2040', '2050', '2060', '2070', '2075', '2080', '2081', '2083', '3010', '3020', '3021', '3110', '3120', '3121', '4010', '4020', '4021', '4022', '4025', '4030', '4031', '4035', '4040', '4045', '4050', '4070', '4080', '4090', '4095', '4100', '4110', '4220', '4230', '4235', '5000', '5010', 'distinct_event_codes']

# mask aggregation stats in train for last session in each installation_id
tile = np.tile(mask, (len(game_stats_trimmed_train.groupby('installation_id').tail(1).index), 1))
frame = pd.DataFrame(tile, columns=mask_cols, index= game_stats_trimmed_train.groupby('installation_id').tail(1).index)
game_stats_trimmed_train.update(frame)


# In[ ]:


features = ['event_count', 'game_time', 'game_time_mean',
                   'game_time_median', 'game_time_std', 'time_start_dow',
                   'time_start_hour', '2000', '2010', '2020', '2025', '2030', '2035',
                   '2040', '2050', '2060', '2070', '2075', '2080', '2081', '2083', '3010',
                   '3020', '3021', '3110', '3120', '3121', '4010', '4020', '4021', '4022',
                   '4025', '4030', '4031', '4035', '4040', '4045', '4050', '4070', '4080',
                   '4090', '4095', '4100', '4110', '4220', '4230', '4235', '5000', '5010',
                   'distinct_event_codes', 
            'activities_by_world_NONE', 'NONE_Clip', 'NONE_Activity', 'NONE_Game', 'NONE_Assessment', 
            'activities_by_world_MAGMAPEAK', 'MAGMAPEAK_Clip', 'MAGMAPEAK_Activity', 'MAGMAPEAK_Game', 'MAGMAPEAK_Assessment', 
            'activities_by_world_CRYSTALCAVES', 'CRYSTALCAVES_Clip', 'CRYSTALCAVES_Activity', 'CRYSTALCAVES_Game', 'CRYSTALCAVES_Assessment', 
            'activities_by_world_TREETOPCITY', 'TREETOPCITY_Clip', 'TREETOPCITY_Activity', 'TREETOPCITY_Game', 'TREETOPCITY_Assessment', 
            'activities_by_type_Clip', 'activities_by_type_Activity', 'activities_by_type_Game', 'activities_by_type_Assessment', 
            'first_install_event_time', 'time_since_previous_game', 
            'NONE_eval_invalid', 'NONE_eval_valid', 'NONE_eval_sum', 'NONE_eval_ratio',
            'MAGMAPEAK_eval_invalid', 'MAGMAPEAK_eval_valid', 'MAGMAPEAK_eval_sum', 'MAGMAPEAK_eval_ratio',
            'TREETOPCITY_eval_invalid', 'TREETOPCITY_eval_valid', 'TREETOPCITY_eval_sum', 'TREETOPCITY_eval_ratio',
            'CRYSTALCAVES_eval_invalid', 'CRYSTALCAVES_eval_valid', 'CRYSTALCAVES_eval_sum', 'CRYSTALCAVES_eval_ratio',
            'Total_eval_invalid', 'Total_eval_valid', 'Total_eval_sum', 'Total_eval_ratio']
train_clean, test_clean = ohe(game_stats_trimmed_train, test_sess_agg_df, 'title', features)
del game_stats_trimmed_train
del test_sess_agg_df
train_clean, test_clean = ohe(train_clean, test_clean, 'type', features)
train_clean, test_clean = ohe(train_clean, test_clean, 'world', features)

mean_encoding(train_clean, test_clean ,'distinct_event_codes')
mean_encoding(train_clean, test_clean ,'time_start_hour')
mean_encoding(train_clean, test_clean ,'event_count')
mean_encoding(train_clean, test_clean ,'game_time')
mean_encoding(train_clean, test_clean ,'first_install_event_time')
mean_encoding(train_clean, test_clean ,'time_since_previous_game')

mean_encoding(train_clean, test_clean ,'game_time_mean')
mean_encoding(train_clean, test_clean ,'game_time_median')
mean_encoding(train_clean, test_clean ,'game_time_std')
test_clean.reset_index(drop=True, inplace=True)


# In[ ]:


sequence_length = 250 
padding_stub = np.repeat(-1., len(features)).astype(np.int8, copy=False)

def create_sequential_input(df, features, pad_val):
    df_grouped = df.groupby('installation_id')            
    
    rnn_input = np.zeros((len(df_grouped), sequence_length, len(features)))
    y_map = list()
    for g, i in zip(df_grouped, range(len(df_grouped))):
        sequence_full = g[1][features].values
        sequence_trimmed = sequence_full[max(0, len(sequence_full)-sequence_length):]        
        current_length = sequence_trimmed.shape[0]       
        
        # padding
        rnn_input[i, :sequence_length-current_length] = np.tile(pad_val, (sequence_length-current_length, 1))
        rnn_input[i, sequence_length-current_length:] = sequence_trimmed
        y_map.append((g[0],g[1].iloc[-1]['game_session']))    
    
    return rnn_input, pd.DataFrame(y_map, columns=['installation_id', 'game_session'])

X_train_input, y_train_map = create_sequential_input(train_clean, features, padding_stub)
X_test_input, y_test_map = create_sequential_input(test_clean, features, padding_stub)


# In[ ]:


train_merge_labels = y_train_map.merge(labels, how='inner', on=['installation_id', 'game_session'])
y_train_input = train_merge_labels['accuracy_group']


# In[ ]:


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2           
            else:
                X_p[i] = 3

        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y, initial_coef):
        loss_partial = partial(self._kappa_loss, X=X, y=y)        
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2            
            else:
                X_p[i] = 3
        return X_p

    def coefficients(self):
        return self.coef_['x']

def get_class_bounds(y, y_pred, N=4, class0_fraction=-1):
    """
    Find boundary values for y_pred to match the known y class percentiles.
    Returns N-1 boundaries in y_pred values that separate y_pred
    into N classes (0, 1, 2, ..., N-1) with same percentiles as y has.
    Can adjust the fraction in Class 0 by the given factor (>=0), if desired. 
    """
    ysort = np.sort(y)
    predsort = np.sort(y_pred)
    bounds = []
    for ibound in range(N-1):
        iy = len(ysort[ysort <= ibound])
        # adjust the number of class 0 predictions?
        if (ibound == 0) and (class0_fraction >= 0.0) :
            iy = int(class0_fraction * iy)
        bounds.append(predsort[iy])
    return bounds


# In[ ]:


class CyclicLR(Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}
        print(self.clr())


        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        # self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        # self.history.setdefault('iterations', []).append(self.trn_iterations)

        # for k, v in logs.items():
        #     self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
        
    def on_epoch_end(self, epoch, logs=None):
        print(self.clr())


# In[ ]:


gc.collect()

epochs = 128
batch_size = 512

def get_model():
    model = Sequential()    
    model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2, input_shape=(sequence_length, len(features))))
    model.add(Dense(1, activation='linear'))  
    model.compile(loss=tf.keras.metrics.mean_squared_error, optimizer='adam', metrics=['mae'])
    
    checkpointer = ModelCheckpoint(filepath='weights.hdf5',verbose=2,save_best_only=True,monitor='val_mae')
#     clr = CyclicLR(base_lr=0.00001, max_lr=0.01,step_size=5, mode='exp_range',gamma=0.99994)
    clr = CyclicLR(base_lr=0.00001, max_lr=0.01,step_size=70., mode='triangular2')
    stopping = EarlyStopping(monitor='val_mae', patience=25, min_delta=0.0001)
    
    callbacks = [checkpointer, clr, stopping]
    return model, callbacks


# In[ ]:


n_folds = 5
kf = StratifiedKFold(n_splits=n_folds)
finals = pd.DataFrame()

for (train_index, test_index), i in zip(kf.split(X_train_input, y_train_input), range(n_folds)):
    model, callbacks = get_model()
    
    X_train, X_test = X_train_input[train_index], X_train_input[test_index]
    y_train, y_test = y_train_input[train_index], y_train_input[test_index]
    
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks,
                        verbose=2)
    
    model.load_weights("weights.hdf5")
    
    test_pred = model.predict(X_test)
    optR = OptimizedRounder()
    bounds = get_class_bounds(y_test, test_pred.reshape(-1))
    optR.fit(test_pred.reshape(-1), y_test, bounds)
    coefficients = optR.coefficients()    
    
    pred_final = model.predict(X_test_input)
    final = optR.predict(pred_final.reshape(-1), coefficients)    
    finals[str(i)] = final


# In[ ]:


probs = {k: v/len(y_train_input) for k,v in Counter(y_train_input.values).items()}

sample_submission = pd.DataFrame()
sample_submission['installation_id']= y_test_map['installation_id']
final_mode_values = finals.mode(axis=1, dropna=True)    .apply(lambda x: int(sorted(x[~np.isnan(x)].values, key=lambda x: probs[x])[-1]), axis=1).values
sample_submission["accuracy_group"] = final_mode_values
sample_submission.to_csv('submission.csv', index=None)

