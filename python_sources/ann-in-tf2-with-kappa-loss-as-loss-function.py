#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv', parse_dates=['timestamp'])
train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
# specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
train = train[train.installation_id.isin(train_labels.installation_id)]

print(train.shape, test.shape, sep = '\n')


# In[ ]:


#Make Mapping
def Make_map_Assessment_attempts():
    Assessment = list(train[train.type == 'Assessment']['title'].drop_duplicates())
    Code = [4110 if x == 'Bird Measurer (Assessment)' else 4100 for x in Assessment]
    map_Assessment_attempts = pd.DataFrame({'title': Assessment, 'event_code': Code})
    return map_Assessment_attempts

map_Assessment_attempts = Make_map_Assessment_attempts()


# In[ ]:


#Make dictionary
def Make_Dict(i):
    lst = list(set(np.append(train[i].unique(), test[i].unique())))
    dct = pd.DataFrame({'Key':lst})
    return dct

def Reset_Dict():
    event_id = Make_Dict('event_id')
    event_code = Make_Dict('event_code')
    title = Make_Dict('title')
    type_ = Make_Dict('type')
    world =  Make_Dict('world')
    attempt = pd.DataFrame({'Key':['Correct', 'Incorrect']})
    return event_id, event_code, title, type_, world, attempt


# In[ ]:


def RemoveRecordAfterLastAssessment(df):
#     df = train[train.installation_id.isin(['0006a69f'])] 
    df2 = df.merge(map_Assessment_attempts, on = ['title', 'event_code'])
    df3 = df2[['installation_id', 'game_session']].drop_duplicates()
    df4 = df.merge(df3, on = ['installation_id', 'game_session'])
    df5 = df4.groupby(['installation_id', 'game_session']).agg({'timestamp':'min'}).reset_index()
    df6 = df5.groupby(['installation_id']).agg({'timestamp':'max'}).reset_index().rename(columns={'timestamp':'timestamp_'})
    df7 = df.merge(df6, on = 'installation_id')
    df8 = df7[df7.timestamp <= df7.timestamp_].drop(columns=['timestamp_'])
    return df8

train = RemoveRecordAfterLastAssessment(train)
print(train.shape, test.shape, sep = '\n')


# In[ ]:


#Prepare Labels
def PrepareLabel():
    df = train.groupby(['installation_id']).agg({'timestamp':'max'}).reset_index()
    df1 = df.merge(train, on = ['installation_id', 'timestamp'])[['installation_id', 'game_session']]
    df2 = df1.merge(train_labels, on = ['installation_id', 'game_session'])[['installation_id', 'accuracy_group']]
    df2 = df2.rename(columns=({'accuracy_group':'accuracy_group_for_train'}))
    return df2


# In[ ]:


#Add column for attempt
def AddColumnForAttempt(df):
    df1 = df[(df['title'] == 'Bird Measurer (Assessment)') & (df['event_code'] == 4110) |    (df['title'] != 'Bird Measurer (Assessment)') & (df['event_code'] == 4100) & (df['type'] == 'Assessment')]
    df2 = df1['event_data'].str.contains('"correct":true')
    df2 = df2.rename('attempt')
    df2 = df2.apply(lambda x: 'Correct' if x == True else 'Incorrect')
    df3 = df.merge(df2, left_index = True, right_index= True, how = 'left')
    return df3

train = AddColumnForAttempt(train)
test = AddColumnForAttempt(test)
print(train.shape, test.shape, sep = '\n')


# In[ ]:


def Count(df, item, item_df):
    df1 = df.groupby(['installation_id', item]).agg({item:'count'}).rename(columns={item:'count'}).reset_index().fillna(0)
    df1 = item_df.merge(df1, left_on = 'Key', right_on = item, how = 'left').drop(columns=[item]).fillna(0)
    df1 = pd.pivot_table(df1, values = 'count', columns=['Key'], index = ['installation_id']).reset_index().fillna(0)
    return df1

def CountByGameSession(df, item, item_df):
    df1 = df.groupby(['installation_id', 'game_session', item]).agg({item:'count'}).rename(columns={item:'count'}).reset_index().fillna(0)
    df1['count'] = 1
    df2 = df1.groupby(['installation_id', item]).agg({'count':'sum'}).reset_index().fillna(0)
    df3 = item_df.merge(df2, left_on = 'Key', right_on = item, how = 'left').drop(columns=[item]).fillna(0)
    df4 = pd.pivot_table(df3, values = 'count', columns=['Key'], index = ['installation_id']).reset_index().fillna(0)
    return df4

def CountLastAttemptbygame_session(df):
    df2 = df.groupby(['installation_id', 'game_session', 'attempt', 'timestamp']).agg({'timestamp':'count'}).rename(columns={'timestamp':'count'}).reset_index()
    df3 = df2.groupby(['installation_id', 'game_session']).agg({'timestamp':'max'}).reset_index().drop(columns=['game_session'] )
    df4 = df3.merge(df2, on = ['installation_id', 'timestamp'])[['installation_id', 'attempt', 'count']]
    df5 = df4.groupby(['installation_id','attempt']).agg({'count':'sum'}).reset_index()
    df6 = pd.pivot_table(df5, values = 'count', columns=['attempt'], index = ['installation_id']).reset_index().fillna(0)
    return df6

def GetMaxMeanGameTime(df):
    df1 = df.groupby(['installation_id']).agg({'game_time':['max', 'mean']}).reset_index()
    df1.columns = [''.join(col).strip() for col in df1.columns.values]
    return df1

def GetMeanGameTime_byGameSession(df):
    df1 = df.groupby(['installation_id', 'game_session']).agg({'game_time':'max'}).reset_index()
    df2 = df1[df1.game_time > 0]
    df3 = df2.groupby(['installation_id']).agg({'game_time':'mean'}).reset_index()
    return df3

def GetAccuracyGroupcount(df):

    AccuracyGroup_dict = pd.DataFrame({'AccuracyGroup':[0, 1, 2, 3]})

    # df = train[train.installation_id.isin(['0006a69f', 'ffeb0b1b'])] 
    df1 = df.groupby(['installation_id', 'game_session', 'attempt']).agg({'attempt':'count'}).rename(columns={'attempt':'count'}).reset_index()
    df2 = pd.pivot_table(df1, values = 'count', columns=['attempt'], index = ['installation_id', 'game_session']).reset_index().fillna(0)
    def AccuracyGroup(row):
        if (row['Correct'] == 1) & (row['Incorrect'] ==0):
            return 3
        elif (row['Correct'] == 1) & (row['Incorrect'] ==1):
            return 2
        elif (row['Correct'] == 1) & (row['Incorrect'] >1):
            return 1
        else:
            return 0
    df2['AccuracyGroup'] = df2.apply(lambda row: AccuracyGroup(row), axis = 1)
    df3 = df2.groupby(['installation_id', 'AccuracyGroup']).agg({'AccuracyGroup':'count'}).rename(columns={'AccuracyGroup':'count'}).reset_index()
    df4 = AccuracyGroup_dict.merge(df3, on = 'AccuracyGroup', how = 'left')
    df5 = pd.pivot_table(df4, values = 'count', columns=['AccuracyGroup'], index = ['installation_id']).reset_index().fillna(0)
    return df5

def GetDayCount(df):
    # df = train[train.installation_id.isin(['0006a69f', 'ffeb0b1b'])] 
    df1 = df.copy()
    df1['timestamp'] = df1['timestamp'].dt.date
    df2 = df1.groupby(['installation_id', 'timestamp']).agg({'timestamp':'nunique'}).rename(columns={'timestamp':'nunique'}).reset_index()
    df3 = df2.groupby(['installation_id']).agg({'nunique': 'sum'}).reset_index()
    return df3

def GetDayDiff(df):
#     df = train[train.installation_id.isin(['0006a69f', 'ffeb0b1b'])] 
    df1 = df.copy()
    df1['timestamp'] = df1['timestamp'].dt.date
    df2 = df1.groupby(['installation_id', 'timestamp']).agg({'timestamp':'nunique'}).rename(columns={'timestamp':'nunique'}).reset_index()
    df3 = df2.groupby(['installation_id']).agg({'timestamp':['min', 'max']}).reset_index()
    df3.columns = [''.join(col).strip() for col in df3.columns.values]
    df3['DateDiff'] = (df3.timestampmax - df3.timestampmin).dt.days
    df4 = df3[['installation_id', 'DateDiff']]
    return df4

def TimeSpendingBy(df, item, item_df):
#     df = train[train.installation_id.isin(['0006a69f', 'ffeb0b1b'])] 
    df2 = df.groupby(['installation_id', 'game_session', item]).agg({'game_time': 'max'}).reset_index().fillna(0)
    df3 = df2.groupby(['installation_id', item]).agg({'game_time':'mean'}).reset_index().fillna(0)
    df4 = item_df.merge(df3, left_on = 'Key', right_on = item, how = 'left').drop(columns=[item]).fillna(0)
    df5 = pd.pivot_table(df4, values = 'game_time', columns=['Key'], index = ['installation_id']).reset_index().fillna(0)
    return df5


# In[ ]:


def PrepareResultSet(df):
    event_id, event_code, title, type_, world, attempt = Reset_Dict()
    event_id_count = Count(df, 'event_id', event_id)
    event_code_count = Count(df, 'event_code', event_code)
    title_count = Count(df, 'title', title)
    type_count = Count(df, 'type', type_)
    world_count = Count(df, 'world', world)
    attempt_count = Count(df, 'attempt', attempt)    
    attempt_count['Attempt'] = attempt_count['Correct'] + attempt_count['Incorrect']
    attempt_count['CorrectRate'] = attempt_count['Correct']/attempt_count['Attempt']
    event_id_count_ = CountByGameSession(df, 'event_id', event_id)
    event_code_count_ = CountByGameSession(df, 'event_code', event_code)
    title_count_ = CountByGameSession(df, 'title', title)
    type_count_ = CountByGameSession(df, 'type', type_)
    world_count_ = CountByGameSession(df, 'world', world)
    attempt_count_ = CountLastAttemptbygame_session(df)
    attempt_count_['Attempt'] = attempt_count_['Correct'] + attempt_count_['Incorrect']
    attempt_count_['CorrectRate'] = attempt_count_['Correct']/attempt_count_['Attempt']
    game_time = GetMaxMeanGameTime(df)
    game_time_ = GetMeanGameTime_byGameSession(df)
    AccuracyGroupcount = GetAccuracyGroupcount(df)
    DayCount = GetDayCount(df)
    DayDiff = GetDayDiff(df)
    event_id_time = TimeSpendingBy(df, 'event_id', event_id)
    event_code_time = TimeSpendingBy(df, 'event_code', event_code)
    title_time = TimeSpendingBy(df, 'title', title)
    type_time = TimeSpendingBy(df, 'type', type_)
    world_time = TimeSpendingBy(df, 'world', world)
    attempt_time = TimeSpendingBy(df, 'attempt', attempt)    
    
    return event_id_count,event_code_count,title_count,           type_count,world_count,attempt_count,event_id_count_,event_code_count_,           title_count_,type_count_,world_count_,attempt_count_,game_time,game_time_,           AccuracyGroupcount,DayCount,DayDiff,event_id_time,event_code_time,title_time,           type_time,world_time,attempt_time


# In[ ]:


def MakeReusltSet(df, IsTrain):
    event_id_count,event_code_count,title_count,    type_count,world_count,attempt_count,event_id_count_,event_code_count_,    title_count_,type_count_,world_count_,attempt_count_,game_time,game_time_,    AccuracyGroupcount,DayCount,DayDiff,event_id_time,event_code_time,title_time,    type_time,world_time,attempt_time = PrepareResultSet(df)
    result_set = pd.DataFrame({'installation_id':df.installation_id.drop_duplicates()})
    result_set2 = result_set    .merge(event_id_count, on = 'installation_id', how = 'left')    .merge(event_code_count, on = 'installation_id', how = 'left')    .merge(title_count, on = 'installation_id', how = 'left')    .merge(type_count, on = 'installation_id', how = 'left')    .merge(world_count, on = 'installation_id', how = 'left')    .merge(attempt_count, on = 'installation_id', how = 'left')    .merge(event_id_count_, on = 'installation_id', how = 'left')    .merge(event_code_count_, on = 'installation_id', how = 'left')    .merge(title_count_, on = 'installation_id', how = 'left')    .merge(type_count_, on = 'installation_id', how = 'left')    .merge(world_count_, on = 'installation_id', how = 'left')    .merge(attempt_count_, on = 'installation_id', how = 'left')    .merge(game_time, on = 'installation_id', how = 'left')    .merge(game_time_, on = 'installation_id', how = 'left')    .merge(AccuracyGroupcount, on = 'installation_id', how = 'left')    .merge(DayCount, on = 'installation_id', how = 'left')    .merge(DayDiff, on = 'installation_id', how = 'left')    .merge(event_id_time, on = 'installation_id', how = 'left')    .merge(event_code_time, on = 'installation_id', how = 'left')    .merge(title_time, on = 'installation_id', how = 'left')    .merge(type_time, on = 'installation_id', how = 'left')    .merge(world_time, on = 'installation_id', how = 'left')    .merge(attempt_time, on = 'installation_id', how = 'left')    .fillna(0)
    if IsTrain==True:
        Labels = PrepareLabel()
        result_set2 = result_set2.merge(Labels, on = 'installation_id', how = 'left')    
    return result_set2


# In[ ]:


def stardardize(df):
    result = df.copy()
    for feature_name in df.columns:
        mean_value = df[feature_name].mean()
        std_value = (df[feature_name].var())**(1/2)
        result[feature_name] = (df[feature_name] - mean_value) / (std_value)
    return result.fillna(0)

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result.fillna(0)

def stardardizeV2(df, df2):
    result = df.copy()
    result2 = df2.copy()
    result3 = pd.concat([df, df2])
    for feature_name in df.columns:
        mean_value = result3[feature_name].mean()
        std_value = (result3[feature_name].var())**(1/2)
        result[feature_name] = (df[feature_name] - mean_value) / (std_value)
    return result.fillna(0)


# In[ ]:


ReusltSettrain = MakeReusltSet(train, IsTrain=True)
ReusltSettest = MakeReusltSet(test, IsTrain=False)


# In[ ]:


x_train = ReusltSettrain[ReusltSettrain.columns[1:-1]]
features_names = np.array(x_train.columns.tolist())
x_pre = ReusltSettest[ReusltSettest.columns[1:]]

y_train = ReusltSettrain['accuracy_group_for_train']
y_train_onehot = np.zeros((y_train.size, y_train.max() + 1))
y_train_onehot[np.arange(y_train.size), y_train] = 1


x_train_std = stardardize(x_train)
x_train_nor = normalize(x_train)
x_train_std_nor = normalize(stardardize(x_train))

x_pre_std = stardardize(x_pre)
x_pre_nor = normalize(x_pre)
x_pre_std_nor = normalize(stardardize(x_pre))                       

y_pre = sample_submission


# In[ ]:


import tensorflow as tf
def kappa_loss(y_pred, y_true, y_pow=2, eps=1e-10, N=4, bsize=256, name='kappa'):

    with tf.name_scope(name):
        y_true = tf.cast(y_true, tf.float64)
        repeat_op = tf.cast(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]), tf.float64)
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.cast((N - 1) ** 2, tf.float64)
    
        pred_ = y_pred ** y_pow
        try:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))
    
        hist_rater_a = tf.reduce_sum(pred_norm, 0)
        hist_rater_b = tf.reduce_sum(y_true, 0)
        
  
        
        conf_mat = tf.linalg.matmul(tf.transpose(pred_norm), y_true)
    
        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(weights * tf.matmul(
            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /
                              tf.cast(bsize, tf.float64))
    
        return nom / (denom + eps)


# **-------------------------------**

# In[ ]:




import tensorflow as tf
from keras import regularizers
from sklearn.model_selection import train_test_split

# # activation_ = ['sigmoid', 'tanh', 'elu', 'hard_sigmoid', 'linear', 'relu', 'selu', 'softplus', 'softsign']
input_shape = x_train.shape[1]
# X_train, X_test, y_train_, y_test = train_test_split(x_train_std, y_train_onehot, test_size=0.3, shuffle=True)
X_train, y_train_ = x_train_std, y_train_onehot

SIZE = X_train.shape[0]
#2, 13, 139
batch = int(X_train.shape[0]/139)

tf.keras.backend.clear_session()
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(5000, input_shape=(input_shape,) , activation='elu'))
#                                 ,kernel_regularizer=regularizers.l2(0.001)
#                                 ,activity_regularizer=regularizers.l1(0.001)))
# model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(70, activation='elu'))
model.add(tf.keras.layers.Dense(8, activation='elu'))
model.add(tf.keras.layers.Dense(4, activation = 'softmax'))
#                                 ,kernel_regularizer=regularizers.l2(0.001)
#                                 ,activity_regularizer=regularizers.l1(0.001)))
opt = tf.keras.optimizers.Adamax()

m = tf.keras.metrics.Accuracy()
m2 = tf.keras.metrics.AUC(num_thresholds=3)
m3 = tf.keras.metrics.CategoricalCrossentropy()

# Here I create the custom learning loop for the model, so that I can use the custom kappa_loss function for loss
for i in range(100):
    for j in range(0, SIZE, batch):
        x = X_train[j+1:j+1+batch].to_numpy()
        y = y_train_[j+1:j+1+batch]
        with tf.GradientTape() as tape:
            y_pre = model(x)
            y_true = y
            loss = kappa_loss(y_true, y_pre, bsize=batch)
        grads = tape.gradient(loss, model.trainable_variables)
        processed_grads = [g for g in grads]
        grads_and_vars = zip(processed_grads, model.trainable_variables)
        opt.apply_gradients(grads_and_vars)
    print ('Model: ', i, ' : ', np.round(loss.numpy(), 2), end = ' ')
    m.update_state(y_true, y_pre)
    m2.update_state(y_true, y_pre)
    m3.update_state(y_true, y_pre)
    print('Result: ', np.round(m.result().numpy(), 2), np.round(m2.result().numpy(), 2), np.round(m3.result().numpy(), 2))
    
sub = model.predict(x_pre_std)
sub = sub.argmax(axis = 1)
sample_submission['accuracy_group'] = sub
sample_submission.to_csv('submission.csv', index = False)


# elu Model:  99  :  0.46 Result:  0.34 0.65 7.74

# In[ ]:


# # Neural Network

# import tensorflow as tf
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train_, y_test = train_test_split(x_train_std, y_train_onehot, test_size=0.3, shuffle=True)
# # activation_ = ['sigmoid', 'tanh', 'elu', 'hard_sigmoid', 'linear', 'relu', 'selu', 'softplus', 'softsign']

# # tfa.metrics.CohenKappa

# EPOCHS = 10
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(3000, input_shape=(x_train.shape[1],) , activation='linear', kernel_initializer='glorot_uniform'))
# model.add(tf.keras.layers.Dropout(0.01))
# model.add(tf.keras.layers.Dense(y_train_onehot.shape[1], activation = 'softmax'))
# model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy', 'mse', 'AUC'])
# model.fit(x=X_train, y=y_train_, epochs=EPOCHS, verbose=1)
# model.evaluate(x=X_test, y=y_test)

# predict = model.predict(X_test)
# predict_int = predict.argmax(axis = 1)
# y_test = y_test.argmax(axis = 1)
# print (qwk3(predict_int, y_test))


# sub = model.predict(x_pre_std)
# sub = sub.argmax(axis = 1)
# sample_submission['accuracy_group'] = sub
# sample_submission.to_csv('submission.csv', index = False)


# In[ ]:


y_train.hist()
sample_submission.hist()
print (y_train.value_counts(normalize = True))
print (sample_submission.accuracy_group.value_counts(normalize=True))

