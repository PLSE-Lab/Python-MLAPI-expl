#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette(n_colors=10)

import warnings
warnings.filterwarnings('ignore')

from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Activation
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
import tensorflow as tf
import keras


# # Step 1 : Get the data

# In[ ]:


# Read in the data CSV files
train = pd.read_csv('../input/data-science-bowl-2019/train.csv')
train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
test = pd.read_csv('../input/data-science-bowl-2019/test.csv')
specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')


# Before we get into detail of the model, we should have a general idea of what is the data, what kind of predictors that the data have. First, we will use .head() to check the first several rows of the data. In this case, we will only care about train, train_label, and specs data, because the predictors in test data are the same with train data. Therefore, we will use train data to train the models and choose the model with best performance, and then apply to the test and get the result. 
# 
# Based on my understanding of this project, I will groupby installation_id and game_session, transform the data into different shapes, and then merge train data with train_label data to train the model. 

# # Step 2: Filter/Clean the data: Define Functions

# 1. Attempts: I will use specs dataset to extract the type of attempts (including correct and incorrect) based on event_info. I will count how many attempts that eachinstallation_id tried. 

# In[ ]:


# create a new column to capture the correct/incorrect info from specs data
specs['attempts'] = ''

for i in range(len(specs)):
    if ('(Correct)' in specs['info'][i]) or ('(Incorrect)' in specs['info'][i]):
        specs['attempts'][i] = 1
    else:        
        specs['attempts'][i] = 0
# It is clearly that some event_id are not in assessment maps, 
# so only some of the event_id have the value from attempts.
# Next, we drop the useless columns to make it clear bacause the next step will be merged with train data

specs_drop = specs.drop(['info','args'],axis=1)

# merge the specs_attempts data with train data
train_cor = pd.merge(train,specs_drop,on='event_id',how='left')

# Finally, I count the total attempts groupby installation_id and game_session.
train_attempts = train_cor.groupby(['installation_id','game_session'],as_index=False)['attempts'].sum()

# Plot
train_attempts['attempts']     .plot(kind='hist',
          figsize=(10, 5),
          xlim = (0,100),
          bins=100,
          title='Total attempts',
         color = color[1])
plt.show()


# > It is clearly to see that a lot of installation_id didn't even try to pass the given assessment.

# 2. Freetime: each game_session has it's unique time_stamp. Check each game_seesion's timestamp is in freetime or not.In order to do this, I will create two new predictors named 'weekend' and 'evening'. If is in weekend/evening, the value will be 'Yes', otherwise, it will be 'No'. (Notice: This predictor is a dummy varlable.). Freetime means either the timestamp is in weekend or in evening.

# In[ ]:


# for each data, create two new variablse named'weekend' and 'evening'
# Because within one game_session, the timestamp is continuous, so we only consider the last timestamp
train_date_temp = train[['installation_id','game_session','timestamp']].groupby(['installation_id','game_session'],as_index=False)['timestamp'].last()
train_date_temp['timestamp'] = pd.to_datetime(train_date_temp['timestamp'])

# transform timestamp into hour and dayofweek(0 represents Monday, 6 represents Sunday.). 
train_date_temp['hour'] = train_date_temp['timestamp'].dt.hour
train_date_temp['weekday']  = train_date_temp['timestamp'].dt.dayofweek

# create a new variable named as weekend.
train_date_temp['weekend'] = ['yes' if index in([5,6])  else 'no' for  index in train_date_temp['weekday']]

# create a new variable named as Evening.
train_date_temp['evening'] = ['yes' if index in([17,23]) else 'no' for index in train_date_temp['hour']]

# create a new variable named as freetime
train_date_temp['freetime'] = ['yes' if (index_1 =='yes' or index_2 == 'yes') else 'no'                                for (index_1,index_2) in zip(train_date_temp['weekend'],train_date_temp['evening'])]

# drop useless variables
train_date = train_date_temp.drop(['timestamp','hour','weekday','weekend','evening'],axis=1)

# merge with train_attempts
train_prep1 = pd.merge(train_attempts,train_date,on=['installation_id','game_session'],how='outer')


# In[ ]:


# make a graph to see the relationship between freetime and total attempts
af_plot = train_prep1.groupby('freetime',as_index=False)['attempts'].sum()
names = list(af_plot['freetime'].unique())
values = list(af_plot['attempts'])

plt.bar(names, values)
plt.title('total attempts in weekend')

plt.show()


# > It is clearly to see that in weekend, children tend not to try again. 

# 3. Unique Event_id: within one installation_id and game_session, how many unique values of event_id.

# In[ ]:


# make a copy of train dataset
train_event_temp = train.copy()
# groupby installation_id and game_sesssion to count the unique values of event_id
train_event = train_event_temp.groupby(['installation_id','game_session'])['event_id'].nunique().reset_index()

# rename the column's name
train_event.columns = ['installation_id','game_session','uni_eventid']

# merge with train_prep1
train_prep2 = pd.merge(train_prep1,train_event,on=['installation_id','game_session'],how='outer')

# plot
train_event['uni_eventid'].plot(kind='hist',
                               figsize=(6, 4), title='Unique values of event_id',
                               color = color[2])

plt.show()


# 4. Total Game Time: count total game time based on game_time. Becase time will increase since the start of the game session, it is easy to keep the last value of each unique game_session within one installation_id.

# In[ ]:


# groupby installation_id and game_session to get the maxium value of game_time.
train_gametime = train.groupby(['installation_id','game_session'],as_index=False)['game_time'].max()

# merge with train_prep2
train_prep3 = pd.merge(train_prep2,train_gametime,on=['installation_id','game_session'],how='outer')

# plot
train_gametime['game_time'].apply(np.log1p).plot(kind='hist',
                                                 figsize=(6, 4), title='Log od total game time',
                                                 color = color[3])

plt.show()


# > Based on this graph, we could see that a lot of installation didn't even try this game becase a lot of game times are 0.

# 5. Total Event Count: event count is the incremental counter of events within a game session. We could find the maximum value of each game session within one installation_id.

# In[ ]:


# groupby installation_id and game_session to get the maxium value of event_count.
train_etcount = train.groupby(['installation_id','game_session'],as_index=False)['event_count'].max()

# merge with train_prep3
train_prep4 = pd.merge(train_prep3,train_etcount,on=['installation_id','game_session'],how='outer')

# plot
train_etcount['event_count'].apply(np.log1p).plot(kind='hist',
                                                 figsize=(6, 4), title='Log od total event count',
                                                 color = color[4])

plt.show()


# 6. Assessment(dummy variable): for each game session, we need to check if the type of game session is assessment or not. 

# In[ ]:


# before we get into detail, we need to check if one game seesion has only one type.
train_gtcheck = train.groupby(['installation_id','game_session'])['type'].nunique().reset_index()
len(train_gtcheck[train_gtcheck['type'] != 1]) # 0
# which means one game_session has only one unique type


# In[ ]:


# groupby installation_id and game_session
train_assess = train.groupby(['installation_id','game_session'],as_index=False)['type'].last()

# create a new variable named as assessment. 'yes' represents it is a assessment type.
train_assess['assessment'] = ['yes' if index =='Assessment'  else 'no' for  index in train_assess['type']]

# drop useless variable and then merge with train_prep4
train_assess = train_assess.drop('type',axis=1)
train_prep5 = pd.merge(train_prep4,train_assess,on=['installation_id','game_session'],how='outer')


# In[ ]:


# make a graph to see the relationship between 'assessment type' and total attempts
at_plot = train_prep5.groupby('assessment',as_index=False)['attempts'].sum()
names_at = list(at_plot['assessment'].unique())
values_at = list(at_plot['attempts'])

plt.bar(names_at, values_at)
plt.title('total attempts in assessment type')

plt.show()


# 7. Unique event_code: how many unique values of event_code in one game session.

# In[ ]:


# count values
train_etcode = train.groupby(['installation_id','game_session'])['event_code'].nunique().reset_index()

# merge data
train_prep6 = pd.merge(train_prep5,train_etcode,on=['installation_id','game_session'],how='outer')

# plot
train_etcode['event_code'].apply(np.log1p).plot(kind='hist',
                                                 figsize=(6, 4), title='Log of total event code',
                                                 color = color[4])

plt.show()


# 8. For each game seesion,which 'title' that the game session belongs to if the type of this game session is 'Assessment'. There are five different assessment titles including Bird Measurer, Cart Balancer, Cauldron Filler, Chest Sorter, and Mushroom Sorter. Note: if the type of game session is not assessment, then all of the five values equals to 0.

# In[ ]:


# before we get into more detail, we need to check is there only one unique title within one game session.
train_gticheck = train.groupby(['installation_id','game_session'])['title'].nunique().reset_index()
len(train_gticheck[train_gtcheck['type'] != 1]) #0


# In[ ]:


train_title = train.copy() # make a copy of train data set.

# write a loop to create five new variables
titles = ['Bird Measurer', 'Cart Balancer', 'Cauldron Filler', 'Chest Sorter', 'Mushroom Sorter']
for each_var in titles:
    train_title[each_var] = 0  # initializing 
    train_title[each_var] = [1 if (each_var) in index else 0 for index in train_title['title']]
    
# groupby installation_id and game_session
train_five_title = train_title.groupby(['installation_id','game_session'],as_index=False)['Bird Measurer', 'Cart Balancer', 'Cauldron Filler', 'Chest Sorter', 'Mushroom Sorter'].last()

# merge with train_prep6
train_prep7 = pd.merge(train_prep6,train_five_title,on=['installation_id','game_session'],how='outer')


# In[ ]:


# plot
train_title_names = list(titles)
train_title_values = []
for var in train_title_names:
    train_title_values.append(len(train_five_title[train_five_title[var] != 0]))

plt.bar(train_title_names, train_title_values)
plt.title('Freq of Each Assessment')
plt.xticks(rotation=45)
plt.show()


# 9. Similar to (8). this time I care about the which world it is in the same game session. There are four different world, including  'NONE' (at the app's start screen), 'TREETOPCITY' (Length/Height), 'MAGMAPEAK' (Capacity/Displacement) and 'CRYSTALCAVES' (Weight).

# In[ ]:


# based on 8, we know that one game_session only has one unique world, therefore we omit the checking procedure for simplicity.

train_world = train.copy() # make a copy of train data set.
# write a loop to create four new variables
world = ['NONE', 'TREETOPCITY','MAGMAPEAK','CRYSTALCAVES']
for each_wor in world:
    train_world[each_wor] = 0  # initializing 
    train_world[each_wor] = [1 if (each_wor) in index else 0 for index in train_world['world']]
    
# groupby installation_id and game_session
train_four_world = train_world.groupby(['installation_id','game_session'],as_index=False)['NONE', 'TREETOPCITY','MAGMAPEAK','CRYSTALCAVES'].last()

# merge with train_prep7
train_prep8 = pd.merge(train_prep7,train_four_world,on=['installation_id','game_session'],how='outer')


# In[ ]:


# plot
train_world_names = list(world)
train_world_values = []
for wor in train_world_names:
    train_world_values.append(len(train_four_world[train_four_world[wor] != 0]))

plt.bar(train_world_names, train_world_values)
plt.title('Freq of Each World')
plt.xticks(rotation=45)
plt.show()


# # Step 3: Filter/Clean Test data set

# Because test data is similar with train data, so we will use same way to clean and filter data

# 1. Attempts

# In[ ]:


# merge the specs_attempts data with test data
test_cor = pd.merge(test,specs_drop,on='event_id',how='left')

# Finally, I count the total attempts groupby installation_id and game_session.
test_attempts = test_cor.groupby(['installation_id','game_session'],as_index=False)['attempts'].sum()

# Plot
test_attempts['attempts']     .plot(kind='hist',
          figsize=(10, 5),
          xlim = (0,100),
          bins=100,
          title='Total attempts',
         color = color[1])
plt.show()


# 2. Freetime

# In[ ]:


# for each data, create two new variablse named'weekend' and 'evening'
# Because within one game_session, the timestamp is continuous, so we only consider the last timestamp
test_date_temp = test[['installation_id','game_session','timestamp']].groupby(['installation_id','game_session'],as_index=False)['timestamp'].last()
test_date_temp['timestamp'] = pd.to_datetime(test_date_temp['timestamp'])

# transform timestamp into hour and dayofweek(0 represents Monday, 6 represents Sunday.). 
test_date_temp['hour'] = test_date_temp['timestamp'].dt.hour
test_date_temp['weekday']  = test_date_temp['timestamp'].dt.dayofweek

# create a new variable named as weekend.
test_date_temp['weekend'] = ['yes' if index in([5,6])  else 'no' for  index in test_date_temp['weekday']]

# create a new variable named as Evening.
test_date_temp['evening'] = ['yes' if index in([17,23]) else 'no' for index in test_date_temp['hour']]

# create a new variable named as freetime
test_date_temp['freetime'] = ['yes' if (index_1 =='yes' or index_2 == 'yes') else 'no'                                for (index_1,index_2) in zip(test_date_temp['weekend'],test_date_temp['evening'])]

# drop useless variables
test_date = test_date_temp.drop(['timestamp','hour','weekday','weekend','evening'],axis=1)

# merge with test_attempts
test_prep1 = pd.merge(test_attempts,test_date,on=['installation_id','game_session'],how='outer')


# In[ ]:


# make a graph to see the relationship between freetime and total attempts
af_plot = test_prep1.groupby('freetime',as_index=False)['attempts'].sum()
names = list(af_plot['freetime'].unique())
values = list(af_plot['attempts'])

plt.bar(names, values)
plt.title('total attempts in weekend')

plt.show()


# 3. Unique Event_id

# In[ ]:


# make a copy of test dataset
test_event_temp = test.copy()
# groupby installation_id and game_sesssion to count the unique values of event_id
test_event = test_event_temp.groupby(['installation_id','game_session'])['event_id'].nunique().reset_index()

# rename the column's name
test_event.columns = ['installation_id','game_session','uni_eventid']

# merge with test_prep1
test_prep2 = pd.merge(test_prep1,test_event,on=['installation_id','game_session'],how='outer')

# plot
test_event['uni_eventid'].plot(kind='hist',
                               figsize=(6, 4), title='Unique values of event_id',
                               color = color[2])

plt.show()


# 4. Total Game Time

# In[ ]:


# groupby installation_id and game_session to get the maxium value of game_time.
test_gametime = test.groupby(['installation_id','game_session'],as_index=False)['game_time'].max()

# merge with test_prep2
test_prep3 = pd.merge(test_prep2,test_gametime,on=['installation_id','game_session'],how='outer')

# plot
test_gametime['game_time'].apply(np.log1p).plot(kind='hist',
                                                 figsize=(6, 4), title='Log od total game time',
                                                 color = color[3])

plt.show()


# 5. Total Event Count

# In[ ]:


# groupby installation_id and game_session to get the maxium value of event_count.
test_etcount = test.groupby(['installation_id','game_session'],as_index=False)['event_count'].max()

# merge with test_prep3
test_prep4 = pd.merge(test_prep3,test_etcount,on=['installation_id','game_session'],how='outer')

# plot
test_etcount['event_count'].apply(np.log1p).plot(kind='hist',
                                                 figsize=(6, 4), title='Log od total event count',
                                                 color = color[4])

plt.show()


# 6. Assessment(dummy variable)

# In[ ]:


# groupby installation_id and game_session
test_assess = test.groupby(['installation_id','game_session'],as_index=False)['type'].last()

# create a new variable named as assessment. 'yes' represents it is a assessment type.
test_assess['assessment'] = ['yes' if index =='Assessment'  else 'no' for  index in test_assess['type']]

# drop useless variable and then merge with test_prep4
test_assess = test_assess.drop('type',axis=1)
test_prep5 = pd.merge(test_prep4,test_assess,on=['installation_id','game_session'],how='outer')

# make a graph to see the relationship between 'assessment type' and total attempts
at_plot = test_prep5.groupby('assessment',as_index=False)['attempts'].sum()
names_at = list(at_plot['assessment'].unique())
values_at = list(at_plot['attempts'])

plt.bar(names_at, values_at)
plt.title('total attempts in assessment type')

plt.show()


# 7. Unique event_code

# In[ ]:


# count values
test_etcode = test.groupby(['installation_id','game_session'])['event_code'].nunique().reset_index()

# merge data
test_prep6 = pd.merge(test_prep5,test_etcode,on=['installation_id','game_session'],how='outer')

# plot
test_etcode['event_code'].apply(np.log1p).plot(kind='hist',
                                                 figsize=(6, 4), title='Log of total event code',
                                                 color = color[4])

plt.show()


# 8. Bird Measurer, Cart Balancer, Cauldron Filler, Chest Sorter, and Mushroom Sorter

# In[ ]:


test_title = test.copy() # make a copy of test data set.

# write a loop to create five new variables
titles = ['Bird Measurer', 'Cart Balancer', 'Cauldron Filler', 'Chest Sorter', 'Mushroom Sorter']
for each_var in titles:
    test_title[each_var] = 0  # initializing 
    test_title[each_var] = [1 if (each_var) in index else 0 for index in test_title['title']]
    
# groupby installation_id and game_session
test_five_title = test_title.groupby(['installation_id','game_session'],as_index=False)['Bird Measurer', 'Cart Balancer', 'Cauldron Filler', 'Chest Sorter', 'Mushroom Sorter'].last()

# merge with test_prep6
test_prep7 = pd.merge(test_prep6,test_five_title,on=['installation_id','game_session'],how='outer')

# plot
test_title_names = list(titles)
test_title_values = []
for var in test_title_names:
    test_title_values.append(len(test_five_title[test_five_title[var] != 0]))

plt.bar(test_title_names, test_title_values)
plt.title('Freq of Each Assessment')
plt.xticks(rotation=45)
plt.show()


# 9. Similar to (8). this time I care about the which world it is in the same game session. There are four different world, including  'NONE' (at the app's start screen), 'TREETOPCITY' (Length/Height), 'MAGMAPEAK' (Capacity/Displacement) and 'CRYSTALCAVES' (Weight).

# In[ ]:


# based on 8, we know that one game_session only has one unique world, therefore we omit the checking procedure for simplicity.

test_world = test.copy() # make a copy of test data set.
# write a loop to create four new variables
world = ['NONE', 'TREETOPCITY','MAGMAPEAK','CRYSTALCAVES']
for each_wor in world:
    test_world[each_wor] = 0  # initializing 
    test_world[each_wor] = [1 if (each_wor) in index else 0 for index in test_world['world']]
    
# groupby installation_id and game_session
test_four_world = test_world.groupby(['installation_id','game_session'],as_index=False)['NONE', 'TREETOPCITY','MAGMAPEAK','CRYSTALCAVES'].last()

# merge with test_prep7
test_prep8 = pd.merge(test_prep7,test_four_world,on=['installation_id','game_session'],how='outer')

# plot
test_world_names = list(world)
test_world_values = []
for wor in test_world_names:
    test_world_values.append(len(test_four_world[test_four_world[wor] != 0]))

plt.bar(test_world_names, test_world_values)
plt.title('Freq of Each World')
plt.xticks(rotation=45)
plt.show()


# # Step 4: Model

# We merge the train_prep7 data with train_labels data. Based on my understanding, this question is actually a both regression and classification problem. In this data set, the num_correct only has two values which are 0 and 1, therefore  predicting num_correct is a classification problem. But the num_incorrect has multiple values, so predicting num_incorrect is a regression problem. Based on train_laebls dataset. We know that accuracy = num_correct/(num_correct + num_incorrect). Based on the accuracy, we also know the accuracy group:
# >  accuracy_group = 3 if accuracy = 1
# 
# >  accuracy_group = 2 if 0.5 <= accuracy < 1
# 
# >  accuracy_group = 1 if 0 < accuracy < 0.5
# 
# >  accuracy_group = 0 if accuracy = 0
#  
# Therefore, according to the rule, we only need to predict the num_correct and num_incorrect.

# 1. Drop useless variables.

# In[ ]:


# merge train data with train label
train_final = pd.merge(train_prep8,train_labels,on=['installation_id','game_session'],how='left')

# fill NaN values with 0
train_final.fillna(0,inplace=True)

# drop some varibales, including installation_id, game_session,title, accuracy and accuracy_group.
trainset = train_final.drop(['title','accuracy','accuracy_group'],axis=1)

# change 'freetime' variable as category variable
trainset['freetime'] = trainset['freetime'].astype('category').cat.codes

# change 'assessment' variable as category variable
trainset['assessment'] = trainset['assessment'].astype('category').cat.codes


# 2. Split train and test in trainset, Define X and y_num, y_bin. X is predictors, y_num is num_incorrect, y_bin is num_correct.

# In[ ]:


trainset_train,trainset_test = train_test_split(trainset,test_size=0.2,random_state=42) # split data

trainset_train_X = trainset_train[['attempts','freetime','uni_eventid','game_time','event_count','assessment',              'event_code','Bird Measurer','Cart Balancer','Cauldron Filler','Chest Sorter','Mushroom Sorter',
                    'NONE','TREETOPCITY','CRYSTALCAVES','MAGMAPEAK']].values.astype('float32')  
trainset_train_ybin = trainset_train[['num_correct']].values.astype('float32')
trainset_train_ynum = trainset_train[['num_incorrect']].values.astype('float32')

trainset_test_X = trainset_test[['attempts','freetime','uni_eventid','game_time','event_count','assessment',              'event_code','Bird Measurer','Cart Balancer','Cauldron Filler','Chest Sorter','Mushroom Sorter',
                    'NONE','TREETOPCITY','CRYSTALCAVES','MAGMAPEAK']].values.astype('float32')  
trainset_test_ybin = trainset_test[['num_correct']].values.astype('float32')
trainset_test_ynum = trainset_test[['num_incorrect']].values.astype('float32')


# 3. Train Model using trainset_train data

# > In order to make a good prediction, I will use netural network, which  are very flexible, and allow you to define one model that does multiple things. For example, you can have one model that predicts both a classification target and a regression target. In this case, I just need to predict both a classfication and regression labels. Therefore, I will use keras neural network to make a good prediction.

# In[ ]:


one_input = Input(shape=(16,), name='one_input') # pass by one input

# show one output: y_bin
y_bin_output = Dense(1, activation='sigmoid', name='y_bin_output')(one_input)
# merge one output with all predictors from input
x = keras.layers.concatenate([one_input, y_bin_output]) 
# stack all other layers
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
#another output
y_num_output = Dense(1, activation='sigmoid', name='y_num_output')(x)


model = Model(inputs=one_input, outputs=[y_bin_output, y_num_output])
model.compile(optimizer='Adam', loss=['binary_crossentropy', 'mean_squared_error'])

model.fit(trainset_train_X, [trainset_train_ybin, trainset_train_ynum],epochs=30,verbose=0)


# 4. apply to trainaet_test to get the error rate and mean square error

# In[ ]:


# predict using trainset_test_X
trainset_pred = model.predict(trainset_test_X)

# transform the data type
trainset_pred_bin = trainset_pred[0].astype('int')
trainset_pred_num = np.around(trainset_pred[1])
trainset_conf_mx = confusion_matrix(trainset_test_ybin,trainset_pred_bin)

# based on the confusion matrix, we could calculate the error rate.
train_pred_er = (trainset_conf_mx[0,1] + trainset_conf_mx[1,0])/len(trainset_pred_bin)

# calculate the mse
trainset_mse = mean_squared_error(trainset_test_ynum,trainset_pred_num)

print('The error rate is: ',train_pred_er)
print('The mean square error is: ',trainset_mse )

# It is clearly to see that the model is pretty good


# # Step 5: Apply to Testset.

# 1. Data Transformation

# In[ ]:


testset = test_prep8.copy()
# change 'freetime' variable as category variable
testset['freetime'] =testset['freetime'].astype('category').cat.codes

# change 'assessment' variable as category variable
testset['assessment'] = testset['assessment'].astype('category').cat.codes


# 2. Write a loop to predict the testset one row by one row.

# In[ ]:


testset['num_correct'] = 0
testset['num_incorrect'] = 0

for i in range(len(testset)):
    value = testset.iloc[i:i+1,2:18].values
    pred_y = model.predict(value)
    testset['num_correct'][i] = pred_y[0].astype('int')
    testset['num_incorrect'][i] = np.around(pred_y[1])


# 3. calculate the accuracy and accuracy_group

# In[ ]:


testset['accuracy'] = testset['num_correct']/(testset['num_correct'] + testset['num_incorrect'])

# fill nan
testset.fillna(0,inplace=True)


# In[ ]:


# calculate accuracy_group
testset['accuracy_group'] = 0
for m in range(len(testset)):
    if testset['accuracy'][m] == 1:
        testset['accuracy_group'][m] =3
    elif 0.5 <= testset['accuracy'][m] < 1:
        testset['accuracy_group'][m] =2
    elif 0 < testset['accuracy'][m] < 0.5:
        testset['accuracy_group'][m] =1
    elif testset['accuracy'][m] == 0:
        testset['accuracy_group'][m] =0
        


# In[ ]:


final_pred_1 = testset[(testset['Bird Measurer'] !=0) | (testset['Cart Balancer'] !=0)| (testset['Cauldron Filler'] !=0)                    | (testset['Chest Sorter'] !=0)| (testset['Mushroom Sorter'] !=0)]

final_pred = final_pred_1.groupby('installation_id',as_index=False)['accuracy_group'].mean()


# In[ ]:


final = final_pred.round(0)
final


# In[ ]:


# save as csv
final.to_csv('submission.csv',index=False)


# In[ ]:




