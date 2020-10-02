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

from kaggle.competitions import twosigmanews
# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# distribution of confidence as a sanity check: they should be distributed as above
import time
import warnings
warnings.simplefilter(action='ignore')


# In[ ]:


env = twosigmanews.make_env()
(mt_df, nt_df) = env.get_training_data()
print("Market data {}".format(mt_df.shape))
print("News data {}".format(nt_df.shape))


# In[ ]:


############ For memory free up 
del nt_df


# In[ ]:


##### Analyze the null values in the dataset ( from some other kernel)
total = mt_df.isnull().sum().sort_values(ascending = False)
percent = round(mt_df.isnull().sum().sort_values(ascending = False)/len(mt_df)*100, 2)
mt_df_null = pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])
mt_df_null


# In[ ]:


# Disable asset codes  for the time being
# np.random.seed(0)
# asset_embedding_dimension = 5
# asset_code_dict = {k: v for v, k in enumerate(mt_df['assetCode'].unique())}
# asset_code_dict['UNK_asset'] = len(asset_code_dict) # for unknown asset in prediction time
# # asset_code_embedding = np.random.uniform(-1,1,(len(asset_code_dict), asset_embedding_dimension)).astype(np.float32)
# asset_code_embedding = np.random.normal(loc=0.0, scale=1.0, size=(len(asset_code_dict), asset_embedding_dimension))
# print(asset_code_embedding.shape)
# asset_code_embedding = dict(zip(list(asset_code_dict.keys()), asset_code_embedding))


# In[ ]:


####### Features for the model 
categorical_features = ['assetCode' , 'universe' , 'time']
numerical_features = ['volume', 
                      'close' ,
                      'open' ,
                      'returnsClosePrevRaw1',
                      'returnsOpenPrevRaw1',
                      'returnsClosePrevMktres1',
                      'returnsOpenPrevMktres1',
                      'returnsClosePrevRaw10',
                      'returnsOpenPrevRaw10',
                      'returnsClosePrevMktres10',
                      'returnsOpenPrevMktres10']
prediction_column = ['returnsOpenNextMktres10']
prediction_column_scaled = ['returnsOpenNextMktres10_scaled']
###### Select only relevant columns 

mt_df = mt_df[categorical_features+numerical_features+prediction_column]
print(mt_df.shape)


# In[ ]:


############## Preprocessing flow

def impute_missing(df, columns):
    X = df[columns].fillna(0.0)
    return X.values

def scaler_fn(df, columns, scaler_to_use = 'minmax'):
    
    if scaler_to_use is 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(df[columns].values)
        scaled_matrix = scaler.transform(df[columns].values)
        for index, num_col in enumerate(numerical_features):
            print("Mean of {} is {:.2f}".format(num_col,scaled_matrix[:,index].mean()))
            print("Std deviation of {} is {:.2f}".format(num_col,scaled_matrix[:,index].std()))
        return scaler, scaled_matrix
    if scaler_to_use is 'standard_scaler':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(df[columns].values)
        scaled_matrix = scaler.transform(df[columns].values)
        return scaler, scaled_matrix

##### Let it be here for future
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

####### Naive programming , i know :-(
def binary_scale(value):
    if value >= 0.0:
        return 1.0
    else:
        return 0.0
    

X_impute = impute_missing(mt_df, numerical_features)
mt_df[numerical_features] = X_impute
scaler, scaled_matrix = scaler_fn(mt_df, numerical_features , scaler_to_use='standard_scaler')
print("Max {} and Min is {} after normalization".format(scaled_matrix.max() , scaled_matrix.min()))
mt_df[prediction_column_scaled[0]] = mt_df[prediction_column[0]].apply(binary_scale)
mt_df[numerical_features] = scaled_matrix

del scaled_matrix # For memory

######## For test data :-)
def test_preprocess(test_df):
    test_df_matrix = impute_missing(test_df, numerical_features)
    test_df[numerical_features] = test_df_matrix
    del test_df_matrix
    X_test = scaler.transform(test_df[numerical_features].values)
    return X_test


# In[ ]:


######### Label counts 
print(mt_df[prediction_column_scaled[0]].value_counts())


# In[ ]:





# In[ ]:


########### Batch generator

def batch_generator(X, Y=None, batch_size = 64, max_seq_len = 20, is_shuffle = False, test=False):
    
    if is_shuffle:
        if not test:
            p = np.random.permutation(len(X))
            X = X[p]
            Y = Y[p]
    combined_len = batch_size*max_seq_len
    if len(X) % (combined_len) == 0:
        iterations = int(len(X) / (combined_len))
    else:
        iterations = int(len(X) / (combined_len)) + 1
        
    for iter_ in range(iterations):
        start = iter_*combined_len
        end   = (iter_+1)*combined_len
        X_batch = X[start:end]
        if not test:
            Y_batch = Y[start:end]
            yield X_batch , Y_batch
        else:
            yield X_batch
        


# In[ ]:


########## Train test split ################


# all_asset_codes_unique = list(pd.unique(mt_df['assetCode']))
# mt_df_grouped = mt_df.groupby("assetCode")
# validation_df = pd.DataFrame()
# train_df      = pd.DataFrame()

# sample_test = 110 # sampling all asset_codes is super expensive time based
# asset_codes_for_test = all_asset_codes_unique[:sample_test]
    
# for count_ , asset_code_ in enumerate(asset_codes_for_test):
#     sample_df = mt_df_grouped.get_group(asset_code_)
#     split_index = int(0.10 * len(sample_df))
#     train_sample    = sample_df[:(len(sample_df)-split_index)]
#     test_sample = sample_df[-1*split_index:]
#     validation_df = pd.concat([validation_df, test_sample])
#     train_df      = pd.concat([train_df, train_sample])
#     print("{} , train {} , validation {}".format(asset_code_,
#                                                  train_sample.shape,
#                                                  test_sample.shape))
# train_df = pd.concat([train_df, mt_df[~mt_df['assetCode'].isin(asset_codes_for_test)]])



from sklearn.model_selection import train_test_split

train_indices, val_indices = train_test_split(mt_df.index.values,test_size=0.25, random_state=23)
train_df = mt_df.loc[train_indices]
validation_df = mt_df.loc[val_indices]


# In[ ]:


print(train_df.shape, validation_df.shape)


# In[ ]:


# For memory
# del mt_df
# del sample_df 


# In[ ]:


BATCH_SIZE  = 256
NUM_EPOCHS  = 11
NUM_FEATURES   = 11


# In[ ]:


import tensorflow as tf
tf.reset_default_graph()

input_ph = tf.placeholder(tf.float32, [None, NUM_FEATURES], 'input_ph')
label_ph = tf.placeholder(tf.float32, [None], 'labels_ph')

W1 = tf.get_variable(
                        name='W1',
                        shape=(NUM_FEATURES,64),
                        initializer=tf.truncated_normal_initializer(-1,1),
                        trainable=True)

b1 = tf.Variable(tf.zeros(shape=[64]),
                                 name='b1', dtype=tf.float32)
    
W2 = tf.get_variable(
                        name='W2',
                        shape=(64,128),
                        initializer=tf.truncated_normal_initializer(-1,1),
                        trainable=True)

b2 = tf.Variable(tf.zeros(shape=[128]),
                                 name='b2', dtype=tf.float32)

W_out = tf.get_variable(
                        name='W_out',
                        shape=(128,1),
                        initializer=tf.truncated_normal_initializer(-1,1),
                        trainable=True)

b_out = tf.Variable(tf.zeros(shape=[1]),
                                 name='b_out', dtype=tf.float32)

h1 = tf.nn.sigmoid(tf.matmul(input_ph, W1) + b1)
h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)
logits = tf.matmul(h2, W_out) + b_out
logits = tf.reshape(logits, [-1])
predictions = tf.nn.sigmoid(logits)
loss   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_ph, logits=logits))

correct_pred = tf.equal(tf.round(predictions), label_ph)
tf_accuracy_score = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

optimizer = tf.train.AdamOptimizer()
gradients, variables = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
train_op = optimizer.apply_gradients(zip(gradients, variables))

print(tf.trainable_variables())


# In[ ]:


try:
    if sess:
        tf.InteractiveSession.close(sess)
except:
    pass
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


# In[ ]:


def check_validation(df, batch_size = 1500, check_loss = True):
    if len(df) < batch_size:
        batch_size = len(df)
    val_epoch_cost = 0.0
    val_prediction = []
    validate_batch = batch_generator(df[numerical_features].values, df[prediction_column_scaled].values,
                                        batch_size = batch_size, test=False)
    epoch_time = 0.0
    val_logits_final = []
    for batch_count, (X_batch , Y_batch) in enumerate(validate_batch):
        start_time = time.time()
        feed_dict = {
                     input_ph: X_batch ,
                     label_ph: Y_batch.flatten()
        }
        val_logits , val_acc  = sess.run((predictions,tf_accuracy_score), feed_dict=feed_dict)
        val_prediction.append(val_acc)
        val_logits_final.extend(val_logits)
        end_time = time.time()
        batch_time = end_time-start_time
        epoch_time += batch_time
        if batch_count % 500 == 0.0:
            print("Validation , done batch {} , time {}".format(batch_count,batch_time ))
    return val_logits_final, np.mean(val_prediction)


# In[ ]:


training_loss = []
training_accuracy = []
validation_accuracy = []
check_validation_step = 1.0
NUM_EPOCHS = 11
for epoch in range(NUM_EPOCHS):
    epoch_cost = 0.0
    epoch_time = 0.0
    train_batch = batch_generator(train_df[numerical_features].values, 
                                  train_df[prediction_column_scaled].values, 
                                    batch_size = BATCH_SIZE,is_shuffle=True)
    for batch_count, (X_batch ,Y_batch) in enumerate(train_batch):
        start_time = time.time()
        Y_batch = Y_batch.flatten() #### For tensorflow placholder
        feed_dict = {
                     input_ph: X_batch ,
                     label_ph : Y_batch
        }
        
        batch_cost , _ = sess.run((loss, train_op) , feed_dict=feed_dict)
        epoch_cost += batch_cost
        end_time = time.time()
        batch_time = end_time-start_time
        epoch_time += batch_time
#         print("Epoch {} , batch {} , loss {} , time {}".format(epoch,batch_count,
#                                                                batch_cost,batch_time ))
        if batch_count % 500 == 0.0:
            print("Epoch {} , Done {} , time {}".format(epoch, batch_count, epoch_time))
    print("Epoch {} ,epoch loss {} , time {}".format(epoch,epoch_cost/batch_count,
                                                               epoch_time ))
    training_loss.append(epoch_cost/batch_count)
    
    val_probs , val_acc = check_validation(validation_df)
    train_probs , train_acc = check_validation(train_df)
    validation_accuracy.append(val_acc)
    training_accuracy.append(train_acc)


# In[ ]:



plt.plot(training_loss)
plt.title('training loss')  
plt.ylabel('loss')  
plt.xlabel('epoch') 
plt.show()


plt.plot(training_accuracy)
plt.title('training accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch') 
plt.show()

plt.plot(validation_accuracy)
plt.title('validation accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch') 
plt.show()


# In[ ]:


# del mt_df
# del X_batch 
# del Y_batch
# del seq_len_batch
# del train_df
# del X_impute


# In[ ]:


# distribution of confidence that will be used as submission
confidence_valid = np.array(val_probs)*2 -1
print(accuracy_score(confidence_valid>0,validation_df[prediction_column_scaled].values))
plt.hist(confidence_valid, bins='auto')
plt.title("predicted confidence")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


# calculation of actual metric that is used to calculate final score
r_valid = validation_df[prediction_column_scaled].values
r_valid = r_valid.clip(-1,1) # get rid of outliers. Where do they come from??
u_valid = validation_df['universe'].values
d_valid = validation_df['time'].dt.date
x_t_i = confidence_valid * r_valid.flatten() * u_valid.flatten()
data = {'day' : d_valid, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)


# In[ ]:


days = env.get_prediction_days()
n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
predicted_confidences = []
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    del news_obs_df
    #########################
    X_test = test_preprocess(market_obs_df)
    test_batch_gen = batch_generator(X_test
                                     ,batch_size = 128,test=True)
    market_prediction = []
    for X_test_batch  in test_batch_gen:
        feed_dict_test = {
             input_ph: X_test_batch ,
            }
        test_batch_prediction = sess.run(predictions,
                                 feed_dict=feed_dict_test)
        market_prediction.extend(test_batch_prediction)
    market_prediction = np.array(market_prediction)
    market_prediction_scaled = (2*market_prediction)-1.0                                       
    n_days +=1
    print(n_days,end=' ')
    predicted_confidences = np.concatenate((predicted_confidences, market_prediction_scaled))
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':market_prediction_scaled})
    # insert predictions to template
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)


# In[ ]:


env.write_submission_file()


# In[ ]:



plt.hist(predicted_confidences, bins='auto')
plt.title("predicted confidence")
plt.show()


# In[ ]:


predicted_confidences


# In[ ]:




