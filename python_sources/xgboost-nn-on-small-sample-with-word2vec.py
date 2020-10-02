#!/usr/bin/env python
# coding: utf-8

# #   Word Embedding Idea.

# The reason i think Word2Vec might working here is dataset is sequential. If you sorted by ip and click time, you will make your datset as a sequence of clicks by different ip. If you take all clicks from one ip, that shows the whole process of that 'ip' 's behavioral pattern. It is sure that for different ips their behavior should be different. But how about commons? How could we extract the relationships of app, device, os and channel? What is the relationship between app 9 and app 10? That is something we need to take into consideration. By applying Word2Vec here, I diminished the specific pattern from different ips and get the 'word embeddings' of each categories in app, device, os and channel based on their own characteristics. You might experienced an ad for game app that pop up during you play a mobile game and after you click that ad another ad for another game might pop up. This is the 'charateristic' that I want to figure out and include in my training dataset.

# Personally I dont think there will be big difference between applying XGBoost on original dataset and on embedded dataset. The reason is obvious that decision tree works in a way that dimension expand doesn't really make effort on increasing the accuracy. But I still apply it here becuase i want to see how are those word embeddings performs and may be find out how long should my word embedding is. I use 3 as length for each predictors here, as you see from feature importance, it is a little bit suprising that for word embedding 'app', x2 doesn't play a same role as x1 and x3. So may be when we increase the dimension of word embeddings and set col_sample by tree a small value, we may get some unexpected good result.

# An update with more features ane larger embedding size is here. https://www.kaggle.com/jingqliu/nn-with-word2vec-large-embedding-more-feats 
# It using a word embeddings with size 300 instead of using 12, the auc goes up to 93% for validation (version 4). 

# # XGBoost on Embedded Dataset

# In[ ]:


import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import gc
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import matplotlib as plt
from gensim.models import Word2Vec


# In[ ]:


train = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/train.csv',nrows = 3698078*2)


# In[ ]:


train = train.drop(columns = ['attributed_time'])


# In[ ]:


train['hour'] = pd.to_datetime(train.click_time).dt.hour.astype('float32')
train['day'] = pd.to_datetime(train.click_time).dt.day.astype('float32')


# ## Some more features that i want to try.

# In[ ]:


#train = train.sort_values(['ip','click_time'])


# In[ ]:


#train['click_time'] = pd.to_datetime(train['click_time'])


# In[ ]:


#train['timediff'] = train.groupby(['ip']).click_time.diff()
#train['appdiff'] = train.groupby(['ip']).app.diff()
#train['osdiff'] = train.groupby(['ip']).os.diff()
#train['channeldiff'] = train.groupby(['ip']).channel.diff()
#train['devicediff'] = train.groupby(['ip']).device.diff()


# In[ ]:


#train['timediff'] = train['timediff'].dt.total_seconds()/1800.0


# In[ ]:


#train['timediff'] = train['timediff'].fillna(0.0)
#train['appdiff'] = train['appdiff'].fillna(0.0)
#train['osdiff'] = train['osdiff'].fillna(0.0)
#train['devicediff'] = train['devicediff'].fillna(0.0)
#train['channeldiff'] = train['channeldiff'].fillna(0.0)


# In[ ]:


#train['timediff'] = train['timediff'].round()


# In[ ]:


#train['appdiff'] = train['appdiff'].astype(bool).astype(float)
#train['osdiff'] = train['osdiff'].astype(bool).astype(float)
#train['devicediff'] = train['devicediff'].astype(bool).astype(float)
#train['channeldiff'] = train['channeldiff'].astype(bool).astype(float)


# In[ ]:


train = train.drop(columns = ['click_time'])


# In[ ]:


train['count'] = train.groupby(['ip','day','hour']).ip.transform('count')


# In[ ]:


train = train.drop(columns = ['ip','day','hour'])


# In[ ]:


train.head()


# In[ ]:


validate = train.iloc[0:1479231,:]


# In[ ]:


train = train.iloc[1479231:,:]


# In[ ]:


def modelfit(alg,dtrain,predictors,useTrainCV = True, cv_folds = 5, early_stopping_rounds = 20):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label = dtrain['is_attributed'].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round = alg.get_params()['n_estimators'], nfold = cv_folds, metrics = 'auc', early_stopping_rounds = early_stopping_rounds)
        alg.set_params(n_estimators = cvresult.shape[0])
        print('Best n_estimator = ' + str(cvresult.shape[0]))
    alg.fit(dtrain[predictors], dtrain['is_attributed'], eval_metric = 'auc')
    
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    print('\nModel Report:')
    print('Acc: %.4g' % metrics.accuracy_score(dtrain['is_attributed'].values, dtrain_predictions))
    print('AUC: %f' % metrics.roc_auc_score(dtrain['is_attributed'], dtrain_predprob))
    
    feat_imp = pd.Series(alg._Booster.get_fscore()).sort_values(ascending = False)
    feat_imp.plot(kind = 'bar', title = 'Feature Importances')


# In[ ]:


predictors = [x for x in train.columns if x not in ['is_attributed']]


# In[ ]:


len(predictors)


# In[ ]:


xgb1 = XGBClassifier(leanring_rate = 0.15, n_estimators = 150, max_depth = 5, min_child_weight = 1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8, objective = 'binary:logistic', nthread = 4, scale_pos_weight = 1, seed = 27)


# In[ ]:


modelfit(xgb1, train, predictors)


# In[ ]:


valid_pred = xgb1.predict(validate[predictors])
valid_prob = xgb1.predict_proba(validate[predictors])[:,1]


# In[ ]:


print('Acc: %.4g' % metrics.accuracy_score(validate['is_attributed'].values, valid_pred))
print('AUC: %f' % metrics.roc_auc_score(validate['is_attributed'], valid_prob))


# In[ ]:


model_app = Word2Vec.load('../input/wordembedding/vec_app.txt')
model_channel = Word2Vec.load('../input/wordembedding/vec_channel.txt')
model_device = Word2Vec.load('../input/wordembedding/vec_device.txt')
model_os = Word2Vec.load('../input/wordembedding/vec_os.txt')


# In[ ]:


train['x1'],train['x2'],train['x3'] = [0.0,0.0,0.0]
train['y1'],train['y2'],train['y3'] = [0.0,0.0,0.0]
train['z1'],train['z2'],train['z3'] = [0.0,0.0,0.0]
train['v1'],train['v2'],train['v3'] = [0.0,0.0,0.0]


# In[ ]:


train['app'] = train['app'].astype(str)
train['device'] = train['device'].astype(str)
train['os'] = train['os'].astype(str)
train['channel'] = train['channel'].astype(str)


# In[ ]:


train[['x1','x2','x3']] = model_app.wv[train.app]
train[['y1','y2','y3']] = model_device.wv[train.device]
train[['z1','z2','z3']] = model_os.wv[train.os]
train[['v1','v2','v3']] = model_channel.wv[train.channel]


# In[ ]:


train = train.drop(columns = ['app','device','os','channel'])


# In[ ]:


train.head()


# In[ ]:


validate['x1'],validate['x2'],validate['x3'] = [0.0,0.0,0.0]
validate['y1'],validate['y2'],validate['y3'] = [0.0,0.0,0.0]
validate['z1'],validate['z2'],validate['z3'] = [0.0,0.0,0.0]
validate['v1'],validate['v2'],validate['v3'] = [0.0,0.0,0.0]


# In[ ]:


validate['app'] = validate['app'].astype(str)
validate['device'] = validate['device'].astype(str)
validate['os'] = validate['os'].astype(str)
validate['channel'] = validate['channel'].astype(str)


# In[ ]:


validate[['x1','x2','x3']] = model_app.wv[validate.app]
validate[['y1','y2','y3']] = model_device.wv[validate.device]
validate[['z1','z2','z3']] = model_os.wv[validate.os]
validate[['v1','v2','v3']] = model_channel.wv[validate.channel]


# In[ ]:


validate = validate.drop(columns = ['app','device','os','channel'])


# In[ ]:


predictors = [x for x in train.columns if x not in ['is_attributed']]


# In[ ]:


len(predictors)


# In[ ]:


xgb2 = XGBClassifier(leanring_rate = 0.15, n_estimators = 150, max_depth = 6, min_child_weight = 1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8, objective = 'binary:logistic', nthread = 4, scale_pos_weight = 1, seed = 27)


# In[ ]:


modelfit(xgb2, train, predictors)


# In[ ]:


valid_pred = xgb2.predict(validate[predictors])
valid_prob = xgb2.predict_proba(validate[predictors])[:,1]


# In[ ]:


print('Acc: %.4g' % metrics.accuracy_score(validate['is_attributed'].values, valid_pred))
print('AUC: %f' % metrics.roc_auc_score(validate['is_attributed'], valid_prob))


# # Neural Network on Embedded dataset.

# In[ ]:


input_x = tf.placeholder(tf.float32, [None, 13])
input_y = tf.placeholder(tf.float32, [None, 2])


# In[ ]:


w1 = tf.Variable(tf.random_normal([13, 13], stddev = 0.05), name = 'w1')
b1 = tf.Variable(tf.random_normal([13], stddev = 0.05), name = 'b1')
w2 = tf.Variable(tf.random_normal([13, 13], stddev = 0.05), name = 'w2')
b2 = tf.Variable(tf.random_normal([13], stddev = 0.05), name = 'b2')
w3 = tf.Variable(tf.random_normal([13, 13], stddev = 0.05), name = 'w3')
b3 = tf.Variable(tf.random_normal([13], stddev = 0.05), name = 'b3')
w4 = tf.Variable(tf.random_normal([13, 13], stddev = 0.05), name = 'w4')
b4 = tf.Variable(tf.random_normal([13], stddev = 0.05), name = 'b4')
w5 = tf.Variable(tf.random_normal([13, 13], stddev = 0.05), name = 'w5')
b5 = tf.Variable(tf.random_normal([13], stddev = 0.05), name = 'b5')
w6 = tf.Variable(tf.random_normal([13, 2], stddev = 0.05), name = 'w6')
b6 = tf.Variable(tf.random_normal([2], stddev = 0.05), name = 'b6')


# In[ ]:


layer1 = tf.nn.xw_plus_b(input_x, w1, b1, name = 'layer1')
layer1 = tf.nn.relu(layer1)
layer2 = tf.nn.xw_plus_b(layer1, w2, b2, name = 'layer2')
layer2 = tf.nn.relu(layer2)
layer3 = tf.nn.xw_plus_b(layer2, w3, b3, name = 'layer3')
layer3 = tf.nn.relu(layer3)
layer4 = tf.nn.xw_plus_b(layer3, w4, b4, name = 'layer4')
layer4 = tf.nn.relu(layer4)
layer5 = tf.nn.xw_plus_b(layer4, w5, b5, name = 'layer5')
layer5 = tf.nn.relu(layer5)
layer6 = tf.nn.xw_plus_b(layer5, w6, b6, name = 'layer6')
prediction = tf.nn.softmax(layer6)


# In[ ]:


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = layer6, labels = input_y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0013).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(input_y, 1), tf.argmax(prediction, 1)), tf.float32))


# In[ ]:


def generate_batch(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    l = 0
    for epoch in range(num_epochs):
        l += 1
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# In[ ]:


batches1 = generate_batch(train,20000,1)
batches2 = generate_batch(train,20000,1)
batches3 = generate_batch(train,20000,1)
batches4 = generate_batch(train,20000,1)
batches5 = generate_batch(train,20000,1)
batches6 = generate_batch(train,20000,1)
batch_bag = [batches1, batches2, batches3,batches4, batches5, batches6]


# In[ ]:


int((len(train)-1)/20000) + 1


# In[ ]:


init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    
    sess.run(init_op)
    
    print('Start!')
    i = 0
    for batches in batch_bag:
        i += 1
        print('Epoch ' + str(i) + ' start!')
        avg_loss = 0
        avg_acc = 0
        avg_auc = 0
        for batch in batches:
            batch = pd.DataFrame(batch, columns = ['is_attributed','count','x1','x2','x3','y1','y2','y3','z1','z2','z3','v1','v2','v3'])
            x_batch = batch.loc[:, batch.columns != 'is_attributed']
            y_batch = batch.loc[:, batch.columns == 'is_attributed']
            y_batch['is_not_attributed'] = 1 - y_batch['is_attributed']
            _, c, acc, pred = sess.run([optimizer, loss, accuracy, prediction],feed_dict = {input_x: x_batch, input_y: y_batch})
            avg_loss += c
            avg_acc += acc
            avg_auc += metrics.roc_auc_score(y_batch['is_attributed'], pred[:,0])
        
        print('AUC: ' + str(avg_auc/296) )
        print('Average loss is: ' + str(avg_loss/296) + ', Average accuracy is: ' + str(avg_acc/296))
    
        
    print('Evaluation Start!')
    x_validate = validate.loc[:, validate.columns != 'is_attributed']
    y_validate = validate.loc[:, validate.columns == 'is_attributed']
    y_validate['is_not_attributed'] = 1 - y_validate['is_attributed']
    pred, acc = sess.run([prediction, accuracy], feed_dict = {input_x: x_validate, input_y: y_validate})
    print('Accuracy is: ' + str(acc))
    print('AUC: %f' % metrics.roc_auc_score(y_validate['is_attributed'], pred[:,0]))
    print('Finish!')
    df = pd.DataFrame(pred)


# In[ ]:


df.round().mean()


# In[ ]:


validate['is_attributed'].round().mean()


# As you can see from the prediction here, this model is not successful as no one is classified as attributed. The reason might be a inapproperate learning rate. This always happens when we try to use NN to classify rare cases. Another possible reason is the NN is not complex enough to figure out the complex patterns. We could either try a high dimensional embedding or try to make the NN deeper. 

# # CNN on Embedded Dataset.

# In[ ]:




