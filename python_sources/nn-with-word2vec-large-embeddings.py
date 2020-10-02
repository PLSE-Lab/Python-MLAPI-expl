#!/usr/bin/env python
# coding: utf-8

# #   Word Embedding Idea.

# The reason i think Word2Vec might working here is dataset is sequential. If you sorted by ip and click time, you will make your datset as a sequence of clicks by different ip. If you take all clicks from one ip, that shows the whole process of that 'ip' 's behavioral pattern. It is sure that for different ips their behavior should be different. But how about commons? How could we extract the relationships of app, device, os and channel? What is the relationship between app 9 and app 10? That is something we need to take into consideration. By applying Word2Vec here, I diminished the specific pattern from different ips and get the 'word embeddings' of each categories in app, device, os and channel based on their own characteristics. You might experienced an ad for game app that pop up during you play a mobile game and after you click that ad another ad for another game might pop up. This is the 'charateristic' that I want to figure out and include in my training dataset.

# Personally I dont think there will be big difference between applying XGBoost on original dataset and on embedded dataset. The reason is obvious that decision tree works in a way that dimension expand doesn't really make effort on increasing the accuracy. But I still apply it here becuase i want to see how are those word embeddings performs and may be find out how long should my word embedding is. I use 3 as length for each predictors here, as you see from feature importance, it is a little bit suprising that for word embedding 'app', x2 doesn't play a same role as x1 and x3. So may be when we increase the dimension of word embeddings and set col_sample by tree a small value, we may get some unexpected good result.

#  Now i increase the dimension of word embeddings to a size of 300. The result AUC score goes up dramatially. (This kernel is an extention of my original kernel. https://www.kaggle.com/jingqliu/xgboost-nn-on-small-sample-with-word2vec)

# In[1]:


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


train = pd.read_csv('../input/readyforuse/ready.csv',nrows = 3698078*4)


# In[ ]:


test = pd.read_csv('../input/readyforuse/testready.csv')


# In[ ]:


train.head()


# In[ ]:


train = train.iloc[1479231:,:]


# In[ ]:


model_app = Word2Vec.load('../input/large-word-embeddings/vec_app.txt')
model_channel = Word2Vec.load('../input/large-word-embeddings/vec_channel.txt')
model_device = Word2Vec.load('../input/large-word-embeddings/vec_device.txt')
model_os = Word2Vec.load('../input/large-word-embeddings/vec_os.txt')


# In[ ]:


train['app'] = train['app'].astype(str)
train['device'] = train['device'].astype(str)
train['os'] = train['os'].astype(str)
train['channel'] = train['channel'].astype(str)


# In[ ]:


test['app'] = test['app'].astype(str)
test['device'] = test['device'].astype(str)
test['os'] = test['os'].astype(str)
test['channel'] = test['channel'].astype(str)


# # Neural Network on Embedded dataset.

# In[ ]:


input_x = tf.placeholder(tf.float32, [None, 301])
input_y = tf.placeholder(tf.float32, [None, 2])


# In[ ]:


w1 = tf.Variable(tf.random_normal([301, 301], stddev = 0.05), name = 'w1')
b1 = tf.Variable(tf.random_normal([301], stddev = 0.05), name = 'b1')
w2 = tf.Variable(tf.random_normal([301, 150], stddev = 0.05), name = 'w2')
b2 = tf.Variable(tf.random_normal([150], stddev = 0.05), name = 'b2')
w3 = tf.Variable(tf.random_normal([150, 150], stddev = 0.05), name = 'w3')
b3 = tf.Variable(tf.random_normal([150], stddev = 0.05), name = 'b3')
w4 = tf.Variable(tf.random_normal([150, 150], stddev = 0.05), name = 'w4')
b4 = tf.Variable(tf.random_normal([150], stddev = 0.05), name = 'b4')
w5 = tf.Variable(tf.random_normal([150, 2], stddev = 0.05), name = 'w5')
b5 = tf.Variable(tf.random_normal([2], stddev = 0.05), name = 'b5')


# In[ ]:


layer1 = tf.nn.xw_plus_b(input_x, w1, b1, name = 'layer1')
layer1 = tf.nn.relu(layer1)
layer1 = tf.nn.dropout(layer1,0.8)
layer2 = tf.nn.xw_plus_b(layer1, w2, b2, name = 'layer2')
layer2 = tf.nn.relu(layer2)
layer3 = tf.nn.xw_plus_b(layer2, w3, b3, name = 'layer3')
layer3 = tf.nn.relu(layer3)
layer4 = tf.nn.xw_plus_b(layer3, w4, b4, name = 'layer4')
layer4 = tf.nn.relu(layer4)
layer5 = tf.nn.xw_plus_b(layer4, w5, b5, name = 'layer5')
prediction = tf.nn.softmax(layer5)


# In[ ]:


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = layer5, labels = input_y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0016).minimize(loss)
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


train1 = train.iloc[0:3698078,:]
train2 = train.iloc[3698078:3698078*2,:]
train3 = train.iloc[3698078*2:3698078*3,:]
train4 = train.iloc[3698078*3:,:]


# In[ ]:


batches1 = generate_batch(train1,10000,1)
batches2 = generate_batch(train2,10000,1)
batches3 = generate_batch(train3,10000,1)
batches4 = generate_batch(train4,10000,1)
batches5 = generate_batch(train1,10000,1)
batches6 = generate_batch(train2,10000,1)
batches7 = generate_batch(train3,10000,1)
batches8 = generate_batch(train4,10000,1)
batches9 = generate_batch(train1,10000,1)
batches10 = generate_batch(train2,10000,1)
batches11 = generate_batch(train3,10000,1)
batches12 = generate_batch(train4,10000,1)
batch_bag1 = [batches1,batches2,batches3,batches4]
batch_bag2 = [batches5,batches6,batches7,batches8]
batch_bag3 = [batches9,batches10,batches11,batches12]
batch_bags = [batch_bag1, batch_bag2, batch_bag3]


# In[ ]:


(int((len(train1)-1)/10000) + 1) * 4


# In[ ]:


test_blocks = generate_batch(test, 20000, 1, shuffle = False)


# In[ ]:


init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    
    sess.run(init_op)
    
    print('Start!')
    i = 0
    for batch_bag in batch_bags:
        i += 1
        print('Epoch ' + str(i) + ' Start!')
        avg_loss = 0
        avg_acc = 0
        avg_auc = 0
        for batches in batch_bag:
            for batch in batches:
                batch = pd.DataFrame(batch, columns = ['app','device','os','channel','is_attributed','count'])
                x_batch = batch.loc[:, batch.columns != 'is_attributed']
                y_batch = batch.loc[:, batch.columns == 'is_attributed']
                x_batch = pd.concat([x_batch, pd.DataFrame(model_app.wv[x_batch['app']]), pd.DataFrame(model_channel.wv[x_batch['channel']]), pd.DataFrame(model_os.wv[x_batch['os']]), pd.DataFrame(model_device.wv[x_batch['device']])],axis = 1)
                x_batch = x_batch.drop(columns = ['app','os','device','channel'])
                y_batch['is_not_attributed'] = 1 - y_batch['is_attributed']
                _,c, acc, pred = sess.run([optimizer, loss, accuracy, prediction],feed_dict = {input_x: x_batch, input_y: y_batch})
                avg_loss += c
                avg_acc += acc
                avg_auc += metrics.roc_auc_score(y_batch['is_attributed'].astype(int), pred[:,0])
        print('Average loss is: ' + str(avg_loss/1480) + ', Average accuracy is: ' + str(avg_acc/1480) + ', Average AUC is: ' + str(avg_auc/1480))
    
    print('Prediction Start!')
    
    df = pd.DataFrame()
    for block in test_blocks:
        block = pd.DataFrame(block, columns = ['app', 'device', 'os', 'channel', 'count'])
        block = pd.concat([block, pd.DataFrame(model_app.wv[block['app']]), pd.DataFrame(model_channel.wv[block['channel']]), pd.DataFrame(model_os.wv[block['os']]), pd.DataFrame(model_device.wv[block['device']])],axis = 1)
        block = block.drop(columns = ['app','device','os','channel'])
        pred = sess.run(prediction, feed_dict = {input_x: block})
        df = df.append(pd.DataFrame(pred))
    
    print('Finish!')


# In[ ]:


df.round().mean()


# In[ ]:


submission = pd.read_csv("../input/talkingdata-adtracking-fraud-detection/sample_submission.csv")
df.columns = ['is_attributed','b']
submission['is_attributed'] = np.array(df['is_attributed'])
submission.to_csv("submission.csv", index=False)

