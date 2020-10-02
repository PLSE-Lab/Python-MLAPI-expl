#!/usr/bin/env python
# coding: utf-8

# In[93]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[94]:


train = pd.read_csv('../input/boston_train.csv')
train.head()


# In[95]:


train.info()


# In[96]:


test = pd.read_csv('../input/boston_test.csv')
test.head()


# In[97]:


cols = train.columns
cols


# In[98]:


train.isna().sum()


# In[99]:


default_types =[[0.0]]*len(cols)
default_types    


# In[100]:


y_name = 'medv'
batch_size = 128
num_epochs = 400
buffer  = 1000
split = 0.7


# In[101]:


def parse_line(line):
    columns = tf.decode_csv(line,default_types)
    features = dict(zip(cols,columns))

    label = features.pop(y_name)
    return features, label


# In[105]:


data = tf.data.TextLineDataset('../input/boston_train.csv').skip(1)


# **Divide data into train and validation data**

# In[106]:


def in_training_set(line):
    num_buckets = 1000000
    bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
    return bucket_id < int(split * num_buckets)

def in_test_set(line):
    return ~in_training_set(line)


# In[107]:


train = (data.filter(in_training_set).map(parse_line))
validation = (data.filter(in_test_set).map(parse_line))


# In[108]:


def X():
    return train.repeat().shuffle(buffer).batch(batch_size).make_one_shot_iterator().get_next()
def Y():
    return validation.shuffle(buffer).batch(batch_size).make_one_shot_iterator().get_next()


# In[120]:


sess = tf.Session()
#sess.run(validation)


# Define model.. train on X

# In[111]:


feature_columns = []
for col in cols[1:-1]:
    feature_columns.append(tf.feature_column.numeric_column(col))


# In[112]:


model = tf.estimator.DNNRegressor(feature_columns=feature_columns, hidden_units=[10,10])


# In[137]:


model.train(input_fn= X,steps=500)


# In[138]:


eval_result = model.evaluate(input_fn=Y)


# In[139]:


for key in sorted(eval_result):
    print('%s: %s' % (key, eval_result[key]))


# **Testing**

# In[128]:


test.head()


# In[146]:


test_in = tf.estimator.inputs.pandas_input_fn(test, shuffle=False)
test_in


# In[151]:


pred_iter = model.predict(input_fn=test_in)
predC = []
for i,pred in enumerate(pred_iter):
    print(test['ID'][i],pred['predictions'][0])
    predC.append(pred['predictions'][0])
    
out_df = pd.DataFrame({"ID":test['ID'], "medv":predC})
file = out_df.to_csv("submission.csv", index=False)


# In[152]:


print(os.listdir('../working'))

