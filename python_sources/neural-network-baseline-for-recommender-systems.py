#!/usr/bin/env python
# coding: utf-8

# # Recurrent Neural Network based Recommender System
# In this kernel I'm going to show you how to implement a pretty well-performing recommender system using Keras.<br>
# Dataset was provided by [MEGOGO](https://megogo.net/) in [Megogo Challenge](https://www.kaggle.com/c/megogochallenge)<br>
# Target metric - [MAP@10](https://habr.com/ru/company/econtenta/blog/303458/) <br>
# The main idea of this approach is to predict the next film, that user will watch, knowing the sequence of films, that user has watched earlier.<br>
# In the terms of ML-engineering we can define this problem as multiclass (with very large amount of classes) sequences classification.

# ## Data Loading

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


train_data = pd.read_csv('../input/train_data_full.csv')
train_data.head()


# Let's find top 10 most popular films. This will be useful later to make recommendations for users, about whom we don't have historical data.

# In[ ]:


top_10_videos = train_data.loc[train_data.session_start_datetime >= '2018-09-01 00:00:00', 
                               'primary_video_id'].value_counts()[:10].index.tolist()


# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission_full.csv')
sample_submission.primary_video_id = ' '.join([str(v) for v in top_10_videos])
test_users = sample_submission.user_id.unique()
sample_submission.head()


# ## Data Preprocessing

# In[ ]:


# dropping samples with kind of 'negative implicit feedback'
train_data = train_data[train_data.watching_percentage >= 0.5]


# Transforming primary_video_id column in a more suitable representation <br>
# For example, if we have primary_video_id column like [1435, 56453, 1245, 76544], we want to transform it in [2, 3, 1, 4]. Zero value will be used later to pad user-video interactions sequences.

# In[ ]:


train_data.primary_video_id = train_data.primary_video_id.astype('category')
train_data['categ_id'] = train_data.primary_video_id.cat.codes + 1


# In[ ]:


# Let`s define inverse transform dictionary
cat_to_element_uid = dict(zip(
    range(1, len(train_data.primary_video_id.cat.categories) + 1),
    train_data.primary_video_id.cat.categories
))

# Assigning most popular film index to inverse transform of zero padding value
cat_to_element_uid[0] = 29114276


# In the next cell we define sequences of films for each user <br>
# Example of transformation:
# <table style="width:50%">
#   <tr>
#     <th align="left">Before</th>
#     <th align="left">After</th> 
#   </tr>
#   <tr>
#     <td><table style="width:100%">
#   <tr>
#     <th>user_id</th>
#     <th>categ_id</th> 
#   </tr>
#   <tr>
#     <td>12</td>
#     <td>2</td> 
#   </tr>
#   <tr>
#     <td>13</td>
#     <td>1</td> 
#   </tr>  
#   <tr>
#     <td>12</td>
#     <td>1</td> 
#   </tr>    
#    <tr>
#     <td>13</td>
#     <td>2</td> 
#   </tr>  
#   <tr>
#     <td>12</td>
#     <td>1</td> 
#   </tr>    
#   <tr>
#     <td>13</td>
#     <td>3</td> 
#   </tr>   
# </table></td>
#     <td><table style="width:50%">
#   <tr>
#     <th>user_id</th>
#     <th>sequence</th> 
#   </tr>
#   <tr>
#     <td>12</td>
#     <td>[2, 1, 1]</td> 
#   </tr>
#   <tr>
#     <td>13</td>
#     <td>[1, 2, 3]</td> 
# </table></td> 
#   </tr>
# </table>

# In[ ]:


get_ipython().run_cell_magic('time', '', "import tqdm\ntqdm.tqdm.pandas()\nsequences = train_data.groupby('user_id')['categ_id'].progress_apply(list)")


# In[ ]:


sequences.head()


# In[ ]:


# Some statistics
print('Median length: {}\nMean length: {}\nMax length: {}'.format(
    sequences.apply(len).median(), sequences.apply(len).mean(), sequences.apply(len).max()))


# In[ ]:


# We will use users with 5 and more wathced films
sequences2use = sequences[sequences.apply(len) >= 5]


# One of the most important part of this solution is to make X and y for our RNN model <br>
# For example, if we define maxlen = 3, we transform sequence [2, 3, 3, 1, 5, 9] to 
# <table style="width:50%">
#   <tr>
#     <th>X</th>
#     <th>y</th> 
#   </tr>
#   <tr>
#     <td>[3, 1, 5]</td>
#     <td>9</td>
#   </tr>
#   <tr>
#     <td>[3, 3, 1]</td>
#     <td>5</td> 
#   </tr>
#   <tr>
#     <td>[2, 3, 3]</td>
#     <td>1</td> 
#   </tr>
# </table><br>
# So, user who watched a lot of films, will be represented by many sequences, and thus, the size of our training dataset will increase significantly.

# In[ ]:


maxlen = 18 # Length of sequences in X
X = []
y = []

def slice_sequence(seq, num_slices):
    for i in range(1, num_slices):
        X.append(seq[-(i+maxlen): -i])
        y.append(seq[-i])
        
for seq in tqdm.tqdm(sequences2use):
    if len(seq) <= 5:
        slice_sequence(seq, 2)
    elif len(seq) <= 6:
        slice_sequence(seq, 3)
    elif len(seq) <= 8:
        slice_sequence(seq, 4)
    elif len(seq) <= 12:
        slice_sequence(seq, 6)
    elif len(seq) <= 16:
        slice_sequence(seq, 8)
    elif len(seq) <= 20:
        slice_sequence(seq, 11)
    elif len(seq) <= 26:
        slice_sequence(seq, 16)
    else:
        slice_sequence(seq, 23)


# In[ ]:


len(X), len(y)


# In[ ]:


lens = [len(x) for x in X]
max(lens), min(lens), np.mean(lens), np.median(lens)


# In[ ]:


from keras.preprocessing.sequence import pad_sequences

# We should pad our sequences with 0 values, so they all will have the same length
X = pad_sequences(X, maxlen=maxlen)
y = np.array(y)
X.shape, y.shape


# ## Let's define the model architecture

# In[ ]:


from keras.layers import Input, Embedding, SpatialDropout1D, CuDNNLSTM, Dropout, Dense
from keras.models import Model

# Let's set random seed
import tensorflow as tf
tf.set_random_seed(42)
np.random.seed(42)


# In[ ]:


train_data.categ_id.unique().size + 1


# In[ ]:


max_features = train_data.categ_id.unique().size + 1
embed_size = 64

def lstm128():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size)(inp)
    x = SpatialDropout1D(0.05)(x)
    x = CuDNNLSTM(128, return_sequences=False)(x)
    x = Dropout(0.02)(x)
    outp = Dense(max_features, activation="softmax")(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop',
                  metrics=['sparse_categorical_accuracy'])
    return model


# In[ ]:


# Let's train our film recommender system
model = lstm128()
model.fit(X, y, batch_size=2048*4, epochs=25, verbose=True, validation_split=0.01, shuffle=True)


# In[ ]:


model_json = model.to_json()
with open('lstm128.json', 'w') as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights('lstm128.h5')
print("Saved model to disk")


# ## Prediction

# In[ ]:


sequences_test = sequences.apply(lambda x: x[-maxlen:])
sequences_test = sequences_test.apply(lambda x: [0 for i in range(maxlen - len(x))] + x)


# In[ ]:


test_users_in_sequences = sorted(set(sequences_test.index) & set(sample_submission.user_id))


# In[ ]:


X_test = np.array(sequences_test[test_users_in_sequences].tolist())


# In[ ]:


get_ipython().run_cell_magic('time', '', "from itertools import chain\nbatch_size = 2048*8\nn_batches = int(X_test.shape[0]/batch_size) + 1\npreds = []\n\nfor batch_ind in tqdm.tqdm(range(n_batches)):\n    batch = X_test[batch_ind*batch_size: (batch_ind + 1)*batch_size]\n    curr_preds = model.predict(batch)\n    curr_preds = np.argsort(-curr_preds)[:, :10]\n    curr_preds = [[cat_to_element_uid[x] for x in row] for row in curr_preds]\n    preds.append([' '.join(map(lambda x: str(x), row)) for row in curr_preds])\n    \npreds = list(chain(*preds))")


# In[ ]:


sample_submission.index = sample_submission.user_id
sample_submission.primary_video_id[test_users_in_sequences] = preds
sample_submission.to_csv('submission_lstm.csv', header=True, index=False)

