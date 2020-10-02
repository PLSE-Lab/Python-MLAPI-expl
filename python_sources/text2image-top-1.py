#!/usr/bin/env python
# coding: utf-8

# ## Using text features to predict image_top_1
# 
# This kernel aims to demonstrate that training an image_top_1 model using the text fields, might come handy for 3 things:
# 
# * Train word embeddings that have more focus on visual content
# * predict image_top_1 which are NaN
# * use the difference of our image top 1 prediction and the actual prediction as feature (not implemented)
# 

# In[1]:


import pandas as pd
import time
import gc


# Let's start by loading train and test data and concatenate for easier processing. We still keep the indices to split it up later again. Additionally we get the indices of items that have nan in the image_top_1 field. We don't want thos in our model.
# 

# In[3]:


text_cols = ['param_1','param_2','param_3','title','description']
print('loading train...')
train = pd.read_csv('../input/train.csv', index_col = 'item_id', usecols = text_cols + ['item_id','image_top_1'])
train_indices = train.index
print('loading test')
test = pd.read_csv('../input/test.csv', index_col = 'item_id', usecols = text_cols + ['item_id','image_top_1'])
test_indices = test.index
print('concat dfs')
df = pd.concat([train,test])
nan_indices = df[pd.isnull(df['image_top_1'])].index
not_nan_indices = df[pd.notnull(df['image_top_1'])].index

#df = df[pd.notnull(df['image_top_1'])]

del train, test


# Join the text for having one text feature.

# In[ ]:


print('cleaning text')

for col in text_cols:
    df[col] = df[col].fillna('nan').astype(str)
print('concat text')
df['text'] = df[text_cols].apply(lambda x: ' '.join(x), axis=1)
df.drop(text_cols,axis = 1, inplace = True)


# Next we fit the keras build in tokenizer on our text, and pad the sequences. We already split by those items that have image_top_1 (X_train) and those which haven't (X_test). Keep in mind that the tokenizer will have our word2index dictionary, whcih we will need later to build our embeddings dictionary.

# In[5]:


from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences

max_features = 100000 # max amount of words considered
max_len = 100 #maximum length of text
dim = 100 #dimension of embedding


print('tokenizing...',end='')
tic = time.time()
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(df['text'].values))
toc = time.time()
print('done. {}'.format(toc-tic))

col = 'text'
print("   Transforming {} to seq...".format(col))
tic = time.time()
df[col] = tokenizer.texts_to_sequences(df[col])
toc = time.time()
print('done. {}'.format(toc-tic))

print('padding X_train')
tic = time.time()
X_train = pad_sequences(df.loc[not_nan_indices,col], maxlen=max_len)
toc = time.time()
print('done. {}'.format(toc-tic))

print('padding X_nan')
tic = time.time()
X_nan = pad_sequences(df.loc[nan_indices,col], maxlen=max_len)
toc = time.time()
print('done. {}'.format(toc-tic))

df.drop(['text'], axis = 1, inplace=True)


# As target value we take the image_top_1 values. I first had them one-hot-encoded with categorical_crossentropy as loss but using sparse_categorical_crossentropy is more memory efficient.

# In[ ]:


y = df.loc[not_nan_indices,'image_top_1'].values


# Ok, now we have our input data and label. So let's patch up a model. Feel free to improve :)

# In[10]:


import numpy as np
from keras.layers import Input,PReLU,BatchNormalization, GlobalMaxPooling1D, GlobalAveragePooling1D, CuDNNGRU, Bidirectional, Dense, Embedding
from keras.layers import Concatenate, Flatten, Bidirectional
from keras.optimizers import Adam
from keras.initializers import he_uniform
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping


from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy



def all_pool(tensor):
    avg_tensor = GlobalAveragePooling1D()(tensor)
    max_tensor = GlobalMaxPooling1D()(tensor)
    res_tensor = Concatenate()([avg_tensor, max_tensor])
    return res_tensor

def build_model():
    inp = Input(shape=(max_len,))

    embedding = Embedding(max_features + 1, dim)(inp)
    x = Bidirectional(CuDNNGRU(64,return_sequences=True))(embedding)
    x = all_pool(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation = 'relu')(x)
    out = Dense(3067, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(optimizer=Adam(lr=0.0005), loss=sparse_categorical_crossentropy)
    return model

model = build_model()
model.summary()


# Let's use some build in functionality of keras to make our life easier. check_point for saving, early stopping to stop at plateau and a validation_split for cv.

# In[11]:


early_stop = EarlyStopping(patience=2)
check_point = ModelCheckpoint('model.hdf5', monitor = "val_loss", mode = "min", save_best_only = True, verbose = 1)

history = model.fit(X_train, y, batch_size = 512, epochs = 10,
                verbose = 1, validation_split=0.1,callbacks=[early_stop,check_point])


# Nice, now we have a model predicting image_top_1 value from text. So what can we do with it? First let's extract the trained embeddings. We can use them directly in any NN architecture. I would not sorely use them but they might be worth concatenating with other embeddings. 

# In[12]:


id2word = {tokenizer.word_index[word]:word for word in tokenizer.word_index}
weights = model.layers[1].get_weights()[0]
embedding_dict = {}
for id in id2word:
    if id <= weights.shape[0]-1:
        embedding_dict[id2word[id]] = weights[id]


# Since now they are already trained, let's save for other people :)

# In[ ]:


import pickle
with open('embedding_dict.p','wb') as f:
    pickle.dump(embedding_dict,f)


# So lets get the predictions for our missing image classes and put them into our train-test dataframe. Here you can play around with a confidence-threshold (I set 0.1) . You don't want to messup your image_top_1 data with classes your not confident with, and since we just used the text there might be some items where we don't have any info at all.

# In[13]:


preds = model.predict(X_nan,verbose=1)


# In[18]:


k = 0
classes = np.zeros(shape=np.argmax(preds,axis = 1).shape)
for i in range(preds.shape[0]):
    if np.max(preds[i]) > 0.1:
        k+=1
        classes[i] = np.argmax(preds[i])
    else:
        classes[i] = np.nan
df.loc[nan_indices,'image_top_1'] = classes


# Let's also save our "corrected" image_top_1 column for others. Cheers and happy kaggling

# In[ ]:


df.loc[train_indices].to_csv('train_image_top_1_features.csv')
df.loc[test_indices].to_csv('test_image_top_1_features.csv')

