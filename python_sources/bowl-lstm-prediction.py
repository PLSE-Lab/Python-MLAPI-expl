#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import json
import seaborn as sns
import datetime
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
import warnings
import random
import plotly.express as px
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
my_pal = sns.color_palette(n_colors=10)


# In[ ]:


n = 11341042 #number of records in file
s = 2000000 #desired sample size
filename = '../input/data-science-bowl-2019/train.csv'
skip = sorted(random.sample(range(n),n-s))
train = pd.read_csv(filename, skiprows=skip)
train.columns = ['event_id','game_session','timestamp','event_data',
            'installation_id','event_count','event_code','game_time','title','type','world']


# In[ ]:


# train = pd.read_csv('../input/data-science-bowl-2019/train.csv')
test = pd.read_csv('../input/data-science-bowl-2019/test.csv')
labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')


# ### LSTM Experiments

# In[ ]:


from more_itertools import sliced
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks.callbacks import EarlyStopping


# In[ ]:


full = train.merge(labels, how='inner', on=['installation_id','game_session'])
train_ls = full[['installation_id','game_session','event_id']]
# convert to str
train_ls['event_id'] = train_ls['event_id'].apply(lambda x: str(x))


# In[ ]:


del train


# In[ ]:


def events_all(aa):
    xx = ''
    for i in aa: 
        xx += i + ' '
    xx = xx.rstrip()
    return xx


# In[ ]:


result = train_ls.groupby(['installation_id','game_session']).sum().reset_index()
result['event_id'] = result['event_id'].apply(lambda x: list(sliced(x, 8)))
result['new_event'] = result['event_id'].apply(events_all)
result = result.merge(labels, how='inner', on=['installation_id','game_session'])[['new_event','accuracy_group']]


# In[ ]:


result.head()
# plt.scatter(result['accuracy_group'], result['event_code'].apply(lambda x: len(x)));


# In[ ]:


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 100
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 500
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(result['new_event'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[ ]:


X = tokenizer.texts_to_sequences(result['new_event'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)


# In[ ]:


Y = pd.get_dummies(result['accuracy_group']).values
print('Shape of label tensor:', Y.shape)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[ ]:


model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# choose epochs and batch_size
epochs = 5
batch_size = 64
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001)])


# In[ ]:


accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


# In[ ]:


plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();


# ### Test prediction:

# In[ ]:


last_test = test[['installation_id','game_session',
                  'timestamp']].groupby(['installation_id']).tail(1)[['installation_id','game_session']]
test_ = test.merge(last_test,how='inner', on=['installation_id','game_session'])


# In[ ]:


test_ls = test_[['installation_id','game_session','event_id']]
# test_ls = test[['installation_id','game_session','event_id']]
test_ls['event_id'] = test_ls['event_id'].apply(lambda x: str(x))
res_test = test_ls.groupby(['installation_id','game_session']).sum().reset_index()
res_test['event_id'] = res_test['event_id'].apply(lambda x: list(sliced(x, 8)))
res_test['new_event'] = res_test['event_id'].apply(events_all)


# In[ ]:


X_ts = tokenizer.texts_to_sequences(res_test['new_event'].values)
X_ts = pad_sequences(X_ts, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X_ts.shape)


# In[ ]:


test_pred = model.predict(X_ts)


# In[ ]:


submission = pd.concat([res_test['installation_id'],
                                     pd.DataFrame(test_pred).idxmax(1)], axis=1)
submission.columns = ['installation_id','accuracy_group']


# In[ ]:


# submission.to_csv('submission.csv')
submission.to_csv('submission.csv', index=None)
submission.head()


# In[ ]:


submission['accuracy_group'].hist();

