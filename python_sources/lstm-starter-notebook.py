#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Let's load in some basics and make sure our files are all here
import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
sample_submission = pd.read_csv('../input/sample_submission.csv')


# Our files are here and load without error. Let's check to make sure the columns match and see the submission format.

# In[ ]:


print(test.columns)
print(train.columns)
print(sample_submission.head(4))


# How much data is there?

# In[ ]:


print(train.shape)
print(test.shape)


# Let's find a row that is labeled with "is_turkey" and play the clip it comes from.

# In[ ]:


print(train[train['is_turkey']==1].index)


# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo(train['vid_id'][1],start=train['start_time_seconds_youtube_clip'][1],end=train['end_time_seconds_youtube_clip'][1])


# Yep, that sounds like a turkey!
# 
# We also have some VGGish audio embeddings to take a look at. Let's see what the shape of those are.

# In[ ]:


print(train['audio_embedding'].head())

#see the possible list lengths of the first dimension
print("train's audio_embedding can have this many frames: "+ str(train['audio_embedding'].apply(lambda x: len(x)).unique())) 
print("test's audio_embedding can have this many frames: "+ str(test['audio_embedding'].apply(lambda x: len(x)).unique())) 

#see the possible list lengths of the first element
print("each frame can have this many features: "+str(train['audio_embedding'].apply(lambda x: len(x[0])).unique()))


# For more information on what these features are and where they come from, see [this page.](https://github.com/tensorflow/models/tree/master/research/audioset#input-audio-features)
# 
# Now that we've seen what the given data looks like, let's make a prediction using a basic LSTM model. 

# In[ ]:


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Bidirectional, LSTM, BatchNormalization, Dropout
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

#split the training data to have a validation set
train_train, train_val = train_test_split(train)
xtrain = [k for k in train_train['audio_embedding']]
ytrain = train_train['is_turkey'].values

xval = [k for k in train_val['audio_embedding']]
yval = train_val['is_turkey'].values

# Pad the audio features so that all are "10 seconds" long
x_train = pad_sequences(xtrain, maxlen=10)
x_val = pad_sequences(xval, maxlen=10)

y_train = np.asarray(ytrain)
y_val = np.asarray(yval)

#Define a basic LSTM model
model = Sequential()
model.add(BatchNormalization(input_shape=(10, 128)))
model.add(Dropout(.5))
model.add(Bidirectional(LSTM(128, activation='relu')))
model.add(Dense(1, activation='sigmoid'))

#maybe there is something better to use, but let's use binary_crossentropy
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#fit on a portion of the training data, and validate on the rest
model.fit(x_train, y_train,
          batch_size=300,
          nb_epoch=4,validation_data=(x_val, y_val))

# Get accuracy of model on validation data. It's not AUC but it's something at least!
score, acc = model.evaluate(x_val, y_val, batch_size=300)
print('Test accuracy:', acc)


# Finally, let's get a prediction to submit. 

# In[ ]:


test_data = [k for k in test['audio_embedding']]
submission = model.predict_classes(pad_sequences(test_data))
submission = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':[x for y in submission for x in y]})


# In[ ]:


print(submission.head()) #check to see that it looks like the sample submission


# In[ ]:


submission.to_csv('lstm_starter.csv', index=False) #drop the index so it matches the submission format.

