#!/usr/bin/env python
# coding: utf-8

# # Bidirectional LSTM

# 
# This notebook is a combination of the data generator from Beluga's notebook: https://www.kaggle.com/gaborfodor/greyscale-mobilenet-lb-0-892 and largely based on Kevin Mader's LSTM code, with modifications in the network architecture https://www.kaggle.com/kmader/quickdraw-baseline-lstm-reading-and-submission. 
# 
# I am grateful for their contributions and can take little credit for this notebook. Running this notebook should achieve 0.823 on the LB. 

# In[ ]:


debug = False
if debug: 
    STEPS = 200
    val_steps = 10
else:
    STEPS = 800
    val_steps = 100
    
STROKE_COUNT = 100
EPOCHS = 45
batchsize = 1000


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from keras.metrics import top_k_categorical_accuracy
def top_3_accuracy(x,y): return top_k_categorical_accuracy(x,y, 3)
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from glob import glob
import gc
gc.enable()
def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# In[ ]:


def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


# In[ ]:


from ast import literal_eval

def _stack_it(raw_strokes):
    """preprocess the string and make 
    a standard Nx3 stroke vector"""
    stroke_vec = literal_eval(raw_strokes) # string->list
    # unwrap the list
    in_strokes = [(xi,yi,i)  
     for i,(x,y) in enumerate(stroke_vec) 
     for xi,yi in zip(x,y)]
    c_strokes = np.stack(in_strokes)
    # replace stroke id with 1 for continue, 2 for new
    c_strokes[:,2] = [1]+np.diff(c_strokes[:,2]).tolist()
    c_strokes[:,2] += 1 # since 0 is no stroke
    # pad the strokes with zeros
    return pad_sequences(c_strokes.swapaxes(0, 1), 
                         maxlen=STROKE_COUNT, 
                         padding='post').swapaxes(0, 1)


# In[ ]:


DP_DIR = '../input/shuffle-csv-50k'
INPUT_DIR = '../input/quickdraw-doodle-recognition'
BASE_SIZE = 256
NCSVS = 100
NCATS = 340
np.random.seed(seed=1987)
tf.set_random_seed(seed=1987)

def f2cat(filename: str) -> str:
    return filename.split('.')[0]

def list_all_categories():
    files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))
    return sorted([f2cat(f) for f in files], key=str.lower)



# In[ ]:


def image_generator_xd( batchsize, ks):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                
                df['drawing'] = df['drawing'].map(_stack_it)
                x2 = np.stack(df['drawing'], 0)
                y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                yield x2, y

def df_to_image_array_xd(df):
    df['drawing'] = df['drawing'].map(_stack_it)
    x2 = np.stack(df['drawing'], 0)
    return x2


# In[ ]:


train_datagen = image_generator_xd(batchsize=batchsize, ks=range(NCSVS - 2))
val_datagen = image_generator_xd(batchsize=batchsize, ks=range(NCSVS - 2, NCSVS))


# ### Stroke-based Classification
# Here we use the stroke information to train a model and see if the strokes give us a better idea of what the shape could be. 

# ### LSTM to Parse Strokes
# The model suggeted from the tutorial is
# 
# ![Suggested Model](https://www.tensorflow.org/versions/master/images/quickdraw_model.png)

# In[ ]:


from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, LSTM, Dense, Dropout, Bidirectional
#if len(get_available_gpus())>0:
    # https://twitter.com/fchollet/status/918170264608817152?lang=en
#    from keras.layers import CuDNNLSTM as LSTM # this one is about 3x faster on GPU instances
stroke_read_model = Sequential()
stroke_read_model.add(BatchNormalization(input_shape = (None,)+(3,)))
# filter count and length are taken from the script https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/train_model.py
stroke_read_model.add(Conv1D(256, (5,), activation = 'relu'))
stroke_read_model.add(Dropout(0.2))
stroke_read_model.add(Conv1D(256, (5,), activation = 'relu'))
stroke_read_model.add(Dropout(0.2))
stroke_read_model.add(Conv1D(256, (3,), activation = 'relu'))
stroke_read_model.add(Dropout(0.2))
stroke_read_model.add(Bidirectional(LSTM(128, dropout = 0.3, recurrent_dropout= 0.3,  return_sequences = True)))
stroke_read_model.add(Bidirectional(LSTM(128,dropout = 0.3, recurrent_dropout= 0.3, return_sequences = True)))
stroke_read_model.add(Bidirectional(LSTM(128,dropout = 0.3, recurrent_dropout= 0.3, return_sequences = False)))
stroke_read_model.add(Dense(512, activation = 'relu'))
stroke_read_model.add(Dropout(0.2))
stroke_read_model.add(Dense(NCATS, activation = 'softmax'))
stroke_read_model.compile(optimizer = 'adam', 
                          loss = 'categorical_crossentropy', 
                          metrics = ['categorical_accuracy', top_3_accuracy])
stroke_read_model.summary()


# In[ ]:


weight_path="{}_weights.best.hdf5".format('stroke_lstm_bidirectional_relu')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=4, 
                                   verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=3) 
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[ ]:


# Change the number of epochs to 20


from IPython.display import clear_output
hist = stroke_read_model.fit_generator(train_datagen, steps_per_epoch=STEPS, epochs=EPOCHS, verbose=1,
                        validation_data=val_datagen, validation_steps = val_steps,
                      callbacks = callbacks_list)
clear_output()


# In[ ]:


hist_df = pd.DataFrame(hist.history) 
hist_df.to_csv('hist_training.csv')
hist_df.index = np.arange(1, len(hist_df)+1)
fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(16, 10))
axs[0].plot(hist_df.val_top_3_accuracy, lw=5, label='Validation Accuracy')
axs[0].plot(hist_df.top_3_accuracy, lw=5, label='Training Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].grid()
axs[0].legend(loc=0)
axs[1].plot(hist_df.val_loss, lw=5, label='Validation MLogLoss')
axs[1].plot(hist_df.loss, lw=5, label='Training MLogLoss')
axs[1].set_ylabel('MLogLoss')
axs[1].set_xlabel('Epoch')
axs[1].grid()
axs[1].legend(loc=0)
fig.savefig('hist.png', dpi=300)
plt.show();


# In[ ]:


valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz.gz'.format(NCSVS - 1)), nrows=34000)
x_valid = df_to_image_array_xd(valid_df)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
lstm_results = stroke_read_model.evaluate(x_valid, y_valid, batch_size = 4096)
print('Accuracy: %2.1f%%, Top 3 Accuracy %2.1f%%' % (100*lstm_results[1], 100*lstm_results[2]))


# # Submission
# 

# In[ ]:


sub_df = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))
sub_df['drawing'] = sub_df['drawing'].map(_stack_it)
sub_vec = np.stack(sub_df['drawing'].values, 0)
sub_pred = stroke_read_model.predict(sub_vec, verbose=True, batch_size=4096)


# In[ ]:


top3 = preds2catids(sub_pred)
cats = list_all_categories()
id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
top3cats = top3.replace(id2cat)
sub_df['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
sub_df.head()


# In[ ]:


sub_df[['key_id', 'word']].to_csv('lstm_relu_datagen.csv', index=False)


# In[ ]:




