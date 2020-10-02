#!/usr/bin/env python
# coding: utf-8

# * * # Keras Using LSTM on full Data
# 
# 

# ## Setup
# Import the necessary libraries and a few helper functions.

# In[ ]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import os
import ast
import datetime as dt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.sequence import pad_sequences
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import time
tic = time.time()


# In[ ]:


np.random.seed(seed=1988)
tf.set_random_seed(seed=1988)


# In[ ]:


DP_DIR = '../input/shuffle-csvs/'
INPUT_DIR = '../input/quickdraw-doodle-recognition/'


BASE_SIZE = 256
NCSVS = 100
NCATS = 340

def f2cat(filename: str) -> str:
    return filename.split('.')[0]

def list_all_categories():
    files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))
    return sorted([f2cat(f) for f in files], key=str.lower)


# In[ ]:


def avg_precision(actual, predicted, k=3):
    if not actual:
        return 0.
    if len(predicted)>k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, n in enumerate(predicted):
        if n in actual and n not in predicted[:i]:
            num_hits += 1.0
            score += num_hits/(i + 1.)
    return score/min(len(actual),k)

def mapk(actual, predicted, k=3):
    return np.mean([avg_precision(a,p,k) for a,p in zip(actual, predicted)])

def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:,:3], columns=['a','b','c'])

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


# 

# In[ ]:


def _stack_it(stroke_vec):
    """preprocess the string and make 
    a standard Nx3 stroke vector"""
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


def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0])-1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                        (stroke[0][i+1], stroke[1][i+1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size,size))
    else:
        return img


# ## Training with Image Generator

# In[ ]:


batchsize = 256
STROKE_COUNT = 196
STEPS = 800
EPOCHS = 20


# In[ ]:


def image_generator_xd( batchsize, ks, lw=6, time_color=True):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
            if not os.path.exists(filename):
                continue
            for df in pd.read_csv(filename, chunksize=batchsize):
                # Generator of multiple batches. Each iter is on single file of batchsize
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                x = np.zeros((len(df), 196, 3))
                for i, raw_strokes in enumerate(df.drawing.values):
                    x[i, :, :] = _stack_it(raw_strokes)
                y=keras.utils.to_categorical(df.y, num_classes=NCATS) # y should be equal to the word
                yield x,y

def df_to_image_array_xd(df, lw=6, time_color=True):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), 196, 3))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i,:,:] = _stack_it(raw_strokes)
    return x


# In[ ]:


valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=34000)
x_valid = df_to_image_array_xd(valid_df)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
print(x_valid.shape, y_valid.shape)
print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))


# In[ ]:


train_datagen = image_generator_xd( batchsize=batchsize, ks=range(NCSVS - 1))
x, y = next(train_datagen)


# In[ ]:


from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, LSTM, Dense, Dropout
stroke_read_model = Sequential()
stroke_read_model.add(BatchNormalization(input_shape = (None,)+x.shape[2:]))
# filter count and length are taken from the script https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/train_model.py
stroke_read_model.add(Conv1D(48, (5,)))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(Conv1D(64, (5,)))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(Conv1D(96, (3,)))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(LSTM(128, return_sequences = True))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(LSTM(128, return_sequences = False))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(Dense(512))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(Dense(NCATS, activation = 'softmax'))
stroke_read_model.compile(optimizer = 'adam', 
                          loss = 'categorical_crossentropy', 
                          metrics = ['categorical_accuracy', top_3_accuracy])
stroke_read_model.summary()


# In[ ]:


weight_path="{}_weights.best.hdf5".format('stroke_lstm_model_generator')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, 
                                   verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=5) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[ ]:


stroke_read_model.fit_generator(train_datagen, 
                      validation_data = (x_valid, y_valid), 
                                verbose=1,
                                steps_per_epoch=STEPS,
                      epochs = EPOCHS,
                      callbacks = callbacks_list)


# In[ ]:


valid_predictions = stroke_read_model.predict(x_valid, batch_size=128, verbose=1)
map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)
print('Map3: {:.3f}'.format(map3))


# ## Create Submission

# In[ ]:


test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))
test.head()
x_test = df_to_image_array_xd(test)
print(test.shape, x_test.shape)
print('Test array memory {:.2f} GB'.format(x_test.nbytes / 1024.**3 ))


# In[ ]:


test_predictions = stroke_read_model.predict(x_test, batch_size=128, verbose=1)

top3 = preds2catids(test_predictions)
top3.head()
top3.shape

cats = list_all_categories()
id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
top3cats = top3.replace(id2cat)
top3cats.head()
top3cats.shape


# In[ ]:


test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
submission = test[['key_id', 'word']]
submission.to_csv('lstm_generator_submission_{}.csv'.format(int(map3 * 10**4)), index=False)
submission.head()
submission.shape


# In[ ]:


toc=time.time()
print('Total time taken: %0.2f sec'%(toc-tic))

