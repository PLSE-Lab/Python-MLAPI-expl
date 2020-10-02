#!/usr/bin/env python
# coding: utf-8

# # Keras Simple CNN Benchmark
# 
# I assume you are already familiar with the competition dataset. This benchmark helps you to reach 0.78 MAP@3.
# 
# This kernel has three main components:
# 
# * Simple configurable Convolutional Network
# * Fast and memory efficient Image Generator
# * Flip Augmentation
# * Full training & submission with Kaggle Kernel
# * TTA
# 
# I did some paramer search but it should not be hard to improve the current score. Simplified versions could be trained without GPU.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
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
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

start = dt.datetime.now()


# In[ ]:


DP_DIR = '../input/shuffle-csvs/'
INPUT_DIR = '../input/quickdraw-doodle-recognition/'
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


def apk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])


# ## Simple ConvNet

# In[ ]:


def custom_single_cnn(size, conv_layers=(8, 16, 32, 64), dense_layers=(512, 256), conv_dropout=0.2,
                      dense_dropout=0.2):
    model = Sequential()
    model.add( Conv2D(conv_layers[0], kernel_size=(3, 3), padding='same', activation='relu', input_shape=(size, size, 1)) )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #if conv_dropout:
    #    model.add(Dropout(conv_dropout))

    for conv_layer_size in conv_layers[1:]:
        model.add(Conv2D(conv_layer_size, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        if conv_dropout:
            model.add(Dropout(conv_dropout))

    model.add(Flatten())
    if dense_dropout:
        model.add(Dropout(dense_dropout))

    for dense_layer_size in dense_layers:
        model.add(Dense(dense_layer_size, activation='relu'))
        model.add(Activation('relu'))
        if dense_dropout:
            model.add(Dropout(dense_dropout))

    model.add(Dense(NCATS, activation='softmax'))
    return model

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top3_acc( tgt, pred ):
    sc = np.mean( (pred[:,0]==tgt) | (pred[:,1]==tgt) | (pred[:,2]==tgt) )
    return sc


# In[ ]:


STEPS = 500
size = 32
batchsize = 512


# In[ ]:


model = custom_single_cnn(size=size,
                          conv_layers=[128, 128],
                          dense_layers=[2048],
                          conv_dropout=False,
                          dense_dropout=0.10 )
model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
print(model.summary())


# ## Training with Image Generator

# In[ ]:


def draw_cv2(raw_strokes, size=256, lw=6):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for stroke in raw_strokes:
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), 255, lw)
    if np.random.rand()>0.5:
        img = np.fliplr(img)
    if np.random.rand()>0.75:
        if np.random.rand()>0.50:
            img = img[ 4:, 4: ]
        else:
            img = img[ :-4, :-4 ]
    if np.random.rand()>0.50:
        img2 = cv2.resize(img, (200, 200))
        img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
        img[18:218,18:218] = img2

    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img

def image_generator(size, batchsize, ks, lw=6):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                x = np.zeros((len(df), size, size))
                for i, raw_strokes in enumerate(df.drawing.values):
                    x[i] = draw_cv2(raw_strokes, size=size, lw=lw)
                x = x / 255.
                x = x.reshape((len(df), size, size, 1)).astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                yield x, y

def df_to_image_array(df, size, lw=6):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), size, size))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i] = draw_cv2(raw_strokes, size=size, lw=lw)
    x = x / 255.
    x = x.reshape((len(df), size, size, 1)).astype(np.float32)
    return x


# In[ ]:


valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=55555)
x_valid = df_to_image_array(valid_df, size)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
print(x_valid.shape, y_valid.shape)
print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))


# In[ ]:


train_datagen = image_generator(size=size, batchsize=batchsize, ks=range(NCSVS - 1))


# In[ ]:


x, y = next(train_datagen)
n = 10
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(10, 10))
for i in range(n**2):
    ax = axs[i // n, i % n]
    ax.imshow(x[i, :, :, 0], cmap=plt.cm.gray)
    ax.axis('off')
plt.tight_layout()
plt.show();


# In[ ]:


callbacks = [
    #EarlyStopping(monitor='val_top_3_accuracy', patience=15, min_delta=0.001, mode='max'),
    ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=5, min_delta=0.005, mode='max', cooldown=3),
    ModelCheckpoint("./black-white-7.model",monitor='val_top_3_accuracy', mode = 'max', save_best_only=True, verbose=1)
]
hist = model.fit_generator(
    train_datagen, steps_per_epoch=STEPS, epochs=10, verbose=1,
    validation_data=(x_valid, y_valid),
    callbacks = callbacks
)


# In[ ]:


hist = model.fit_generator(
    train_datagen, steps_per_epoch=STEPS, epochs=10, verbose=1,
    validation_data=(x_valid, y_valid),
    callbacks = callbacks
)


# In[ ]:


hist = model.fit_generator(
    train_datagen, steps_per_epoch=STEPS, epochs=10, verbose=1,
    validation_data=(x_valid, y_valid),
    callbacks = callbacks
)


# In[ ]:


hist = model.fit_generator(
    train_datagen, steps_per_epoch=STEPS, epochs=10, verbose=1,
    validation_data=(x_valid, y_valid),
    callbacks = callbacks
)


# In[ ]:


hist = model.fit_generator(
    train_datagen, steps_per_epoch=STEPS, epochs=10, verbose=1,
    validation_data=(x_valid, y_valid),
    callbacks = callbacks
)


# In[ ]:


hist = model.fit_generator(
    train_datagen, steps_per_epoch=STEPS, epochs=10, verbose=1,
    validation_data=(x_valid, y_valid),
    callbacks = callbacks
)


# In[ ]:


hist_df = pd.DataFrame(hist.history)
fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(16, 10))
axs[0].plot(hist_df.val_categorical_accuracy, lw=5)
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].grid()
axs[1].plot(hist_df.val_categorical_crossentropy, lw=5)
axs[1].set_ylabel('MLogLoss')
axs[1].grid()
plt.show();


# In[ ]:


#Load Best Epoch
#model = load_model("./black-white-1.model", custom_objects={'top_3_accuracy': top_3_accuracy } )
valid_predictions1 = model.predict(x_valid, batch_size=128, verbose=1)
map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions1).values)
top3 = top3_acc(valid_df[['y']].values.flatten(), preds2catids(valid_predictions1).values)
print('Map3: {:.3f}'.format(map3))
print('Top3: {:.3f}'.format(top3))

x_valid2 = np.array( [ np.fliplr(x_valid[i]) for i in range(x_valid.shape[0])] )
valid_predictions2 = model.predict(x_valid2, batch_size=128, verbose=1)
map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions2).values)
top3 = top3_acc(valid_df[['y']].values.flatten(), preds2catids(valid_predictions2).values)
print('Map3: {:.3f}'.format(map3))
print('Top3: {:.3f}'.format(top3))

map3 = mapk(valid_df[['y']].values, preds2catids(0.5*valid_predictions1+0.5*valid_predictions2).values)
top3 = top3_acc(valid_df[['y']].values.flatten(), preds2catids(0.5*valid_predictions1+0.5*valid_predictions2).values)
print('Map3: {:.3f}'.format(map3))
print('Top3: {:.3f}'.format(top3))


# ## Create Submission

# In[ ]:


test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))
test.head()
x_test = df_to_image_array(test, size)
print(test.shape, x_test.shape)
print('Test array memory {:.2f} GB'.format(x_test.nbytes / 1024.**3 ))


# In[ ]:


test_predictions1 = model.predict(x_test, batch_size=128, verbose=1)

x_test2 = np.array( [ np.fliplr(x_test[i]) for i in range(x_test.shape[0])] )
test_predictions2 = model.predict(x_test2, batch_size=128, verbose=1)

test_predictions = 0.5*test_predictions1 + 0.5*test_predictions2


# In[ ]:


top3 = preds2catids(test_predictions)
top3.head()
top3.shape


# In[ ]:


cats = list_all_categories()
id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
top3cats = top3.replace(id2cat)
top3cats.head()
top3cats.shape


# In[ ]:


test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
submission = test[['key_id', 'word']]
submission.to_csv('bw_cnn_submission_{}.csv'.format(int(map3 * 10**4)), index=False)
submission.head()
submission.shape


# In[ ]:


end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))


# ## Acknowledments
# Thanks for @jpmiller and @mihaskalic for publishing their starter kernels quite early.
# 
#  [1] https://www.kaggle.com/jpmiller/image-based-cnn
#  
# [2] https://www.kaggle.com/mihaskalic/when-in-doubt-convnets
