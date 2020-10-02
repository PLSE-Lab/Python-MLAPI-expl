#!/usr/bin/env python
# coding: utf-8

# As you have probably noticed (or calculated), there is no way how to get all the dataset into the RAM given the 16 GB of Kaggle Kernels (for float32 representation of inputs, you would need around 128 GB of RAM just for training dataset itself, if my calculations are correct).

# In[ ]:


import sys
import numpy as np

oneImage8 = np.zeros((1, 512, 512, 4), dtype=np.int8)
oneImage16 = np.zeros((1, 512, 512, 4), dtype=np.float16)
oneImage32 = np.zeros((1, 512, 512, 4), dtype=np.float32)
print('Size of all training images, if encoded as int8: ', sys.getsizeof(oneImage8) * 31072 / 1024 / 1024 / 1024, ' GB')
print('Size of all training images, if encoded as float16: ', sys.getsizeof(oneImage16) * 31072 / 1024 / 1024 / 1024, ' GB')
print('Size of all training images, if encoded as float32: ', sys.getsizeof(oneImage32) * 31072 / 1024 / 1024 / 1024, ' GB')


# This kernel tries to play around with custom data generator for fast on-fly data loading. It's written as a descendant of **keras.utils.Sequence**, which has the nice property to be compatible with the **use_multiprocessing=True** setting in *model.fit_generator()* function.
# 
# Now with added caching: If you have enough RAM (e.g. running this script on Google Cloud highmem VC), you can let the Data Generator save all the data into RAM and then use these in the following epochs. Do not try this on Kaggle Kernels with full training dataset, though.
# Currently, using cache cannot be combined with **use_multiprocessing=True** because this parameter leads to creating multiple DataGenerator objects that do not have any shared variables. Solutions to this are welcome.

# In[ ]:


import keras
from keras.utils import Sequence
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import os


# In[ ]:


BATCH_SIZE = 16
SEED = 777
SHAPE = (512, 512, 4)
DIR = '../input'
VAL_RATIO = 0.1 # 10 % as validation
DEBUG = True
THRESHOLD = 0.05 # due to different cost of True Positive vs False Positive, this is the probability threshold to predict the class as 'yes'


# In[ ]:


def getTrainDataset():
    
    path_to_train = DIR + '/train/'
    data = pd.read_csv(DIR + '/train.csv')

    paths = []
    labels = []
    
    for name, lbl in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(28)
        for key in lbl:
            y[int(key)] = 1
        paths.append(os.path.join(path_to_train, name))
        labels.append(y)

    return np.array(paths), np.array(labels)

def getTestDataset():
    
    path_to_test = DIR + '/test/'
    data = pd.read_csv(DIR + '/sample_submission.csv')

    paths = []
    labels = []
    
    for name in data['Id']:
        y = np.ones(28)
        paths.append(os.path.join(path_to_test, name))
        labels.append(y)

    return np.array(paths), np.array(labels)


# In[ ]:


# credits: https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L302
# credits: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class ProteinDataGenerator(keras.utils.Sequence):
            
    def __init__(self, paths, labels, batch_size, shape, shuffle = False, use_cache = False):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.use_cache = use_cache
        if use_cache == True:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]))
            self.is_cached = np.zeros((paths.shape[0]))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx+1) * self.batch_size]

        paths = self.paths[indexes]
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        # Generate data
        if self.use_cache == True:
            X = self.cache[indexes]
            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(path)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, path in enumerate(paths):
                X[i] = self.__load_image(path)

        y = self.labels[indexes]
        
        return X, y
    
    def on_epoch_end(self):
        
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item
            
    def __load_image(self, path):
        R = Image.open(path + '_red.png')
        G = Image.open(path + '_green.png')
        B = Image.open(path + '_blue.png')
        Y = Image.open(path + '_yellow.png')

        im = np.stack((
            np.array(R), 
            np.array(G), 
            np.array(B),
            np.array(Y)), -1)
        
        im = np.divide(im, 255)
        return im


# In[ ]:


paths, labels = getTrainDataset()
tg = ProteinDataGenerator(paths, labels, BATCH_SIZE, SHAPE)


# Let's measure the time to get 128 images. Standard methods are around ~4 seconds on Kaggle Kernels.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'for i in range(8):\n    im, lbl = tg[i]')


# Let's test the RAM caching functionality (and measure the loading time):

# In[ ]:


tg_cache = ProteinDataGenerator(paths[0:200], labels[0:200], BATCH_SIZE, SHAPE, use_cache=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', '#first reading of 128 images should take same time as before\nfor i in range(8):\n    im, lbl = tg_cache[i]')


# In[ ]:


get_ipython().run_cell_magic('time', '', '#second reading from RAM should be MUCH faster\nfor i in range(8):\n    im, lbl = tg_cache[i]')


# In[ ]:


# read data from DB
im, lbl = tg[0]

fig, ax = plt.subplots(4, 4, figsize=(50,50))

for row in range(4):
    for col in range(4):
        ax[row, col].imshow(im[row*4+col, :, :, 0:3])
        #plt.imshow(image[row*4+col, :, :, 0:3]) # first three channels are RGB, fourth is yellow

plt.show()
del(tg, tg_cache)


# # Using in Keras
# Let's try to test the multi_processing.

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization
from keras import metrics
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import tensorflow as tf

from tensorflow import set_random_seed
set_random_seed(SEED)


# In[ ]:


# credits: https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras

def f1(y_true, y_pred):
    #y_pred = K.round(y_pred)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


# In[ ]:


# some basic useless model
def create_model(input_shape):
    
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dropout(0.5))
    #model.add(Dense(28))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(28))
    model.add(Activation('sigmoid'))
    
    return model


# In[ ]:


model = create_model((512,512,4))
model.compile(
    loss='binary_crossentropy', 
    optimizer=Adam(0.0001),
    metrics=['acc',f1])

model.summary()


# In[ ]:


paths, labels = getTrainDataset()

# divide to 
keys = np.arange(paths.shape[0], dtype=np.int)  
np.random.seed(SEED)
np.random.shuffle(keys)
lastTrainIndex = int((1-VAL_RATIO) * paths.shape[0])

if DEBUG == True:  # use only small subset for debugging, Kaggle's RAM is limited
    pathsTrain = paths[0:256]
    labelsTrain = labels[0:256]
    pathsVal = paths[lastTrainIndex:lastTrainIndex+256]
    labelsVal = labels[lastTrainIndex:lastTrainIndex+256]
    use_cache = True
else:
    pathsTrain = paths[0:lastTrainIndex]
    labelsTrain = labels[0:lastTrainIndex]
    pathsVal = paths[lastTrainIndex:]
    labelsVal = labels[lastTrainIndex:]
    use_cache = False

print(paths.shape, labels.shape)
print(pathsTrain.shape, labelsTrain.shape, pathsVal.shape, labelsVal.shape)

tg = ProteinDataGenerator(pathsTrain, labelsTrain, BATCH_SIZE, SHAPE, use_cache=use_cache)
vg = ProteinDataGenerator(pathsVal, labelsVal, BATCH_SIZE, SHAPE, use_cache=use_cache)

# https://keras.io/callbacks/#modelcheckpoint
checkpoint = ModelCheckpoint('./base.model', monitor='val_f1', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)


# In[ ]:


epochs = 50

if DEBUG == True:
    use_multiprocessing = False # DO NOT COMBINE WITH CACHE! 
    workers = 1 # DO NOT COMBINE WITH CACHE! 
else:
    use_multiprocessing = True
    workers = 2

hist = model.fit_generator(
    tg,
    steps_per_epoch=len(tg),
    validation_data=vg,
    validation_steps=8,
    epochs=epochs,
    use_multiprocessing=use_multiprocessing, # you have to train the model on GPU in order to this to be benefitial
    workers=workers, # you have to train the model on GPU in order to this to be benefitial
    verbose=1,
    callbacks=[checkpoint])


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('loss')
ax[0].plot(hist.epoch, hist.history["loss"], label="Train loss")
ax[0].plot(hist.epoch, hist.history["val_loss"], label="Validation loss")
ax[1].set_title('acc')
ax[1].plot(hist.epoch, hist.history["f1"], label="Train F1")
ax[1].plot(hist.epoch, hist.history["val_f1"], label="Validation F1")
ax[0].legend()
ax[1].legend()


# In[ ]:


#fullValGen = ProteinDataGenerator(paths[lastTrainIndex:], labels[lastTrainIndex:], BATCH_SIZE, SHAPE)
#fullValPred = np.zeros((paths[lastTrainIndex:].shape[0], 28))
#for i in tqdm(range(len(fullValGen))):
bestModel = load_model('./base.model', custom_objects={'f1': f1})


# In[ ]:


pathsTest, labelsTest = getTestDataset()

testg = ProteinDataGenerator(pathsTest, labelsTest, BATCH_SIZE, SHAPE)
submit = pd.read_csv(DIR + '/sample_submission.csv')
P = np.zeros((pathsTest.shape[0], 28))
for i in tqdm(range(len(testg))):
    images, labels = testg[i]
    score = bestModel.predict(images)
    P[i*BATCH_SIZE:i*BATCH_SIZE+score.shape[0]] = score


# In[ ]:


PP = np.array(P)


# In[ ]:


prediction = []

for row in tqdm(range(submit.shape[0])):
    
    str_label = ''
    
    for col in range(PP.shape[1]):
        if(PP[row, col] < THRESHOLD):   # to account for losing TP is more costly than decreasing FP
            #print(PP[row])
            str_label += ''
        else:
            str_label += str(col) + ' '
    prediction.append(str_label.strip())
    
submit['Predicted'] = np.array(prediction)
submit.to_csv('datagenerator_model_v1.csv', index=False)


# In[ ]:




