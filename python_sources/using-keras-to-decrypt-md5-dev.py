#!/usr/bin/env python
# coding: utf-8

# **Using Keras to Decrypt MD5.**
# 
# Current results is 16% accuracy.
# 
# While this prediction rate is extremely low, it proves hash decryption is possible.
# 
# Further refinement should improve accuracy.
# 
# Additional features have been streamlined to increase dev speed.
# 
# Testing has been limited to 8 character passwords.

# In[ ]:


# Math Imports
import numpy as np  # linear algebra
# fix random seed for reproducibility
np.random.seed(0)

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# from tqdm import tqdm
import hashlib
import gc

import matplotlib as mp
import matplotlib.pyplot as plt


# In[ ]:


# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
import tensorflow as tf 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
print(tf.test.gpu_device_name())
sess = tf.Session()


# In[ ]:


# Keras Imports
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback


# GetBest(Callback) class hidden, used to track and reload best model weights.

# In[ ]:


# https://github.com/keras-team/keras/issues/2768
class GetBest(Callback):

    def __init__(self, monitor='val_loss', verbose=0,
                 mode='auto', period=1):
        super(GetBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.best_epochs = 0
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('GetBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
                
    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            #filepath = self.filepath.format(epoch=epoch + 1, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can pick best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' storing weights.'
                              % (epoch + 1, self.monitor, self.best,
                                 current))
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve' %
                              (epoch + 1, self.monitor))            
                    
    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f' % (self.best_epochs, self.monitor,
                                                       self.best))
        self.model.set_weights(self.best_weights)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'dataframe = pd.read_csv(\'../input/rockyou.txt\',\n                        delimiter = "\\n", \n                        header = None, \n                        names = ["Passwords"],\n                        encoding = "ISO-8859-1",\n                        nrows = 100000)\n\n# BUG: ord throws error due to malformed data near 900000 row')


# In[ ]:


get_ipython().run_cell_magic('time', '', "# # delete all rows with password over 25 letters and less than 3\n# -> testing at 8 char password\nclutter = dataframe[ (dataframe['Passwords'].str.len() >= 9) \n                   | (dataframe['Passwords'].str.len() <= 7) ].index\n# print(dataframe[ (dataframe['Passwords'].str.len() >= 26) ])\ndataframe.drop(clutter, inplace=True)\n\ndataframe = dataframe.reset_index(drop=True)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Drop duplicate password\ndataframe.drop_duplicates(subset=['Passwords'], keep=False, inplace=True)\n\ndataframe = dataframe.reset_index(drop=True)")


# In[ ]:


dataframe['MD5'] = [hashlib.md5(str.encode(str(i))).hexdigest() 
                    for i in dataframe['Passwords'].fillna(0).astype(str)]


# In[ ]:


# training data and keys
passwords = pd.DataFrame(dataframe['Passwords'])
hashes = pd.DataFrame(dataframe['MD5'])

hashes.head()


# In[ ]:


# Digest MD5 Hash to decimal array converter  

# split 32 chr hash into 16 hexadecimal pairs
h01 = dataframe['MD5'].str[:2].apply(int, base=16).astype(float)/256
h02 = dataframe['MD5'].str[2:4].apply(int, base=16).astype(float)/256
h03 = dataframe['MD5'].str[4:6].apply(int, base=16).astype(float)/256
h04 = dataframe['MD5'].str[6:8].apply(int, base=16).astype(float)/256

h05 = dataframe['MD5'].str[8:10].apply(int, base=16).astype(float)/256
h06 = dataframe['MD5'].str[10:12].apply(int, base=16).astype(float)/256
h07 = dataframe['MD5'].str[12:14].apply(int, base=16).astype(float)/256
h08 = dataframe['MD5'].str[14:16].apply(int, base=16).astype(float)/256

h09 = dataframe['MD5'].str[16:18].apply(int, base=16).astype(float)/256
h10 = dataframe['MD5'].str[18:20].apply(int, base=16).astype(float)/256
h11 = dataframe['MD5'].str[20:22].apply(int, base=16).astype(float)/256
h12 = dataframe['MD5'].str[22:24].apply(int, base=16).astype(float)/256

h13 = dataframe['MD5'].str[24:26].apply(int, base=16).astype(float)/256
h14 = dataframe['MD5'].str[26:28].apply(int, base=16).astype(float)/256
h15 = dataframe['MD5'].str[28:30].apply(int, base=16).astype(float)/256
h16 = dataframe['MD5'].str[30:32].apply(int, base=16).astype(float)/256
# convert hex to dec and divide by 255 to normalize

train = np.column_stack((h01, h02, h03, h04,
                         h05, h06, h07, h08,
                         h09, h10, h11, h12,
                         h13, h14, h15, h16))

train[1:3]


# In[ ]:


# Digest password into encoded decimal array.

#df['ascii'] = [ord(x) for x in df['label']]

# ascii ord values can be as high as

xp01 = [ord(x) for x in dataframe['Passwords'].str[0:1]]
p01 = np.array(xp01, dtype=np.float32)/128
xp02 = [ord(x) for x in dataframe['Passwords'].str[1:2]]
p02 = np.array(xp02, dtype=np.float32)/128
xp03 = [ord(x) for x in dataframe['Passwords'].str[2:3]]
p03 = np.array(xp03, dtype=np.float32)/128
xp04 = [ord(x) for x in dataframe['Passwords'].str[3:4]]
p04 = np.array(xp04, dtype=np.float32)/128
xp05 = [ord(x) for x in dataframe['Passwords'].str[4:5]]
p05 = np.array(xp05, dtype=np.float32)/128
xp06 = [ord(x) for x in dataframe['Passwords'].str[5:6]]
p06 = np.array(xp06, dtype=np.float32)/128
xp07 = [ord(x) for x in dataframe['Passwords'].str[6:7]]
p07 = np.array(xp07, dtype=np.float32)/128
xp08 = [ord(x) for x in dataframe['Passwords'].str[7:8]]
p08 = np.array(xp08, dtype=np.float32)/128

key = np.column_stack((p01, p02, p03, p04, p05, p06, p07, p08))

key[1:5]


# In[ ]:


hashcode = train[0:1]
plt.imshow(np.reshape(hashcode,[4,4]), interpolation="nearest", cmap="gray")


# In[ ]:


password = key[0:1]
key_shape = key.shape[1]
plt.imshow(np.reshape(password,[1,key_shape]), interpolation="nearest", cmap="gray")


# In[ ]:


# Set input shape based on training data. 
# Digested hash shape = 16.
train_dim = train.shape[1]


# In[ ]:


# Fuzzy dropout model to force learning.
# Fuzz disabled for testing.
model = Sequential()

model.add(Dense(256, input_dim=train_dim, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))


model.add(Dense(64, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(32, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='sigmoid'))


# In[ ]:


# Compile Model // 
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer='adam')

callbacks = [EarlyStopping(monitor='val_acc', patience=100),
             GetBest(monitor='val_acc', verbose=1, mode='max')]

model.summary()


# In[ ]:


history = model.fit(train, key,
                    callbacks=callbacks,
                    epochs=3000,
                    batch_size=128,
                    # shuffle=True,
                    validation_split=0.1,
                    verbose=2)


# In[ ]:


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:


# evaluate model
score = model.evaluate(train, key, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


# MD5 test array

test = ('test1234','test2345','testtest','password')

test[0]


# In[ ]:


# Digest MD5 hashes to test.


# In[ ]:


# Predict password decimal array from unique MD5.
predictions = model.predict(train, batch_size=None, verbose=0)


# In[ ]:


predictions[0]


# In[ ]:


# predictions = predictions*128


# In[ ]:


# predictions[0:10].astype(int)


# In[ ]:


passes = (key[0:17000]*128).astype(int)

passes[1:10]


# In[ ]:


drift=1
guess = ((predictions[0:17000]*128)/drift).astype(int)

guess[1:10]


# In[ ]:


key_id = 7

decoded = [chr(x) for x in passes[key_id]]

print(decoded)

decrypt = [chr(x) for x in guess[key_id]]

print(decrypt)


# In[ ]:


# set(decoded) & set(decrypt)

answers = set(decoded).intersection(decrypt)

print (answers)


# In[ ]:


[x for x in decoded if x in decrypt]


# In[ ]:


list(set(decoded).intersection(set(decrypt)))


# In[ ]:


# Decode predicted decimal array into characters.
# ascii ord values can be as high as

# p01 = np.array(xp01, dtype=np.float32)*255
# xp01 = [chr(x) for x in predictions['Passwords'].str[0:1]]
# p02 = np.array(xp02, dtype=np.float32)*255
# xp02 = [chr(x) for x in predictions['Passwords'].str[1:2]]
# p03 = np.array(xp03, dtype=np.float32)*255
# xp03 = [chr(x) for x in predictions['Passwords'].str[2:3]]
# p04 = np.array(xp04, dtype=np.float32)*255
# xp04 = [chr(x) for x in predictions['Passwords'].str[3:4]]
# p05 = np.array(xp05, dtype=np.float32)*255
# xp05 = [chr(x) for x in predictions['Passwords'].str[4:5]]
# p06 = np.array(xp06, dtype=np.float32)*255
# xp06 = [chr(x) for x in predictions['Passwords'].str[5:6]]
# p07 = np.array(xp07, dtype=np.float32)*255
# xp07 = [chr(x) for x in predictions['Passwords'].str[6:7]]
# p08 = np.array(xp08, dtype=np.float32)*255
# xp08 = [chr(x) for x in predictions['Passwords'].str[7:8]]


# test_key = np.column_stack((p01, p02, p03, p04, p05, p06, p07, p08))

# test_key[1:5]


# In[ ]:


# submission = pd.DataFrame({'Hash': test_id, 'Password': predictions})

# submission.head()

