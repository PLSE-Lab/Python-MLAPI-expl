#!/usr/bin/env python
# coding: utf-8

# A Keras CNN to classify set A. As many files lacked classification, and I wasn't happy with the classification on the other files, I have re-labelled all files and obtained good results on this set.

# In[ ]:


import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from scipy.signal import decimate


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, GlobalAvgPool1D, Dropout, BatchNormalization, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


# In[ ]:


INPUT_LIB = '../input/'
SAMPLE_RATE = 44100


# ## Load the data
# We will need to preprocess the unclassified data, as they are marked NaN and also have incorrect file names. For now, we will also make all time series have equal length. We will do all this by defining element wise functions, that we can pass to Pandas.

# In[ ]:


def clean_filename(fname, string):   
    file_name = fname.split('/')[1]
    if file_name[:2] == '__':        
        file_name = string + file_name
    return file_name


# In[ ]:


def load_wav_file(name, path):
    _, b = wavfile.read(path + name)
    assert _ == SAMPLE_RATE
    return b


# In[ ]:


def zero_pad(arr, length):
    """Adds zeros to end of numpy array, for total length len, and makes datatype float.
    Not used currently."""
    result = np.zeros((length, ), dtype='float')
    result[:len(arr)] = arr
    return result


# In[ ]:


def repeat_to_length(arr, length):
    """Repeats the numpy 1D array to given length, and makes datatype float
    Needs adjustment to preserve phase, and could also be optimized"""
    result = np.empty((length, ), dtype = 'float32')
    l = len(arr)
    pos = 0
    while pos + l <= length:
        result[pos:pos+l] = arr
        pos += l
    if pos < length:
        result[pos:length] = arr[:length-pos]
    return result


# In[ ]:


file_info = pd.read_csv(INPUT_LIB + 'set_a.csv')
new_info = pd.DataFrame({'file_name' : file_info['fname'].apply(clean_filename, 
                                                                string='Aunlabelledtest'),
                         'target' : file_info['label'].fillna('unclassified')})   
new_info['time_series'] = new_info['file_name'].apply(load_wav_file, 
                                                      path=INPUT_LIB + 'set_a/')    
new_info['len_series'] = new_info['time_series'].apply(len)  


# In[ ]:


MAX_LEN = max(new_info['len_series'])
CLASSES = ['artifact', 'normal', 'murmur']
CODE_BOOK = {x:i for i,x in enumerate(CLASSES)}   
NB_CLASSES = len(CLASSES)


# ## Convert data to numpy arrays
# Let's leave the unclassified files for validation, and make a training set of the others. We will zero-pad the time series at the end to make the all the same length, then collect all in 2D numpy arrays, that can be used for neural network training.

# In[ ]:


new_info['time_series'] = new_info['time_series'].apply(repeat_to_length, length=MAX_LEN) 
x_train = np.stack(new_info.loc[new_info['target'] 
                                != 'unclassified']['time_series'].values, axis=0)
x_test = np.stack(new_info.loc[new_info['target'] 
                               == 'unclassified']['time_series'].values, axis=0)


# In[ ]:


#At least here in the kernel, we don't need this fine time resoluton, so we downsample it first.
x_train = decimate(x_train, 8, axis=1, zero_phase=True)
x_train = decimate(x_train, 8, axis=1, zero_phase=True)
x_train = decimate(x_train, 4, axis=1, zero_phase=True)
x_test = decimate(x_test, 8, axis=1, zero_phase=True)
x_test = decimate(x_test, 8, axis=1, zero_phase=True)
x_test = decimate(x_test, 4, axis=1, zero_phase=True)


# In[ ]:


#Scale each observation to zero mean and unit variance.
#For a neural net, we also need a channel axis at the end.
x_train = ((x_train - np.mean(x_train, axis=1).reshape(-1,1)) / 
           np.std(x_train, axis=1).reshape(-1,1))
x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1,1)) / 
          np.std(x_test, axis=1).reshape(-1,1))


# In[ ]:


x_train = x_train[:,:,np.newaxis]
x_test = x_test[:,:,np.newaxis]


# Now, as explained in another [notebook][1], I will not use the classification from new_info['target'] but instead my own labels, with three classes 0=artifact, 1=normal/extrahls, and 2=murmur.
# 
# 
#   [1]: https://www.kaggle.com/toregil/d/kinguistics/heartbeat-sounds/misclassified-files-in-set-a/editnb "notebook"

# In[ ]:


train_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 
                1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
                2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 
                1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1]

test_labels = [0, 2, 1, 1, 1, 1, 1, 1, 0, 2, 0, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 
               0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]


# In[ ]:


def one_hot(labels, nb_classes=None):
    if nb_classes is None:
        nb_classes = max(labels) + 1
    else:
        assert nb_classes > max(labels)
    result = np.zeros((len(labels), nb_classes), dtype="int")
    for i, l in enumerate(labels):
        result[i, l] = 1
    return result


# In[ ]:


y_train = one_hot(train_labels, 3)
y_test = one_hot(test_labels, 3)


# ##Train the model

# In[ ]:


model = Sequential()
model.add(Conv1D(filters=4, kernel_size=11, activation='relu', input_shape=x_train.shape[1:]))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=8, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=16, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=128, kernel_size=11, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
model.add(Dropout(0.75))
model.add(GlobalAvgPool1D())
model.add(Dense(3, activation='softmax'))


# This version of the net has 40.000 parameters, and I suspect it will overfit our dataset of 124 time series. 

# In[ ]:


def batch_generator(x_train, y_train, batch_size):
    """
    Rotates the time series randomly in time
    """
    x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32')
    y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')
    full_idx = range(x_train.shape[0])
    
    while True:
        batch_idx = np.random.choice(full_idx, batch_size)
        x_batch = x_train[batch_idx]
        y_batch = y_train[batch_idx]
    
        for i in range(batch_size):
            sz = np.random.randint(x_batch.shape[1])
            x_batch[i] = np.roll(x_batch[i], sz, axis = 0)
     
        yield x_batch, y_batch


# In[ ]:


weight_saver = ModelCheckpoint('set_a_weights.h5', monitor='val_loss', 
                               save_best_only=True, save_weights_only=True)


# In[ ]:


model.compile(optimizer=Adam(2e-5), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


hist = model.fit_generator(batch_generator(x_train, y_train, 8),
                   epochs=40, steps_per_epoch=1000,
                   validation_data = (x_test, y_test),
                   callbacks=[weight_saver],
                   verbose=2)


# ## Evaluation

# In[ ]:


plt.plot(hist.history['loss'], color='b')
plt.plot(hist.history['val_loss'], color='r')
plt.show()
plt.plot(hist.history['acc'], color='b')
plt.plot(hist.history['val_acc'], color='r')
plt.show()


# In[ ]:


y_hat = model.predict(x_test)
np.set_printoptions(precision=2, suppress=True)
for i in range(3):
    plt.plot(y_hat[:,i])
    plt.plot(y_test[:,i])
    plt.show()
print(CLASSES)


# In[ ]:


y_pred = np.argmax(y_hat, axis=1)
y_true = np.argmax(y_test, axis=1)
for i in range(len(y_true)):
    if y_pred[i] != y_true[i]:
        print("Index: {}, Pred: {}, True: {}".format(i, CLASSES[y_pred[i]], CLASSES[y_true[i]]))
        plt.plot(x_test[i])
        plt.show()


# Not too bad. I'd be grateful if someone could double check my labelling.

# I'll be back when I checked set B.

# In[ ]:




