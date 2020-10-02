#!/usr/bin/env python
# coding: utf-8

# # Overview
# The notebook takes the preprocessed data from the QuickDraw step (thumbnails and strokes) and trains 4 different models on the data to see which gets the best performance. The outcome variable (y) is always the same (category). The input for the first two models is the thumbnail and for the last is the stroke data.
# ## Classification Models
# The first is a basic multi-layer perceptron as a sort of baseline. It takes the flattened image vector and with 2 dense or fully-connected layers produces an outcome. 
# 
# THe second model is a basic convolutional model similar to AlexNet. The model uses convolutions with max pooling rather than fully-connected layers to allow for some degree of spatial invariance.
# 
# The third model uses the stroke data instead of the image data and has a series of 1D convolutions and Global Average Pooling layers to 'process' but not 'read' the stroke sequences.
# 
# The last model is the stroke-based LSTM. The model takes the stroke data and 'preprocesses' it a bit using 1D convolutions and then uses two stacked LSTMs followed by two dense layers to make the classification. The model can be thought to 'read' the drawing stroke by stroke.
# 
# 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.io_utils import HDF5Matrix
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.metrics import top_k_categorical_accuracy
def top_5_accuracy(x,y): return top_k_categorical_accuracy(x,y, 5)
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
base_dir = os.path.join('..', 'input', 'quickdraw-overview')


# In[2]:


train_path = os.path.join(base_dir, 'quickdraw_train.h5')
valid_path = os.path.join(base_dir, 'quickdraw_valid.h5')
test_path = os.path.join(base_dir, 'quickdraw_test.h5')
word_encoder = LabelEncoder()
word_encoder.fit(HDF5Matrix(train_path, 'word')[:])
print('words', len(word_encoder.classes_), '=>', word_encoder.classes_)


# # Thumbnail Classification
# Here we try and classify the thumbnails well.

# In[3]:


def get_Xy(in_path):
    X = HDF5Matrix(in_path, 'thumbnail')[:]
    y = to_categorical(word_encoder.transform(HDF5Matrix(in_path, 'word')[:]))
    return X, y


# In[4]:


train_X, train_y = get_Xy(train_path)
valid_X, valid_y = get_Xy(valid_path)
test_X, test_y = get_Xy(test_path)
print(train_X.shape)


# In[5]:


fig, m_axs = plt.subplots(3,3, figsize = (16, 16))
rand_idxs = np.random.choice(range(train_X.shape[0]), size = 9)
for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):
    c_ax.imshow(train_X[c_id][:,:,0], cmap = 'bone')
    c_ax.set_title(word_encoder.classes_[np.argmax(train_y[c_id])].decode())


# # Multilayer Perceptron
# Here we start with a very simple model to try and classify the thumbnails using 2 hidden layers

# In[6]:


from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPool2D, Flatten, Dense, Dropout
mlp_model = Sequential()
mlp_model.add(BatchNormalization(input_shape = train_X.shape[1:]))
mlp_model.add(Flatten())
mlp_model.add(Dropout(0.5))
mlp_model.add(Dense(256))
mlp_model.add(Dense(len(word_encoder.classes_), activation = 'softmax'))
mlp_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy', top_5_accuracy])
mlp_model.summary()


# In[7]:


weight_path="{}_weights.best.hdf5".format('thumb_mlp_model')
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=5) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[8]:


mlp_model.fit(train_X, train_y,
                      validation_data = (valid_X, valid_y), 
                      batch_size = 1024,
                      epochs = 30,
                      shuffle = 'batch',
                      callbacks = callbacks_list)


# In[9]:


mlp_model.load_weights(weight_path)
mlp_results = mlp_model.evaluate(test_X, test_y)
print('Accuracy: %2.1f%%, Top 2 Accuracy %2.1f%%' % (100*mlp_results[1], 100*mlp_results[2]))


# # AlexNet-like CNN
# Here we use a slightly more complicated model trying to get a bit at the hierarchical structure of the data

# In[10]:


from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPool2D, Flatten, Dense, Dropout
thumb_class_model = Sequential()
thumb_class_model.add(BatchNormalization(input_shape = train_X.shape[1:]))
thumb_class_model.add(Conv2D(16, (3,3), padding = 'same'))
thumb_class_model.add(Conv2D(16, (3,3)))
thumb_class_model.add(MaxPool2D(2,2))
thumb_class_model.add(Conv2D(32, (3,3), padding = 'same'))
thumb_class_model.add(Conv2D(32, (3,3)))
thumb_class_model.add(MaxPool2D(2,2))
thumb_class_model.add(Conv2D(64, (3,3), padding = 'same'))
thumb_class_model.add(Flatten())
thumb_class_model.add(Dropout(0.5))
thumb_class_model.add(Dense(256))
thumb_class_model.add(Dense(len(word_encoder.classes_), activation = 'softmax'))
thumb_class_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy', top_5_accuracy])
thumb_class_model.summary()


# In[ ]:


weight_path="{}_weights.best.hdf5".format('thumb_cnn_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=5) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[ ]:


thumb_class_model.fit(train_X, train_y,
                      validation_data = (valid_X, valid_y), 
                      batch_size = 1024,
                      epochs = 40,
                      shuffle = 'batch',
                      callbacks = callbacks_list)


# In[ ]:


thumb_class_model.load_weights(weight_path)
cnn_results = thumb_class_model.evaluate(test_X, test_y)
print('Accuracy: %2.1f%%, Top 2 Accuracy %2.1f%%' % (100*cnn_results[1], 100*cnn_results[2]))


# # Stroke-based Classification
# Here we use the stroke information to train a model and see if the strokes give us a better idea of what the shape could be. 

# In[ ]:


def get_Xy(in_path):
    X = HDF5Matrix(in_path, 'strokes')[:]
    y = to_categorical(word_encoder.transform(HDF5Matrix(in_path, 'word')[:]))
    return X, y
train_X, train_y = get_Xy(train_path)
valid_X, valid_y = get_Xy(valid_path)
test_X, test_y = get_Xy(test_path)
print(train_X.shape)


# In[ ]:


fig, m_axs = plt.subplots(3,3, figsize = (16, 16))
rand_idxs = np.random.choice(range(train_X.shape[0]), size = 9)
for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):
    test_arr = train_X[c_id]
    test_arr = test_arr[test_arr[:,2]>0, :] # only keep valid points
    lab_idx = np.cumsum(test_arr[:,2]-1)
    for i in np.unique(lab_idx):
        c_ax.plot(test_arr[lab_idx==i,0], 
                test_arr[lab_idx==i,1], '.-')
    c_ax.axis('off')
    c_ax.set_title(word_encoder.classes_[np.argmax(train_y[c_id])].decode())


# # Stroke baseline Model
# We can make a baseline stroke model by copying the general structure from the MLP we made before. The model here captures in a sense the sort of information in the strokes themselves without 'reading them'. The convolutions serve to preprocess and then a global average pooling layer to summarize the information in a single vector that can then be processed using standard fully-connected layers.
# 

# In[ ]:


from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, Dense, Dropout, GlobalAveragePooling1D
mlp_stroke_model = Sequential()
mlp_stroke_model.add(BatchNormalization(input_shape = (None,)+train_X.shape[2:]))
# filter count and length are taken from the script https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/train_model.py
mlp_stroke_model.add(Conv1D(48, (5,)))
mlp_stroke_model.add(Dropout(0.3))
mlp_stroke_model.add(Conv1D(64, (5,)))
mlp_stroke_model.add(Dropout(0.3))
mlp_stroke_model.add(Conv1D(128, (3,)))
mlp_stroke_model.add(Dropout(0.3))
mlp_stroke_model.add(GlobalAveragePooling1D())
mlp_stroke_model.add(Dropout(0.3))
mlp_stroke_model.add(Dense(256))
mlp_stroke_model.add(Dense(len(word_encoder.classes_), activation = 'softmax'))
mlp_stroke_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy', top_5_accuracy])
mlp_stroke_model.summary()


# In[ ]:


weight_path="{}_weights.best.hdf5".format('stroke_mlp_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=5) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[ ]:


mlp_stroke_model.fit(train_X, train_y,
                      validation_data = (valid_X, valid_y), 
                      batch_size = 1024,
                      epochs = 30,
                      shuffle = 'batch',
                      callbacks = callbacks_list)


# In[ ]:


mlp_stroke_model.load_weights(weight_path)
mlp_stroke_results = mlp_stroke_model.evaluate(test_X, test_y, batch_size = 2048)
print('Accuracy: %2.1f%%, Top 2 Accuracy %2.1f%%' % (100*mlp_stroke_results[1], 100*mlp_stroke_results[2]))


# # LSTM to Parse Strokes
# The model suggeted from the tutorial is
# 
# ![Suggested Model](https://www.tensorflow.org/versions/master/images/quickdraw_model.png)

# In[ ]:


from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, LSTM, Dense, Dropout
if len(get_available_gpus())>0:
    # https://twitter.com/fchollet/status/918170264608817152?lang=en
    from keras.layers import CuDNNLSTM as LSTM # this one is about 3x faster on GPU instances
stroke_read_model = Sequential()
stroke_read_model.add(BatchNormalization(input_shape = (None,)+train_X.shape[2:]))
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
stroke_read_model.add(Dense(256))
stroke_read_model.add(Dense(len(word_encoder.classes_), activation = 'softmax'))
stroke_read_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy', top_5_accuracy])
stroke_read_model.summary()


# In[ ]:


weight_path="{}_weights.best.hdf5".format('stroke_lstm_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=5) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[ ]:


stroke_read_model.fit(train_X, train_y,
                      validation_data = (valid_X, valid_y), 
                      batch_size = 1024,
                      epochs = 30,
                      shuffle = 'batch',
                      callbacks = callbacks_list)


# In[ ]:


stroke_read_model.load_weights(weight_path)
lstm_results = stroke_read_model.evaluate(test_X, test_y, batch_size = 2048)
print('Accuracy: %2.1f%%, Top 2 Accuracy %2.1f%%' % (100*lstm_results[1], 100*lstm_results[2]))


# In[ ]:




