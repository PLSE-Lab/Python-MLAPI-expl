#!/usr/bin/env python
# coding: utf-8

# # Overview
# The notebook takes the preprocessed data from the QuickDraw step (thumbnails and strokes) and trains an LSTM. The outcome variable (y) is always the same (category). The stroke-based LSTM. The model takes the stroke data and 'preprocesses' it a bit using 1D convolutions and then uses two stacked LSTMs followed by two dense layers to make the classification. The model can be thought to 'read' the drawing stroke by stroke.
# 
# ## Fun Models
# 
# After the classification models, we try to build a few models to understand what the LSTM actually does. We take the thought vector from the LSTM (before the classification output) and try to predict the original image. This could then give us insight into what the LSTM was actually paying attention to and able to reconstruct. 

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
import gc
gc.enable()
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
print('words', len(word_encoder.classes_), '=>', ', '.join([x.decode() for x in word_encoder.classes_]))


# # Stroke-based Classification
# Here we use the stroke information to train a model and see if the strokes give us a better idea of what the shape could be. 

# In[3]:


def get_Xy(in_path):
    X = HDF5Matrix(in_path, 'strokes')[:]
    y = to_categorical(word_encoder.transform(HDF5Matrix(in_path, 'word')[:]))
    return X, y
train_X, train_y = get_Xy(train_path)
valid_X, valid_y = get_Xy(valid_path)
test_X, test_y = get_Xy(test_path)
print(train_X.shape)


# In[22]:


fig, m_axs = plt.subplots(3,3, figsize = (16, 16))
rand_idxs = np.random.choice(range(train_X.shape[0]), size = 9)
for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):
    test_arr = train_X[c_id]
    test_arr = test_arr[test_arr[:,2]>0, :] # only keep valid points
    lab_idx = np.cumsum(test_arr[:,2]-1)
    for i in np.unique(lab_idx):
        c_ax.plot(test_arr[lab_idx==i,0], 
                np.max(test_arr[:,1])-test_arr[lab_idx==i,1], '.-')
    c_ax.axis('off')
    c_ax.set_title(word_encoder.classes_[np.argmax(train_y[c_id])].decode())


# # LSTM to Parse Strokes
# The model suggeted from the tutorial is
# 
# ![Suggested Model](https://www.tensorflow.org/versions/master/images/quickdraw_model.png)

# In[5]:


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
stroke_read_model.add(Dense(512))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(Dense(len(word_encoder.classes_), activation = 'softmax'))
stroke_read_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy', top_5_accuracy])
stroke_read_model.summary()


# In[6]:


weight_path="{}_weights.best.hdf5".format('stroke_lstm_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=5) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[7]:


stroke_read_model.fit(train_X, train_y,
                      validation_data = (valid_X, valid_y), 
                      batch_size = 2048,
                      epochs = 50,
                      callbacks = callbacks_list)


# In[ ]:


stroke_read_model.load_weights(weight_path)
lstm_results = stroke_read_model.evaluate(test_X, test_y, batch_size = 4096)
print('Accuracy: %2.1f%%, Top 2 Accuracy %2.1f%%' % (100*lstm_results[1], 100*lstm_results[2]))


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
test_cat = np.argmax(test_y, 1)
pred_y = stroke_read_model.predict(test_X, batch_size = 4096)
pred_cat = np.argmax(pred_y, 1)
plt.matshow(confusion_matrix(test_cat, pred_cat))
print(classification_report(test_cat, pred_cat, 
                            target_names = [x.decode() for x in word_encoder.classes_]))


# # Reading Point by Point

# In[25]:


points_to_use = [5, 15, 20, 30, 40, 50]
points_to_user = [108]
samples = 12
word_dex = lambda x: word_encoder.classes_[x].decode()
rand_idxs = np.random.choice(range(test_X.shape[0]), size = samples)
fig, m_axs = plt.subplots(len(rand_idxs), len(points_to_use), figsize = (24, samples/8*24))
for c_id, c_axs in zip(rand_idxs, m_axs):
    res_idx = np.argmax(test_y[c_id])
    goal_cat = word_encoder.classes_[res_idx].decode()
    
    for pt_idx, (pts, c_ax) in enumerate(zip(points_to_use, c_axs)):
        test_arr = test_X[c_id, :].copy()
        test_arr[pts:] = 0 # short sequences make CudnnLSTM crash, ugh 
        stroke_pred = stroke_read_model.predict(np.expand_dims(test_arr,0))[0]
        top_10_idx = np.argsort(-1*stroke_pred)[:10]
        top_10_sum = np.sum(stroke_pred[top_10_idx])
        
        test_arr = test_arr[test_arr[:,2]>0, :] # only keep valid points
        lab_idx = np.cumsum(test_arr[:,2]-1)
        for i in np.unique(lab_idx):
            c_ax.plot(test_arr[lab_idx==i,0], 
                    np.max(test_arr[:,1])-test_arr[lab_idx==i,1], # flip y
                      '.-')
        c_ax.axis('off')
        if pt_idx == (len(points_to_use)-1):
            c_ax.set_title('Answer: %s (%2.1f%%) \nPredicted: %s (%2.1f%%)' % (goal_cat, 100*stroke_pred[res_idx]/top_10_sum, word_dex(top_10_idx[0]), 100*stroke_pred[top_10_idx[0]]/top_10_sum))
        else:
            c_ax.set_title('%s (%2.1f%%), %s (%2.1f%%)\nCorrect: (%2.1f%%)' % (word_dex(top_10_idx[0]), 100*stroke_pred[top_10_idx[0]]/top_10_sum, 
                                                                 word_dex(top_10_idx[1]), 100*stroke_pred[top_10_idx[1]]/top_10_sum, 
                                                                 100*stroke_pred[res_idx]/top_10_sum))


# In[14]:





# In[ ]:





# In[ ]:


# keras crashes kernel when the kernel length is too short
from tqdm import tqdm_notebook
if False:
    for seq_len in tqdm_notebook(reversed(range(108))):
        try:
            stroke_read_model.predict(np.zeros((1, seq_len, 3)));
        except Exception as e:
            print(seq_len, 'is too short', e)
            max_len = seq_len+1
            break


# In[ ]:





# In[ ]:




