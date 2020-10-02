#!/usr/bin/env python
# coding: utf-8

# # Overview
# The notebook takes the preprocessed data from the QuickDraw step (thumbnails and strokes) and trains an LSTM. The outcome variable (y) is always the same (category). The stroke-based LSTM. The model takes the stroke data and 'preprocesses' it a bit using 1D convolutions and then uses two stacked LSTMs followed by two dense layers to make the classification. The model can be thought to 'read' the drawing stroke by stroke.
# 
# ## Fun Models
# 
# After the classification models, we try to build a few models to understand what the LSTM actually does. We take the thought vector from the LSTM (before the classification output) and try to predict the original image. This could then give us insight into what the LSTM was actually paying attention to and able to reconstruct. 

# In[ ]:


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


# In[ ]:


train_path = os.path.join(base_dir, 'quickdraw_train.h5')
valid_path = os.path.join(base_dir, 'quickdraw_valid.h5')
test_path = os.path.join(base_dir, 'quickdraw_test.h5')
word_encoder = LabelEncoder()
word_encoder.fit(HDF5Matrix(train_path, 'word')[:])
print('words', len(word_encoder.classes_), '=>', ', '.join([x.decode() for x in word_encoder.classes_]))


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
                np.max(test_arr[:,1])-test_arr[lab_idx==i,1], '.-')
    c_ax.axis('off')
    c_ax.set_title(word_encoder.classes_[np.argmax(train_y[c_id])].decode())


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
stroke_read_model.add(LSTM(512, return_sequences = False))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(Dense(128))
stroke_read_model.add(Dropout(0.3))
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


from IPython.display import clear_output
stroke_read_model.fit(train_X, train_y,
                      validation_data = (valid_X, valid_y), 
                      batch_size = 4096,
                      epochs = 25,
                      callbacks = callbacks_list)
clear_output() # the training makes a large mess


# In[ ]:


stroke_read_model.load_weights(weight_path)
stroke_read_model.save('stroke_model.h5')
lstm_results = stroke_read_model.evaluate(test_X, test_y, batch_size = 4096)
print('Accuracy: %2.1f%%, Top 5 Accuracy %2.1f%%' % (100*lstm_results[1], 100*lstm_results[2]))


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
test_cat = np.argmax(test_y, 1)
pred_y = stroke_read_model.predict(test_X, batch_size = 4096)
pred_cat = np.argmax(pred_y, 1)
cmat = confusion_matrix(test_cat, pred_cat)
plt.matshow(cmat, cmap = 'nipy_spectral', vmin = 0, vmax = np.mean(cmat[cmat>0]))


# # From Thoughts to Images
# Since understanding what RNNs think about is difficult, we can try to see if it is possible to go from the thought vector back to the original image. We can use the ```stroke_read_model``` as a baseline and extract the last LSTM output to train a new model for predicting the image

# In[ ]:


del train_X, train_y, valid_X, valid_y, test_X, test_y
gc.collect()


# In[ ]:


# the backend approach doesn't work since it tries to run all at once
from keras import backend as K
get_thought_vec_f = K.function(inputs = [stroke_read_model.get_input_at(0), K.learning_phase()],
          outputs = [stroke_read_model.layers[-4].get_output_at(0)])
get_thought_vec = lambda x: get_thought_vec_f([x, 0])
# so we build a submodel
from keras.models import Model
thought_model = Model(inputs = [stroke_read_model.get_input_at(0)],
                     outputs = [stroke_read_model.layers[-4].get_output_at(0)])
get_thought_vec = lambda x: thought_model.predict(x, batch_size = 2048, verbose = True)


# In[ ]:


def get_Xy(in_path):
    X = get_thought_vec(HDF5Matrix(in_path, 'strokes')[:])
    y = HDF5Matrix(in_path, 'thumbnail')[:]
    yn = np.zeros(y.shape, np.float32)
    for i in range(y.shape[0]):
        yn[i] = y[i]*1.0/y[i].max()
    return X, yn
train_X, train_y = get_Xy(train_path)
valid_X, valid_y = get_Xy(valid_path)
print(train_X.shape, '=>', train_y.shape)


# In[ ]:


from keras import backend as K
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


# In[ ]:


from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, Dense, Dropout, Reshape
thumb_dream_model = Sequential()
base_size = (8, 8, 32)

thumb_dream_model.add(BatchNormalization(input_shape = train_X.shape[1:]))
thumb_dream_model.add(Dropout(0.5))
thumb_dream_model.add(Dense(np.prod(base_size)))
thumb_dream_model.add(Reshape(base_size))
thumb_dream_model.add(Conv2D(32, (2,2), padding = 'same'))
thumb_dream_model.add(Conv2D(32, (3,3), padding = 'same'))
thumb_dream_model.add(Dropout(0.3))
thumb_dream_model.add(UpSampling2D((2,2)))
thumb_dream_model.add(Conv2D(16, (3,3), padding = 'same'))
thumb_dream_model.add(Conv2D(16, (3,3), padding = 'same'))
thumb_dream_model.add(Dropout(0.3))
thumb_dream_model.add(UpSampling2D((2,2)))
thumb_dream_model.add(Conv2D(8, (3, 3), padding = 'same'))
thumb_dream_model.add(Conv2D(8, (3, 3), padding = 'valid'))
thumb_dream_model.add(Conv2D(1, (3, 3), padding = 'valid', activation = 'sigmoid'))
thumb_dream_model.compile(optimizer = 'adam', 
                          loss = dice_coef_loss, # bce/mean squared error are terrible loss functions, something GAN-based would be much better suited since we do not care what it is or the exact scale
                          metrics = ['mae', 'binary_accuracy'])
thumb_dream_model.summary()


# In[ ]:


weight_path="{}_weights.best.hdf5".format('thought_dreamer')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=5) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[ ]:


thumb_dream_model.fit(train_X, train_y,
                      validation_data = (valid_X, valid_y), 
                      batch_size = 2048, # maybe smaller batches work better
                      epochs = 25,
                      callbacks = callbacks_list)
clear_output() # the training makes a large mess


# In[ ]:


thumb_dream_model.load_weights(weight_path)
thumb_dream_model.save('dream_model.h5')


# # Show some thoughts
# The top (bw) is the actual drawing and the bottom (blue-red) is the image generated from the thought-vector. The results are not very inspiring. The LSTM model before was clearly very focused on the classification task and has thrown away almost all of the spatial information. It will take a cleverer approach to get meaningful thought vectors out.

# In[ ]:


del train_X, train_y, valid_X, valid_y
gc.collect()
test_X, test_y = get_Xy(test_path)


# In[ ]:


fig, m_axs = plt.subplots(4, 8, figsize = (20, 16))
rand_idxs = np.random.choice(range(test_X.shape[0]), size = 32)
for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):
    goal_out = test_y[c_id][:,:,0]
    goal_out = plt.cm.RdBu(goal_out)
    pred_out = thumb_dream_model.predict(test_X[c_id:(c_id+1)])[0,:,:,0]
    pred_out = plt.cm.jet(pred_out*2.0)
    c_ax.imshow(np.concatenate([goal_out, pred_out], 0)[::-1]) # things are upside-down
    c_ax.axis('off')
fig.savefig('first_thoughts.png')


# In[ ]:


fig, m_axs = plt.subplots(4, 8, figsize = (20, 16))
rand_idxs = np.random.choice(range(test_X.shape[0]), size = 32)
for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):
    goal_out = test_y[c_id][:,:,0]
    goal_out = plt.cm.RdBu(goal_out)
    pred_out = thumb_dream_model.predict(test_X[c_id:(c_id+1)])[0,:,:,0]
    pred_out = plt.cm.jet(pred_out*2.0)
    c_ax.imshow(np.concatenate([goal_out, pred_out], 0)[::-1]) # things are upside-down
    c_ax.axis('off')
fig.savefig('second_thoughts.png')


# # Now Massively Overfit
# Now we can massively overfit the results on a small chunk of the dataset

# In[ ]:


get_ipython().run_cell_magic('time', '', "thumb_dream_model.compile(optimizer = 'adam', loss = 'binary_crossentropy')\nthumb_dream_model.fit(test_X[:2048], test_y[:2048],\n                      batch_size = 256, # maybe smaller batches work better\n                      epochs = 400, verbose = False)")


# In[ ]:


fig, m_axs = plt.subplots(4, 8, figsize = (20, 16))
rand_idxs = np.random.choice(range(2048), size = 32)
for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):
    goal_out = test_y[c_id][:,:,0]
    goal_out = plt.cm.bone_r(goal_out)
    pred_out = thumb_dream_model.predict(test_X[c_id:(c_id+1)])[0,:,:,0]
    pred_out = plt.cm.jet(pred_out*2.0)
    c_ax.imshow(np.concatenate([goal_out, pred_out], 0)[::-1]) # things are upside-down
    c_ax.axis('off')
fig.savefig('just_overfit_thoughts.png')


# In[ ]:


fig, m_axs = plt.subplots(4, 8, figsize = (20, 16))
rand_idxs = np.random.choice(range(2048, test_X.shape[0]), size = 32)
for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):
    goal_out = test_y[c_id][:,:,0]
    goal_out = plt.cm.bone_r(goal_out)
    pred_out = thumb_dream_model.predict(test_X[c_id:(c_id+1)])[0,:,:,0]
    pred_out = plt.cm.jet(pred_out*2.0)
    c_ax.imshow(np.concatenate([goal_out, pred_out], 0)[::-1]) # things are upside-down
    c_ax.axis('off')
fig.savefig('generalized_overfit_thoughts.png')


# In[ ]:





# In[ ]:




