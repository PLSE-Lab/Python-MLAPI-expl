#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm #progress bars
import datetime as dt
import tensorflow as tf

import collections
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
# Input data files are available in the "../input/" directory.


# In[ ]:


train = pd.read_feather("../input/preprocessor-for-data-bowl-2019/train_processed.fth")
#test = pd.read_feather("../input/preprocessor-for-data-bowl-2019/test_processed.fth")
train_labels = pd.read_feather("../input/preprocessor-for-data-bowl-2019/train_labels_processed.fth").set_index('installation_id')
#test_sessions_map = pd.read_feather("../input/preprocessor-for-data-bowl-2019/test_labels_processed.fth").to_dict()


# In[ ]:


# The longest sequence has length 58988
SEQ_LENGTH = 2000 #13000
# see preprocessor -- this is 99 or 95 percentile of length


# In[ ]:


#### Parameters
params = {'dim': (SEQ_LENGTH,4),
          'batch_size': 16,  
          'shuffle': True
}

model_params = {
          'LEARNING_RATE': 0.0001, #default is 0.001
          'LOSS_FN': tf.keras.losses.CategoricalCrossentropy(),
          'METRICS': ['categorical_accuracy'],
                     #{'output0': ['accuracy', qwk],
                     # 'output1': ['accuracy', qwk],
                     # 'output2': ['accuracy', qwk],
                     # 'output3': ['accuracy', qwk],
                     # 'output4': ['accuracy', qwk],}
         'CLIP_NORM': 1,
         'LSTM_L2': 0.000001,
    'DENSE_DROPOUT': 0.5,
    'LSTM_DROPOUT': 0.4
}

BATCH_SIZE = params['batch_size']


# In[ ]:


from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import L1L2

np.random.seed(0) # set random seed for reproducibility
# input which receives correct_assessment, session_number, game_time
# these are not embedded!
non_embedded_input = tf.keras.Input(dtype='float32',name='non_embedded_input', shape=(SEQ_LENGTH, 3))# batch_shape=(BATCH_SIZE, SEQ_LENGTH, 3))

event_id_input = tf.keras.Input(dtype='float32',name='event_id_input',shape=(SEQ_LENGTH))# batch_shape=(BATCH_SIZE, SEQ_LENGTH))
type_input = tf.keras.Input(dtype='float32',name='type_input',  shape=(SEQ_LENGTH))
world_input = tf.keras.Input(dtype='float32',name='world_input',shape=(SEQ_LENGTH))

event_id_embedding_layer = (tf.keras.layers.Embedding(input_dim=390, # +1 bc of masking
                                                     output_dim=10,  # TODO: is this too high?
                                                     input_length=SEQ_LENGTH, # should I set this?
                                                     embeddings_initializer='uniform', 
                                                     #embeddings_regularizer=l2(.000001), 
                                                     #activity_regularizer=l2(.000001), 
                                                     embeddings_constraint=None, 
                                                     mask_zero=True 
                                                    )(event_id_input))
type_embedding_layer =      (tf.keras.layers.Embedding(input_dim=5, # +1 bc of masking
                                                     output_dim=3, 
                                                     input_length=SEQ_LENGTH, # should I set this?
                                                     embeddings_initializer='uniform', 
                                                     #embeddings_regularizer=l2(.000001), 
                                                     #activity_regularizer=l2(.000001), 
                                                     embeddings_constraint=None, 
                                                     mask_zero=True 
                                                    )(type_input))
world_embedding_layer =    (tf.keras.layers.Embedding(input_dim=5, # +1 bc of masking
                                                     output_dim=3, 
                                                     input_length=SEQ_LENGTH, # should I set this?
                                                     embeddings_initializer='uniform', 
                                                     #embeddings_regularizer=l2(.000001), 
                                                     #activity_regularizer=l2(.000001), 
                                                     embeddings_constraint=None, 
                                                     mask_zero=True 
                                                    )(world_input))


# In[ ]:


# should consider regularizing here or with an auxiliary output?
full_input = tf.keras.layers.concatenate([non_embedded_input,
                                         event_id_embedding_layer,
                                         type_embedding_layer,
                                         world_embedding_layer], axis=-1)


# In[ ]:


#MASK_VALUE = 0
#masked_input = tf.keras.layers.Masking(mask_value=MASK_VALUE)(full_input)
LSTM_LAYERS = 32 #
lstm_layer = tf.keras.layers.LSTM(units=LSTM_LAYERS,
                                 kernel_regularizer=l2(model_params['LSTM_L2']),
                                 #activity_regularizer=l2(model_params['LSTM_L2']),# there's a bengio article saying that this is bad, could be added to the dense layer?
                               #  dropout=model_params['LSTM_DROPOUT'],
                                 #kernel_regularizer = L1L2(l1=0.01, l2=0.01),
                                  return_sequences=True)(full_input) 
lstm_layer2 = tf.keras.layers.LSTM(units=LSTM_LAYERS,
                                 kernel_regularizer=l2(model_params['LSTM_L2']),
                                 #activity_regularizer=l2(model_params['LSTM_L2']),# there's a bengio article saying that this is bad, could be added to the dense layer?
                                 dropout=model_params['LSTM_DROPOUT'],
                                 #kernel_regularizer = L1L2(l1=0.01, l2=0.01
                                                           )(lstm_layer) 


# should consider:
#  stateful = True
## DO NOT:  this holds the state *between aligned samples*.  E.g. you slice your long sequence into several, then feed those sequences in one at a time.


# In[ ]:


normalized_layer = tf.keras.layers.BatchNormalization()(lstm_layer2)
x = tf.keras.layers.Dropout(model_params['DENSE_DROPOUT'])(tf.keras.layers.Dense(64, activation='relu')(normalized_layer))
#x = tf.keras.layers.Dropout(model_params['DENSE_DROPOUT'])(tf.keras.layers.Dense(64, activation='relu')(x))
#x = tf.keras.layers.Dropout(model_params['DENSE_DROPOUT'])(tf.keras.layers.Dense(64, activation='relu')(x))
#x = tf.keras.layers.Dropout(model_params['DENSE_DROPOUT'])(tf.keras.layers.Dense(64, activation='relu')(x))

output0 = tf.keras.layers.Dropout(0)(tf.keras.layers.Dense(4, activation='softmax', name='output0')(lstm_layer2))
#output1 = tf.keras.layers.Dense(4, activation='softmax', name='output1')(x)
#output2 = tf.keras.layers.Dense(4, activation='softmax', name='output2')(x)
#output3 = tf.keras.layers.Dense(4, activation='softmax', name='output3')(x)
#output4 = tf.keras.layers.Dense(4, activation='softmax', name='output4')(x)


# In[ ]:


train_labels = train_labels.groupby('installation_id')['accuracy_group'].apply(lambda x : x.to_numpy()).to_frame()
train_labels = train_labels.to_dict(orient='dict')['accuracy_group']


# In[ ]:


model_inputs = [non_embedded_input, event_id_input, type_input, world_input]
model = tf.keras.Model(inputs=model_inputs, outputs=output0)

my_optimizer = tf.keras.optimizers.Adam(learning_rate=model_params['LEARNING_RATE'],  
                                        beta_1=0.9, 
                                        beta_2=0.999, 
                                        amsgrad=True,)
#                                        clipnorm = model_params['CLIP_NORM'])

#my_optimizer_2 = tf.keras.optimizers.Adamax(learning_rate=model_params['LEARNING_RATE'])

import pickle
with open('../input/preprocessor-for-data-bowl-2019/event_ids_map.pkl', 'rb') as file:
    event_ids_map = pickle.load(file)
with open('../input/preprocessor-for-data-bowl-2019/took_assessments_map.pkl','rb') as file:
    took_assessments_map = pickle.load(file)
#assessment_codes = [k for k in reverse_activities_map if 'Assessment' in reverse_activities_map[k]]
the_models = {i : tf.keras.models.clone_model(model) for i in took_assessments_map}


# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
def save_name(activity):
    return ModelCheckpoint(filepath=str(activity), save_best_only = True)

def useful_callbacks(activity):
    return [#save_name(activity), 
            EarlyStopping(patience=50, restore_best_weights=True)]

history = {}

for assessment in the_models:  
    print("Starting model for " + assessment + ".")
    the_models[assessment].compile(optimizer = my_optimizer,
                        loss = model_params['LOSS_FN'],
                        metrics= model_params['METRICS']
                            )
    print("Compiled model for " + assessment + ".")
    print("Time to fit.")
    with np.load("../input/preprocessor-for-data-bowl-2019/data/X_" + assessment + ".npz", allow_pickle=True) as data_X: # TODO: could consider mmap_mode?  dunno
        X0 = data_X['x0']#[:,-SEQ_LENGTH:]
        X1 = data_X['x1']#[:,-SEQ_LENGTH:]
        X2 = data_X['x2']#[:,-SEQ_LENGTH:]
        X3 = data_X['x3']#[:,-SEQ_LENGTH:]
    Y = np.load("../input/preprocessor-for-data-bowl-2019/data/Y_" + assessment + ".npy", allow_pickle=True)
    history[assessment] = the_models[assessment].fit([X0, X1, X2, X3], Y,
                         validation_split=.1,
                         epochs = 200,
                         callbacks = useful_callbacks(assessment),
                         verbose=1)


# In[ ]:


for assessment in the_models:
    tf.keras.models.save_model(the_models[assessment], assessment + '.h5', save_format='h5')
    with open('history_' + assessment + '.pkl', 'wb') as open_file:
        pickle.dump(history[assessment].history,open_file)


# for assessment in history:
#     # Plot training & validation accuracy values
#     plt.plot(history[assessment].history['accuracy'])
#     plt.plot(history[assessment].history['validation_accuracy'])
#     plt.title('Model accuracy: ' + assessment)
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Test'], loc='upper left')
#     plt.show()
# 
# # Plot training & validation loss values
#     plt.plot(history[assessment].history['loss'])
#     plt.plot(history[assessment].history['val_loss'])
#     plt.title('Model loss: ' + assessment)
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Test'], loc='upper left')
#     plt.show()
