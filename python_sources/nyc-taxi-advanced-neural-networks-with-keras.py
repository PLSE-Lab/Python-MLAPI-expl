#!/usr/bin/env python
# coding: utf-8

# # NYC Taxi - Advanced Neural Networks with Keras
# 
# ## Introduction
# 
# ### Foreword
# 
# This kernel is a study of mainly two Neural Network techniques using the data from the [New York City Taxi Trip Duration competition](https://www.kaggle.com/c/nyc-taxi-trip-duration).
# 
# My aim here is not to present the data exploration and data preparation. Kernels like [NYC Taxi EDA - Update: The fast & the curious](https://www.kaggle.com/headsortails/nyc-taxi-eda-update-the-fast-the-curious) and [From EDA to the Top (LB 0.367)
# ](https://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-367) do an amazing job at that and my small data management is directly inspired from them.
# 
# Moreover the goal is not to have a perfect model at the end but to showcase some ideas. At the end of the day the model shown here scores a RMSLE around 0.40 in the public leaderboard
# 
# Finally, this is my first real kernel (appart from a test one on the Titanic Competition). Feel free to comment and point out my mistakes. I am here to learn.
# 
# ### The techniques of interest
# 
# What I want to explain today is the creation of a Neural Network with share weights on the one hand, using Keras [functional API](https://keras.io/getting-started/functional-api-guide/) and the usage of multiple train metrics to improve the learning speed and accuracy.
# 

# ## Quick data management
# 
# To make the predictions relevant a minimum of data management is in order. I kept it simple and only took ideas from other kernels.
# 
# ### Importing libraries

# In[1]:


# No need for presentations I guess
import numpy as np
import pandas as pd

# I import keras from the tensorflow library
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input

# Preprocessing and evaluation metric
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error


# ### Data processessing
# 
# The haversine function you may recognise. It comes directly from the [From EDA to the Top (LB 0.367)
# ](https://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-367) kernel. It allows to get an "as the crow flies" distance between pickup and dropout.

# In[2]:


def haversine(lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        R = 6371

        lat = lat2 - lat1
        lng = lng2 - lng1

        d = np.sin(lat * 0.5)**2             + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5)**2
        h = 2 * R * np.arcsin(np.sqrt(d))

        return h


# What the next function does is :
# * Remove the `id` column
# * Calculate the haversine distance of each trip
# * Split the `pickup_datetime` column into sub elements
# * One Hot encode categorical variables
# * Drop the `dropoff_datetime` column if processing the train set
# * Removes some outliers on the train set based on duration and speed
# 
# It also put the GPS coordinates to be the first four columns of the set. We will soon see why.

# In[3]:


def data_prep(raw, pred_data=False):
    out = raw.copy()
    # drop the 'id' column
    out.drop('id', axis=1, inplace=True)
    
    # add the haversine distance
    out.loc[:, 'distance'] = haversine(out.loc[:, 'pickup_latitude'],                                         out.loc[:, 'pickup_longitude'],                                        out.loc[:, 'dropoff_latitude'],                                        out.loc[:, 'dropoff_longitude'])
    
    # split date_times
    elts = ['month', 'day', 'hour', 'minute', 'second']
    
    col = 'pickup_datetime'
    out[col] = pd.to_datetime(out[col])

    for el in elts:
        out[col + '_' + el] = out[col].map(lambda x: getattr(x, el))

    out[col + '_day_of_week'] = out[col].map(lambda x: x.isoweekday())
    
    # remove the original datetime column    
    out.drop('pickup_datetime', axis=1, inplace=True)
    
    # one hot encode categoricals :
    out = pd.get_dummies(out, columns=['vendor_id', 
                                       'store_and_fwd_flag', 
                                       'pickup_datetime_day_of_week'])
    
    # remove some outliers : trip longer than 22 hours and avg speed > 100km/h
    if not pred_data:
        out = out[out['trip_duration'] < 22 * 3600]
        out = out[out['distance'] / out['trip_duration'] * 3600 < 100 ]
        out.drop('dropoff_datetime', axis=1, inplace=True)
        
    # split the gps locations out of `out` 
    coords = ['pickup_latitude', 'pickup_longitude', 
              'dropoff_latitude' ,'dropoff_longitude']
    
    out_gps = out.loc[:, coords]
    out.drop(coords, axis=1, inplace=True)

    return pd.concat([out_gps, out], axis=1)


# ### Data preparation
# 
# I apply the processing on the train set.

# In[4]:


train = pd.read_csv('../input/train.csv')
train = data_prep(train)  

X = train.drop('trip_duration', axis=1)
y = train['trip_duration']

train = None


# Now that I have a `X` and `Y` I use the `StandardScaler` from scikit learn to scale the inputs.
# 
# I also split `X` into `X_flat`, and `X_coords` 0 and 1. These last two are respectively the pickup and dropoff GPS coordinates. `X_flat` contains the rest of the training data.

# In[5]:


scaler = StandardScaler()
X = scaler.fit_transform(X)

X_flat = X[:, 4:]
X_coords0 = X[:, :2]
X_coords1 = X[:, 2:4]


# ## Let's build a Neural Network
# 
# First I set the seed for reproductibility and define `tb_path` as the folder where my TensorBoard's log will go

# In[6]:


tb_path = 'tbGraphs/taxi/mult_shared/'
np.random.seed(1)


# ### Multiple input network with shared weights
# 
# The first technique I want to present is "Multiple input network with shared weights". 
# 
# In this data set we have two pieces of really similar information, the GPS coordinates. The neighborhood these coordinates describe are the same for 'pickup' and 'dropoff'.
# 
# My network will feature a sub-network shared both set of GPS coordinates. This means that the weights are the same for 'pickup' and 'dropoff' coordinates. It is in a way similar to convolutional networks.
# 
# Below is a representation of that idea.

# ![model](http://interactive.blockdiag.com/image?compression=deflate&encoding=base64&src=eJx1jDEKgDAQBHtfcR8QtBb8ypGYi4oxJ8kFC_HvJoWNRNhqZ3a142kzq5rhagAm5mA6aEeIiwpkMCa9syE3vLCvwkw_XdECRUG2KAuhJzk5bOUnxzoluPojya-XrUpfdE6Sh0NzP63MQQs)

# ### Creating the network
# 
# The problem we have is that we can't define our network only with the "Sequential()" API. What we need is the [functional API](https://keras.io/getting-started/functional-api-guide/) and the `concatenate` layer.
# 
# I first set some parameters for the dimensions of the network

# In[8]:


sub_n = 3 # number of layers in the shared subnet
lvl_n = 4 # number of layers for the second part of the network
n_node = 200 # number of neuron for each layer


# #### Coordinates subnet
# 
# I will first create a Sequential model and add to it `sub_n - 1` dense layers

# In[9]:


coord_mod = Sequential()

coord_mod.add(Dense(n_node, activation='relu', input_dim=2))
for _ in range(sub_n - 1):
    coord_mod.add(Dense(n_node, activation='relu'))


# Then using I create two `Input` layers accepting two parameters each (the latitude and longitude)

# In[10]:


coord_inputs0 = Input(shape=(2, ))
coord_inputs1 = Input(shape=(2, ))


# ![model](http://interactive.blockdiag.com/image?compression=deflate&encoding=base64&src=eJyVkMEKwjAMhu97ioBXBT0XfRGR0rVZW1YXaVOmiO_uQMWh3aHX5P8S_q8NpHvjlYV7A6CJotmCWB01BYqwh-CtYxvxtgbGK_-NT-KD7aqx5FREI1Nuz2QwQB3dBcXSD5fMlWDExJI6yQ7lgDxS7CsvUOb6t1-9m8Nv9ZnE0rLgaooVerz_zNQs5RY8TPFXOdE8nv6YvUs) 

# After that I apply my previously designed model to both of these inputs. Because it is the same model everything is shared : architecture of the model but more importantly weights.

# In[12]:


shared_coord0 = coord_mod(coord_inputs0)
shared_coord1 = coord_mod(coord_inputs1)


# Finally I concatenate these two into one layer using the `concatenate` layer.

# In[14]:


merged_coord = keras.layers.concatenate([shared_coord0, shared_coord1])


# ![model](http://interactive.blockdiag.com/image?compression=deflate&encoding=base64&src=eJyVkMEKwjAMhu97ioBXBT0XfRGR0rVZW1YXaVOmiO_uQMWh3aHX5P8S_q8NpHvjlYV7A6CJotmCWB01BYqwh-CtYxvxtgbGK_-NT-KD7aqx5FREI1Nuz2QwVPNdUCz9cMkMdWDExJI6yQ7lgDxS7CsvUOb6t1_Bm8Nv-ZnG0rJga4oVerz_zNQs5RY8TPFXOdE8nvMWvak)

# #### The rest of the net
# 
# Similarly I create a `Input` layer for the flat part of my data

# In[16]:


flat_inputs = Input(shape=(X_flat.shape[1], ))


# Again I concatenate it with my last part

# In[18]:


l = keras.layers.concatenate([merged_coord, flat_inputs])


# ![model](http://interactive.blockdiag.com/image?compression=deflate&encoding=base64&src=eJyVkMEKwjAMhu97ioBXBT0XfRGR0rVZW1YXaVNUxHd3ouLQ7tBr8n8J_9cG0r3xysKtAdBE0axBLPaaAkXYQvDWsY14XQLjhf_GB_HBNtVYciqikSm3RzIYqvkuKJZ-OGWuRiMmltRJdigH5DPFHuouUObn4zroq3i1-60_EVlaFnyNsUKP95-JnLncjIcx_ionmvsD_Fa-Bw)

# Now I add`lvl_n` dense layers on top of this concatenated layer

# In[21]:


for lnl in range(lvl_n):
    l = Dense(n_node, activation='relu')(l)


# ![model](http://interactive.blockdiag.com/image?compression=deflate&encoding=base64&src=eJyVkMEKwjAMhu97ioBXBT0XfRGR0rVZW1YXaVNUxHd3ouLQ7tBr8n_5ydcG0r3xysKtAdBE0axBLPaaAkXYQvDWsY14XQLjhf_GB_HBNtVYciqikSm3RzIYqvkuKJZ-OGWuRiMmltRJdigH5DPFvvoGZX5W10Ffyavdr4CJytKyYGyMFT5590z0zOVmTIzx13OiuT8AHqO-ZQ)

# And finally I add the output layer. Because the duration is a positive number of seconds I find the `relu` activation relevant.

# In[23]:


main_output = Dense(1, activation='relu', name='main_output')(l) 


# ![model](http://interactive.blockdiag.com/image?compression=deflate&encoding=base64&src=eJydkMEKwjAMhu97ioBXBT0XfRGR0rVZW1YXaVNUxHd3oOLQ7lCvyf_lJ18bSPfGKwu3BkATRbMGsdhrChRhC8FbxzbidQmMF_4ZH8Qb21RjyamIRqbcHslgqOa7oFj64ZS5Go2YWFIn2aEckM8U--oblPmf6o_m1e5bwURmaVlwNsYKv7x6JoLmcjMuxvjzPdHcH1QRvsM)

# The last part is to use `Model()` to turn that beautiful architecture into a reality.
# 
# What we need to do is list all the input layers and all the output layers in it.

# In[25]:


model = Model(inputs=[coord_inputs0, coord_inputs1, flat_inputs], 
              outputs=main_output)


# That's it our network is built !!

# ### Compiling and training the network
# 
# Now that we have a model we need to train it. In order to do so I will use the `adam` optimizer.
# 
# The second thing that I want adress here is the choice of loss function. 
# 
# Naively using MSLE seems the best option because it is the metric of evaluation. However doing so result in an slow training . 
# 
# When using the MSE the learning is way faster but the RMSE calculated in the end is not as good.
# 
# What I decided to do was to train a first time for 30 epochs using MSE then to load the best weights (on the validation set MSLE) and train again for 30 epochs usings MSLE.
# 

# #### Using MSE
# 
# So first I compile the model using MSE as my loss function

# In[26]:


model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['msle', 'mse'])


# Then I define my batch size and setup two callbacks
# * `tbCallBack` to be able to view the learning process in TensorBoard
# * `ckCallBack` that will save the best model at each epoch

# In[27]:


batches = 512

path_grph = tb_path + 'shared/test' + str(sub_n) + '_' + str(lvl_n) + '_' + str(n_node)
tbCallBack = keras.callbacks.TensorBoard(log_dir=path_grph,
                                         histogram_freq=0, 
                                         write_graph=True, 
                                         write_images=False)


path_mdl = 'shared_model' + str(sub_n) + '_' + str(lvl_n) + '_' + str(n_node) + '.hdf5'
ckCallBack = keras.callbacks.ModelCheckpoint(path_mdl, 
                                             monitor='val_mean_squared_logarithmic_error',
                                             save_best_only=True,
                                             mode='min')


# Then I start the fitting

# In[28]:


model.fit([X_coords0, X_coords1, X_flat], y, 
          batch_size=batches,
          epochs=30,
          validation_split=0.2,
          verbose=1,
          callbacks=[tbCallBack, ckCallBack])


# #### Using MSLE
# 
# For this second part of the training I load the best weights saved by the CheckPoint callback and relaunch the training from there.
# 
# To change the loss function all that needs to be done is recompiling the model. That does not reset the weights

# In[29]:


model.load_weights(path_mdl)
model.compile(loss='mean_squared_logarithmic_error',
              optimizer='adam',
              metrics=['msle', 'mse'])


# I remake the TensorBoard callback to be able to see both trainings separately

# In[30]:


path_grph = tb_path + 'shared/test_msle' + str(sub_n) + '_' + str(lvl_n) + '_' + str(n_node)
tbCallBack = keras.callbacks.TensorBoard(log_dir=path_grph,
                                         histogram_freq=0, 
                                         write_graph=True, 
                                         write_images=False)


# And the fitting itself

# In[ ]:


model.fit([X_coords0, X_coords1, X_flat], y, 
          batch_size=batches,
          epochs=30,
          validation_split=0.2,
          verbose=1,
          callbacks=[tbCallBack, ckCallBack])


# After that step I have a file named `'shared_model' + str(sub_n) + '_' + str(lvl_n) + '_' + str(n_node) + '.hdf5'` that contains the best weights after the second training

# ## Predict
# 
# With all that done I can make a submission.
# 
# Before that I prepare the test set.

# In[23]:


validation = pd.read_csv('../input/test.csv')
validation = data_prep(validation, pred_data=True)  
validation = scaler.transform(validation)

X_flat_val = validation[:, 4:]
X_coords_val0 = validation[:, :2]
X_coords_val1 = validation[:, 2:4]


# I import the submission template

# In[24]:


submission = pd.read_csv('../input/sample_submission.csv')


# I load my best weights and make my prediction

# In[25]:


model.load_weights(path_mdl)
keras_preds = model.predict([X_coords_val0, X_coords_val1, X_flat_val])


# Finally I create my 'keras_submission.csv'

# In[26]:


keras_submission = submission.copy()
keras_submission.trip_duration = keras_preds

keras_submission.to_csv('keras_submission.csv', index=False)


# The file submitted got a score of 0.40266 on kaggle that would place it around rank 496
# 
# ## Conclusion
# 
# I hope you enjoyed that kernel as much as I enjoyed making it and learning about all the things in it. 
# 
# I am far from being an expert on the subject so don't hesitate to make comment and to suggest modification.
# 
# The final network presented is not by far perfect and I voluntarily chose to not tweak every aspect of it to avoid cluttering the kernel
