#!/usr/bin/env python
# coding: utf-8

# Forked from (https://www.kaggle.com/donkeys/my-little-eda-with-random-forest/log) My Little EDA with Random Forest; Courtesy of "averagemn"

# # Kernel for CareerCon 2019 
# (see contest for details https://www.kaggle.com/c/career-con-2019)
# ## Paul A. Nussbaum
# ### Demonstrates Signal Analysis and Pattern Recognition for Machine Learning
# #### Executive Summary
# Inertial measurement sensors on a moving robot record signals representing different accelerations as they vary over time. This algorithm seeks to observe these signals of bouncing and bumping, and from that determine which category of floor type the robot is rolling over. To accomplish this, the original signals are analyzed and presented to a machine learning algorithm which seeks to recognize the different patterns.
# #### Details
# * There are eight floor type classifications, numbered in the training data 0 through 7 in the following order: 'carpet', 'concrete', 'fine_concrete', 'hard_tiles', 'hard_tiles_large_space',  'soft_pvc', 'soft_tiles', 'tiled', and 'wood'
# 
# #### Revision History
# * v03 - Current revision. Added one-hot encoding, and turned on GPU. Tuned 2D layers a bit.
# * v02 - Converted to Keras CNN using Numpy arrays.
# * v01 - original fork (see top of page)
# 

# In[1]:


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from keras.layers import * 
from keras.models import Model, Sequential, load_model
from keras import backend as K 
from keras import optimizers 
from keras.callbacks import * 
from keras.backend import clear_session

print(os.listdir("../input"))


# ## Load the Data

# In[2]:


df_train = pd.read_csv("../input/X_train.csv")
df_test = pd.read_csv("../input/X_test.csv")
df_y = pd.read_csv("../input/y_train.csv")


# In[3]:


from sklearn.preprocessing import LabelEncoder

# encode class values as integers so they work as targets for the prediction algorithm
encoder = LabelEncoder()
y = encoder.fit_transform(df_y["surface"])
y_count = len(list(encoder.classes_))


# In[4]:


label_mapping = {i: l for i, l in enumerate(encoder.classes_)}


# In[5]:


print("Data Frame Shape (Train then Test, then correct training labels)")
print(df_train.shape, df_test.shape, y.shape)
print("Numpy Array Shape (Train then Test then correct training labels)")

# --- Convert Training, Testing, and Labels into Numpy arrays

num_train = int(df_train.shape[0] / 128)

# Use this for 1D Convolutions
# X_train = np.reshape(np.array(df_train), (num_train,128,13))
# remove potential leakage info 
# X_train = X_train[:,:,3:14]

# Use this for 2D Convolutions
X_train = np.reshape(np.array(df_train), (num_train,128,13,1))
# remove potential leakage info 
X_train = X_train[:,:,3:14,:]

num_test = int(df_test.shape[0] / 128)

# Use this for 1D Convolutions
# X_test = np.reshape(np.array(df_test), (num_test,128,13))
# remove potential leakage info 
# X_test = X_test[:,:,3:14]

# Use this for 2D Convolutions
X_test = np.reshape(np.array(df_test), (num_test,128,13,1))
# remove potential leakage info 
X_test = X_test[:,:,3:14,:]

y_array = np.array(y)
# use one hot encoding
y_one_hot = np.zeros((y_array.shape[0],y_count))
y_one_hot[np.arange(y_array.shape[0]),y_array] = 1
print(X_train.shape, X_test.shape, y_one_hot.shape)
num_features = X_train.shape[2]


# In[6]:


df_train.describe()


# In[7]:


# Sanity Check of conversion from data frame to numpy
# average values in TRAINING array conversion below - should be equal to the individual data frame feature "Mean" values above

for i in range (num_features) :
    print(X_train[:,:,i].mean())


# # Run CNN classifier:

# In[43]:


def make_model() :
    
    scale = 15
    # scale = 100

    # use a simple sequential convolutional neural network model
    model2 = Sequential()
    
    # Start with a droput to slow learning and avoid local minima which in turn will prevent overfitting on the training set 
    model2.add(Dropout(0.1, input_shape=(128,10,1)))
    
    # The first colvolutional layer looks for basic patterns (such as slope) within a sensor's time sequenced data
    # but will also look across all of the sensors for patterns where the may correlate
    # Assume there are three basic kinds of slope (+, 0, and -) and there are 2 groups of 3 sensors, and 1 group of 3 sensors, 
    # so (3^3) * (3^3) * (3^4) = 3^10 = 59 k combinations of slopes across the sensors
    # We can use this as a rough measure of how big to dimension our first layer - let's assume that only "scale" out of 1000 of these are "important"
    model2.add(Conv2D(int(60 * scale), (3,1), strides = 1, activation='relu'))
    
    # The following convolutional layer(s) are really 1D, and look for larger and larger time patterns (second order, scale increase, etc.)
    # The use of the strides > 1 in the convolution layer risks the possibility of phase dependence 
    # This could conceivably cause the same time data shifted by one or two samples to appear different, so instead we use pooling strides
    # These strides can be reduced at the cost of network size, speed, and possible overfitting 
    # If it is found to be needed, we can compensate for overfitting with larger dropout, or more dropouts between layers, or larger "batch_size"
    model2.add(Dropout(0.1))
    model2.add(MaxPooling2D(pool_size = (2,1), strides = (2,1)))
    model2.add(Conv2D(int(10 * scale), (3,1), strides = 1, activation='relu'))
    model2.add(Dropout(0.1))
    model2.add(MaxPooling2D(pool_size = (2,1), strides = (2,1)))
    model2.add(Conv2D(int(5 * scale), (3,1), strides = 1, activation='relu'))
    model2.add(Dropout(0.1))
    model2.add(MaxPooling2D(pool_size = (2,1), strides = (2,1)))
    model2.add(Conv2D(int(5 * scale), (3,1), strides = 1, activation='relu'))
    model2.add(Dropout(0.1))
    model2.add(MaxPooling2D(pool_size = (2,1), strides = (2,1)))
    model2.add(Conv2D(int(2 * scale), (3,1), strides = 1, activation='relu'))
    model2.add(Dropout(0.1))
    model2.add(Conv2D(int(2 * scale), (3,1), strides = 1, activation='relu'))
    
    # After these convolutional layers, we are covering a large chunk of the entire sensor data set of 128 samples
    # and are ready to classify these "wavelet-like" learned patterns using perceptron-style neural netowrks ("Dense")
    model2.add(Flatten())
    
    # Include a couple of dense layers in case the classes are not linearly seperable by this point
    model2.add(Dropout(0.1))
    model2.add(Dense(int(2 * scale), activation='relu'))
    model2.add(Dropout(0.1))
    model2.add(Dense(int(1 * scale), activation='relu'))
    
    # Finally, mirror the "one-hot" classification scheme with a softmax output layer
    model2.add(Dropout(0.1))
    model2.add(Dense(y_count, activation = 'softmax'))
    
    # Binary Cross Entropy (or log loss) is used as the error function
    model2.compile(loss='categorical_crossentropy', optimizer='adam')
    return model2


# In[ ]:


# we will be slowing down the learning using "Dropouts" (see above) so the patience needed to exit local minima can be large
patience = 15
# Probably will never reach this many epochs, but want to use a number larger than what we expect
epochs = 300
# Divide the data into 15 different versions of training/validation
n_fold = 15
# Using KFold instead of StratifiedKFold becuase there is a low degree of confidence that the test classification distributions 
# or more importantly, the real world classification probabilities, are equal to those found in the training set
folds = KFold(n_splits=n_fold, shuffle=False, random_state=1234)

sam = X_train.shape[0]
col = X_train.shape[1]

sam_test = X_test.shape[0]
col_test = X_test.shape[1]

prediction = np.zeros((sam_test, y_count))
prediction_train = np.zeros((sam, y_count))
for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train,y_one_hot)):
    print('Fold', fold_n)
    X_train2, X_valid2 = X_train[train_index], X_train[valid_index]
    y_train2, y_valid2 = y_one_hot[train_index], y_one_hot[valid_index]
    K.clear_session()
    model = make_model()
    print(model.summary())
    checkpointer = ModelCheckpoint('Net1', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(patience = patience, verbose=0) 
    results = model.fit(X_train2, y_train2, epochs = epochs, batch_size = 32,
                    callbacks=[earlystopper, checkpointer], validation_data=[X_valid2, y_valid2])
    model = load_model('Net1')
    # For each fold, we will accummulate our opinion of the final classification
    prediction_train  += model.predict(X_train)/n_fold
    # Note, in a real world deployment, we may have "n_fold" neural networks, each rendering their opinion - but here
    # to save memory and disk space, we dispose of each NN when the classification is done
    prediction += model.predict(X_test)/n_fold 
    print()


# In[ ]:


pred_y = prediction_train.argmax(axis = 1)
num_correct = 0
for i in range(y_array.shape[0]) :
    if pred_y[i] == y_array[i] :
        num_correct += 1
        
print("Score on Training Data =", num_correct / y_array.shape[0])


# In[ ]:


prediction


# In[ ]:


prediction.shape


# In[ ]:


ss = pd.read_csv('../input/sample_submission.csv')
# ss['surface'] = encoder.inverse_transform(prediction.astype(int))
ss['surface'] = encoder.inverse_transform(prediction.argmax(axis = 1))
ss.to_csv('rf.csv', index=False)
ss.head(10)

