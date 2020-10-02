#!/usr/bin/env python
# coding: utf-8

# ### Goal
# 
# Layers weights initialization is a very important aspect of deep learning that can improve or decrease perfomance of your model. In this short kernel I want to run an experiment about using Glorot (Xavier) and He distributions for layer weights initialization and it's influence on training process with ReLU and PReLU activations.
# 
# To be more concrete - I want to use glorot/he normal and glorot/he uniform distributions and ReLU/PReLU activations, so in total I will get 8 models to train: 
# * 'ReLU-glorot_normal
# * 'ReLU-glorot_uniform
# * 'ReLU-he_normal'
# * 'ReLU-he_uniform'
# * 'PReLU-glorot_normal'
# * 'PReLU-glorot_uniform'
# * 'PReLU-he_normal':
# * 'PReLU-he_uniform'
# 
# Let's start coding:
# 

# In[ ]:


# Importing libraries
import pandas as pd
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

from keras import backend as K
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, Dense, Flatten, PReLU, Activation, MaxPooling2D
from keras.optimizers import SGD
from keras.initializers import glorot_normal, glorot_uniform, he_normal, he_uniform, Constant


# ### Data loading and preprocessing
# 
# In this experiment I want to keep things simple - I'm not going to use a data augmentation, because I want to reduce randomness of the experiment to minimum. 

# In[ ]:


def preprocess_dataset(dataset):
    '''The function converts pandas DataFrame to numpy array, reshapes data and scales values between 0 and 1'''
    dataset = (dataset.to_numpy().reshape(-1, 28, 28, 1) / 255.0).astype(np.float32)
    return dataset

# Loading train, test datasets and submission file
train_df = pd.read_csv('../input/digit-recognizer/train.csv')
test_df = pd.read_csv('../input/digit-recognizer/test.csv')
sample = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

# Train labels
Y = train_df['label']
Y = to_categorical(Y, num_classes = 10)
train_df.drop('label', axis = 1, inplace = True)

# Converting and scaling train / test datasets
train_df = preprocess_dataset(train_df)
test_df = preprocess_dataset(test_df)


# ### Model creation and training
# 
# Here I will use simple VGG-like architecture with 6 hidden layers. There will be 8 models in total, each model will be trained on 20 epochs and after that - predictions on test data will be made and submitted.

# In[ ]:


def learning_curves(m, title):
    '''The function plots learning curves of specific model "m"'''
    H = m.history.history
    plt.plot(H['accuracy'], label = 'acc')
    plt.plot(H['val_accuracy'], label = 'val_acc')
    plt.plot(H['loss'], label = 'loss')
    plt.plot(H['val_loss'], label = 'val_loss')
    plt.title(f'{title} learning curves')
    plt.grid(); plt.legend()
    
def plot_weights(title):
    '''Plot weights of model layers'''
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            ax.hist(weights[0].flatten(), bins = 100, label = layer.name, alpha = 0.5)
        plt.legend()
        plt.title(title)
    plt.yscale('log')


# In[ ]:


seed = 666

# Defining glorot and he distributions
g_norm = glorot_normal(seed = seed)
g_unif = glorot_uniform(seed = seed)
he_norm = he_normal(seed = seed)
he_unif = he_uniform(seed = seed)

# Creating combinations of activation/distribution
models = {
         'ReLU-glorot_normal': ('relu', g_norm),
         'ReLU-glorot_uniform': ('relu', g_unif),
         'ReLU-he_normal': ('relu', he_norm),
         'ReLU-he_uniform': ('relu', he_unif),
         'PReLU-glorot_normal': ('prelu', g_norm),
         'PReLU-glorot_uniform': ('prelu', g_unif),
         'PReLU-he_normal': ('prelu', he_norm),
         'PReLU-he_uniform': ('prelu', he_unif)
         }

# Loop through all models
for m in models:
    init = models[m][1] # Weights initializer
    act = models[m][0] # Activation
    
    model = Sequential()
    model.add(Conv2D(32, 3, input_shape = (28, 28, 1), kernel_initializer = init, padding = 'same'))
    model.add(Activation('relu')) if act == 'relu' else model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(Conv2D(32, 3, kernel_initializer = init, padding = 'same'))
    model.add(Activation('relu')) if act == 'relu' else model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, 3, kernel_initializer = init, padding = 'same'))
    model.add(Activation('relu')) if act == 'relu' else model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(Conv2D(64, 3, kernel_initializer = init, padding = 'same'))
    model.add(Activation('relu')) if act == 'relu' else model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(128, kernel_initializer = init))
    model.add(Activation('relu')) if act == 'relu' else model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(Dense(100, kernel_initializer = init))
    model.add(Activation('relu')) if act == 'relu' else model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(Dense(10, activation = 'softmax', kernel_initializer = init))

    model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    print('_'*20, m, '_'*20) # Print model name to separate plots
    
    # Creating plot of layers weights before train process
    fig = plt.figure(figsize = (19, 12))
    gs = gridspec.GridSpec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    plot_weights(f'{m} initial weights')
    
    # Training model
    model.fit(train_df, Y, epochs = 20, batch_size = 64, validation_split = 0.1, verbose = 0)
    
    # Predicting test data and creating submission file
    preds = model.predict(test_df)
    preds = np.argmax(preds, axis = 1)
    sample['Label'] = preds
    sample.to_csv(f'{m}.csv', index = False)
    
    # Creating plot of layers weights after training process
    ax = fig.add_subplot(gs[0, 1])
    plot_weights(f'{m} learned weights')    
    
    # Plot learning curves
    ax = fig.add_subplot(gs[1, :])
    learning_curves(model, m)
    plt.show()


# ### Results
# 
# He distribution was designed specifically for ReLU activation, which is non differentiable at 0 and, as we can see from these plots - the  models, that use he normal/uniform distribution converges faster and overall training process seems to be more stable than models that use glorot distribution for weights initialization.
# 
# When I submitted predictions on test dataset, I got next results:
# * ReLU-he_uniform      - 0.98928
# * PReLU-he_normal      - 0.98885
# * ReLU-he_normal       - 0.98785
# * PReLU-he_uniform     - 0.98728
# * ReLU-glorot_normal   - 0.98700
# * PReLU-glorot_uniform - 0.98657
# * PReLU-glorot_normal  - 0.98585
# * ReLU-glorot_uniform  - 0.98471
# 
# The models, that use he distribution for weights initialization gave us slightly better score than glorot, but it can be critical for kaggle competitions. Also looks like standard ReLU is more prefferable on such simple networks.

# In[ ]:




