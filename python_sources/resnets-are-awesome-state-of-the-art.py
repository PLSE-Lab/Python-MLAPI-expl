#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# The notebook contains an implementation of a residual neural network in Keras. If you read the references and study the code, you should have an understanding of ResNets. By the way, this method achieved above 99.75%, this is a state-of-the-art solution, hope you like it!  
# 
# ![soluions](http://playagricola.com/Kaggle/KaggleMNISThist3.png)

# In[ ]:


import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation
from keras.layers import Add, Flatten, AveragePooling2D, Dense, Dropout
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model


# In[ ]:


# Load the data
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")

# Reshape and normalize
X = train.drop(columns=['label']).values.reshape(-1, 28, 28, 1) / 255
y = train['label'].values

test = test.values.reshape(-1, 28, 28, 1) / 255

# Get training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# # ResNets
# 
# <img src="https://i.stack.imgur.com/msvse.png" alt="Drawing" style="width: 400px;"/>
# 
# In short, residual neural networks introduce a shortcut connection to prevent vanishing gradients problem. Therefore, enabling to train much deeper models. I won't focus here on how ResNets work as there is a lot of good material available. I want to provide you with an easy to follow implementation that you can study and later use, modify as you wish.
# 
# If you want to know more about ResNets I highly recommend that you read this article  
# https://towardsdatascience.com/introduction-to-resnets-c0a830a288a4. 
#   
# More on ConvNets  
# https://medium.com/zylapp/review-of-deep-learning-algorithms-for-image-classification-5fdbca4a05e2

# In[ ]:


def residual_block(inputs, filters, strides=1):
    """Residual block
    
    Shortcut after Conv2D -> ReLU -> BatchNorm -> Conv2D
    
    Arguments:
        inputs (tensor): input
        filters (int): Conv2D number of filterns
        strides (int): Conv2D square stride dimensions

    Returns:
        x (tensor): input Tensor for the next layer
    """
    y = inputs # Shortcut path
    
    # Main path
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=strides,
        padding='same',
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding='same',
    )(x)
    x = BatchNormalization()(x)
    
    # Fit shortcut path dimenstions
    if strides > 1:
        y = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=strides,
        padding='same',
        )(y)
        y = BatchNormalization()(y)
    
    # Concatenate paths
    x = Add()([x, y])
    x = Activation('relu')(x)
    
    return x
    
    
def resnet(input_shape, num_classes, filters, stages):
    """ResNet 
    
    At the beginning of each stage downsample feature map size 
    by a convolutional layer with strides=2, and double the number of filters.
    The kernel size is the same for each residual block.
    
    Arguments:
        input_shape (3D tuple): shape of input Tensor
        filters (int): Conv2D number of filterns
        stages (1D list): list of number of resiual block in each stage eg. [2, 5, 5, 2]
    
    Returns:
        model (Model): Keras model
    """
    # Start model definition
    inputs = Input(shape=input_shape)
    x = Conv2D(
        filters=filters,
        kernel_size=7,
        strides=1,
        padding='same',
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Stack residual blocks
    for stage in stages:
        x = residual_block(x, filters, strides=2)
        for i in range(stage-1):
            x = residual_block(x, filters)
        filters *= 2
        
    # Pool -> Flatten -> Classify
    x = AveragePooling2D(4)(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(int(filters/4), activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Instantiate model
    model = Model(inputs=inputs, outputs=outputs)
    return model    


# ### Now lets see how our implementation looks like

# In[ ]:


simple_model = resnet(
    input_shape=X[0].shape, 
    num_classes=np.unique(y).shape[-1], 
    filters=64, 
    stages=[2]
)
simple_architecture = plot_model(simple_model, show_shapes=True, show_layer_names=False)
simple_architecture.width = 600
simple_architecture


# Keep in mind that to match the dimensions of the tensors in Add() layer, in the shortcut path there is also a Conv2D layer with strides equal to the strides in the main path. What is more, the AveragePooling2D kernel size should satisfy $\frac{prevKernelSize}{avgPoolKernelSize}={1, 2, 3, ...}$

# ## How to win?
# 
# If you want to win competitions, it is a good practice to train several networks independently and average the predictions (ensemble). It produces more robust results so you don't have to make several submissions and hope to be lucky. The code below is meant for that purpose. Though, here I train just a single network but feel free to change the number of iterations. One more thing, for the submission I used the whole dataset (X, y) instead of (X_train, y_train) for 40 epochs and ensembled 5 models - 99.757% accuracy.

# In[ ]:


def train_model(epochs, filters, stages, batch_size, visualize=False):
    """Helper function for tuning and training the model
    
    Arguments:
        epoch (int): number of epochs
        filters (int): Conv2D number of filterns
        stages (1D list): list of number of resiual block in each stage eg. [2, 5, 5, 2]
        batch_size (int): size of one batch
        visualize (bool): if True then plot training results 
    
    Returns:
        model (Model): Keras model
    """
    # Create and compile model
    model = resnet(
        input_shape=X[0].shape,
        num_classes=np.unique(y).shape[-1],
        filters=filters, 
        stages=stages
    )
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Define callbacks
    checkpoint = ModelCheckpoint(
        filepath=f'resnet-{int(time.time())}.dhf5',
        monitor='loss',
        save_best_only=True
    )

    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8**x)

    callbacks = [checkpoint, annealer]

    # Define data generator
    datagen = ImageDataGenerator(  
        rotation_range=10,  
        zoom_range=0.1, 
        width_shift_range=0.1, 
        height_shift_range=0.1
    )
    datagen.fit(X)

    # Fit model
    history = model.fit_generator(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_test, y_test),
        epochs=epochs, 
        verbose=2, 
        workers=12,
        callbacks=callbacks
    )
    
    if visualize:
        fig, axarr = plt.subplots(1, 2, figsize=(16, 8))
        # Plot training & validation accuracy values
        axarr[0].plot(history.history['accuracy'])
        axarr[0].plot(history.history['val_accuracy'])
        axarr[0].set_title('Model accuracy')
        axarr[0].set_ylabel('Accuracy')
        axarr[0].set_xlabel('Epoch')
        axarr[0].legend(['Train', 'Test'], loc='upper left')
        # Plot training & validation loss values
        axarr[1].plot(history.history['loss'])
        axarr[1].plot(history.history['val_loss'])
        axarr[1].set_title('Model loss')
        axarr[1].set_ylabel('Loss')
        axarr[1].set_xlabel('Epoch')
        axarr[1].legend(['Train', 'Test'], loc='upper left')

        plt.show()

    return model


# In[ ]:


# Train models
models = []
for i in range(1):
    print('-------------------------')
    print('Model: ', i+1)
    print('-------------------------')
    model = train_model(
        epochs=10,
        filters=64,
        stages=[3, 3, 3],
        batch_size=128,
        visualize=True
    )
    models.append(model)


# In[ ]:


# Get predictions, ensemble and create submission csv
predictions = []
for model in models:
    predictions.append(model.predict(test))
predictions = np.sum(predictions, axis=0)
predictions = np.argmax(predictions, axis=1)
submission = pd.DataFrame({'ImageId': np.arange(1, 28001, 1), 'Label': predictions})
submission.to_csv('mnist_resnet_submission.csv', index=False)


# Now that you know a thing or two about ResNets, I encourage you to play with the model yourself. Try various configurations, implement a different ResNet (FYI this is ResNet V1). Maybe you will reach 99.79%, best of luck.
# 
# ### Please upvote if you found this notebook useful, thank you :)
