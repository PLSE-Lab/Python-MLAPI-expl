#!/usr/bin/env python
# coding: utf-8

# # Distinguishing Cat and Dog Images with Convolutional Neural Networks
# 
# Here I study the effects of successively adding components to a CNN. The goal is to label images as either Cat or Dog, based on training images taken from the [Dogs vs. Cats competition](https://www.kaggle.com/c/dogs-vs-cats). The evaluation metric used in the competition is simply the classification accuracy, though here I will also be examining the ROC curves of my models for a bit of extra insight.
# 
# The final model is inspired by the one provided in [this excellent notebook](https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification). (The only real changes I make are to use 150x150 images instead of 128x128, and a single sigmoid output instead of a two-output softmax.)

# In[ ]:


import os
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ## Initial Model
# 
# The images are color and in a variety of shapes. In order to train a neural network to handle them, they will need to have uniform size. Note the input shape parameter: it is 150x150 for the size and 3 for the color depth. 
# 
# Because we are facing a two-class classification problem, i.e. a *binary classification problem*, we will end our network with a [*sigmoid* activation](https://wikipedia.org/wiki/Sigmoid_function), so that the output of our network will be a single scalar between 0 and 1, encoding the probability that the current image is class 1 (as opposed to class 0).

# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(1, activation='sigmoid')  
])


# In[ ]:


model.summary()


# Next, we'll configure the specifications for model training. We will train our model with the `binary_crossentropy` loss, because it's a binary classification problem and our final activation is a sigmoid. We will use the `rmsprop` optimizer with a learning rate of `0.001`.

# In[ ]:


model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['acc'])


# ### Training
# I have packaged up the data into a couple of "pickled" DataFrames. The full training dataset comprises 25000 images, split evenly between cat and dog images. `train_df` comprises 90% of the full training data, and `validate_df` the remaining 10%. These have been split at random, in a stratified fashion so as to keep equal amounts of cats and dogs in the validation set (and hence the training set as well).
# 
# Let's train on all training images available, for up to 15 epochs to start, and validate on all validation images. The raw image data has pixel activations between 0 and 255, which we will scale to lie between 0 and 1; this can help prevent our NN from multiplying (or dividing) by very large numbers.

# In[ ]:


# Load data
train_df = pd.read_pickle('../input/cat-vs-dog-data/train_df.pkl')
train_df.filename = train_df.filename.map(lambda s: s.split('\\')[1])
validate_df = pd.read_pickle('../input/cat-vs-dog-data/validate_df.pkl')
validate_df.filename = validate_df.filename.map(lambda s: s.split('\\')[1])
train_dir = '../input/dogs-vs-cats/train/train'


# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1./255.)
validation_datagen = ImageDataGenerator(rescale = 1./255.)

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    train_dir,
                                                    x_col='filename',
                                                    y_col='category',
                                                    batch_size=50,
                                                    class_mode='binary',
                                                    target_size=(150, 150),
                                                       seed=42)     
validation_generator =  validation_datagen.flow_from_dataframe(validate_df,
                                                          train_dir,
                                                          x_col='filename',
                                                          y_col='category',
                                                         batch_size=50,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150),
                                                            seed=42)


# Early Stopping is basically always a good callback to use, especially if you have a good idea of how much patience is needed.

# In[ ]:


earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, restore_best_weights=True)


# **NOTE:** Not actually training in this notebook, since that would take awhile to upload. Instead, I'll load up the results from training on my machine.

# In[ ]:


# history = model.fit_generator(train_generator,
#                               epochs=15,
#                               validation_data=validation_generator,
#                              callbacks=[earlystopping])


# ### Save model, so retraining is not necessary in future notebook runs

# In[ ]:


def save_history(history, filename):
    with open(filename, 'wb') as file_pi:
        pickle.dump((history.epoch, history.history), file_pi)


# In[ ]:


def load_history(filename):
    history = keras.callbacks.History()
    with open(filename, 'rb') as file_pi:
        (history.epoch, history.history) = pickle.load(file_pi)
    return history


# In[ ]:


def save_earlystopping(earlystopping, filename):
    with open(filename, 'wb') as file_pi:
        pickle.dump((earlystopping.stopped_epoch, earlystopping.patience,
                    earlystopping.monitor, earlystopping.min_delta,
                    earlystopping.monitor_op, earlystopping.restore_best_weights,
                    earlystopping.wait, earlystopping.baseline), file_pi)


# In[ ]:


def load_earlystopping(filename):
    earlystopping = keras.callbacks.EarlyStopping()
    with open(filename, 'rb') as file_pi:
        (earlystopping.stopped_epoch, earlystopping.patience,
         earlystopping.monitor, earlystopping.min_delta,
         earlystopping.monitor_op, earlystopping.restore_best_weights,
         earlystopping.wait, earlystopping.baseline) = pickle.load(file_pi)
    return earlystopping


# In[ ]:


# # Save model
# model.save('initial_model.h5')
# save_history(history, 'initial_history.pkl')
# save_earlystopping(earlystopping, 'initial_earlystopping.pkl')


# In[ ]:


# Load model
model = keras.models.load_model('../input/cat-vs-dog-cnn-models/initial_model.h5')
history = load_history('../input/cat-vs-dog-cnn-models/initial_history.pkl')
earlystopping = load_earlystopping('../input/cat-vs-dog-cnn-models/initial_earlystopping.pkl')


# ### Define various helper methods, used for reporting training history and validation performance

# In[ ]:


def check_early_stopping(history, earlystopping):
    print('Early stopping')
    monitor = history.history[earlystopping.monitor]
    fun1, fun2 = np.min, np.argmin
    best = 'Lowest'
    if earlystopping.monitor_op == np.greater:
        fun1, fun2 = np.max, np.argmax
        best = 'Highest'
    print(f'  Monitor: {earlystopping.monitor}')
    print(f'    {best} value: {fun1(monitor)}')
    print(f'    Epoch: {fun2(monitor)+1}')
    stopped_epoch = earlystopping.stopped_epoch - earlystopping.patience + 1
    if earlystopping.stopped_epoch == 0:
        stopped_epoch = 'None (stopped_epoch==0)'
    print(f'  Epoch detected by early stopping: {stopped_epoch}')
    if not earlystopping.restore_best_weights or earlystopping.stopped_epoch == 0:
        print('  Best weights NOT returned')
    else:
        print('  Best weights returned')


# In[ ]:


def plot_history(history):
    acc      = history.history[     'acc' ]
    val_acc  = history.history[ 'val_acc' ]
    loss     = history.history[    'loss' ]
    val_loss = history.history['val_loss' ]

    epochs   = range(len(acc))

    plt.plot  ( epochs,     acc , label='Training')
    plt.plot  ( epochs, val_acc , label='Validation')
    plt.xlabel ('Epoch')
    plt.ylabel ('Accuracy')
    plt.legend ()
    plt.title ('Training and validation accuracy')
    plt.figure()

    plt.plot  ( epochs,     loss, label='Training')
    plt.plot  ( epochs, val_loss, label='Validation')
    plt.xlabel ('Epoch')
    plt.ylabel ('Loss')
    plt.legend ()
    plt.title ('Training and validation loss'   )


# In[ ]:


def report(model, validation_generator, history=None, earlystopping=None):
    if earlystopping is not None:
        check_early_stopping(history, earlystopping)
        print()
    
    if history is not None:
        plot_history(history)
    
    # Evaluate trained model on validation set
    validation_generator.reset()
    [val_loss, val_acc] = model.evaluate_generator(validation_generator)
    print('Model evaluation')
    print(f'val_loss: {val_loss}, val_acc: {val_acc}')
    print()
    
    # Compute ROC curve
    validation_generator.reset()
    validation_set = [validation_generator.next() for _ in range(len(validation_generator))]
    val_images = np.concatenate([validation_set[i][0] for i in range(len(validation_set))])
    val_y = np.concatenate([validation_set[i][1] for i in range(len(validation_set))])
    fpr, tpr, thresholds = roc_curve(val_y, model.predict(val_images))
    
    return fpr, tpr


# In[ ]:


class ROCCurveParams():
    def __init__(self, fpr, tpr, color, linestyle):
        self.fpr, self.tpr, self.color, self.linestyle = fpr, tpr, color, linestyle


# In[ ]:


def plot_roc_curves(roc_curves):
    plt.figure()
    for curvename in roc_curves:
        c = roc_curves[curvename]
        plt.plot(c.fpr, c.tpr, color=c.color, lw=2, label=curvename, linestyle=c.linestyle)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Validation set ROC')
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:


roc_curves = dict()


# In[ ]:


fpr, tpr = report(model, validation_generator, history, earlystopping)
roc_curves['Initial model'] = ROCCurveParams(fpr, tpr, 'darkorange', '-')


# In[ ]:


plot_roc_curves(roc_curves)


# ### Summary
# 
# Initial model with 83.5% validation accuracy. Trains quickly. Overfits considerably.

# ## Add Dropout, increase early stopping patience

# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  
])
model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['acc'])


# In[ ]:


model.summary()


# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1./255.)
validation_datagen = ImageDataGenerator(rescale = 1./255.)

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    train_dir,
                                                    x_col='filename',
                                                    y_col='category',
                                                    batch_size=50,
                                                    class_mode='binary',
                                                    target_size=(150, 150),
                                                       seed=42)     
validation_generator =  validation_datagen.flow_from_dataframe(validate_df,
                                                          train_dir,
                                                          x_col='filename',
                                                          y_col='category',
                                                         batch_size=50,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150),
                                                            seed=42)


# In[ ]:


earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, restore_best_weights=True)


# In[ ]:


# history = model.fit_generator(train_generator,
#                               epochs=30,
#                               validation_data=validation_generator,
#                              callbacks=[earlystopping])


# In[ ]:


# # Save model
# model.save('dropout_model.h5')
# save_history(history, 'dropout_history.pkl')
# save_earlystopping(earlystopping, 'dropout_earlystopping.pkl')


# In[ ]:


# Load model
model = keras.models.load_model('../input/cat-vs-dog-cnn-models/dropout_model.h5')
history = load_history('../input/cat-vs-dog-cnn-models/dropout_history.pkl')
earlystopping = load_earlystopping('../input/cat-vs-dog-cnn-models/dropout_earlystopping.pkl')


# In[ ]:


fpr, tpr = report(model, validation_generator, history, earlystopping)
roc_curves['Dropout'] = ROCCurveParams(fpr, tpr, 'green', '-')
plot_roc_curves(roc_curves)


# ### Summary
# 
# Dropout has reduced overfitting and improved the validation set performance.

# ## Reduce LR on plateau, further increase early stopping patience

# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  
])
model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['acc'])


# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1./255.)
validation_datagen = ImageDataGenerator(rescale = 1./255.)

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    train_dir,
                                                    x_col='filename',
                                                    y_col='category',
                                                    batch_size=50,
                                                    class_mode='binary',
                                                    target_size=(150, 150),
                                                       seed=42)     
validation_generator =  validation_datagen.flow_from_dataframe(validate_df,
                                                          train_dir,
                                                          x_col='filename',
                                                          y_col='category',
                                                         batch_size=50,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150),
                                                            seed=42)


# In[ ]:


earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=9, restore_best_weights=True)
reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=0.00001)


# In[ ]:


# history = model.fit_generator(train_generator,
#                               epochs=40,
#                               validation_data=validation_generator,
#                              callbacks=[earlystopping, reducelr])


# In[ ]:


# # Save model
# model.save('redlr_model.h5')
# save_history(history, 'redlr_history.pkl')
# save_earlystopping(earlystopping, 'redlr_earlystopping.pkl')


# In[ ]:


# Load model
model = keras.models.load_model('../input/cat-vs-dog-cnn-models/redlr_model.h5')
history = load_history('../input/cat-vs-dog-cnn-models/redlr_history.pkl')
earlystopping = load_earlystopping('../input/cat-vs-dog-cnn-models/redlr_earlystopping.pkl')


# In[ ]:


fpr, tpr = report(model, validation_generator, history, earlystopping)
roc_curves['Reduce LR'] = ROCCurveParams(fpr, tpr, 'red', '--')
plot_roc_curves(roc_curves)


# ### Summary
# 
# LR Reduction on Plateaus stabilizes learning near the end. It takes several more epochs to reach optimum, and results in nearly identical validation set performance. The training set performance is markedly improved, and some amount of overfitting is now revealed.
# 
# **NOTE:** From this point onward, the "Dropout" ROC curve (and all successive curves) includes LR Reduction on Plateaus.

# In[ ]:


roc_curves['Dropout'] = roc_curves.pop('Reduce LR')


# ## Batch normalization

# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  
])
model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['acc'])


# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1./255.)
validation_datagen = ImageDataGenerator(rescale = 1./255.)

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    train_dir,
                                                    x_col='filename',
                                                    y_col='category',
                                                    batch_size=50,
                                                    class_mode='binary',
                                                    target_size=(150, 150),
                                                       seed=42)     
validation_generator =  validation_datagen.flow_from_dataframe(validate_df,
                                                          train_dir,
                                                          x_col='filename',
                                                          y_col='category',
                                                         batch_size=50,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150),
                                                            seed=42)


# In[ ]:


earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=9, restore_best_weights=True)
reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=0.00001)


# In[ ]:


# history = model.fit_generator(train_generator,
#                               epochs=40,
#                               validation_data=validation_generator,
#                              callbacks=[earlystopping, reducelr])


# In[ ]:


# # Save model
# model.save('bnorm_model.h5')
# save_history(history, 'bnorm_history.pkl')
# save_earlystopping(earlystopping, 'bnorm_earlystopping.pkl')


# In[ ]:


# Load model
model = keras.models.load_model('../input/cat-vs-dog-cnn-models/bnorm_model.h5')
history = load_history('../input/cat-vs-dog-cnn-models/bnorm_history.pkl')
earlystopping = load_earlystopping('../input/cat-vs-dog-cnn-models/bnorm_earlystopping.pkl')


# In[ ]:


fpr, tpr = report(model, validation_generator, history, earlystopping)
roc_curves['Batch Norm'] = ROCCurveParams(fpr, tpr, 'Blue', '-')
plot_roc_curves(roc_curves)


# ### Summary
# 
# Batch normalization sped up the training considerably, and has improved the training performance, but did not affect the validation performance very much. Overfitting is now extremely evident.
# 
# **NOTE:** From now on, the "Dropout" and all successive ROC curves include Batch Normalization.

# In[ ]:


roc_curves['Dropout'] = roc_curves.pop('Batch Norm')


# ## Double filters in all conv layers

# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  
])
model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['acc'])


# In[ ]:


model.summary()


# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1./255.)
validation_datagen = ImageDataGenerator(rescale = 1./255.)

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    train_dir,
                                                    x_col='filename',
                                                    y_col='category',
                                                    batch_size=50,
                                                    class_mode='binary',
                                                    target_size=(150, 150),
                                                       seed=42)     
validation_generator =  validation_datagen.flow_from_dataframe(validate_df,
                                                          train_dir,
                                                          x_col='filename',
                                                          y_col='category',
                                                         batch_size=50,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150),
                                                            seed=42)


# In[ ]:


earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=9, restore_best_weights=True)
reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=0.00001)


# In[ ]:


# history = model.fit_generator(train_generator,
#                               epochs=50,
#                               validation_data=validation_generator,
#                              callbacks=[earlystopping, reducelr])


# In[ ]:


# # Save model
# model.save('big_model.h5')
# save_history(history, 'big_history.pkl')
# save_earlystopping(earlystopping, 'big_earlystopping.pkl')


# In[ ]:


# Load model
model = keras.models.load_model('../input/cat-vs-dog-cnn-models/big_model.h5')
history = load_history('../input/cat-vs-dog-cnn-models/big_history.pkl')
earlystopping = load_earlystopping('../input/cat-vs-dog-cnn-models/big_earlystopping.pkl')


# In[ ]:


fpr, tpr = report(model, validation_generator, history, earlystopping)
roc_curves['Double filters'] = ROCCurveParams(fpr, tpr, 'Green', '-')
plot_roc_curves(roc_curves)


# ### Summary
# 
# Doubling the number of filters in each convolutional layer greatly improves training performance, and improves the speed of training to reach the same level of performance as the smaller model (even with longer epoch durations taken into account).
# 
# Validation accuracy at the lowest validation loss is roughly the same, as illustrated in the ROC curve. But validation accuracy continues to increase for roughly constant loss, and in particular is almost two percentage points higher for nearly the same loss in epoch 15 compared to epoch 7. The lowest validation loss has slightly improved, but the learning curves still indicate major overfitting.

# ## Dataset augmentation

# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  
])
model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['acc'])


# In[ ]:


model.summary()


# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale = 1./255.)

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    train_dir,
                                                    x_col='filename',
                                                    y_col='category',
                                                    batch_size=50,
                                                    class_mode='binary',
                                                    target_size=(150, 150),
                                                       seed=42)     
validation_generator =  validation_datagen.flow_from_dataframe(validate_df,
                                                          train_dir,
                                                          x_col='filename',
                                                          y_col='category',
                                                         batch_size=50,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150),
                                                              seed=42)


# In[ ]:


earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=9, restore_best_weights=True)
reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=0.00001)


# In[ ]:


# history = model.fit_generator(train_generator,
#                               epochs=50,
#                               validation_data=validation_generator,
#                              callbacks=[earlystopping, reducelr])


# In[ ]:


# # Save model
# model.save('daug_model.h5')
# save_history(history, 'daug_history.pkl')
# save_earlystopping(earlystopping, 'daug_earlystopping.pkl')


# In[ ]:


# Load model
model = keras.models.load_model('../input/cat-vs-dog-cnn-models/daug_model.h5')
history = load_history('../input/cat-vs-dog-cnn-models/daug_history.pkl')
earlystopping = load_earlystopping('../input/cat-vs-dog-cnn-models/daug_earlystopping.pkl')


# In[ ]:


fpr, tpr = report(model, validation_generator, history, earlystopping)
roc_curves['Data augment'] = ROCCurveParams(fpr, tpr, 'Purple', '-')
plot_roc_curves(roc_curves)


# ### Summary
# 
# Due to the extra image preprocessing, the time per epoch is much longer. The training set accuracy has gone down, but of course the training data is now inconsistent, due to the random image manipulations.
# 
# After a few initial epochs of variability, the validation performance improves quite a lot. Both validation loss and accuracy beat the corresponding training measures. In terms of validation performance, this is by far the best model in this notebook.
# 
# Some mathematical speculation: In some sense, the training image manipulations introduce an extra dimension of variation to the probability manifold from which training images are drawn. The validation set is drawn from the center of all that variation (the mean; all variations exactly zero) and is enveloped by the expanding "cloud" of training set points. Thus it ends up sitting nicely in the middle of the training distribution and is therefore particularly well-described by a model trained on the training set.
