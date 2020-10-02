#!/usr/bin/env python
# coding: utf-8

# <a id='top'></a>
# <div class="list-group" id="list-tab" role="tablist">
#   <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Notebook Content!</h3>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#begin" role="tab" aria-controls="profile">Begin<span class="badge badge-primary badge-pill">1</span></a>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#import" role="tab" aria-controls="messages">Import Everything<span class="badge badge-primary badge-pill">2</span></a>
#   <a class="list-group-item list-group-item-action"  data-toggle="list" href="#read" role="tab" aria-controls="settings">Read, Load and Pre-process Data<span class="badge badge-primary badge-pill">3</span></a>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#model" role="tab" aria-controls="settings">Design Model<span class="badge badge-primary badge-pill">4</span></a> 
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#image" role="tab" aria-controls="settings">Overview of Model<span class="badge badge-primary badge-pill">5</span></a>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#augment" role="tab" aria-controls="settings">Data Augmentation<span class="badge badge-primary badge-pill">6</span></a>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#training" role="tab" aria-controls="settings">Training<span class="badge badge-primary badge-pill">7</span></a>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#overview" role="tab" aria-controls="settings">Overview<span class="badge badge-primary badge-pill">8</span></a>  
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#output" role="tab" aria-controls="settings">Output<span class="badge badge-primary badge-pill">9</span></a>  
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#bottom" role="tab" aria-controls="settings">End<span class="badge badge-primary badge-pill">10</span></a>  

# <a id='begin'></a>
# # Begin

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import time
start_time = time.time()


# <a id='import'></a>
# # Import Everything

# In[ ]:


random_seed = 2020
np.random.seed(random_seed)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers import Input, BatchNormalization, Add, Activation
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


# <a id='read'></a>
# ## Read, Load and Pre-process Data

# In[ ]:


#Load train and test data
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')

#Extract and separate prediction (Y) and Inputs (X)
Y = train['label']
X = train.drop(labels="label", axis=1)

#Reshape: Original image is a 28x28 pixel image
X = X.values.reshape(-1, 28, 28, 1) / 255
test = test.values.reshape(-1, 28, 28, 1) / 255

print(X.shape, test.shape)


# <a id='model'></a>
# # Design Model

# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',
                                           patience = 3,
                                           verbose = 1,
                                           factor = 0.5,
                                           min_lr = 0.0001)

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   verbose=1,
                   patience=15,
                   restore_best_weights=True)

def new_model(hidden=512, learning_rate=0.00128):
    #Input layer
    INPUT   = Input((28, 28, 1))
    
    #First Convolution
    inputs  = Conv2D(64, (5, 5), activation='relu', padding='same')(INPUT)
    inputs  = MaxPool2D(pool_size=(3,3), strides=(1,1))(inputs)
    inputs  = BatchNormalization()(inputs)
    inputs  = Activation('relu')(inputs)
    inputs  = Dropout(0.25)(inputs)
    
    #Branch off to Three Towers
    #First Tower
    tower_1 = Conv2D(64, (1, 1), activation='relu', padding='same')(inputs)
    tower_1 = Conv2D(128, (2, 2), activation='relu', padding='same')(tower_1)
    tower_1 = Dropout(0.5)(tower_1)
    tower_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(tower_1)
    tower_1 = MaxPool2D(pool_size=(3,3), strides=(2,2))(tower_1)
    tower_1 = BatchNormalization()(tower_1)
    
    #Second Tower
    tower_2 = Conv2D(64, (2, 2), activation='relu', padding='same')(inputs)
    tower_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(tower_2)
    tower_2 = Dropout(0.5)(tower_2)
    tower_2 = Conv2D(256, (5, 5), activation='relu', padding='same')(tower_2)
    tower_2 = MaxPool2D(pool_size=(3,3), strides=(2,2))(tower_2)
    tower_2 = BatchNormalization()(tower_2)
    
    #Third Tower
    tower_3 = Conv2D(64, (1, 1), activation='relu', padding='same')(inputs)
    tower_3 = Conv2D(128, (3, 3), activation='relu', padding='same')(tower_3)
    tower_3 = Dropout(0.5)(tower_3)
    tower_3 = Conv2D(256, (5, 5), activation='relu', padding='same')(tower_3)
    tower_3 = MaxPool2D(pool_size=(3,3), strides=(2,2))(tower_3)
    tower_3 = BatchNormalization()(tower_3)
    
    #Combine Three Towers
    x       = Add()([tower_1, tower_2, tower_3])
    x       = Activation('relu')(x)
    
    #Last Convolution
    x       = Conv2D(256, (5, 5), activation='relu', padding='same')(x)
    x       = MaxPool2D(pool_size=(5,5), strides=(4,4))(x)
    x       = BatchNormalization()(x)
    x       = Activation('relu')(x)
    #Flatten Data
    x       = Flatten()(x)
    
    #Dense Hidden Network
    x       = Dense(hidden, activation='relu')(x)
    x       = Dropout(0.5)(x)
    x       = Dense(hidden//4, activation='relu')(x)
    x       = Dropout(0.5)(x)
    
    #Model Output
    preds   = Dense(10, activation='softmax', name='preds')(x)
    
    #Build Model
    model   = Model(inputs=INPUT, outputs=preds)
    
    #Define Optimizer
    optimizer = Adam(lr=learning_rate)
    #Compile model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])
    
    return model

model = new_model()


# <a id='image'></a>
# # Overview of Model

# In[ ]:


model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# <a id='augment'></a>
# # Data Augmentation

# In[ ]:


#Data Augmentation to prevent overfitting
datagen = ImageDataGenerator(featurewise_center=False,
                            samplewise_center=False,
                            featurewise_std_normalization=False,
                            samplewise_std_normalization=False,
                            zca_whitening=False,
                            rotation_range=10,
                            zoom_range=0.1,
                            shear_range=0.02,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            horizontal_flip=False,
                            vertical_flip=False)


# <a id='training'></a>
# # Training

# In[ ]:


epochs = 200
batch_size = 128

print("Learning Properties: Epoch:%i \t Batch Size:%i" %(epochs, batch_size))
predict_accumulator = np.zeros(model.predict(test).shape)

accumulated_history = []
for i in range(1, 6):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=random_seed*i)
    model = new_model(512, 0.01)
    #Fit the model
    datagen.fit(X_train)
    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                 epochs=epochs, validation_data=(X_val, Y_val), verbose=1,
                                 steps_per_epoch=X_train.shape[0]//batch_size,
                                 callbacks=[learning_rate_reduction, es],
                                 workers=4)
    loss, acc = model.evaluate(X, Y)
    if acc > 0.99:
        predict_accumulator += model.predict(test)*acc
        accumulated_history.append(history)
        print("Current Predictions on fold number %i" %i)
        print(*np.argmax(predict_accumulator, axis=1), sep='\t')


# <a id='overview'></a>
# # Overview

# In[ ]:


def graph(full_history):
    '''Show and save the historical graph of the training model.'''
    print('Accuracy:')
    n=1
    for history in full_history:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy Fold No. {}'.format(n))
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='lower right')
        plt.savefig('history_acc_{}.png'.format(n))
        plt.show()
        n+=1

    print('Loss:')
    n=1
    for history in full_history:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss Fold No. {}'.format(n))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig('history_loss_{}.png'.format(n))
        plt.show()
        n+=1

graph(accumulated_history)


# <a id='output'></a>
# # Save Output

# In[ ]:


print("Completed Training.")
results = np.argmax(predict_accumulator, axis=1)
results = pd.Series(results, name="Label")
print("Saving prediction to output...")
submission = pd.concat([pd.Series(range(1, 1+test.shape[0]), name="ImageId"), results], axis=1)
submission.to_csv('submission.csv', index=False)


# <a id='bottom'></a>
# # End
# 
# I hope this notebook could be of help to others! If you liked the notebook, please upvote!

# In[ ]:


end_time = time.time()
total_time = int(end_time - start_time)
print("Total time spent: %i hours, %i minutes, %i seconds"       %((total_time//3600), (total_time%3600)//60, (total_time%60)))

