#!/usr/bin/env python
# coding: utf-8

# # Load dataset

# In[ ]:


from matplotlib import pyplot as plt
import os
import numpy as np


# In[ ]:


dataset_path = "../input/pet-dataset/"
x_train_file = os.path.join(dataset_path,'x_train_cls.npy')
y_train_file = os.path.join(dataset_path,'y_train_cls.npy')
x_test_file = os.path.join(dataset_path,'x_test_cls.npy')
y_test_file = os.path.join(dataset_path,'y_test_cls.npy')

x_train = np.load(x_train_file)
y_train = np.load(y_train_file)
x_test = np.load(x_test_file)
y_test = np.load(y_test_file)

print("X_train: {},{}".format(x_train.shape,x_train.dtype))
print("Y_train: {}".format(y_train.shape))
print("X_test: {}".format(x_test.shape))
print("Y_test: {}".format(y_test.shape))


# # Normalize data

# In[ ]:


from keras.utils import to_categorical
x_train = x_train/255.0
x_test  = x_test/255.0
y_train = to_categorical(y_train,37)
y_test = to_categorical(y_test,37)


# # Import keras

# In[ ]:


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D,Activation
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping


# # Load pretrained model VGG19

# In[ ]:


img_width, img_height = 64, 64

base_model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))


# In[ ]:


print(base_model.summary())


# # Fineturning Model (Cut output layer) Function

# In[ ]:


def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    # Freeze Parameters for train
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x) 
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model


# # Contruct last layer

# In[ ]:


FC_LAYERS = [1024, 1024, 1024]
dropout = 0.5

finetune_model = build_finetune_model(base_model, 
                                      dropout=dropout, 
                                      fc_layers=FC_LAYERS, 
                                      num_classes=37)


# In[ ]:


print(finetune_model.summary())


# # Define learning algorithm

# In[ ]:


from keras.optimizers import SGD
# Compile the model
epochs = 100
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
finetune_model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])


# # Training it

# In[ ]:


# Train the model
finetune_model.fit(x_train, y_train,
              batch_size=512,
              shuffle=True,
              epochs=epochs,
              validation_data=(x_test, y_test),
              callbacks=[EarlyStopping(min_delta=0.001, patience=3)])


# # Evaluate the model

# In[ ]:


scores = finetune_model.evaluate(x_test , y_test)

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])

