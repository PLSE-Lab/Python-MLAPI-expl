#!/usr/bin/env python
# coding: utf-8

# # Product Detection with Pre-Trained MobileNetV2 on Tensorflow 2.0

# This notebook demonstrate product detection using the pre-trained MobileNetV2 model from Tensorflow 2.0 framework. The goal is to get an accuracy score between 50 to 60 percent with minimal parameters and model size, so the experiment could be done in a short time.
# 
# According to [this page](https://keras.io/api/applications/) on Keras documentation, MobileNetV2 has the least model parameters (~3M) and least memory required (14MB). It is worth to note that with this minimal specifications, the accuracy on ImageNet validation set is equal to VGG model which has ~37x larger model parameters and ~46x larger memory size.
# 
# Let's get started by importing library needed.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import tensorflow as tf


# It is recommended to use GPU for fast model training, inference, or dataset loading using ImageDataGenerator class provided by Tensorflow. To check what GPU is we assigned to, we could check using this Linux command.

# In[ ]:


get_ipython().system('nvidia-smi')


# ## Dataset

# There are roughly 105k samples on the training set and ~12k samples on the testing set provided in this competition. However, there are samples considered to be broken, that apparently they have file name with length more than 36 (32 characters long for hex encoding and `.jpg` format). We can detect their path using this script and generate it into a `.txt` file for deleting purposes, though keeping them won't cause any problem when loading the data.

# In[ ]:


train_path = '../input/shopee-product-detection-student/train/train/train/'
test_path = '../input/shopee-product-detection-student/test/test/test/'

broken_fnames = []
for label in os.listdir(train_path):
    label_path = train_path + label + '/'
    for filename in os.listdir(label_path):
        if len(filename) > 36:
            print(label_path + filename)
            broken_fnames.append(label_path + filename)
            
print()
for filename in os.listdir(test_path):
    if len(filename) > 36:
        print(test_path + filename)
        broken_fnames.append(test_path + filename)
        
f = open('broken-file-names.txt', 'w')
f.write('\n'.join(broken_fnames))
f.close()


# Now is time to prepare the generator for the dataset. There are some notes for choosing the training and testing specification:
# 
# - All of the images (train, validation, and test) are resized to shape of 224x224 px (the default image size MobileNetV2 is trained on)
# - Only 2,990 samples out of 105,390 training samples (~2%) are used for the validation set. This should be enough to validate the training result, with the remaining 102,400 samples for training are also large enough for the model to learn
# - Batching of the datasets with a size of 128 samples each. You could set smaller batch size if you prefer a more smooth gradient and more steps in one epoch, or larger batch size for more samples to be considered by the model to step over on the loss surface
# 
# However we don't need to clean or preprocess it to keep the time short, and let the AI do the magic.

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 128
SEED = 0

def get_set():
    train_path = '../input/shopee-product-detection-student/train/train/train/'
    test_path = '../input/shopee-product-detection-student/test/test/'

    train_gen = ImageDataGenerator(rescale=1./255, validation_split=3007./105390)
    train_set = train_gen.flow_from_directory(train_path, target_size=IMAGE_SIZE,                                               batch_size=BATCH_SIZE, seed=SEED,                                               subset='training')
    val_set = train_gen.flow_from_directory(train_path, target_size=IMAGE_SIZE,                                             batch_size=BATCH_SIZE, seed=SEED,                                             subset='validation')

    test_gen = ImageDataGenerator(rescale=1./255)
    test_set = train_gen.flow_from_directory(test_path, target_size=IMAGE_SIZE,                                              batch_size=BATCH_SIZE, seed=SEED,                                              shuffle=False, class_mode=None)
    
    return train_set, val_set, test_set

train_set, val_set, test_set = get_set()


# ## The Model

# We instantiate the MobileNetV2 as the feature extractor and set it as untrainable to train faster. The weights used are that used on ImageNet, and at the top of the extractor, we used global average pooling layer to minimize the output vector dimension. For the classifier, we only use a single softmax layer to output the predictions. Adam is used as the optimizer with its default parameter.

# In[ ]:


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

def get_model():
    base = MobileNetV2(input_shape=IMAGE_SIZE+(3,), include_top=False,                        pooling='avg', weights='imagenet')
    base.trainable = False
    dense = Dense(42, activation='softmax', name='dense')(base.output)

    model = Model(inputs=base.inputs, outputs=dense, name='mobilenetv2')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model

model = get_model()


# ## Training Phase

# Since we have 102,400 training samples and batch size of 128, then there should be 102,400/128 = 800 steps or weights updates per epoch. It should be enough to train the model for 3 epoch (total 2,400 weights updates). At the end of the training phase, we save the model to a `model-mobilenetv2.hdf5` file.

# In[ ]:


EPOCHS = 3

hist = model.fit(train_set, epochs=EPOCHS, batch_size=BATCH_SIZE,                  validation_data=val_set, shuffle=False)
model.save('model-mobilenetv2.hdf5')


# We could recheck the final accuracy for the validation set.

# In[ ]:


loss, acc = model.evaluate(val_set, batch_size=BATCH_SIZE)
print('Validation acc (percent): %.2f' % (100 * acc))


# ## Generate Prediction

# We make a function to generate `.csv` prediction with the model and saving directory as the arguments.

# In[ ]:


def generate_prediction(model, save_name):
    subm = pd.read_csv('../input/shopee-product-detection-student/test.csv')
    subm = subm.sort_values(by='filename', ignore_index=True)
    
    fnames = sorted(os.listdir('../input/shopee-product-detection-student/test/test/test'))
    unbroken_index = np.where(np.vectorize(len)(np.array(fnames)) == 36)[0]
    
    y_pred = model.predict(test_set, batch_size=BATCH_SIZE)
    pred = y_pred.argmax(axis=1)
    pred = pred[unbroken_index]
    subm['category'] = pred
    subm['category'] = subm['category'].apply(lambda x : '%02d' % x) # zero pad
    
    subm.to_csv(save_name, index=False)
    return subm


# In[ ]:


from tensorflow.keras.models import load_model

model = load_model('model-mobilenetv2.hdf5')
subm = generate_prediction(model, './submission.csv')
subm


# ## Improvements

# This kernel is just used as a baseline, since the goal is only to get fairly enough accuracy in a short time. Further improvements could be done by augmenting the data or try different feature extractor, classifier, or hyperparameters.
