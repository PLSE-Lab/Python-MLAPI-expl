#!/usr/bin/env python
# coding: utf-8

# # Import everything & find data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras import Input, Model, optimizers
from keras.layers import Dense, regularizers, Dropout, GlobalAveragePooling2D
from keras.applications import InceptionV3, inception_v3
from sklearn.metrics import classification_report

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

image_root = '/kaggle/input/marvel/marvel'
# Any results you write to the current directory are saved as output.


# Helper functions to rescale image pixel values from [0, 255] to [-1, 1] and restore them back.

# In[ ]:


def normalize_image(images):
    return images / 127.5 - 1


def restore_image(images):
    return (images + 1) / 2


# # Loading data in batches
# 
# This function will create 2 generators responsible for loading training and testing datasets in batches. Since many times we deal with image data that's too large to fit into memory space, usually we read images in batches.

# In[ ]:


def get_image_loader(image_size, batch_size, color_mode='rgb', shuffle=True):
    data_gen = ImageDataGenerator(
        preprocessing_function=normalize_image
    )
    train_loader = data_gen.flow_from_directory(
        os.path.join(image_root, 'train'),
        color_mode=color_mode,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle
    )
    valid_loader = data_gen.flow_from_directory(
        os.path.join(image_root, 'valid'),
        color_mode=color_mode,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle
    )
    return train_loader, valid_loader


train, valid = get_image_loader(image_size=224, batch_size=64, shuffle=True)
print("There are {} batches in training set, {} batches in validation set".format(len(train), len(valid)))


# Visualize a few sample images from the dataset.

# In[ ]:


def display_image(*images, col=None, width=20):
    from matplotlib import pyplot as plt

    if col is None:
        col = len(images)
    row = np.math.ceil(len(images) / col)
    plt.figure(figsize=(width, (width + 1) * row / col))
    for i, image in enumerate(images):
        plt.subplot(row, col, i + 1)
        plt.axis('off')
        plt.imshow(image, cmap='gray')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


batch_x, _ = train[0]
restored = restore_image(batch_x)[:16]
display_image(*restored, col=8)


# # Extract important features from images
# 
# This function creates a pre-trained deep learning model based on Google's work named [Inception v3](https://arxiv.org/abs/1512.00567) used to extract image features. In other words, each image will be fed into the deep learning model in the format of 3-dimensional tensors (3 224x224 matrices stacking together), and the model will extract 1280 features from that image represented as a 1280-dimensional vector.

# In[ ]:


def get_feature_extractor():
    base = InceptionV3(include_top=False, input_shape=(224, 224, 3), pooling='avg')
    btnk = base.get_layer('mixed8').output
    features = GlobalAveragePooling2D()(btnk)
    return Model(base.input, features)

feature_extractor = get_feature_extractor()
features = feature_extractor.predict(batch_x)
print(features.shape)


# extract features from both datasets

# In[ ]:


def extract_features(extractor, dataset):
    x, y = None, None
    n = len(dataset)
    for i in range(n):
        print("\r[info] extracting features: batch #{}/{}...".format(i + 1, n), end='')
        batch_x, batch_y = dataset[i]
        features = extractor.predict(batch_x)
        if x is None:
            x = features
            y = batch_y
        else:
            x = np.vstack([x, features])
            y = np.vstack([y, batch_y])
    return x, y

print("\n[info] extracting training set image features:")
x_train, y_train = extract_features(feature_extractor, train)
print("\n[info] training set complete!\n")

print("\n[info] extracting validation set image features:")
x_valid, y_valid = extract_features(feature_extractor, valid)
print("\n[info] validation set complete!\n")

print("training set input shape = {}, output shape = {}".format(x_train.shape, y_train.shape))
print("validation set input shape = {}, output shape = {}".format(x_valid.shape, y_valid.shape))


# Now that we have the images represented as 1280 features, we create a fully connected neural network to try to find the mapping between the feature vectors and the images' labels.

# In[ ]:


def create_nn(input_size, num_classes):
    """this function creates a neural network model that 
    accepts vectors of 'input_size' dimensions and outputs 
    `num_classes` dimensional one-hot vectors to represent
    the corresponding class label"""
    inputs = Input(shape=(input_size,))
    x = Dropout(.5)(inputs)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(.5)(x)
    out = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(.1))(x)
    return Model(inputs, out)

model = create_nn(x_train.shape[1], y_train.shape[1])
opt = optimizers.Adam(lr=.0001, beta_1=.99, beta_2=.999, epsilon=1e-8)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

print("\n[info] start training...")
H = model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_valid, y_valid), verbose=2)
plt.plot(H.history['loss'], label='train loss')
plt.plot(H.history['val_loss'], label='valid loss')
plt.legend()

plt.figure()
plt.plot(H.history['acc'], label='train acc')
plt.plot(H.history['val_acc'], label='valid acc')
plt.legend()


# evaluate the trained performance and visualize some of them 

# In[ ]:


test_results = model.predict(x_valid)

id2label = {val: key for key, val in valid.class_indices.items()}

print(classification_report(
    y_valid.argmax(axis=1),
    test_results.argmax(axis=1),
    target_names=id2label.values()
))

sample_images, sample_labels = valid[0]
sample_images = sample_images[:16]
sample_images = (sample_images + 1) / 2
plt.figure(figsize=(20, 25))
col = 4
row = np.math.ceil(len(sample_images) / col)
for i, image in enumerate(sample_images):
    plt.subplot(row, col, (i+1))
    plt.title(id2label[sample_labels[i].argmax()], fontsize=30)
    plt.text(0, 256, id2label[test_results[i].argmax()], color='red', fontsize=30)
    plt.axis('off')
    plt.imshow(image)


# Now that we have the trained models, how do we use them for later without retraining the models? Let's combine the 2 models and save it on disk

# In[ ]:


bottleneck = feature_extractor.output
out = model(bottleneck)
hero_recognizer = Model(feature_extractor.input, out)
hero_recognizer.save("hero_recognizer.h5")


# We load the model like this in any other python programs

# In[ ]:


import cv2
from keras.engine.saving import load_model

hero_recognizer = load_model('hero_recognizer.h5')


# And use the model to recognize Marvel heroes like a pro!

# In[ ]:


import requests
from io import BytesIO
from PIL import Image

url = 'https://i.ytimg.com/vi/bPTZ43nM688/maxresdefault.jpg'
response = requests.get(url)
test_image = Image.open(BytesIO(response.content))
test_image = np.array(test_image)[:, :, :3]

input_image = cv2.resize(test_image, (224, 224))
input_image = normalize_image(input_image)

pred = hero_recognizer.predict(np.array([input_image]))

predicted_hero = id2label[pred.argmax()]
print("The model thinks this image is of [{}].".format(predicted_hero))
display_image(test_image)

