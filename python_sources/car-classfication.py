#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


# In[ ]:


get_ipython().system('pip install -q -U tf-hub-nightly')
get_ipython().system('pip install -q tfds-nightly')
import tensorflow_hub as hub

from tensorflow.keras import layers


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train_dir = '/kaggle/input/car-classificationproject-vision/Train/Train/'
test_dir = '/kaggle/input/car-classificationproject-vision/Test/Test/Test1/'


# In[ ]:


len(train_dir)


# In[ ]:


for dirname, _, filenames in os.walk(train_dir):
#     print(dirname)
#     print(filenames)
    print(dirname[59:])
#     for filename in filenames:
#         print(filename)


# In[ ]:


car_map = pd.read_csv('/kaggle/input/car-classificationproject-vision/car_map.csv')
car_map=car_map.set_index('Cars')
car_map_dict=car_map.to_dict('index')


# In[ ]:


car_map_dict


# In[ ]:


batch_size = 64
IMG_HEIGHT = 224
IMG_WIDTH = 224


# In[ ]:


train_image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True, rotation_range=5, shear_range=0.2, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2) # Generator for our training data
# train_image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data


# In[ ]:


def oh_to_class(Y):
    result = np.zeros((Y.shape[0], 1))
    for i in range(Y.shape[0]):
        result[i]=np.argmax(Y[i])
    return result


# In[ ]:


train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')


# In[ ]:


for dirname, _, filenames in os.walk(test_dir):
#     print(dirname)
    print(filenames)
#     print(dirname)
#     for filename in filenames:
#         print(filename)


# In[ ]:


test_dir='/kaggle/input/car-classificationproject-vision/Test/Test/'


# In[ ]:


test_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=test_dir,
                                                           shuffle=False,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode=None)


# In[ ]:


sample_training_images, labels = next(train_data_gen)


# In[ ]:


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# In[ ]:


len(sample_training_images)


# In[ ]:


plotImages(sample_training_images[:5])


# In[ ]:


labels[:5]


# In[ ]:


oh_to_class(labels[:5])


# In[ ]:


train_data_gen.class_indices


# In[ ]:


def output_labels_to_map_labels(Y, output_label_dict, map_label_dict):
#     Y input is not one hot encoded, but in the form of labels
    result = np.zeros(Y.shape) 
    for i in range(Y.shape[0]):
        curr_label_ind = Y[i][0]
        car_name=0
        for key, val in output_label_dict.items(): 
#             print(key, val)
            if val == curr_label_ind: 
                car_name = key
                break
        print
        req_label_index = map_label_dict[car_name]['Class Numbers']
        result[i][0]=req_label_index
    return result


# In[ ]:


plotImages(sample_training_images[10:15])


# In[ ]:


oh_to_class(labels[10:15])


# In[ ]:


output_labels_to_map_labels(oh_to_class(labels[10:15]),train_data_gen.class_indices, car_map_dict)


# In[ ]:


# model = Sequential()
# model.add(Conv2D(64, kernel_size=(3, 3),input_shape=(224,224,3), padding='same', kernel_initializer='he_normal', activation='relu'))
# model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
# model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# model.add(Dropout(0.4))
# model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
# model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
# model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# model.add(Dropout(0.4))
# model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
# model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
# model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# model.add(Dropout(0.4))
# model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
# model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
# model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# model.add(Flatten())
# model.add(Dense(4096))
# model.add(Dense(4096, activation='relu'))
# model.add(Dense(45, activation='softmax'))


# In[ ]:


feature_extractor_url = "https://tfhub.dev/tensorflow/resnet_50/feature_vector/1" 


# In[ ]:


feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))


# In[ ]:


model = tf.keras.Sequential([
  feature_extractor_layer,
  layers.Dense(45, activation='softmax')
])

model.summary()


# In[ ]:


model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['acc'])


# In[ ]:


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()
        
#     def on_epoch_end(self, epoch, logs={}):
#         if(logs.get('acc')>0.95):
#             print("Reached 95% accuracy so cancelling training!")
#             self.model.stop_training = True


# In[ ]:


steps_per_epoch = 4050//64

batch_stats_callback = CollectBatchStats()

history = model.fit_generator(train_data_gen, epochs=370,
                              steps_per_epoch=steps_per_epoch,
                              callbacks = [batch_stats_callback])


# In[ ]:


model.evaluate(train_data_gen)


# In[ ]:


train_predicted_oh = model.predict(train_data_gen)


# In[ ]:


oh_to_class(train_predicted_oh)


# In[ ]:


test_predicted_oh = model.predict_generator(test_data_gen)


# In[ ]:


test_predicted=oh_to_class(test_predicted_oh)


# In[ ]:


test_predicted


# In[ ]:


sample_testing_images = next(test_data_gen)


# In[ ]:


plotImages(sample_testing_images[0:5])


# In[ ]:


result_test = output_labels_to_map_labels(test_predicted,train_data_gen.class_indices, car_map_dict)


# In[ ]:


result_test


# In[ ]:


test_data_gen.filenames


# In[ ]:


file_names = []
for name in test_data_gen.filenames:
    file_names+=[name[6:]]


# In[ ]:


file_names=np.array(file_names).reshape(-1, 1)


# In[ ]:


result_image_prediction = np.concatenate((file_names, result_test.astype(np.int32)), axis = 1)


# In[ ]:


result_image_prediction


# In[ ]:


result_df = pd.DataFrame(result_image_prediction, columns=['image', 'predictions'])


# In[ ]:


result_df.set_index('image')


# In[ ]:


result_df.to_csv('submissionCommit12WithAugmentedDataE370.csv', index=False)


# In[ ]:




