#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.preprocessing.image import save_img
import tensorflow as tf
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

sns.set()


# # Data preparation
# ## 2.1 Load data

# In[ ]:


#load the data and pop the label
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
labels = train.pop('label')

X_train = train.values
X_test = test.values


# ## 2.4 label encoding

# In[ ]:


#if we do the one hot vector encoding, we will use 'categorical_crossentropy' for loss function
#otherwise, with integer labels as our targets, we use  'sparse_categorical_crossentropy'

from keras.utils.np_utils import to_categorical
#encode labels to one hot vector
Y_train = to_categorical(labels,num_classes=10);


# encode the label to one hot vector

# ## 2.5 split training and validation

# In[ ]:


from sklearn.model_selection import train_test_split

#set therandom seed
random_seed=1;
#split the train and validation sets
x_train, x_val, y_train, y_val = train_test_split(X_train,labels, test_size =.2, random_state = random_seed)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

scaled_x_train = scaler.transform(x_train)
scaled_x_train = scaled_x_train.reshape(-1,28,28,1)

scaled_x_val = scaler.transform(x_val)
scaled_x_val = scaled_x_val.reshape(-1,28,28,1)


# # 3. CNN
# ## 3.1 Define the model

# In[ ]:


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator #For data augmentation
from tensorflow.keras.callbacks import ReduceLROnPlateau #for annealer


print(tf.__version__)


# In[ ]:


#build the keras NN model
new_model =Sequential([
    Conv2D(filters = 64, activation = 'relu',kernel_size=(5,5),padding='Same',input_shape = (28,28,1)),
    Conv2D(filters = 64, activation = 'relu',kernel_size=(5,5),padding='Same'),
    MaxPool2D(pool_size=(2,2),strides=(2,2)),
    Dropout(0.25),

    Conv2D(filters = 64, activation = 'relu',kernel_size=(3,3),padding='Same'),
    Conv2D(filters = 64, activation = 'relu',kernel_size=(3,3),padding='Same'),
    MaxPool2D(pool_size=(2,2),strides=(2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    
    Dropout(0.5),
    Dense(10, activation='softmax'), #a normalized exponential functions for probability
])

#compile the model
optimizer = RMSprop(lr=0.001)
new_model.compile(optimizer = optimizer,
              loss= 'sparse_categorical_crossentropy',
              metrics = ['acc']
             )


# In[ ]:


#prepare for submission

scaled_x_test = scaler.transform(X_test)
scaled_x_test = scaled_x_test.reshape(-1,28,28,1)


# In[ ]:


loss, acc = new_model.evaluate(scaled_x_train.reshape(-1,28,28,1),y_train)


# In[ ]:


new_model.load_weights("../input/digit-recognizer-with-keras-cnn/model_fitted.h5")
loss, acc = new_model.evaluate(scaled_x_train.reshape(-1,28,28,1),y_train)


# The models weights are loaded

# # Visualizing intermediate activations

# ## We first look at all layers

# In[ ]:


for i,layer in enumerate(new_model.layers):
    print((i,layer))


# ## visualize the weights in the last hidden layer(Dense 256->10)

# ## Build Code vector model(input =scaled_image, output =256*1 code vector)

# In[ ]:


#code vector model
dense_layer_outputs = [new_model.layers[9].output]
print(dense_layer_outputs)
code_vector_act_model = keras.models.Model(inputs = new_model.input,
                                     outputs = dense_layer_outputs)


# ## group data of the same label together

# In[ ]:


#the position of i's in the training set
label_position_in_train=[];
for label in range(10):
    label_position_in_train.append(np.where(y_train==label)[0][:])
    
imgs,img_tensors = [0]*10,[0]*10;
for label in range(10):
    imgs[label] = np.array([scaled_x_train[index_of_image].reshape(-1,28,28) for index_of_image in label_position_in_train[label]])
    img_tensors[label] = imgs[label].reshape(-1,28,28,1)
    



# In[ ]:


code_vectors = [code_vector_act_model.predict(img_tensors[label]) for label in range(10)]


# # Plots of code vectors

# In[ ]:


f,ax = plt.subplots(2,5, figsize=(30,6))
for label in range(10):
    ax[label//5][label%5].plot(code_vectors[label][0], label = 'sample',alpha=.7)
    ax[label//5][label%5].plot(code_vectors[label].mean(axis=0),label = 'mean',alpha=.7)
    ax[label//5][label%5].set_title('label='+str(label))
    ax[label//5][label%5].legend()
    f.savefig('code_vector consistency.png')


# ## shift the images

# In[ ]:


index =0;
img = scaled_x_val[index];
# plt.figure()
# plt.imshow(img.reshape(28,28))
transformed_images = [img];

transform_parameters_list = [{'ty':-1},{'tx':-2},{'tx':-3},{'tx':-4},{'tx':-5}]

for i in range(len(transform_parameters_list)):
    image_datagen = ImageDataGenerator()
    transform_parameters =transform_parameters_list[i]
    transformed_images.append(image_datagen.apply_transform(img,transform_parameters))
#     ax[i//3,i%3].imshow(transformed_images[i+1].reshape(28,28))

transformed_images = np.array(transformed_images)
print(transformed_images.shape)
f, ax = plt.subplots(1,6, figsize=(36,6))
for i in range(transformed_images.shape[0]):
    ax[i].imshow(transformed_images[i].reshape(28,28))
    ax[i].set_title('shift ='+str(-i))
f.savefig('shifted_images.png')
    
f, ax = plt.subplots(1,6, figsize =(36,6))    
transformed_code_vectors = code_vector_act_model.predict(transformed_images);
ax[0].plot(transformed_code_vectors[0])
ax[0].set_title('original code vector')
for i in range(1,transformed_images.shape[0]):
    ax[i].plot(transformed_code_vectors[i-1],alpha=.7, label = 'shift='+str(-i+1))
    ax[i].plot(transformed_code_vectors[i],alpha=.7,label = 'shift='+str(-i))
    ax[i].legend()
    ax[i].set_title('code vectors comparison')
f.savefig('shifted_code_vectors.png')


# # End
