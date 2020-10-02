#!/usr/bin/env python
# coding: utf-8

# # ARE YOU ATTRACTIVE OR NOT? (LET SEE WHAT NEURAL NETWORKS HAVE TO SAY)
# 
# ## INTRODUCTION 
# 
# The objective of this project is trying to predict if a person is attractive or not, based on almost 200k images of celebrities. 
# 
# This dataset has information of attributes like gender, hair color, the eyebrows, big lips, big nose, makeup, all of these binary attributes. 
# 
# For this project I have two approaches, both of them using neural networks, the firs approach is using attractive as target and using the other attributes as predictors. For the second approach I'll be using the images on the dataset to predict the attribute 'attractive'.
# 
# This data was collected by the Chinese University of Hong Kong.
# 
# The original images have different backgrounds, sizes and angles for each person. 
# 
# 
# 

# ### Imports
# 
# For this project I'll be  using Tensorflow version 1.13.1, the available one at the kaggle kernels**.

# In[2]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))


# In[3]:


import pandas as pd
import numpy as np
import cv2    
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from time import time
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
from PIL import Image
from io import BytesIO
import base64
from keras.utils import np_utils

from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation 

from keras.models import Sequential, Model 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
from IPython.core.display import display, HTML


from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard 

plt.style.use('ggplot')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import tensorflow as tf
print(tf.__version__)


# ## Exploring the dataset
# 
# As the target variable is the attractive attribute it is important to see it's relationship with other attibutes. 
# 
# The most correlated attributes with the target variable are the lipstick with a 48% correlation, young with a 38% correlation, 47% correlation with heavy makeup and a weak correlation of 25% with arched eyebrows.
# 

# In[5]:


main_folder = '../input/'
images_folder = main_folder + 'img_align_celeba/img_align_celeba/'

EXAMPLE_PIC = images_folder + '000225.jpg'

TRAINING_SAMPLES = 10000
VALIDATION_SAMPLES = 2000
TEST_SAMPLES = 2000
IMG_WIDTH = 178
IMG_HEIGHT = 218
BATCH_SIZE = 16
NUM_EPOCHS = 20


# In[6]:


# import the data set that include the attribute for each picture
data = pd.read_csv(main_folder + 'list_attr_celeba.csv')
data.set_index('image_id', inplace=True)
data.replace(to_replace=-1, value=0, inplace=True) #replace -1 by 0
data.shape


# In[7]:


img = load_img(EXAMPLE_PIC)
plt.grid(False)
plt.imshow(img)
data.loc[EXAMPLE_PIC.split('/')[-1]][['Attractive','Male','Young']] #some attributes


# The attractive attribute class is almost balanced, therefore I'm not going to use any class balance algorithm. Other important attributes are almost balanced as well.
# 
# In the plots below it is able to see the class balance for some important attributes.

# In[8]:


plt.title('Attractiveness')
sns.countplot(y='Attractive', data=data, color="c")
plt.show()


# In[9]:


plt.title('Gender')
sns.countplot(y='Male', data=data, color="c")
plt.show()


# In[10]:


plt.title('Age')
sns.countplot(y='Young', data=data, color="c")
plt.show()


# In[12]:


plt.title('Heavy_Makeup')
sns.countplot(y='Wearing_Lipstick', data=data, color="c")
plt.show()


# In[11]:


plt.title('Arched_Eyebrows')
sns.countplot(y='Wearing_Lipstick', data=data, color="c")
plt.show()


# In[13]:


corr = data.corr()
sns.heatmap(corr)


# In[14]:


corr


# ### Train test split
# 
# Train and test split for the moodels using attributes as **preditors.

# In[15]:


y = data["Attractive"]
X = data.drop(['Attractive'], axis=1)


# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[19]:


X_train.shape, X_test.shape


# In[21]:


df = []
labels = []
for img in images_folder:
    try:
        img_read = plt.imread( images_folder + "/" + img)
        img_resize = cv2.resize(img_read, (50, 50))
        img_array = img_to_array(img_resize)
        df.append(img_array)
        labels.append(1)
    except:
        None


# In[ ]:


X_train.shape, X_test.shape


# # METHODS 
# 
# For this project the neural networks models were Sequential models. This models are linear stacks of layers. 
# 
# With model.add it is possible to add as much layers as you want. Usually, multilayer models have an input layer, hidden layers and an output layer.
# 
# For the first layer of a sequential model it is needed to specify the information about the input, like the input shape and the input dimension. 
# 
# Before getting to testing the model it is needed to compile the model. In the compilation process you can decide which optimizer and metric you are going to use to evaluate your model.
# 
# After compilation, with the fit function it is possible to train the model.
# 
# In this model a multilayer perceptron for binary classification model was used. 
# 
# Neural networks have some basic components: input nodes, connections, transfer or activation function, output node and bias.
# 
# Each connection of each input node has weights associated.Then all the values of the input nodes are used as input as a weighted sum wich will be the input for the transfer function. Transfer funcitons are activated only when an specific threshold is exceeded. As a result there's an output node (it is associated with the function).
# 
# ### Convolutional Neural Networks
# 
# A Convolutional Neural Network is a deep learning algorithm that takes images as input.
# 
# For these models images are used as different levels of pixels matrix, each "level" specifies the color (trhough RGB, 3 layers) and each image has height and width dimensions. 
# 
# These algorithms can easily turn computationally expensive.

# ## Only using the csv
# 
# ### First Neural network

# In[ ]:


model = tf.keras.Sequential()
model.add(Dense(64, input_dim=39, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(X_test, Y_test, batch_size=128)


# In[ ]:


model.history


# In[ ]:


test_loss, test_acc = model.evaluate(X_test, Y_test)

print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)


# In[ ]:


predictions = model.predict(X_test)


# 
# ### Second Neural network

# In[ ]:


model = tf.keras.Sequential()
model.add(Dense(64, input_dim=39, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(X_test, Y_test, batch_size=128)


# In[ ]:


test_loss, test_acc = model.evaluate(X_test, Y_test)

print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)


# ## Using the images

# In[ ]:


first_model = tf.keras.Sequential()


# In[ ]:


first_model.add(Flatten(input_shape=(39,1)))


# In[ ]:


first_model.add(Dense(64, activation='sigmoid'))


# In[ ]:


first_model.add(Dense(10, ))


# In[ ]:


first_model.compile(optimizer = 'sgd',
                    loss = 'sparse_categorical_crossentropy',
                    metrics = ['accuracy'])


# In[ ]:


model_history = first_model.fit(x = X_train,
                                y = y_train,
                                batch_size = 128,
                                epochs = 5,
                                validation_split = 0.2,
                                shuffle=True)


# In[ ]:


df_partition = pd.read_csv(main_folder + 'list_eval_partition.csv')
df_partition.head()


# In[ ]:


#df_partition.set_index('image_id', inplace=True)
#df_par_attr = df_partition.join(data['Attractive'], how='inner')
df_par_attr.head()


# In[ ]:


def load_reshape_img(fname):
    img = load_img(fname)
    x = img_to_array(img)/255.
    x = x.reshape((1,) + x.shape)

    return x


def generate_df(partition, attr, num_samples):
    df_ = df_par_attr[(df_par_attr['partition'] == partition) 
                           & (df_par_attr[attr] == 0)].sample(int(num_samples/2))
    df_ = pd.concat([df_,
                      df_par_attr[(df_par_attr['partition'] == partition) 
                                  & (df_par_attr[attr] == 1)].sample(int(num_samples/2))])

    # for Train and Validation
    if partition != 2:
        x_ = np.array([load_reshape_img(images_folder + fname) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0], 218, 178, 3)
        y_ = np_utils.to_categorical(df_[attr],2)
    # for Test
    else:
        x_ = []
        y_ = []

        for index, target in df_.iterrows():
            im = cv2.imread(images_folder + index)
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (50, 50)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis =0)
            x_.append(im)
            y_.append(target[attr])

    return x_, y_


# In[ ]:


##TESTING FUNCTION

def data(data)
    x_ = []
        y_ = []

        for index, target in df_.iterrows():
            im = cv2.imread(images_folder + index)
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis =0)
            x_.append(im)
            y_.append(target[attr])

    return x_, y_


# In[ ]:


x_train, y_train = generate_df(1, "Attractive", 10000)
x_test, y_test = generate_df(0, "Attractive", 2000)


# In[ ]:


x_train.shape, x_test.shape


# In[ ]:


y_train.shape, y_test.shape


# In[ ]:


first_model = tf.keras.Sequential()
first_model.add(Flatten(input_shape=(50, 50)))
first_model.add(Dense(64, activation='sigmoid'))
first_model.add(Dense(10, ))
first_model.compile(optimizer = 'sgd',
                    loss = 'sparse_categorical_crossentropy',
                    metrics = ['accuracy'])


# In[ ]:


model_history = first_model.fit(x = x_train,
                                y = y_train,
                                batch_size = 128,
                                epochs = 5,
                                validation_split = 0.2,
                                shuffle=True)


# In[ ]:


model = tf.keras.Sequential()
model.add(Dense(64, input_dim=39, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(X_test, y_test, batch_size=128)


# ## CONCLUSIONS
# 
# Convolutional Neural Networks can be helpful when trying to work with images. This models can easily turn computationally expensive.
# 

# ## APPENDIX
# * https://keras.io/getting-started/sequential-model-guide/
# * https://www.tensorflow.org/tutorials/keras/basic_classification
# * https://www.datacamp.com/community/tutorials/deep-learning-python

# 
