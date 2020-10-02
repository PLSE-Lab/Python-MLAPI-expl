#!/usr/bin/env python
# coding: utf-8

# # Tensorflow Keras Tutorial - Using Pretrained Models and Multiclass Classification (Part 4)
# 
# **What is Keras?** Keras is a wrapper that allows you to implement Deep Neural Network without getting into intrinsic details of the Network. It can use Tensorflow or Theano as backend. This tutorial series will cover Keras from beginner to intermediate level.
# 
# 
# <p style="color:red">IF YOU HAVEN'T GONE THROUGH THE PART 1-3 OF THIS TUTORIAL, IT'S RECOMMENDED FOR YOU TO GO THROUGH THAT FIRST.</p>
# [LINK TO PART 1](https://www.kaggle.com/akashkr/tf-keras-tutorial-neural-network-part-1)<br>
# [LINK TO PART 2](https://www.kaggle.com/akashkr/tf-keras-tutorial-cnn-part-2)<br>
# [LINK TO PART 3](https://www.kaggle.com/akashkr/tf-keras-tutorial-binary-classification-part-3)
# 

# ## Importing Libraries

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import os


# ## Download Pre-trained Model

# In[ ]:


get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5     -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


# ## Load Pretrained Model
# InceptionV3 is pretrained model on billons on image classified into thousands of classes. `InceptionV3` returns a skeleton of model and `load_weights` loads pretrained model weights into the skeleton.
# 
# #### InceptionV3
# > * **input_shape** Shape of the input layer
# * **include_top** Whether to include the first dense layer in the model
# * **weights** Weight name to load

# In[ ]:


pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

type(pre_trained_model)


# In[ ]:


# This model layers are long and complex
pre_trained_model.summary()


# ## Customizing the model
# 
# To make our custom model using the pretrained models, there are a few things that we need to take into account.
# 1. Freeze the loaded model weights to ensure that the pretrained weights do not get modified while fitting our dataset
# 2. Choose the appropriate layer for output
# 3. Add few more layers which get trained as custom model
# 4. Use dropouts in order to avoid overfitting in the model

# In[ ]:


# Freezing all layers to avoid modification of weights in pretrained model
for layer in pre_trained_model.layers:
    layer.trainable = False


# In[ ]:


# Extract "mixed7" named layer
# The name of the layer is according to the dimension (7x7) of the layer
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


# In[ ]:


# Making our own custom model

# Flatten the output layer to 1 dimension
x = tf.keras.layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = tf.keras.layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = tf.keras.layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = tf.keras.layers.Dense(6, activation='softmax')(x)           

model = tf.keras.Model(pre_trained_model.input, x) 

model.compile(
    optimizer = RMSprop(lr=0.0001),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)


# ## Defining Dataset Generator

# In[ ]:


training_images_path = '../input/intel-image-classification/seg_train/seg_train'
validation_images_path = '../input/intel-image-classification/seg_test/seg_test'


# In[ ]:


# Defining training image generator with all the augmentation sample parameters
# This ensures correct classification for different image in validation
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Defining validation image generator
# We don't pass augmentation parameters as we model haven't seen this data earlier
validation_generator = ImageDataGenerator(rescale=1/255)

# Loading training data from path
train_generator = train_datagen.flow_from_directory(
    training_images_path,
    target_size=(150, 150),
    batch_size=40,
    class_mode='categorical'
)

# Loading validation data from path
validation_generator = validation_generator.flow_from_directory(
    validation_images_path,
    target_size=(150, 150),
    batch_size=40,
    class_mode='categorical'
)


# ## Training Model

# In[ ]:


history = model.fit(
    train_generator,
    steps_per_epoch=350,
    epochs=4,
    verbose=1,
    validation_data=validation_generator
)


# ## Loss and Accuracy
# Let's see the Loss and Accuracy graph for training and Validation Data

# In[ ]:


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()


# **IN THE NEXT TUTORIAL WE WILL SEE APPLICATION OF NEURAL NETWORK IN NATURAL LANGUAGE PROCESSING.**
# 
# > # PART 5 [Basics of NLP](https://www.kaggle.com/akashkr/tf-keras-tutorial-basics-of-nlp-part-5)
