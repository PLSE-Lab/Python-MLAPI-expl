#!/usr/bin/env python
# coding: utf-8

# # Transfer Learning in Medical Imaging(X-ray image)
# **_Note: If you like this kernel and/or choose to fork it, please help us out by up-voting the kernel with the <kbd>^</kbd> button above._**
# [Note : The model is badly overfitted,if you can solve this model from high variance problem,then please provide your kernels link in the comment box and it will be highly appreciated]
# 

# ## 1. Loading Python modules
# 

# In[ ]:


# Import the appropriate Keras modules
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet import preprocess_input
from tensorflow.python.keras.applications import ResNet50
import tensorflow as tf

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 2. Taking a look at the data
# I've preloaded the data that we'll be using into this kernel. Let's take a look at the directory structure and contents, then create some variables to help us as we proceed.

# In[ ]:


import os

# Set path variable to the directory where the data is located
path = os.path.join('..', 'input', 'hello-world-deep-learning-siim', 'data')

# Command line "magic" command to show directory contents
get_ipython().system('ls {path}/*/*')


# As you can see, the `data` directory contains subdirectories `train`, `val` and `test`, which contain the *training*, *validation* and *test* data for our experiment. `train` and `val` contain subdirectories `abd` and `chst` containing abdominal and chest radiographs for each data set. There are 65 training images and 10 validation images with *balanced distributions* over our *target classes* (i.e. approximately equal numbers of abdominal and chest radiographs in each data set).

# In[ ]:


# set variables for paths to directories for training & validation data
train_dir = os.path.join(path, 'train')
val_dir = os.path.join(path, 'val')


# we'll need to import additional modules to look at an example image
import numpy as np    # this is a standard convention
from keras.preprocessing import image
import matplotlib.pyplot as plt    # also by convention

# set the path to a chest radiograph, then load it and show
img_path = os.path.join(train_dir, 'chst/chst33.png')
img = image.load_img(img_path, target_size=(229, 229))
plt.imshow(img)
plt.title('Example chest radiograph')
plt.show()

# set the path to an abdominal radiograph, then load it and show
img2_path = os.path.join(train_dir, 'abd/abd1.png')
img2 = image.load_img(img2_path, target_size=(229, 229))
plt.imshow(img2)
plt.title("Example abdominal radiograph")
plt.show()


# ## 3. Setting up our generators & selecting hyperparameters
# A neural network model consists of:
# - An **architecture** defining the structure of the connections between **layers** of **artificial neurons (_units_)** in the network
#     - *Within a layer*, units are NOT connected to each other (with some exceptions, not relevant to this module)
#     - The connections that define the architecture are between neurons in *different layers* of the network
# - The **weights** (*or* parameters) that determine the strength of those connections.
# 
# 

# In[ ]:




image_size = 224 #The default input size for this model is 224x224.
nb_train_samples = 65 # number of files in training set
num_of_test_samples = 10 # number of files in test set
batch_size = 8 #the model will take 8 random batches of files at a time during training

EPOCHS = 50 #we will run this model for 50 epochs(1 epoch = whole dataset traversion during training)
STEPS = nb_train_samples // batch_size #the model will take 326 steps to complete per batch training


# ## 4. Creating the model with transfer learning
# We'll employ a technique called **transfer learning** with the `resnet50` model, using weights obtained by **pretraining** the model on the **ImageNet** data set.

# In[ ]:


resnet_weights_path = '../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
num_classes = 2 
from tensorflow.python.keras.layers import Dense, GlobalMaxPooling2D
model = tf.keras.Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

model.layers[0].trainable = False
# Unfreeze the model backbone before we train a little more
#for layer in model.layers[10:]:
#    layer.trainable = True
'''
for layer in model.layers[:10]:
    layer.trainable=False
for layer in model.layers[10:]:
    layer.add(Dropout(0.7))
    layer.trainable=True
'''
    
from tensorflow.python.keras import optimizers

adamopt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0)

    

    
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(optimizer=adamopt, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# In[ ]:


## Specify the values for all arguments to data_generator_with_aug.
data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                             horizontal_flip = True,
                                             width_shift_range = 0.2,
                                             height_shift_range = 0.2,
                                             shear_range = 0.2,
                                             zoom_range = 0.2
                                            )


# In[ ]:


data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input            
                                          )


# In[ ]:


train_generator = data_generator_with_aug.flow_from_directory(
       directory = train_dir,
       target_size = (image_size, image_size),
       batch_size = batch_size,
       class_mode = 'categorical')

validation_generator = data_generator_no_aug.flow_from_directory(
       directory = val_dir,
       target_size = (image_size, image_size), 
       class_mode = 'categorical')


# In[ ]:


# Early stopping & checkpointing the best model in ../working dir & restoring that as our model for prediction
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = 3)
cb_checkpointer = ModelCheckpoint(filepath = '../working/best.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')


# In[ ]:


#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(
       train_generator, # specify where model gets training data
       epochs = EPOCHS,
       steps_per_epoch=STEPS,
       validation_data=validation_generator,
      callbacks=[cb_checkpointer, cb_early_stopper]
       ) # specify where model gets validation data


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



scores = model.evaluate_generator(validation_generator) 
print("validation Accuracy = ", scores[1])


# In[ ]:


Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['abdominal', 'chest']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))


# In[ ]:


print(history.history.keys())

fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['acc'], 'orange', label='Training accuracy')
ax[0].plot(history.history['val_acc'], 'blue', label='Validation accuracy')
ax[1].plot(history.history['loss'], 'red', label='Training loss')
ax[1].plot(history.history['val_loss'], 'green', label='Validation loss')
ax[0].legend()
ax[1].legend()
plt.show()


# ## Reference : https://www.kaggle.com/wfwiggins203/hello-world-for-deep-learning-siim
