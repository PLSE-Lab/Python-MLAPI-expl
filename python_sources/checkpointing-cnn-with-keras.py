#!/usr/bin/env python
# coding: utf-8

# # **Goal**
# 
# The goal of this notebook is to illustrate
# * how to create a callback that creates checkpoint on a certain model being trained
# * resume the training process from a checkpoint
# 
# ## **Context**
# The following demonstration is using the **Blood Cell Images** which are to be **categorized by blood cell types** and there are 4 cell types to classify it into. Since the above objectives are the main goal of this notebook, I won't be going into much detail as to the choice of the model or the pre-processing methods employed.

# # **0. Import Libraries and Dependencies**

# In[ ]:


import numpy as np
import os
from os import listdir
import pandas as pd

import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix

import itertools
import seaborn as sns
from tensorflow import keras


from PIL import Image
from tqdm import tqdm

print('Setup completed!')


# In[ ]:


get_ipython().system('pip install -q -U tf-hub-nightly')
get_ipython().system('pip install -q tfds-nightly')
print('Tensorflow Hub requirements successfully installed!')


# In[ ]:


train_path = '../input/blood-cells/dataset2-master/dataset2-master/images/TRAIN/'
test_path = '../input/blood-cells/dataset2-master/dataset2-master/images/TEST/'

print('Paths ready!')


# In[ ]:


num_classes = len(listdir(train_path))
num_classes


# In[ ]:


# Pre-defined functions
def key_extractor(dictionary, value):
    '''
    Input:
    - Dictionary of any key,value pair
    - value to extract
    
    Return:
    - key of that value
    
    Example: dict = {'a':4, 'b':6, 'y':9,'z':3}
    key_extractor(dict, 3) => 'z'
    
    Caveat: Works only if all values are unique!
    '''
    for k,v in dictionary.items():
        if value == v:
            return k


# # **1. Image Preprocessing and Generation**

# ## **1.1. Initial Inspection**

# Check first the size of data contained within each class. If the data is too small, then augmentation might be needed. If data is enough, then proceed as usual.

# In[ ]:


class_dirs = [(train_path + '/' + category) for category in listdir(train_path)]
class_dirs


# In[ ]:


num_imgs_per_class = [len(listdir(class_dir)) for class_dir in class_dirs]


# In[ ]:


plt.figure(figsize=(5,5))
plt.title('Number of Images per Class')
sns.barplot(x=listdir(train_path), y=num_imgs_per_class)
plt.ylabel('Number of images per class')


# Since each class has enough size, then proceed as usual in training.

# ## **1.2. Generate Image Data**

# In[ ]:


from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Tensorflow's Keras has an API that already handles converting RAW images into their array form
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   validation_split=0.2)

print('Ready to generate image data!')


# In[ ]:


image_size = 224
batch_size = 32

train_generator = data_generator.flow_from_directory(train_path,
                                                    target_size=(image_size, image_size),
                                                    class_mode='categorical',
                                                    batch_size=batch_size,
                                                    subset='training')

valid_generator = data_generator.flow_from_directory(train_path,
                                                    target_size=(image_size, image_size),
                                                    class_mode='categorical',
                                                    batch_size=batch_size,
                                                    subset='validation')

# I turned on the shuffle=False for convenience later when I need to extract the associated filename for the
# predicted classes
test_generator = data_generator.flow_from_directory(test_path,
                                                    target_size=(image_size, image_size),
                                                    class_mode='categorical',
                                                    shuffle=False,
                                                    batch_size=batch_size)


# ## **1.3. Visualize 1 Batch of Training Images**

# In[ ]:


for image_batch, label_batch in train_generator:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break


# For the sake of easier visualization, note that ```ImageDataGenerator``` implements **OneHotEncoding** which means that the labels are in numbers. We formulate a way to convert this back to the string labels, for visualization purposes only!

# In[ ]:


train_generator.class_indices


# In[ ]:


plt.figure(figsize=(20,8))
plt.subplots_adjust(hspace=0.5)
show_num_images = train_generator.batch_size
row = 3
col = np.ceil(show_num_images/row)

for i in range(show_num_images):
    plt.subplot(row,col,i+1)
    plt.imshow(image_batch[i])
#     plt.title(label_batch[i])
    plt.title(key_extractor(train_generator.class_indices, np.argmax(label_batch[i])))
    plt.axis('off')
_ = plt.suptitle("One Batch of Training Images (Labeled Accordingly)")


# # **2. Build Model**
# 
# ## **2.1. Model Architecture**
# 
# Several implementations were already attempted prior to this one. However, the accuracy was incredibly low. Hence, a new model built from Tensorflow Hub's classifiers was opted instead. (For the sake of the main objective of this notebook, I won't focus on possible improvements of this technique in particular).

# ### 2.1.1 **Load the Classifier**
#  
# This is similar as reloading a pre-trained model such as ResNet50, MobileNetV2, etc.

# In[ ]:


classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"

IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])

print("TF Hub's classifier successfully loaded!")


# ### **2.1.2. Download the Headless Model**
# 
# To improve the model with respect to the current categorization, we reload the MobileNetV2 without the top prediction layer and retrain that last layer to get the new weights for the prediction layer of 4 classes. In other words, we use MobileNetV2 just as a FEATURE EXTRACTOR and not as the final classifier!

# In[ ]:


feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"

# Create the feature extractor
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))

# Apply the feature_extractor on the first batch of images generated (for trial purposes only)
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)


# ### **2.1.3. Freeze the Feature Extractor Layer**
# 
# Since we are treating the headless MobileNetV2 as feature extractor, we set its layers to not trainable.

# In[ ]:


feature_extractor_layer.trainable = False
print('Feature extraction layer frozen!')


# ### **2.1.4. Attach a Classification Head**
# 
# As reiterated previously, we remove the original head and attach a new layer for our custom classification of the 4 labels.

# In[ ]:


model = tf.keras.Sequential([
  feature_extractor_layer,
  layers.Dense(train_generator.num_classes)
])

model.summary()


# ## **2.2. Compile for Training**
# 
# Since the last Dense layer was just added with no weight, we retrain the model, where the weights for the first feature extractor layers will no longer be trained and only the last layer's weights will be trained. Hence, the very few number of trainable parameters indicated in the summary.

# In[ ]:


# Use compile to configure the training process:
model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['acc'])

print('Model compiled! \nReady for training!')


# ## **2.3. Create Callbacks**
# 
# This is where the juice of this notebook will be. We will use Keras' ```ModelCheckpoint``` to create checkpoints as the training progresses. This will allow us to resume training later if the training process is interrupted.
# 
# **WARNING!**
# 
# In Kaggle, **training neural networks** with very long training time exceeding the kernel limit which is I think 9 hours of straight activity will lead to the training being unfinished! So, if you're prototyping your model on Kaggle's resources and it trains more than this, the model's training parameters such as learned weights and current state won't be saved. When you come back to the kernel, you have to restart all over again!
# 
# **SOLUTION:**
# 
# The only solutions I found possible are
# * implement a ModelCheckpoint and callback during training. **Don't run yet the kernell cell where training happens**. Instead
# 
# > 1) Implement the code on that cell.
# 
# > 2) Click on **Save Version** button on the top-right corner.
# 
# > 3) Check **Save & Run All (Commit)** option so that the kernel/notebook's output will be saved!
# 
# > 4) Run the entire notebook (with an accelerator if needed).
# 
# > 5) Since model training is the last part of your notebook, IF in case the model training gets interrupted, whatever version is committed in the kernel will include the checkpoints created so far!
# 
# **Saving the output is important. With the ModelCheckpoint, the kernel's output would INCLUDE the checkpoints created during training. Since these are saved, you can then create another kernel/notebook and use the PREVIOUS kernel's output as an input, reload the checkpoints in the new notebook, and resume training!**
# 
# **Alternative**
# 
# The above methodology will FORCE you to create ANOTHER notebook with a different name which can make "version control" very messy and seemingly manual. As I was reviewing this notebook, I was able to find a way to **save the output of a kernel and re-use it as input data on THE SAME kernel.** Take note that we "save" the first output the same as described above. Now, **to reload the previous output on the same kernel:**
# > 1) Go to the kernel/notebook you've saved.
# > 2) Check the an output was indeed saved from the previous commit. Outputs such as checkpoints or an 'h5' version of the best-model depending on your implemented methodology.
# > 3) Click on EDIT on the top-right corner of that notebook.
# > 4) On the kernel/notebook, click on ADD DATA.
# > 5) Upload
# > 6) Click on the CODE button </> on the left panel of the upload box.
# > 7) Choose YOUR WORK on the dropdown to be re-directed to the list of notebooks/kernels you've saved.
# > 8) Choose the kernel/notebook (which can be literally the same one you are currently working on right now, a.k.a. previous version of your current notebook). If there were outputs saved, you should be able to see them.
# > 9) Provide a description.
# > 10) Upload!

# In[ ]:


get_ipython().system('pip install h5py')
print('Ready to save models in the h5 format.')


# In[ ]:


# from tf.keras.callbacks import ModelCheckPoint
# filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.hdf5',
                                                monitor='val_loss',
                                                verbose=1,
                                                save_best_only=True,
                                                mode='min',
                                                period=1)
    
print('Callback successfully created!')


# # **3. Train the Model**
# 
# In this part, I demonstrated how to implement a model training with the checkpoint/callback initiated prior.
# 
# ## **3.1. Checkpoint Model Improvements**
# 
# Here, we illustrate how to **track** the model as it improves over training. Essentially, it will save versions/checkpoints of the model everytime it detects an improvement in whichever metric you choose. Here we choose to save the best model based on improvements on validation accuracy.

# In[ ]:


steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size)
valid_steps_per_epoch = np.ceil(valid_generator.samples/valid_generator.batch_size)

history = model.fit(train_generator,
                      steps_per_epoch=steps_per_epoch,
                      validation_data=valid_generator,
                      validation_steps=valid_steps_per_epoch,
                      epochs=3,
                      callbacks = [checkpoint],
                      verbose=2)

print('Model trained successfully!')


# Now, if you refer back to the output where ```/kaggle/working``` directory is defined, you can see a new file **best_model.hdf5** which contains your saved model. Now, we can check if this can be reloaded and continued for training.
