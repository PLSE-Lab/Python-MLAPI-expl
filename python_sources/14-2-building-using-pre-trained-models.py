#!/usr/bin/env python
# coding: utf-8

# # Working with pretrained models
# 
# This notebook outlines how, and why, to work with pretrained models for image classification.
# 
# ## What are pretrained models
# Pretrained models are models that where built and trained by others, usually for a different but related task. This process is called **transfer learning**. Transfer learning is especially popular in computer vision, but is gaining popularity in pretty much all other fields of machine learning, too. 
# 
# The most popular benchmark for computer vision models is called [ImageNet](http://www.image-net.org/), a collection of millions of images showing 1000 different objects. There is an annual challenge to build a model that performs best on classifying ImageNet images and many useful models have come out of it. In this case, we will use the the [Xception model](https://arxiv.org/abs/1610.02357) as a basis from which we will build a cat and dog classifier.
# 
# ## Some preparation:
# We will now copy the pretrained model in the right location to load it with Keras. This is only nessecary since Kaggle Kernels have no internet connection, otherwise Keras download the models automatically.

# In[ ]:


# Check that we have access to the models:
get_ipython().system('ls ../input/keras-pretrained-models')


# In[ ]:


# Create paths for model
import os
cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# In[ ]:


# Copy model over
get_ipython().system('cp ../input/keras-pretrained-models/xception* ~/.keras/models/')


# In[ ]:


# Check that model is in place
get_ipython().system('ls ~/.keras/models')


# ## Preparing the data
# In this section we will copy and sort the data, just as we did when we built our image classifier from scratch. You can find the kernel explaining these steps here: https://www.kaggle.com/jannesklaas/14-building-an-image-classifier

# In[ ]:


# Import matplotlib for plotting
import matplotlib.pyplot as plt
# Import matplotlibs image tool
import matplotlib.image as mpimg
# Flip the switch to get easier matplotlib rendering
get_ipython().run_line_magic('matplotlib', 'inline')
# for file listing
import os
# for file moving
from shutil import copyfile


# In[ ]:


# Create destination directories
if not os.path.exists('train'):
    os.mkdir('train')
if not os.path.exists('train/cat'):
    os.mkdir('train/cat')
if not os.path.exists('train/dog'):  
    os.mkdir('train/dog')
if not os.path.exists('validation'):    
    os.mkdir('validation')
if not os.path.exists('validation/cat'):
    os.mkdir('validation/cat')
if not os.path.exists('validation/dog'):
    os.mkdir('validation/dog')
# define paths
source_path = '../input/dogs-vs-cats-redux-kernels-edition/train/'

cat_train_path = 'train/cat'
dog_train_path = 'train/dog'

cat_validation_path = 'validation/cat'
dog_validation_path = 'validation/dog'
# Loop over image numbering
for i in range(110):
    cat = 'cat.' + str(i) + '.jpg'
    dog = 'dog.' + str(i) + '.jpg'
    # Get source paths
    cat_source = os.path.join(source_path,cat)
    dog_source = os.path.join(source_path,dog)
    # Get destination paths
    if i < 100:
        cat_dest = os.path.join(cat_train_path,cat)
        dog_dest = os.path.join(dog_train_path,dog)
    else: 
        cat_dest = os.path.join(cat_validation_path,cat)
        dog_dest = os.path.join(dog_validation_path,dog)
    # Move file
    copyfile(cat_source,cat_dest)
    copyfile(dog_source,dog_dest)
    print('Copied',(i+1)*2,'out of 220 files',end='\r')


# In[ ]:


# Check that images are in position
img=mpimg.imread('train/cat/cat.1.jpg')
imgplot = plt.imshow(img)
plt.show()


# ## Load the pretrained model
# In this section we will load the pretrained model. The [Keras Applications](https://keras.io/applications/) model provides a range of useful pretrained model, but we will work with Xception in this case.
# 
# Models trained on ImageNet usually have a structure consisting of a 'base' made out of convolutional layers which extract features and a 'top' made out of densely connected layers which do the classification. We want to use the feature extraction capabilities of our model but completely retrain the classification part of the model. Luckily, Keras offers us the option to load just the base of the model. You might want to experiement with retraining different layers though. For example it might be worth trying to retrain the last convolutional layer as well.

# In[ ]:


from keras.applications import Xception
from keras.models import Sequential


# In[ ]:


model = Xception(weights='imagenet', include_top=False)
model.summary()


# As you can see, the model is pretty large and contains some layer types you might not know. But don't worry, you don't have to understand how and why those bigger models work to use them.
# 
# ## Extracting bottleneck features
# We will now extract some bottleneck features. Bottleneck features are the outputs of the last convolutional layer. We do this by running all of our training and validation images through the model, creating a new dataset which we will later use to train our new top.
# 
# To limit runtime, this kernel uses only 200 training and 20 validation images. The generators load images in order, so we know the labels and can create a label vector easily (the first 100 images are cats, the second 100 images are dogs).  To save runtime we also don't do much image augumentation besides rescaling.

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

# Only rescaling for training too this time
datagen = ImageDataGenerator(rescale=1./255)


# In[ ]:


# only rescaling
validation_datagen = ImageDataGenerator(rescale=1./255)


# In[ ]:


# Set up batch size
batch_size = 1
train_generator = datagen.flow_from_directory(
        'train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size, # How many images do we need at a time
        class_mode=None, # Generator will yield data without labels
        shuffle= False) # Generator will read files in order


# In[ ]:


validation_generator = validation_datagen.flow_from_directory(
        'validation',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size, # How many images do we need at a time
        class_mode=None,
        shuffle=False)


# Now we can create the bottleneck features of our training set

# In[ ]:


bottleneck_features_train = model.predict_generator(train_generator, 200, verbose = 1)


# as well as the label vector of our training set.

# In[ ]:


import numpy as np
train_labels = np.array([0] * 100 + [1] * 100)


# We do the same with the validation data:

# In[ ]:


bottleneck_features_validation = model.predict_generator(validation_generator, 20)


# In[ ]:


validation_labels = np.array([0] * 10 + [1] * 10)


# ## Building the new top
# Now we can build the new top of our model. We start with a `Flatten`layer, which reshapes the 3D convolutional outputs into 1D inputs that can be used by densely connected networks. Then we add a dense layer, relu, some dropout and a final classification layer.

# In[ ]:


from keras.layers import Flatten, Dense, Dropout, Activation


# In[ ]:


model = Sequential()
model.add(Flatten(input_shape=bottleneck_features_train.shape[1:]))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[ ]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Then we fit this top model to the bottleneck training data we created:

# In[ ]:


model.fit(bottleneck_features_train, train_labels,
          epochs=10,
          batch_size=32,
          validation_data=(bottleneck_features_validation, validation_labels))


# Et voila! 95% accuracy after training from only 200 images!

# In[ ]:


# Cleanup for kaggle
get_ipython().system('rm -r train')
get_ipython().system('rm -r validation')


# In[ ]:




