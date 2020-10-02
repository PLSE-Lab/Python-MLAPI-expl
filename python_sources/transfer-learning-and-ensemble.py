#!/usr/bin/env python
# coding: utf-8

# Importing deepstack for powerful ensembles, [more info](https://github.com/jcborges/DeepStack)

# In[ ]:


get_ipython().system('pip install deepstack')


# **Importing essential pacakages**

# In[ ]:


import pandas as pd
import numpy as np 
import os
import tensorflow as tf
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import PReLU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau


# Specifying global parameters

# In[ ]:


img_h,img_w= (300,300)
batch_size=128
epochs=10
n_class=48


# *Concatenating train and test directory paths..*

# In[ ]:


base_dir = '../input/devnagri-handwritten-character/DEVNAGARI_NEW'
train_dir = os.path.join(base_dir, 'TRAIN')
validation_dir = os.path.join(base_dir, 'TEST')


# I'm going to use two D-CNN models namely VGG-19 and Inception V3.If you are new to transfer learning , plaese refer [this](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a).
# 
# Quick Info:- 
# 
# **Transfer learning:-** Transfer learning is the improvement of learning in a new task through the transfer of knowledge from a related task that has already been learned.
# 
# # Vgg-19
# VGG is a Convolutional Neural Network (CNN) architecture that secured first and second positions in the localisation and classification tasks respectively in ImageNet challenge 2014.All the CNNs have more or less similar architecture, stack of convolution and pooling layers at start and ending with fully connected and soft-max layers.
# The VGG architecture is also similar and is clearly explained in [this paper](https://arxiv.org/pdf/1409.1556.pdf).
# 
# The main contribution of VGG is to show that classification/localisation accuracy can be improved by increasing the depth of CNN inspite of using small receptive fields in the layers. (especially earlier layers).
# 
# Neural networks prior to VGG used bigger receptive fields ( 7*7 and 11*11) as compared to 3*3 in VGG, but they were not as deep as VGG. There are few variants of VGG, the deepest one is with 19 weight layers.
# 
# # Inception-V3
# 
# Using multiple features from multiple filters improve the performance of the network. Other than that, there is another fact that makes the inception architecture better than others. All the architectures prior to inception, performed convolution on the spatial and channel wise domain together. By performing the 1x1 convolution, the inception block is doing cross-channel correlations, ignoring the spatial dimensions. This is followed by cross-spatial and cross-channel correlations via the 3x3 and 5x5 filters.
# For more details refer [this paper](http://https://arxiv.org/pdf/1512.00567.pdf).
# 

# Importing and initializing VGG-19

# In[ ]:


from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

base_model_1=VGG19(include_top=False, weights='imagenet',input_shape=(img_h,img_w,3), pooling='avg')

# Making last layers trainable, because our dataset is much diiferent from the imagenet dataset 
for layer in base_model_1.layers[:-6]:
    layer.trainable=False
    
model_1=Sequential()
model_1.add(base_model_1)

model_1.add(Flatten())
model_1.add(BatchNormalization())
model_1.add(Dropout(0.35))
model_1.add(Dense(n_class,activation='softmax'))
            
model_1.summary()


# Plotting Vgg-19 model

# In[ ]:


tf.keras.utils.plot_model(
    model_1,
    to_file="model_1.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=100,
)


# Importing and initializing Inception_V3

# In[ ]:


from tensorflow.keras.applications.inception_v3 import InceptionV3

base_model_2= InceptionV3(include_top=False, weights='imagenet',
                                        input_tensor=None, input_shape=(img_h,img_w,3), pooling='avg')

for layer in base_model_2.layers[:-30]:
    layer.trainable=False
model_2=Sequential()
model_2.add(base_model_2)
model_2.add(Flatten())
model_2.add(BatchNormalization())
model_2.add(Dense(1024,activation='relu'))
model_2.add(BatchNormalization())

model_2.add(Dense(512,activation='relu'))
model_2.add(Dropout(0.35))
model_2.add(BatchNormalization())

model_2.add(Dense(256,activation='relu'))
model_2.add(Dropout(0.35))
model_2.add(BatchNormalization())

model_2.add(Dense(n_class,activation='softmax'))

model_2.summary()


# Plotting Inception_V3 model

# In[ ]:


tf.keras.utils.plot_model(
    model_2,
    to_file="model_2.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=100,
)


# Initializing train and test datagenerators

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
         rescale=1./255,
         rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

test_datagen= ImageDataGenerator(rescale=1./255)


# Creating callbacks and optimizers. You may try out diiferent optimizers

# In[ ]:


from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=3,
                                         cooldown=2,
                                         min_lr=1e-10,
                                         verbose=1)

callbacks = [reduce_learning_rate]
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# Compiling Models

# In[ ]:


model_1.compile( loss='categorical_crossentropy',optimizer= optimizer, metrics=['accuracy'])
model_2.compile( loss='categorical_crossentropy',optimizer= optimizer, metrics=['accuracy'])


# In[ ]:


train_generator = train_datagen.flow_from_directory(
                    train_dir,                   # This is the source directory for training images
                    target_size=(img_h, img_w),  # All images will be resized to 300x300
                    batch_size=batch_size,
                    class_mode='categorical')


from tensorflow.keras.preprocessing.image import ImageDataGenerator
validation_generator = test_datagen.flow_from_directory(
                        validation_dir,
                        target_size=(img_h, img_w),
                        batch_size=batch_size,
                        class_mode='categorical')


# Now lets fit the models on our dataset

# In[ ]:


history_1 = model_1.fit(
      train_generator,
      steps_per_epoch=6528//batch_size, 
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps=3312//batch_size,  
      callbacks=callbacks,
      verbose=1)


# In[ ]:


history_2 = model_2.fit(
      train_generator,
      steps_per_epoch=6528//batch_size, 
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps=3312//batch_size,  
      callbacks=callbacks,
      verbose=1)


# Now Let us plot the curves for train and val losses.

# For VGG-19 model

# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history_1.history['accuracy'])
plt.plot(history_1.history['val_accuracy'])
plt.title('VGG-19 model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_1.history['loss'])
plt.plot(history_1.history['val_loss'])
plt.title('VGG-19 model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# For Inception V3 model

# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history_2.history['accuracy'])
plt.plot(history_2.history['val_accuracy'])
plt.title('Inception V3 model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_2.history['loss'])
plt.plot(history_2.history['val_loss'])
plt.title('Inception V3 model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# You can see the overall achieved accuracy after training for very less time.
# Now let us use two major kind of ensembles namely
# * Weighted Average ensemble
# * Stacking ensemble with meta learners
# 
# We will be using Deepstack Library for implementing these.

# You can add as many members as you want for ensemble.

# In[ ]:


from deepstack.base import KerasMember

member1 = KerasMember(name="model1", keras_model=model_1, train_batches=train_generator, val_batches=validation_generator)
member2 = KerasMember(name="model2", keras_model=model_2, train_batches=train_generator, val_batches=validation_generator)


# Now let's do weighted average ensemble.
# # **Weighted Average Ensemble:**
# This method weights the contribution of each ensemble member based on their performance on a hold-out validation dataset. Models with better contribution receive a higher weight.
# ![](https://miro.medium.com/max/1384/1*5CnIeN_BtByepM_4JWrdvQ.png)

# Constructor of a Dirichlet Weighted Ensemble
# Args:
# *     N: the number of times weights should be (randomly) tried out,
#           sampled from a dirichlet distribution
# *     metric: (optional) evaluation metric function.
#          Default: `sklearn.metrics.roc_auc_score`
# *     maximize: if evaluation metric should be maximized (otherwise minimized)         

# In[ ]:


from deepstack.ensemble import DirichletEnsemble
from sklearn.metrics import accuracy_score

wAvgEnsemble = DirichletEnsemble(N=10000, metric=accuracy_score)
wAvgEnsemble.add_members([member1, member2])
wAvgEnsemble.fit()
wAvgEnsemble.describe()


# # Stacking
# 
# In Stacking there are two types of learners called Base Learners and a Meta Learner.Base Learners and Meta Learners are the normal machine learning algorithms like Random Forests, SVM, Perceptron etc.Base Learners try to fit the normal data sets where as Meta learner fit on the predictions of the base Learner.
# ![](https://miro.medium.com/max/1166/0*L-yEiG9ONP-AWJ82.png)
# 
# Stacking Technique involves the following Steps:-
# 1. Split the training data into 2 disjoint sets
# 2. Train several Base Learners on the first part
# 3. Test the Base Learners on the second part and make predictions
# 4. Using the predictions from (3) as inputs,the correct responses from the output,train the higher level learner.
# 
# Meta Learner is kind of trying to find the optimal combination of base learners.
# 
# ![](https://miro.medium.com/max/1400/1*1ArQEf8OFkxVOckdWi7mSA.png)
# 
# 

# In[ ]:


from deepstack.ensemble import StackEnsemble
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

#Ensure you have the scikit-learn version >= 0.22 installed
print("sklearn version must be >= 0.22. You have:", sklearn.__version__)

stack = StackEnsemble()

# 2nd Level Meta-Learner
estimators = [
    ('rf', RandomForestClassifier(verbose=0, n_estimators=100, max_depth=15, n_jobs=20, min_samples_split=30)),
    ('etr', ExtraTreesClassifier(verbose=0, n_estimators=100, max_depth=10, n_jobs=20, min_samples_split=20)),
    ('dtc',DecisionTreeClassifier(random_state=0, max_depth=3))
]
# 3rd Level Meta-Learner
clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)

stack.model = clf
stack.add_members([member1, member2])
stack.fit()
stack.describe(metric=sklearn.metrics.accuracy_score)


# Now lets save our stack-ensemble model.

# Saves meta-learner and base-learner of ensemble into folder / directory.
# 
# **Args:**
# * folder: the folder where models should be saved to(Create if not exists).           

# In[ ]:


stack.save()


# Loading the model is as simple as this.

# In[ ]:


stack.load()


# Don't forget to save the DCNN itself, otherwise you may have trouble.

# In[ ]:


model_json = model_1.to_json()
with open("VGG_19.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_1.save_weights("VGG_19_weights.h5")
print("Saved VGG19 to disk")


# In[ ]:


model_json = model_2.to_json()
with open("Inception_V3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_2.save_weights("Inception_V3_weights.h5")
print("Saved Inception_V3 to disk")


# **I encourage all of you to play around with this kernel, change the hyperparameters and meta learners. This ensembling is going to help you a lot in different kaggle competitions.And please give me an upvote, it motivates me to work more. :)**
