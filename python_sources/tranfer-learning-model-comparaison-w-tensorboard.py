#!/usr/bin/env python
# coding: utf-8

# This Kernel was made to compare different models for tranfer learning using Tensorboard visualisation tools, all while using Keras.
# Not really knowing what architecture to go with in our model, we decided to try to compare different models that are known for image classification and compare how they did in this specific problem.
# 
# This competition was the very first we did, and so there might be errors, or incoherent things. If you think something could have been better, in a more efficient manner, or simply if you think something is wrong, please let us know! We did this competition mostly to learn as much as possible and it has been a great learning experience for the past few months! But we still have a lot to learn and even more willing to!
# 
# The logic was to craete a function, where the only input would be the model we wanted to try, and it would run it and then allow to be compared in Tensorboard.

# In[1]:


import numpy as np 
import pandas as pd
from datetime import datetime

from keras.models import Sequential
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense,Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy
from keras import regularizers, optimizers
from keras.optimizers import Adam

from keras.applications import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.nasnet import NASNetMobile


from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau

import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = (20,10)

def append_ext(fn):
    return fn+".tif"


import os
print(os.listdir("../input"))


# First we want to get the test and train dataset. Notice that we will only take 180,000 images, that is because we want to keep a balanced dataset (in the train set there are about 220,000 images, but the labels are not balanced).

# In[ ]:


traindf=pd.read_csv("../input/train_labels.csv",dtype=str)
train_size = 180000
traindf = traindf.sort_values(by=['label','id'])
traindf = traindf.iloc[:int(train_size/2)].append(traindf.iloc[-int(train_size/2):])
testdf=pd.read_csv("../input/sample_submission.csv",dtype=str)
traindf["id"]=traindf["id"].apply(append_ext)
testdf["id"]=testdf["id"].apply(append_ext)
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)


# While creating the datagens we used a batch_size of 512, because we want to go for a relatively rapid testing. In practise it seems like a lower batch_size (around 32 or 64) gives better results. But we simply want to see which model seems the best for this probelm compared to the others, so we prefer going for a code that runs fater.

# In[ ]:


B_size = 512

train_generator=datagen.flow_from_dataframe(
                                            dataframe=traindf,
                                            directory="../input/train/",
                                            x_col="id",
                                            y_col="label",
                                            subset="training",
                                            batch_size=B_size,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="binary",
                                            target_size=(96, 96)
)

valid_generator=datagen.flow_from_dataframe(
                                            dataframe=traindf,
                                            directory="../input/train/",
                                            x_col="id",
                                            y_col="label",
                                            subset="validation",
                                            batch_size=B_size,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="binary",
                                            target_size=(96, 96)
)

test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(
                                                dataframe=testdf,
                                                directory="../input/test/",
                                                x_col="id",
                                                y_col=None,
                                                batch_size=B_size,
                                                seed=42,
                                                shuffle=False,
                                                class_mode=None,
                                                target_size=(96, 96)
)


# In[ ]:


train_generator.n//train_generator.batch_size


# The metric used for this competition was a ROC AUC, so we made a metric to be able to measure this. We seem to have gotten the wrong metric though, because most of the time in our submits, we had a validation AUC very high in our program (sometimes even going as high as 0.99) but when submitting, it would fall down to 0.94. This is something we need to look into for future competitions.

# In[ ]:


def auc(y_true, y_pred):
    """ROC AUC metric evaluator"""
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


# In[ ]:


def make_model(model_choice, model_name, input_tensor):
    '''Function to create a model
    Input:
    - model_choice, for ex: VGG19(include_top=False, input_tensor=input_tensor)
    - model_name, (str), name that will be given to the model in tensorboard
    
    Output:
    - model made with keras.model.Model'''
    
    base_model = model_choice
    x = base_model(input_tensor)
    out = Flatten()(x)
    out = Dense(1, activation="sigmoid")(out)
    model = Model(input_tensor, out)
    
    #The only callback we will use is TensorBoard, we could use early stopping or modifying the learning rate
    #but we wanted to compare the models as they were, with the same parameters for each.
    tensorboard=TensorBoard(log_dir = './logs/{}'.format(model_name),
                            histogram_freq=0,
                            batch_size=B_size,
                            write_graph=True,
                            write_grads=True,
                            write_images=False)
    
    model.compile(optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=['accuracy', auc])
    model.summary()
    
    history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=5,
                    callbacks=[tensorboard])
    

    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title(model_name +  ' Model AUC')
    plt.legend([model_name +  ' Training',model_name +  ' Validation'])
    plt.ylabel('AUC')
    plt.xlabel('epoch')
    
    return model


# In[ ]:


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size


# We decided to test some of the models available with Keras that were known for image classification. In theory, it would have been interesting to test all of the available, as well as a CNN that we made ourselves, but this was our very first competition, so we mostly wanted to try making something that worked in the first place.

# In[ ]:


input_tensor = Input((96, 96, 3))

VGG19_model = make_model(VGG19(include_top=False, weights='imagenet', input_tensor=input_tensor), 'VGG19', input_tensor)


# In[ ]:


VGG16_model = make_model(VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor), 'VGG16', input_tensor)


# In[ ]:


MobileNet_model = make_model(MobileNet(include_top=False, weights='imagenet', input_tensor=input_tensor), 'MobileNet', input_tensor)


# In[ ]:


NASNetMobile_model = make_model(NASNetMobile(include_top=False, weights='imagenet', input_tensor=input_tensor), 'NASNetMobile', input_tensor)


# In[ ]:


InceptionV3_model = make_model(InceptionV3(include_top=False, weights='imagenet', input_tensor=input_tensor), 'InceptionV3', input_tensor)


# The ResNet50 didn't seem to work with input_tensor, so instead of using our function, we just copied it and changed the inputs with input_shape, and it worked great.

# In[ ]:


input_shape = (96, 96, 3)

ResNet50_model=ResNet50(include_top=False, input_tensor=None, weights='imagenet', input_shape = input_shape)
Rx = ResNet50_model.output
Rx = Flatten()(Rx)
prediction = Dense(1, activation="sigmoid")(Rx)
Rmodel = Model(ResNet50_model.input, prediction)

ResNet50_tensorboard = TensorBoard(log_dir = './logs/{}'.format('ResNet50'),
                                            histogram_freq=0,
                                            batch_size=B_size,
                                            write_graph=True,
                                            write_grads=True,
                                            write_images=False)

Rmodel.compile(optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=['accuracy', auc])


# In[ ]:


history = Rmodel.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=5,
                    callbacks=[ResNet50_tensorboard]
)
end = datetime.now()


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('ResNet50 model accuracy')
plt.legend(['ResNet50_training','ResNet50_validation'])
plt.ylabel('accuracy')
plt.xlabel('epoch')


# In[ ]:


plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('ResNet50 model accuracy')
plt.legend(['ResNet50_training','ResNet50_validation'])
plt.ylabel('AUC')
plt.xlabel('epoch')


# Finally, this last part generates a link that leads us to TensorBoard to see how each model performs according to the metric we have given him.
# 
# (though it does not seem to work once the code is commited, only when running the code in the Notebook. If anybody has a solution to this, we would be very interested!)

# In[ ]:


get_ipython().system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
get_ipython().system('unzip ngrok-stable-linux-amd64.zip')
LOG_DIR = './logs'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 8080 &'
    .format(LOG_DIR)
)
get_ipython().system_raw('./ngrok http 8080 &')
get_ipython().system(' curl -s http://localhost:4040/api/tunnels | python3 -c     "import sys, json; print(json.load(sys.stdin)[\'tunnels\'][0][\'public_url\'])"')


# **Conclusion**
# 
# From the information we get from TensorBoard, it seems like the ResNet50 is the best model we could use as we have tested it here. The MobileNet and InceptionV3 also seem like good candidates, though a bit behind. Both VGG seem to be under performing compared to the other models, while the NasNet seems good, but its validation loss doesn't seem to change much, which is a bit concerning.
# 
# Again, we know that our AUC metric is a bit flawed, so these should be taken with a grain of salt. Especially since in our testings, it seems like the VGG19 outperformed the ResNet50 on the public leaderboard sumbit.
# 
# It should also be noted that we could try to tweak some things in the transfer learning, untraining more layers and adding others. The choice of an Adam optimizer over something like a SGD is also a questionnable choice.
# 
# If you have any questions and/or remarks, please let us know! This was a very interesting competition, but we also want to learn as much as possible from it now that it is over! :-)
