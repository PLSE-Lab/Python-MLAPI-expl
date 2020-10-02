#!/usr/bin/env python
# coding: utf-8

# <center><font size="5" color="red">Many thanks for feedback and upvote^______^.</font></center>

# <a class="anchor" id="0."></a>
# **Contents**
# * [1. Summary](#1.)
# * * [1.1. Helper Function:show_model_history(modelHistory, model_name)](#1.1.)
# * * [1.2. Helper Function:evaluate_conv_model(model, model_name, X, y)](#1.2.)
# * * [1.3. Helper Function:show_image_classes(image, label, n=10)](#1.3.)
# * [2. Naive Model](#2.)
# * [3. Convolutional Model 1](#3.)
# * [4. Convolutional Model 2](#4.)
# * [5. Convolutional Model 3](#5.)
# * [6. Convolutional Model 4](#6.)
# * [7. Convolutional Model 5](#7.)
# * [8. Convolutional Model 6](#8.)
# * [9. Convolutional Model 7](#9.)
# * [10. Convolutional Model 8](#10.)
# * [11. Play with Optimizers](#11.)
# * [12. Saving and Loading Deep Learning Model with Serialization](#12.)
# * * [12.1. HDF5: Hierarchal Data Format](#12.1.)
# * * [12.2. JSON(JavaScript Object Notation) Format](#12.2.)
# * * [12.3. YAML(YAML Ain't Markup Language) Format](#12.3.)
# * * [12.4. Serialization classes](#12.4.)
# * * [12.5. Saving and Loading Model](#12.5.)
# * * [15.7. Test JSON Serialization](#12.6.)
# * * [15.7. Test YAML Serialization](#12.7.)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras import layers
from keras import optimizers

from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))


# [Go To Content Menu](#0.)
# 
# <a class="anchor" id="1."></a>
# # **1. Summary**
# 
# In this kernel, implementing and performance of Convolutional Neural Network will be demonstrated on Sign Language dataset. CNN networks basically consist of three main parts; Conv, Pooling and Dense layers. Conv layers consist of filters and feature maps. The Pooling layer reduces the feature obtained in the previous Conv layer. The Dense layer is the normal feedforward network layer.
# 
# The summary of the study is explained in the following paragraphs.
# 
# Auxiliary functions are defined to reduce code redundancy and confusion in the study. These functions are defined as a function of the same process used in many parts of the study and then  called when it needed.
# 
# In the **first** CNN model, a CNN containing a single Conv layer. The CNN structure is as follows;
# 
# Conv ==> MaxPooling ==> Dense (relu) ==> Dense (softmax).
# 
# Obtained train and validation accuracy rates are low and there are problems of overfitting(excessive adaptation) and high variance.
# 
# In the **second** CNN model, two Conv layers were used. The network structure is as follows:
# 
# Conv ==> MaxPooling ==> Conv ==> MaxPooling ==> Dense (relu) ==> Dense (softmax).
# 
# Despite the increased success rates of train and validation of the model, there is still excessive adaption and high variance prolemia.
# 
# In the **third** CNN model, three Conv layers were used. The network structure is as follows:
# 
# Conv ==> MaxPooling ==> Conv ==> MaxPooling ==> Conv ==> MaxPooling ==> Dense (relu) ==> Dense (softmax).
# 
# Since there is little improvement compared to the previous model, a Dropout layer was added to the next model instead of adding a Conv layer.
# 
# In the **fourth** CNN model, three Conv layers and Dropout layers were used. The network structure is as follows:
# 
# Conv ==> MaxPooling ==> Dropout ==> Conv ==> MaxPooling ==> Dropout ==> Conv ==> MaxPooling ==> Dropout ==> Dense (relu) ==> Dense (softmax).
# 
# Although the model has reduced overfitting,high variance still exists.
# 
# The **fifth** CNN model has one Conv ==> MaxPooling ==> Dropout block than the fourth CNN model.
# 
# Conv ==> MaxPooling ==> Dropout ==> Conv ==> MaxPooling ==> Dropout ==> Conv ==> MaxPooling ==> Dropout ==> Conv ==> MaxPooling ==> Dropout ==> Dense ( relu) = Dense (softmax).
# 
# Overfitting and high variance were solved in the model, but the train and validation performance of the model decreased.
# 
# In the **sixth** CNN model, the Dropout layer is used between two Dense layers. The network structure is as follows:
# 
# Conv ==> MaxPooling ==> Conv ==> MaxPooling ==> Conv ==> MaxPooling ==> Conv ==> MaxPooling ==> Dense (relu) ==> Dropout ==> Dense (softmax).
# 
# The success rate of the model increased without excessive fit of the model and high variance.
# 
# In the **seventh** model, BatchNormalization layer is used to improve performance. The network structure is as follows:
# 
# Conv ==> MaxPooling ==> BatchNormalization ==> Conv ==> MaxPooling ==> BatchNormalization ==> Conv ==> MaxPooling ==> BatchNormalization ==> Conv ==> MaxPooling ==> BatchNormalization ==> Dense ( relu) ==> Dropout ==> Dense (softmax).
# 
# The model performance increased but there was an excessive adaptation problem.
# 
# In the **eighth** model, Dropout layer is used after Conv layers again. The network structure is as follows:
# 
# Conv ==> MaxPooling ==> BatchNormalization ==> Dropout ==> Conv ==> MaxPooling ==> BatchNormalization ==> Dropout ==> Conv ==> MaxPooling ==> BatchNormalization ==> Dropout ==> Conv = => MaxPooling ==> BatchNormalization ==> Dropout ==> Dense (relu) ==> Dropout ==> Dense (softmax).
# 
# Overfitting has been removed and performance has been improved to a little bit.
# 
# Finally, other parameters of the CNN model have been adjusted to improve model performance.
# 

# In[ ]:


X=np.load("../input/sign-language-digits-dataset/Sign-language-digits-dataset/X.npy")
y=np.load("../input/sign-language-digits-dataset/Sign-language-digits-dataset/Y.npy")
print("The dataset loaded...")


# [Go To Content Menu](#0.)
# 
# <a class="anchor" id="1.1."></a>
# **1.1. Helper Function:show_model_history(modelHistory, model_name)**

# In[ ]:


def show_model_history(modelHistory, model_name):
    history=pd.DataFrame()
    history["Train Loss"]=modelHistory.history['loss']
    history["Validation Loss"]=modelHistory.history['val_loss']
    history["Train Accuracy"]=modelHistory.history['accuracy']
    history["Validation Accuracy"]=modelHistory.history['val_accuracy']
    
    fig, axarr=plt.subplots(nrows=2, ncols=1 ,figsize=(12,8))
    axarr[0].set_title("History of Loss in Train and Validation Datasets")
    history[["Train Loss", "Validation Loss"]].plot(ax=axarr[0])
    axarr[1].set_title("History of Accuracy in Train and Validation Datasets")
    history[["Train Accuracy", "Validation Accuracy"]].plot(ax=axarr[1]) 
    plt.suptitle(" Convulutional Model {} Loss and Accuracy in Train and Validation Datasets".format(model_name))
    plt.show()


# [Go To Content Menu](#0.)
# 
# <a class="anchor" id="1.2."></a>
# **1.2. Helper Function:evaluate_conv_model(model, model_name, X, y)**

# In[ ]:


from keras.callbacks import EarlyStopping
def split_dataset(X, y, test_size=0.3, random_state=42):
    X_conv=X.reshape(X.shape[0], X.shape[1], X.shape[2],1)
    
    

    return train_test_split(X_conv,y, stratify=y,test_size=test_size,random_state=random_state)

def evaluate_conv_model(model, model_name, X, y, epochs=100,
                        optimizer=optimizers.RMSprop(lr=0.0001), callbacks=None):
    print("[INFO]:Convolutional Model {} created...".format(model_name))
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    print("[INFO]:Convolutional Model {} compiled...".format(model_name))
    
    print("[INFO]:Convolutional Model {} training....".format(model_name))
    earlyStopping = EarlyStopping(monitor = 'val_loss', patience=20, verbose = 1) 
    if callbacks is None:
        callbacks = [earlyStopping]
    modelHistory=model.fit(X_train, y_train, 
             validation_data=(X_test, y_test),
             callbacks=callbacks,
             epochs=epochs,
             verbose=0)
    print("[INFO]:Convolutional Model {} trained....".format(model_name))

    test_scores=model.evaluate(X_test, y_test, verbose=0)
    train_scores=model.evaluate(X_train, y_train, verbose=0)
    print("[INFO]:Train Accuracy:{:.3f}".format(train_scores[1]))
    print("[INFO]:Validation Accuracy:{:.3f}".format(test_scores[1]))
    
    show_model_history(modelHistory=modelHistory, model_name=model_name)
    return model


# [Go To Content Menu](#0.)
# 
# <a class="anchor" id="1.3."></a>
# **1.3. Helper Function:show_image_classes(image, label, n=10)**

# In[ ]:


def decode_OneHotEncoding(label):
    label_new=list()
    for target in label:
        label_new.append(np.argmax(target))
    label=np.array(label_new)
    
    return label
def correct_mismatches(label):
    label_map={0:9,1:0, 2:7, 3:6, 4:1, 5:8, 6:4, 7:3, 8:2, 9:5}
    label_new=list()
    for s in label:
        label_new.append(label_map[s])
    label_new=np.array(label_new)
    
    return label_new
    
def show_image_classes(image, label, n=10):
    label=decode_OneHotEncoding(label)
    label=correct_mismatches(label)
    fig, axarr=plt.subplots(nrows=n, ncols=n, figsize=(18, 18))
    axarr=axarr.flatten()
    plt_id=0
    start_index=0
    for sign in range(10):
        sign_indexes=np.where(label==sign)[0]
        for i in range(n):

            image_index=sign_indexes[i]
            axarr[plt_id].imshow(image[image_index], cmap='gray')
            axarr[plt_id].set_xticks([])
            axarr[plt_id].set_yticks([])
            axarr[plt_id].set_title("Sign :{}".format(sign))
            plt_id=plt_id+1
    plt.suptitle("{} Sample for Each Classes".format(n))
    plt.show()


# [Go To Content Menu](#0.)
# 
# <a class="anchor" id="2."></a>
# # **2. About the Dataset**

# In[ ]:


number_of_pixels=X.shape[1]*X.shape[2]
number_of_classes=y.shape[1]
print(20*"*", "SUMMARY of the DATASET",20*"*")
print("an image size:{}x{}".format(X.shape[1], X.shape[2]))
print("number of pixels:",number_of_pixels)
print("number of classes:",number_of_classes)

y_decoded=decode_OneHotEncoding(y.copy())
sample_per_class=np.unique(y_decoded, return_counts=True)
print("Number of Samples:{}".format(X.shape[0]))
for sign, number_of_sample in zip(sample_per_class[0], sample_per_class[1]):
    print("  {} sign has {} samples.".format(sign, number_of_sample))
print(65*"*")


# The dataset consists of images with one-handed display of digits 0 to 9 in sign language. The images are 64X64 in size and gray in color. It was obtained by 218 people making 10 different signs once. There should be a total of 2180 samples, while there are 2062 samples in the data set. This is probably because some unfavorable images have been removed by creator of the dataset..

# In[ ]:


show_image_classes(image=X, label=y.copy())


# [Go To Content Menu](#0.)
# 
# <a class="anchor" id="3."></a>
# # **3. Convolutional Model 1**

# In[ ]:


def build_conv_model_1():
    model=Sequential()
    
    model.add(layers.Conv2D(64, kernel_size=(3,3),
                           padding="same",
                           activation="relu", 
                           input_shape=(64, 64,1)))
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(number_of_classes, activation="softmax"))
        
    return model


# In[ ]:


trained_models=dict()
model=build_conv_model_1()
trained_model_1=evaluate_conv_model(model=model, model_name=1, X=X, y=y)

#Will be used for serialization
trained_models["model_1"]=(trained_model_1,optimizers.RMSprop(lr=0.0001) )


# When the above graphs are examined, it can be seen that the model has a low training accuracy rate and a lower validation accuracy rate. This means that  high bias and high varience, which is too bad for machine learning model. In addition, the zigzags in the validation graph show that the robustness of validation results is very low.
# 
# Considering the above evaluations, it would be useful to add a new Convolution layer to the model. 

# [Go To Content Menu](#0.)
# 
# <a class="anchor" id="4."></a>
# # **4. Convolutional Model 2**

# In[ ]:


def build_conv_model_2():
    model = Sequential()
    model.add(layers.Convolution2D(64, (3, 3), activation='relu', padding="same", input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
       
    model.add(layers.Convolution2D(64, (3, 3), activation='relu', padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
        
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
      
    return model


# In[ ]:


model=build_conv_model_2()
trained_model_2=evaluate_conv_model(model=model, model_name=2, X=X, y=y)

#Will be used for serialization
trained_models["model_2"]=(trained_model_2,optimizers.RMSprop(lr=0.0001) )


# When the above graphs are examined, it can be seen that the model has a high training accuracy rate and a lower validation accuracy rate. This means that low bias and high varience. In addition, although the zigzags in the validation chart are reduced, they still exist. It can be assessed that the robustness of validation results is still low.
# 
# In view of the above considerations, it is useful to add a new Conv layer or Dropout layer to avoid overfitted the model. First let's add a new Conv layer.

# [Go To Content Menu](#0.)
# 
# <a class="anchor" id="5."></a>
# # **5. Convolutional Model 3**

# In[ ]:


def build_conv_model_3():
    model = Sequential()
    model.add(layers.Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
           
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
        
    return model


# In[ ]:


model=build_conv_model_3()
trained_model_3=evaluate_conv_model(model=model, model_name=3, X=X, y=y)
#Will be used for serialization
trained_models["model_3"]=(trained_model_3,optimizers.RMSprop(lr=0.0001) )


# Although the validation accuracy rate has increased, the problem of overfitting of the model still exists. We can assume that adding a new Conv layer is not useful. In addition, although the zigzags in the validation chart are reduced, they still exist. It can be assessed that the robusness of validation results is still low.
# 
# Let's try using the Dropout layer, one of the solutions to the problem of overfitting in deep networks.

# [Go To Content Menu](#0.)
# 
# <a class="anchor" id="6."></a>
# # **6. Convolutional Model 4**

# In[ ]:


def build_conv_model_4():
    model = Sequential()
    model.add(layers.Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
       
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model


# In[ ]:


model=build_conv_model_4()
trained_model_4=evaluate_conv_model(model=model, model_name=4, X=X, y=y)

#Will be used for serialization
trained_models["model_4"]=(trained_model_4,optimizers.RMSprop(lr=0.0001) )


# Although the validation success accuracy has increased, the problem of overfitting(high variance) of the model still exists. 
# 
# Let's try adding a new Conv ==> MaxPool ==> Dropout layer.

# [Go To Content Menu](#0.)
# 
# <a class="anchor" id="7."></a>
# # **7. Convolutional Model 5**

# In[ ]:


def build_conv_model_5():
    model = Sequential()
    model.add(layers.Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
       
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
        
    return model


# In[ ]:


model=build_conv_model_5()
trained_model_5=evaluate_conv_model(model=model, model_name=5, X=X, y=y)
#Will be used for serialization
trained_models["model_5"]=(trained_model_5,optimizers.RMSprop(lr=0.0001) )


# Overfitting  and low robustness problems were resolved, but the training and validation performance of the model was very poor. Let's remove the last Conv ==> MaxPool ==> Dropout layer added to Model 4 and try different things.
# 
# We can fine tunne another parameters to improve model performance. It is better to use Dropout layers between full connected layers and perhaps after pooling layers. We can also increase the number of nodes 128 to 256 in full connected layers.

# [Go To Content Menu](#0.)
# 
# <a class="anchor" id="8."></a>
# # **8. Convolutional Model 6**

# In[ ]:


def build_conv_model_6():
    model = Sequential()
    model.add(layers.Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
        
    return model


# In[ ]:


model=build_conv_model_6()
trained_model_6=evaluate_conv_model(model=model, model_name=6, X=X, y=y)
#Will be used for serialization
trained_models["model_6"]=(trained_model_6,optimizers.RMSprop(lr=0.0001) )


# Beside,overfitting  and low robustness problems were resolved, the training and validation performance of the model lifted up.
# 
# We can also fine tunne the number of filters in Conv layers. Filters are the feature detectors. Generally fewer filters are used at the input layer and increasingly more filters used at deeper layers.
# 
# Filter size is another parameter we can fine tunne it. The filter size should be as small as possible, but large enough to see features in the input data. It is common to use 3x3 on small images and 5x5 or 7x7 and more on larger image sizes.
# 
# BatchNormalization is another layer can be used in CNN. Although the BatchNormalization layer prolongs the training time of deep networks, it has a positive effect on the results. Let's add the BatchNormalization layer to Model 4 and see the results.
# 

# [Go To Content Menu](#0.)
# 
# <a class="anchor" id="9."></a>
# # **9. Convolutional Model 7**

# In[ ]:


def build_conv_model_7():
    model = Sequential()
    model.add(layers.Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
        
    return model


# In[ ]:


model=build_conv_model_7()
trained_model_7=evaluate_conv_model(model=model, model_name=7, X=X, y=y)
#Will be used for serialization
trained_models["model_7"]=(trained_model_7,optimizers.RMSprop(lr=0.0001) )


# As we expect BatchNormalization increase the model performans. But there is overfitting problem in the model. To deal with that we will use Dropout layer in Conv blocks. 

# [Go To Content Menu](#0.)
# 
# <a class="anchor" id="10."></a>
# # **10. Convolutional Model 8**

# In[ ]:


def build_conv_model_8():
    model = Sequential()
    model.add(layers.Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
        
    return model


# In[ ]:


model=build_conv_model_8()
trained_model_8_1=evaluate_conv_model(model=model, model_name=8, X=X, y=y)
#Will be used for serialization
trained_models["model_8_1"]=(trained_model_8_1,optimizers.RMSprop(lr=0.0001) )


# [Go To Content Menu](#0.)
# 
# <a class="anchor" id="11."></a>
# # **11. Playing with Optimizers**

# In[ ]:


model=build_conv_model_8()
optimizer=optimizers.RMSprop(lr=1e-4)# our default optimizer in evaluate_conv_model function
trained_model_8_2=evaluate_conv_model(model=model, model_name=8, X=X, y=y,optimizer=optimizer, epochs=200)

#Will be used for serialization
trained_models["model_8_2"]=(trained_model_8_2,optimizer )


# In[ ]:


model=build_conv_model_8()
optimizer=optimizers.Adam(lr=0.001)
trained_model_8_3=evaluate_conv_model(model=model, model_name=8, X=X, y=y, optimizer=optimizer, epochs=250)
#Will be used for serialization
trained_models["model_8_3"]=(trained_model_8_3,optimizer )


# In[ ]:


model=build_conv_model_8()
optimizer_8_4=optimizers.Adam(lr=0.001)
trained_model_8_4=evaluate_conv_model(model=model, model_name=8, X=X, y=y, optimizer=optimizer_8_4, epochs=300)
#Will be used for serialization
trained_models["model_8_4"]=(trained_model_8_4,optimizer )


# [Go to Content Menu](#0.)
# 
# <a class="anchor" id="12."></a>
# # 12. Saving and Loading Deep Learning Model with Serialization
# 
# Traingin deep learning model can takes hours or weeks. It is important to know how to save trained model to disk and to load saved model from disk. In this section we will see two different types of serialization for saving and loading of Keras models; JSON and YAML.
# 
# Before serizalization, we will see HDF5 format which is used save and load weihgt of trained model. There are two stage to save and load trained model. First, weights of the model saved and loaded to disk. Then, trained model saved and loaded depend on saved and loaded weights. 
# 
# <a class="achor" id="12.1."></a>
# **12.1. HDF5: Hierarchal Data Format**
# 
# HDF5 allows flexible and easy data storage format to save and load large the weights of trained deep learning model. 
# 
# HDF5 is not only for storing weights of networks, but it is optimized way of storing extracted features and images. It is necessary to use for traing on large dataset like ImageNet.
# 
# Keras model has save_weights() function for saving weights to disk and load_weihgts() function for loading from disk. 
# 
# <a class="anchor" id="12.2."></a>
# **12.2. JSON(JavaScript Object Notation) Format**
# 
# JSON is a simple data-interchange file format, which is easy for human to read and write. It also is for machines to parse and generate.   
# 
# JSON built on  structures of, a collection of name/value pairs and an ordered list of values. These are universel data structures which are supporte by all modern programming languages. It makes sense that JSON is interchangeable with programming languages. 
# 
# Keras provides two handy function for serialization with JSON; to_json(), model_from_json().
# * to_json(): describe any model in JSON format.
# * model_from_json(): allows to load JSON file. 
# 
# 
# <a class="achor" id="12.3."></a>
# **12.3. YAML(YAML Ain't Markup Language) Format**
# 
# YAML is an other human friendly data serialization standard for all programming languages. 
# 
# Keras provides two handy function for serialization with YAML; to_yaml(), model_from_yaml().
# * to_yaml(): describe any model in YAML format.
# * model_from_yaml(): allows to load YAML file. 
# 
# We will do both JSON and YAML serialization in class implementation. 

# [Go to Content Menu](#0.)
# 
# <a class="anchor" id="12.4."></a>
# **12.4. Serialization Classes**

# In[ ]:


from keras.models import model_from_json, model_from_yaml
class Save:
    @classmethod
    def save(self, model, model_file_name, hdf5_file_name):
        if "json" in model_file_name:
            model_format=model.to_json()
        else:
            model_format=model.to_yaml()
        with open(model_file_name, "w") as file:
            file.write(model_format)
        model.save_weights(hdf5_file_name)
class Load:
    @classmethod
    def load(self, model_file_name, hdf5_file_name):
        format_file=open(model_file_name)
        loaded_file=format_file.read()
        format_file.close()
        if "json" in model_file_name:
            model=model_from_json(loaded_file)
        else:
            model=model_from_yaml(loaded_file)
        model.load_weights(hdf5_file_name)
        
        return model
class YAML:
    def __init__(self):
        self.yaml_file_name=None
        self.hdf5_file_name=None
    
    def save(self, model, model_name):
        self.yaml_file_name=model_name+".yaml"
        self.hdf5_file_name=model_name+"_yaml.hdf5"
        Save.save(model,
                  self.yaml_file_name,
                  self.hdf5_file_name)        
        
        print("YAML model and HDF5 weights saved to disk...")
        print("Model file name:{}".format(self.yaml_file_name))
        print("Weights file name:{}".format(self.hdf5_file_name))
    
    def load(self):
              
        print("YAML model and HDF5 loaded from disk...")
        return  Load.load(self.yaml_file_name, self.hdf5_file_name)
        
class JSON:
    def __init__(self):
        self.json_file_name=None
        self.hdf5_file_name=None
        
    def save(self, model, model_name):
        self.json_file_name=model_name+".json"
        self.hdf5_file_name=model_name+"_json.hdf5"
        Save.save(model,
                  self.json_file_name,
                  self.hdf5_file_name)
        
        print("JSON model and HDF5 weights saved to disk...")
        print("Model file name:{}".format(self.json_file_name))
        print("Weights file name:{}".format(self.hdf5_file_name))
        
    
    def load(self):

        print("JSON model and HDF5 weights loaded from disk...")
        return Load.load(self.json_file_name, self.hdf5_file_name)
        
class Serialization():
    def __init__(self, file_format):
        assert file_format in ["json", "yaml"], "There is no such a serialization format"
        self.file_format=file_format
        if self.file_format=="json":
            self.serialization_type=JSON()
        else:
            self.serialization_type=YAML()
    
    def save(self, model, model_name="model"):
        self.serialization_type.save(model, model_name)
    def load(self):
        return self.serialization_type.load()


# [Go to Content Menu](#0.)
# 
# <a class="anchor" id="12.5."></a>
# **15.5. Saving and Loading Model**

# In[ ]:


def test_serialization(trained_models, format_type):
    for model_name, model_pack in trained_models.items():
        model, optimizer=model_pack
        serialization=Serialization(format_type)
        serialization.save(model, model_name=model_name)

        loaded_model=serialization.load()
        X_train, X_test, y_train, y_test=split_dataset(X, y)


        #optimizer=optimizers.RMSprop(lr=0.0001)
        loaded_model.compile(loss="categorical_crossentropy", 
                             optimizer=optimizer,
                             metrics=["accuracy"])

        train_scores = loaded_model.evaluate(X_train, y_train, verbose=0)
        test_scores  = loaded_model.evaluate(X_test, y_test, verbose=0)
        print("Train accuracy:{:.3f}".format(train_scores[1]))
        print("Test accuracy:{:.3f}".format(test_scores[1]))
        print()


# [Go to Content Menu](#0.)
# 
# <a class="anchor" id="12.6."></a>
# **15.6. Test JOSN Serialization**

# In[ ]:


test_serialization(trained_models, format_type="json")


# [Go to Content Menu](#0.)
# 
# <a class="anchor" id="12.7."></a>
# **15.7. Test YAML Serialization**

# In[ ]:


import yaml
yaml.warnings({'YAMLLoadWarning': False})
test_serialization(trained_models, format_type="yaml")


# In[ ]:


print("Created files in Kaggle working directory:")
for file in sorted(os.listdir("../working")):
    if "ipynb" in file:
        continue
    print(file)


# Many thanks for your feedbacks and upvotes ^______^.

# In[ ]:




