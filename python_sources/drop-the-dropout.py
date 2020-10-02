#!/usr/bin/env python
# coding: utf-8

# # Drop the Dropout
# In this kernel, we train an (existing) kernel that uses Dropout, and in a second stage we suppress the Dropout and prevent overfitting by training only a thousand parameters. The base kernel we use is the excellent public one intitled "[25 Millions Images! [0.99757] MNIST](https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist/notebook)".
# 
# Besides the code past from there, we create a second model where we remove dropout layers and introduce new layers to multiply activations from previous layers. This is done in keras using depthwise-pointwise convolution. While this method is natural for the first layers when we have 3D tensors with the CNNs, it is a little awkward for the Dense layer , since we need to reshape the 1D tensor into a 3D tensor.
# 
# The Dropout technique is very interesting to prevent overfitting, but it has the drawback that the model used to train is different from the model used to make inference.  Here we first train for 45 epochs 15 neural nets, then train for 20 epochs 15 new neural nets based on the previous ones, with "multiplicative" layers, all the old Convolutional and Dense layers set to not trainable, and without Dropout. Our rational is that Dropout is important/essential when we have millions of parameters, but not so when the model has less than two thousands trainable parameters. Then we use the 30 models built to make the inference.
# 
# Below we indicate "(copy)" when there is no modification from the code this notebook is based.

# # Load Libraries (copy)

# In[ ]:


# LOAD LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, DepthwiseConv2D, Reshape, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import keras


# # Load Kaggle's 42,000 training images (copy)

# In[ ]:


# LOAD THE DATA
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


# PREPARE DATA FOR NEURAL NETWORK
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)
X_train = X_train / 255.0
X_test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)


# # Generate 25 million more images!! (copy)
# by randomly rotating, scaling, and shifting Kaggle's 42,000 images.

# In[ ]:


# Fixing the seed
np.random.seed(82)


# In[ ]:


# CREATE MORE IMAGES VIA DATA AUGMENTATION
datagen = ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)


# # Build 15 Convolutional Neural Networks! (copy)

# In[ ]:


# BUILD CONVOLUTIONAL NEURAL NETWORKS
nets = 8
model = [0] *nets
for j in range(nets):
    model[j] = Sequential()
    model[j].add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))#0
    model[j].add(BatchNormalization())#1
    model[j].add(Conv2D(32, kernel_size = 3, activation='relu'))#2
    model[j].add(BatchNormalization())#3
    model[j].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))#4
    model[j].add(BatchNormalization())#5
    model[j].add(Dropout(0.4))#6
    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))#7
    model[j].add(BatchNormalization())#8
    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))#9
    model[j].add(BatchNormalization())#10
    model[j].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))#11
    model[j].add(BatchNormalization())#12
    model[j].add(Dropout(0.4))#13
    model[j].add(Conv2D(128, kernel_size = 4, activation='relu'))#14
    model[j].add(BatchNormalization())#15
    model[j].add(Flatten())#16
    model[j].add(Dropout(0.4))#17
    model[j].add(Dense(10, activation='softmax'))#18
    # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# # Build 15 Neural Networks without Dropout (new)
# We remove all the Dropout Layers. After each convolutional layer, we add a depthwise-pointwise convolution (which will be initialized to one) that will multiply by learned parameters each previous features. So after a convolution layer with 32 features, the new layer will have 32 parameters. 
# These new layers commute with the ReLu activation, which is not the case for other types of activations. In those cases, we could separate the activations from the layers, and we could place depthwise-pointwise layers before and after the activations.

# In[ ]:


model2 = [0] *nets
for j in range(nets):
    model2[j] = Sequential()
    model2[j].add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1), trainable=False))#0
    model2[j].add(BatchNormalization())#1
    model2[j].add(DepthwiseConv2D((1,1), use_bias=False))#2
    model2[j].add(Conv2D(32, kernel_size = 3, activation='relu', trainable=False))#3
    model2[j].add(BatchNormalization())#4
    model2[j].add(DepthwiseConv2D((1,1), use_bias=False))#5
    model2[j].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu', trainable=False))#6
    model2[j].add(BatchNormalization())#7
    model2[j].add(DepthwiseConv2D((1,1), use_bias=False))#8
    model2[j].add(Conv2D(64, kernel_size = 3, activation='relu', trainable=False))#9
    model2[j].add(BatchNormalization())#10
    model2[j].add(DepthwiseConv2D((1,1), use_bias=False))#11
    model2[j].add(Conv2D(64, kernel_size = 3, activation='relu', trainable=False))#12
    model2[j].add(BatchNormalization())#13
    model2[j].add(DepthwiseConv2D((1,1), use_bias=False))#14
    model2[j].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu', trainable=False))#15
    model2[j].add(BatchNormalization())#16
    model2[j].add(DepthwiseConv2D((1,1), use_bias=False))#17
    model2[j].add(Conv2D(128, kernel_size = 4, activation='relu', trainable=False))#18
    model2[j].add(BatchNormalization())#19
    model2[j].add(DepthwiseConv2D((1,1), use_bias=False))#20
    model2[j].add(Flatten())#21
    model2[j].add(Dense(10, trainable=False))#22
    model2[j].add(Reshape((1,1,10)))#23
    model2[j].add(DepthwiseConv2D((1,1), use_bias=False))#24
    model2[j].add(Flatten())#25
    model2[j].add(Activation("softmax"))#26
    # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
    model2[j].compile(optimizer=keras.optimizers.Adadelta(), loss="categorical_crossentropy", metrics=["accuracy"])


# # A function to init the weights of the new Models 
# The layers similar to the previous ones are initialized with the weights of the previous models, the depthwise-pointwise layers are initalized to one.

# In[ ]:


def initModel2(j):
    model2[j].get_layer(index=0).set_weights(model[j].get_layer(index=0).get_weights())
    model2[j].get_layer(index=1).set_weights(model[j].get_layer(index=1).get_weights())#batchnorm1
    model2[j].get_layer(index=3).set_weights(model[j].get_layer(index=2).get_weights())
    model2[j].get_layer(index=4).set_weights(model[j].get_layer(index=3).get_weights())#batchnorm4
    model2[j].get_layer(index=6).set_weights(model[j].get_layer(index=4).get_weights())
    model2[j].get_layer(index=7).set_weights(model[j].get_layer(index=5).get_weights())#batchnorm7
    model2[j].get_layer(index=9).set_weights(model[j].get_layer(index=7).get_weights())
    model2[j].get_layer(index=10).set_weights(model[j].get_layer(index=8).get_weights())#batchnorm10
    model2[j].get_layer(index=12).set_weights(model[j].get_layer(index=9).get_weights())
    model2[j].get_layer(index=13).set_weights(model[j].get_layer(index=10).get_weights())#batchnorm13
    model2[j].get_layer(index=15).set_weights(model[j].get_layer(index=11).get_weights())
    model2[j].get_layer(index=16).set_weights(model[j].get_layer(index=12).get_weights())#batchnorm16
    model2[j].get_layer(index=18).set_weights(model[j].get_layer(index=14).get_weights())
    model2[j].get_layer(index=19).set_weights(model[j].get_layer(index=15).get_weights())#batchnorm19
    model2[j].get_layer(index=22).set_weights(model[j].get_layer(index=18).get_weights())
    a=np.ones((1,1,1,32,1))
    model2[j].get_layer(index=2).set_weights(a)
    a=np.ones((1,1,1,32,1))
    model2[j].get_layer(index=5).set_weights(a)
    a=np.ones((1,1,1,32,1))
    model2[j].get_layer(index=8).set_weights(a)
    a=np.ones((1,1,1,64,1))
    model2[j].get_layer(index=11).set_weights(a)
    a=np.ones((1,1,1,64,1))
    model2[j].get_layer(index=14).set_weights(a)
    a=np.ones((1,1,1,64,1))
    model2[j].get_layer(index=17).set_weights(a)
    a=np.ones((1,1,1,128,1))
    model2[j].get_layer(index=20).set_weights(a)
    a=np.ones((1,1,1,10,1))
    model2[j].get_layer(index=24).set_weights(a)


# In[ ]:


initModel2(0)


# # Train 15+15 CNNs (modified to add the training of the new models)

# In[ ]:


# DECREASE LEARNING RATE EACH EPOCH
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
# TRAIN NETWORKS
history = [0] * nets
history2 = [0] * nets
epochs = 45
for j in range(nets):
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.1)
    history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),
        epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,  
        validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))
    initModel2(j)
    history2[j] = model2[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),
        epochs = 5, steps_per_epoch = X_train2.shape[0]//64,  
        validation_data = (X_val2,Y_val2), verbose=0)
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        j+1,epochs,max(history2[j].history['acc']),max(history2[j].history['val_acc']) ))


# # Ensemble CNN predictions

# In[ ]:


# ENSEMBLE PREDICTIONS AND SUBMIT
results = np.zeros( (X_test.shape[0],10) ) 
for j in range(nets):
    results = results + model[j].predict(X_test)
resultsa = np.argmax(results,axis = 1)
resultsa = pd.Series(resultsa,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),resultsa],axis = 1)
submission.to_csv("MNIST-CNN-ENSEMBLE1.csv",index=False)


# In[ ]:


results2 = np.zeros( (X_test.shape[0],10) ) 
for j in range(nets):
    results2 = results2 + model2[j].predict(X_test)
resultsa = np.argmax(results2,axis = 1)
resultsa = pd.Series(resultsa,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),resultsa],axis = 1)
submission.to_csv("MNIST-CNN-ENSEMBLE2.csv",index=False)


# In[ ]:


res=results+results2
resultsa = np.argmax(res,axis = 1)
resultsa = pd.Series(resultsa,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),resultsa],axis = 1)
submission.to_csv("MNIST-CNN-ENSEMBLE3.csv",index=False)


# # Credits
# As already said, the code is based on "[25 Millions Images! [0.99757] MNIST](https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist/notebook)", in which credits was given to :
# 
# * [Yassine Ghouzam][1] - [Introduction to CNN Keras - 0.997 (top 6%)][2]
# * [Peter Grenholm][5] - [Welcome to deep learning (CNN 99%)][6]
# * [Ding Li][3] - [Digits Recognition With CNN Keras][4]
# * [Aditya Soni][7] - [MNIST with Keras for Beginners(.99457)][8]
# 
# [1]:https://www.kaggle.com/yassineghouzam
# [2]:https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
# [3]:https://www.kaggle.com/dingli
# [4]:https://www.kaggle.com/dingli/digits-recognition-with-cnn-keras
# [5]:https://www.kaggle.com/toregil
# [6]:https://www.kaggle.com/toregil/welcome-to-deep-learning-cnn-99/
# [8]:https://www.kaggle.com/adityaecdrid/mnist-with-keras-for-beginners-99457/
# [7]:https://www.kaggle.com/adityaecdrid
