#!/usr/bin/env python
# coding: utf-8

# <center>
# <h1> Danbury AI June 2018: Workshop Part 4</h1>
# <h2>Street View House Numbers</h2>
# </center>
# SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits), but comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images.
# 
# * 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10.
# *  MNIST-like 32-by-32 images centered around a single character (many of the images do contain some distractors at the sides).
# 
# Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng Reading Digits in Natural Images with Unsupervised Feature Learning NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011. 
# 
# Web source: http://ufldl.stanford.edu/housenumbers/
# 
# Here is a selection of what the individual training images look like:
# ![](http://ufldl.stanford.edu/housenumbers/32x32eg.png)

# In[29]:


import pandas as pd
import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from ipywidgets import interact
from keras.layers import Input, Dense, Dropout,Conv2D,MaxPooling2D,Flatten
from keras.layers import GlobalMaxPooling2D,UpSampling2D,GlobalMaxPooling1D
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Print out the folders where our datasets live. 
print("Datasets: {0}".format(os.listdir("../input/danbury-ai-june-2018")))


# Here we load in numpy matricies for the training images (train_x), training labels (train_y), and test images (test_x). 

# In[34]:


# The training images. 
X = np.load("../input/danbury-ai-june-2018/train_x.npy")
y = np.load("../input/danbury-ai-june-2018/train_y.npy")

# We subtract 1 from the labels in order to scale the the labels between 0,9. 
y = y - 1

# These are the images we will need to predict lables for. 
test  = np.load("../input/danbury-ai-june-2018/test.npy")


# We will now do our standard training/validation split so we can evaluate our model's generalization characteristics. 

# In[35]:


X_train, X_validation, y_train , y_validation = train_test_split(X,to_categorical(y,10), test_size=0.2)


# Here we define a convolutional neural network with a max pooling layer and gobal pooling layer. 

# In[36]:


def makeModel(inputSize):
    inputs = Input(shape=inputSize,name="input")
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = GlobalMaxPooling2D()(x)
    out = Dense(10,activation='softmax', name="output")(x)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['acc'])
    
    return model


# We will now train our model. 

# In[37]:


model2 = makeModel((32,32,3,))
model2.summary()
hist2 = model2.fit(X_train, y_train, batch_size=100,epochs=10, validation_data=(X_validation,y_validation))


# Now that we have trained our network, let's look at the training loss over the training epochs. 

# In[ ]:


def learningCurves(hist):
    histAcc_train = hist.history['acc']
    histLoss_train = hist.history['loss']
    histAcc_validation = hist.history['val_acc']
    histLoss_validation = hist.history['val_loss']
    maxValAcc = np.max(histAcc_validation)
    minValLoss = np.min(histLoss_validation)

    plt.figure(figsize=(12,12))
    epochs = len(histAcc_train)

    plt.plot(range(epochs),histLoss_train, label="Training Loss", color="#acc6ef")
    plt.plot(range(epochs),histLoss_validation, label="Validation Loss", color="#a7e295")

    plt.scatter(np.argmin(histLoss_validation),minValLoss,zorder=10,color="green")

    plt.xlabel('Epochs',fontsize=14)
    plt.title("Learning Curves",fontsize=20)

    plt.legend()
    plt.show()

    print("Max validation accuracy: {0}".format(maxValAcc))
    print("Minimum validation loss: {0}".format(minValLoss))
    
learningCurves(hist2)


# Predict the class lables with our model 

# In[38]:


pred = model2.predict(test)
pred = np.argmax(pred,1)
pred = pred + 1


# In[39]:


submission = pd.DataFrame.from_items([
    ('id',list(range(pred.shape[0]))),
    ('label', pred)])

submission.to_csv('submission.csv', index = False)

