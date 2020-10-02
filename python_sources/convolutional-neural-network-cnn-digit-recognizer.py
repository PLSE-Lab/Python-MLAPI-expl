#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Networks (CNNs / ConvNets)

# # Orhan SERTKAYA

# Content:
# * [Introduction](#1):
# * [Loading the Data Set](#2):
# * [Normalization, Reshape and Label Encoding](#3):
# * [Train-Test Split](#4):
# * [Convolutional Neural Network(Implementing with Keras)](#5):
# * [Define Optimizer](#6):
# * [Compile Model](#7):
# * [Epochs and Batch Size](#8):
# * [Data Augmentation](#9):
# * [Fit the Model](#10):
# * [Evaluate the model](#11):
# * [Predict For Random Sample](#12):
# * [Wrong Predicted Digit Values](#13):
# * [Predict Test Data](#14):
# * [Conclusion](#15):

# <a id="1"></a> <br>
# # INTRODUCTION
# * In this kernel, we will be working on Digit Recognizer Dataset (Implementing with Keras).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a id="2"></a> <br>
# ## Loading the Data Set
# * In this part we load and visualize the data.

# In[ ]:


# read train
train = pd.read_csv("../input/train.csv")
print(train.shape)
train.head()


# * For example,let's look at first sample pixel values

# In[ ]:


train.iloc[0].value_counts()


# In[ ]:


# read test
test = pd.read_csv("../input/test.csv")
print(test.shape)
test.head()


# In[ ]:


# put labels into y_train variable
Y_train = train["label"]
# Drop 'label' column
X_train = train.drop(labels = ["label"], axis = 1)
X_train.head()


# In[ ]:


# visualize number of digits classes
plt.figure(figsize=(15,7))
g = sns.countplot(Y_train, palette="icefire")
plt.title("Number of digit classes")
Y_train.value_counts()


# In[ ]:


# plot some samples
#as_matrix = Converting to matrix
img = X_train.iloc[0].as_matrix()
img = img.reshape((28,28))
plt.imshow(img,cmap="gray")
plt.title(Y_train.iloc[0]) #or plt.title(train.iloc[0,0]) both of them are okay.
plt.axis("off")
plt.show()


# In[ ]:


# plot some samples
img = X_train.iloc[7].as_matrix()
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(Y_train.iloc[7]) #or plt.title(train.iloc[7,0]) both of them are okay.
plt.axis("off")
plt.show()


# In[ ]:


# plot some samples
img = X_train.iloc[7223].as_matrix()
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(Y_train.iloc[7223]) #or plt.title(train.iloc[7223,0]) both of them are okay.
plt.axis("off")
plt.show()


# <a id="3"></a> <br>
# ## Normalization, Reshape and Label Encoding 
# * Normalization
#     * We perform a grayscale normalization to reduce the effect of illumination's differences.
#     * If we perform normalization, CNN works faster.
# * Reshape
#     * Train and test images (28 x 28) 
#     * We reshape all data to 28x28x1 3D matrices.
#     * Keras needs an extra dimension in the end which correspond to channels. Our images are gray scaled so it use only one channel. 
# * Label Encoding  
#     * Encode labels to one hot vectors 
#         * 2 => [0,0,1,0,0,0,0,0,0,0]
#         * 4 => [0,0,0,0,1,0,0,0,0,0]

# In[ ]:


# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
print("X_train shape: ",X_train.shape)
print("test shape: ",test.shape)


# In[ ]:


# Reshape
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("X_train shape: ",X_train.shape)
print("test shape: ",test.shape)


# In[ ]:


# Label Encoding 
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding(one hot vectors)
Y_train = to_categorical(Y_train, num_classes = 10)


# <a id="4"></a>
# ## Train-Test Split
# * We split the data into train and test sets.
# * test size is 10%.
# * train size is 90%.

# In[ ]:


# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 2)
print("x_train shape: ",x_train.shape)
print("x_val shape: ",x_val.shape)
print("y_train shape: ",y_train.shape)
print("y_val shape :",y_val.shape)


# In[ ]:


# Some examples
plt.imshow(x_train[4][:,:,0], cmap="gray")
plt.axis("off")
plt.show()


# <a id="5"></a>
# ## Convolutional Neural Network 

# ## Implementing with Keras

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential # to create a cnn model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.optimizers import RMSprop,Adam
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'Same'))
model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'Same'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = (2,2)))

# fully connected
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation = "softmax"))


# In[ ]:


model.summary()


# <a id="6"></a>
# ### Define Optimizer   
# * Adam optimizer: Change the learning rate

# In[ ]:


# Define the optimizer
#optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)

from keras.optimizers import RMSprop,Adam,SGD,Adagrad,Adadelta,Adamax,Nadam
#optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#optimizer=Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#optimizer=Adadelta(lr=0.001, rho=0.95, epsilon=0.1, decay=0.1)
#optimizer=Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

#optimizer=Adagrad(lr=0.01, epsilon=None, decay=0.0)


# In[ ]:


# # Define the optimizer
# optimizer = RMSprop(lr = 0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# <a id="7"></a>
# ### Compile Model
# * categorical crossentropy
# * We make binary cross entropy at previous parts and in machine learning tutorial
# * At this time we use categorical crossentropy. That means that we have multi class.
# * <a href="https://ibb.co/jm1bpp"><img src="https://preview.ibb.co/nN3ZaU/cce.jpg" alt="cce" border="0"></a>

# In[ ]:


# Compile the model
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# <a id="8"></a>
# ### Epochs and Batch Size

# In[ ]:


epochs = 25  # for better result increase the epochs
batch_size = 250


# <a id="9"></a>
# ### Data Augmentation

# In[ ]:


# data augmentation
datagen = ImageDataGenerator(   
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=10,  # randomly rotate images in the range 10 degrees
        zoom_range = 0.1, # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False  # randomly flip images
        )
datagen.fit(x_train)


# <a id="10"></a>
# ### Fit the Model

# In[ ]:


# # Fit the model
# history = model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size),
#                    epochs = epochs, validation_data = (x_val,y_val), steps_per_epoch = x_train.shape[0] // batch_size)


# In[ ]:





# In[ ]:


history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                              epochs=epochs, validation_data = (x_val, y_val),
                              verbose = 2, steps_per_epoch=x_train.shape[0] // batch_size,
                              callbacks=[learning_rate_reduction]) 


# <a id="11"></a>
# ## Evaluate the model
# * Validation and Loss visualization
# * Confusion matrix

# In[ ]:


# Plot the loss curve for training
plt.plot(history.history['loss'], color='r', label="Train Loss")
plt.title("Train Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[ ]:


# Plot the accuracy curve for training
plt.plot(history.history['acc'], color='g', label="Train Accuracy")
plt.title("Train Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[ ]:


# Plot the loss curve for validation 
plt.plot(history.history['val_loss'], color='r', label="Validation Loss")
plt.title("Validation Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[ ]:


# Plot the accuracy curve for validation 
plt.plot(history.history['val_acc'], color='g', label="Validation Accuracy")
plt.title("Validation Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[ ]:


print('Train accuracy of the model: ',history.history['acc'][-1])


# In[ ]:


print('Train loss of the model: ',history.history['loss'][-1])


# In[ ]:


print('Validation accuracy of the model: ',history.history['val_acc'][-1])


# In[ ]:


print('Validation loss of the model: ',history.history['val_loss'][-1])


# <a id="12"></a>
# ### Predict For Random Sample

# In[ ]:


print(x_val.shape)
plt.imshow(x_val[100].reshape(28,28),cmap="gray")
plt.axis("off")
plt.show()


# In[ ]:


trueY = y_val[100]
img = x_val[100]
test_img = img.reshape(1,28,28,1)

preds = model.predict_classes(test_img)
prob = model.predict_proba(test_img)

print("trueY: ",np.argmax(trueY))#show the max value in trueY values(trueY is one hot vector!)
print("Preds: ",preds)
print("Prob: ",prob)


# <a id="13"></a>
# ### Let's show the wrong predicted digit values

# In[ ]:


d = {'pred': model.predict_classes(x_val), 'true': np.argmax(y_val,axis=1)} #axis=1!important!
df = pd.DataFrame(data=d)

#looking at wrong predicted values(For 1! you can change it.)
array1 = np.array(df[(df.pred != df.true) & (df.true==1)].index)
print(array1)

# shows total mistakes
df2 = df[(df.pred != df.true)]
df2


# In[ ]:


plt.figure(figsize = (12,12))

for i in range(len(df2)):
    plt.subplot(6, 4, i+1)
    img = x_val[df2.index[i]]
    img = img.reshape((28,28))
    plt.imshow(img, cmap='gray')
    plt.title("True Class: " + str(df2["true"].iloc[i])+"    Pred Class: " + str(df2["pred"].iloc[i]))
    plt.axis('off')
    
plt.show()


# * **As you can see, some numbers are hard to understand!**

# In[ ]:


# confusion matrix
import seaborn as sns
# Predict the values from the validation dataset
Y_pred = model.predict(x_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap = "gist_yarg_r", linecolor="black", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:


for i in range(len(confusion_mtx)):
    print("Class:",str(i))
    print("Number of Wrong Prediction:", str(sum(confusion_mtx[i])-confusion_mtx[i][i]), "out of "+str(sum(confusion_mtx[i])))
    print("Percentage of True Prediction: {:.2f}%".format(confusion_mtx[i][i] / (sum(confusion_mtx[i])/100) ))
    print("***********************************************************")


# <a id="14"></a>
# ## PREDICT TEST DATA

# In[ ]:


test = pd.read_csv('../input/test.csv')
test = test.values.reshape(-1,28,28,1)
test.shape


# In[ ]:


# predict results
results = model.predict(test)


# In[ ]:


# select the index with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,len(test)+1),name = "ImageId"),results],axis = 1)

submission.to_csv("Digit_Recognizer_CNN_Result_2.csv",index=False)


# In[ ]:


submission.head(10)


# <a id="15"></a>
# # Conclusion
# * If you like it, please upvote.
# * If you have any question, I will be appreciate to hear it.
