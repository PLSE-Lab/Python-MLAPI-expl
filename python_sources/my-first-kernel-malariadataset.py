#!/usr/bin/env python
# coding: utf-8

# # Still in progress, I am a beginner so incase of any correction

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import cv2
import glob
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Nadam, Adadelta, Adamax
from keras.utils import to_categorical
from matplotlib import pyplot



import os
print(os.listdir("../input/cell_images/cell_images/"))

# Any results you write to the current directory are saved as output.


# # Loading and Preprocessing the Data

# In[ ]:


images = []
labels = []

path1 = "../input/cell_images/cell_images/Parasitized/"
path2 = "../input/cell_images/cell_images/Uninfected/"


for i in glob.glob(os.path.join(path1,'*png')):
    img = cv2.imread(i)
    #img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # convert to grayscale
    img = cv2.resize(img,(50,50))  #resize
    images.append(np.array(img))
    labels.append(0)
    
for j in glob.glob(os.path.join(path2,'*png')):
    img = cv2.imread(j)
    #img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # convert to grayscale
    img = cv2.resize(img,(50,50))  #resize
    images.append(np.array(img))
    labels.append(1)    
    



# In[ ]:



 
cells , labels = np.array(images), np.array(labels)    
np.save("cells",cells)
np.save("labels",labels)
the_cells, the_labels = np.load("cells.npy") , np.load("labels.npy")


# # Plotting some images of the labelled dataset

# In[ ]:



fig=plt.figure(figsize=(10, 8))
n = 16
for i in range(n):
    img = np.random.randint(0, the_cells.shape[0] , 1)
    fig.add_subplot(n**(.5), n**(.5), i+1)
    plt.imshow(the_cells[img[0]])
    plt.title('{} : {}'.format('Unifected' if the_labels[img[0]] == 1 else 'Parasitized' ,
                                the_labels[img[0]]) )
    plt.xticks([]) , plt.yticks([])
        
plt.show()


# # Splitting and Normalizing the data

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(the_cells, the_labels, test_size=0.2)

#converting to float and normalizing
X_train = X_train.astype('float32')/255 
X_test = X_test.astype('float32')/255

#getting the numbr of unique classes in the labels
num_classes=len(np.unique(the_labels))

#One hot encoding as classifier since we  has multiple classes
Y_train=keras.utils.to_categorical(Y_train,num_classes)
Y_test=keras.utils.to_categorical(Y_test,num_classes)


# # Defining a simple model

# ## fully connected neural network

# In[ ]:


# import regularizer
from keras.regularizers import l1
# instantiate regularizer
reg = l1(0.001)


# In[ ]:


from keras.layers import Dense, Activation


nnmodel = Sequential()
nnmodel.add(Dense(32, input_shape=(50,50,3)))
nnmodel.add(Activation('relu'))
nnmodel.add(Flatten())
nnmodel.add(Dense(2,activation="softmax"))#, activity_regularizer=l1(0.001))) 
nnmodel.summary()


# In[ ]:


nnmodel.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.001),metrics=['mae', 'acc'])

nnhistory =nnmodel.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, verbose=1)


# In[ ]:


# summarize history for accuracy
plt.plot(nnhistory.history['acc'])
plt.plot(nnhistory.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test' ], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(nnhistory.history['loss'])
plt.plot(nnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show() 


# In[ ]:


plt.plot(nnhistory.history['mean_absolute_error'])
plt.title('Model training mean_absolute_error')
plt.ylabel('mean_absolute_error')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(nnhistory.history['val_mean_absolute_error'])
plt.title('Model validation mean_absolute_error')
plt.ylabel('mean_absolute_error')
plt.xlabel('epoch')
plt.legend(['test' ], loc='upper left')
plt.show()


# ## Convolutional Neural Network

# In[ ]:


model = Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(50,activation="relu"))
model.add(Dense(2,activation="softmax",activity_regularizer=l1(0.001)))#2 represent output layer neurons 
model.summary()


# # Plotting the Loss and Accuracy

# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['mae', 'acc'])

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=1)


# In[ ]:


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show() 


# In[ ]:


plt.plot(history.history['mean_absolute_error'])
plt.title('Model training mean_absolute_error')
plt.ylabel('mean_absolute_error')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(history.history['val_mean_absolute_error'])
plt.title('Model validation mean_absolute_error')
plt.ylabel('mean_absolute_error')
plt.xlabel('epoch')
plt.legend(['test' ], loc='upper left')
plt.show()


# # Organizing the Model

# In[ ]:


# split data set 
def loadingdata():
    #splitting
    X_train, X_test, Y_train, Y_test = train_test_split(the_cells, the_labels, test_size=0.2)
    
    #converting to float and normalizing
    X_train = X_train.astype('float32')/255 
    X_test = X_test.astype('float32')/255
    
    #getting the numbr of unique classes in the labels
    num_classes=len(np.unique(the_labels))

    #One hot encoding as classifier since we  has multiple classes
    Y_train=keras.utils.to_categorical(Y_train,num_classes)
    Y_test=keras.utils.to_categorical(Y_test,num_classes)

    return X_train, Y_train, X_test, Y_test


# In[ ]:


# here the model is initialized and fitted, accuracy and losses as well as varying learnig curves are plotted 
def MyCNNmodel(X_train, Y_train, X_test, Y_test, lrate):

    model = Sequential()
    model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    #model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))    
    model.add(Dense(50,activation="relu"))
    model.add(Dense(2,activation="softmax"))#2 represent output layer neurons 
    #model.summary()
    ## compiling the model
    opt = SGD(lr=lrate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['mae', 'acc'])
    # fit model
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=1)
    #history = model.fit(X_train, Y_train, validation_split=0.33, epochs=20, batch_size=10, verbose=1)
    #print(history.history.keys())
    
#     # summarize history for accuracy
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
#     # summarize history for loss
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     #plt.show()
    
    # plot learning curves
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='val')
    plt.title('lrate='+str(lrate), pad=-50)

     


# # Plotting the Learning Rates

# In[ ]:



# prepare dataset
X_train, Y_train, X_test, Y_test = loadingdata()
# create learning curves for different learning rates
learning_rates = [1E-0, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7]
for i in range(len(learning_rates)):
    # determine the plot number
    plot_no = 420 + (i+1)
    plt.subplot(plot_no)
    # fit model and plot learning curves for a learning rate
    MyCNNmodel(X_train, Y_train, X_test, Y_test, learning_rates[i])
# show learning curves
plt.show()


# # Classification Report and Confusion Matrix

# In[ ]:


from sklearn.metrics import classification_report

y_pred=model.predict(X_test) 
y_pred=np.argmax(y_pred, axis=1) 
y_true = Y_test
y_true=np.argmax(y_true, axis=1) 


print(classification_report(y_true, y_pred))


# In[ ]:


#Making confusion matrix that checks accuracy of the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
cm


# In[ ]:


import scikitplot 
import matplotlib.pyplot as plt

#y_true = # ground truth labels
#y_probas = # predicted probabilities generated by sklearn classifier
y_probas = model.predict(X_test)
scikitplot.metrics.plot_roc(y_true, y_probas)
plt.show()


# In[ ]:


from keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)


# # Variations of the models for Hyperparameter Optimization

# view how the models loss and accuracy and loss changes
# 1. fcn2layer, lr = 0.01, beta = 0,reg=0, epochs = 10
# 2. fcn2layer+reg, lr = 0.01, beta = 0,reg=0.001, epochs = 10
# 3. fcn2layer+lr1, lr = 0.001, beta = 0,reg=0, epochs = 10
# 4. fcn2layer+lr1+reg, lr = 0.001, beta = 0,reg=0.001, epochs = 10
# 5. fcn3layer+lr1+reg, lr = 0.001, beta = 0,reg=0.001, epochs = 10
# 6. fcn3layer+lr1+reg, lr = 0.001, beta = 0,reg=0.001, epochs = 10
# 7. fcn4layer+lr1+reg, lr = 0.001, beta = 0,reg=0.001, epochs = 10
# 
# and also when the nodes changes
# 1. fcn2layer, nodes = 32
# 2. fcn2layer, nodes = 64
# 3. fcn2layer, nodes = 128
# 
# 
# where: lr = learning rate, reg = regularization
# 
# 
# 

# In[ ]:


# import regularizer
from keras.regularizers import l1,l2
# instantiate regularizer
reg = l1(0.001)
reg2 = l2(0.001)


# In[ ]:


nn1model = Sequential()
nn1model.add(Dense(32, input_shape=(50,50,3)))
nn1model.add(Activation('relu'))
nn1model.add(Flatten())
nn1model.add(Dense(2,activation="softmax"))#, activity_regularizer=l1(0.001))) 
#nn1model.summary()

nn1model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.01),metrics=['accuracy'])

nn1history =nn1model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=300, verbose=1)


# In[ ]:


nn1model.summary()


# In[ ]:


# summarize history for accuracy
plt.plot(nn1history.history['acc'])
plt.plot(nn1history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test' ], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(nn1history.history['loss'])
plt.plot(nn1history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


score = nn1model.evaluate(X_test, Y_test, verbose=1)
print('\n', 'Test_Loss:-', score[0])
print('\n', 'Test_Accuracy:-', score[1])


# In[ ]:


from keras.layers import  Dropout, BatchNormalization


# In[ ]:


cnn1model = Sequential()
cnn1model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
#cnn1model.add(BatchNormalization())
cnn1model.add(MaxPooling2D(pool_size=2))
cnn1model.add(Dropout(0.25))      
cnn1model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
#cnn1model.add(BatchNormalization())
cnn1model.add(MaxPooling2D(pool_size=2))
cnn1model.add(Dropout(0.25))      
# cnn1model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
# cnn1model.add(MaxPooling2D(pool_size=2))
# cnn1model.add(BatchNormalization())
#cnn1model.add(Dropout(0.25))      
cnn1model.add(Flatten())
cnn1model.add(Dense(50,activation="relu"))
cnn1model.add(Dense(2,activation="softmax", kernel_regularizer=l2(0.001)))#activity_regularizer=l1(0.001)))
#cnn1model.summary()
cnn1model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
cnn1history = cnn1model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=300, verbose=1)


# In[ ]:


cnn1model.summary()


# In[ ]:


score1 = cnn1model.evaluate(X_test, Y_test, verbose=1)
print('\n', 'Test_Loss:-', score1[0])
print('\n', 'Test_Accuracy:-', score1[1])


# In[ ]:


# summarize history for accuracy
plt.plot(cnn1history.history['acc'])
plt.plot(cnn1history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test' ], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(cnn1history.history['loss'])
plt.plot(cnn1history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show() 


# In[ ]:





# # Comparing Several Keras Optimizers

# ## Optimizer 1: RMSprop

# In[ ]:


cnnmodel_rmsprop = Sequential()
cnnmodel_rmsprop.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
cnnmodel_rmsprop.add(MaxPooling2D(pool_size=2))
cnnmodel_rmsprop.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
cnnmodel_rmsprop.add(MaxPooling2D(pool_size=2))
cnnmodel_rmsprop.add(Flatten())
cnnmodel_rmsprop.add(Dense(50,activation="relu"))
cnnmodel_rmsprop.add(Dense(2,activation="softmax"))#2 represent output layer neurons 
#cnnmodel_rmsprop.summary()
cnnmodel_rmsprop.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
hist_rmsprop = cnnmodel_rmsprop.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=1, callbacks=[reduce_lr])


# ## Optimizer 2: Adam

# In[ ]:


cnnmodel_adam = Sequential()
cnnmodel_adam.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
cnnmodel_adam.add(MaxPooling2D(pool_size=2))
cnnmodel_adam.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
cnnmodel_adam.add(MaxPooling2D(pool_size=2))
cnnmodel_adam.add(Flatten())
cnnmodel_adam.add(Dense(50,activation="relu"))
cnnmodel_adam.add(Dense(2,activation="softmax"))#2 represent output layer neurons 
cnnmodel_adam.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
hist_adam = cnnmodel_adam.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=1, callbacks=[reduce_lr])


# ## Optimizer 3: Nadam

# In[ ]:


cnnmodel_nadam = Sequential()
cnnmodel_nadam.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
cnnmodel_nadam.add(MaxPooling2D(pool_size=2))
cnnmodel_nadam.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
cnnmodel_nadam.add(MaxPooling2D(pool_size=2))
cnnmodel_nadam.add(Flatten())
cnnmodel_nadam.add(Dense(50,activation="relu"))
cnnmodel_nadam.add(Dense(2,activation="softmax"))#2 represent output layer neurons 
cnnmodel_nadam.compile(optimizer=Nadam(), loss='binary_crossentropy', metrics=['accuracy'])
hist_nadam = cnnmodel_nadam.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=1, callbacks=[reduce_lr])


# ## Optimizer 4: SGD

# In[ ]:


cnnmodel_sgd= Sequential()
cnnmodel_sgd.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
cnnmodel_sgd.add(MaxPooling2D(pool_size=2))
cnnmodel_sgd.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
cnnmodel_sgd.add(MaxPooling2D(pool_size=2))
cnnmodel_sgd.add(Flatten())
cnnmodel_sgd.add(Dense(50,activation="relu"))
cnnmodel_sgd.add(Dense(2,activation="softmax"))#2 represent output layer neurons 
#cnnmodel_rmsprop.summary()
cnnmodel_sgd.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
#hist_rmsprop = cnnmodel_rmsprop.fit(X_train, Y_train, batch_size=batch_size*2,  nb_epoch=0, validation_data=(X_test,Y_test), callbacks=[reduce_lr])
hist_sgd = cnnmodel_sgd.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=1, callbacks=[reduce_lr])


# ## Optimizer 5: SGD + Nesterov

# In[ ]:


cnnmodel_sgdnesterov = Sequential()
cnnmodel_sgdnesterov.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
cnnmodel_sgdnesterov.add(MaxPooling2D(pool_size=2))
cnnmodel_sgdnesterov.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
cnnmodel_sgdnesterov.add(MaxPooling2D(pool_size=2))
cnnmodel_sgdnesterov.add(Flatten())
cnnmodel_sgdnesterov.add(Dense(50,activation="relu"))
cnnmodel_sgdnesterov.add(Dense(2,activation="softmax"))#2 represent output layer neurons 
#cnnmodel_rmsprop.summary()
cnnmodel_sgdnesterov.compile(optimizer=SGD(nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])
#hist_rmsprop = cnnmodel_rmsprop.fit(X_train, Y_train, batch_size=batch_size*2,  nb_epoch=0, validation_data=(X_test,Y_test), callbacks=[reduce_lr])
hist_sgdnesterov = cnnmodel_sgdnesterov.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=1, callbacks=[reduce_lr])


# ## Optimizer 6: SGD with momentum=0.9

# In[ ]:


cnnmodel_sgdmomentum = Sequential()
cnnmodel_sgdmomentum.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
cnnmodel_sgdmomentum.add(MaxPooling2D(pool_size=2))
cnnmodel_sgdmomentum.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
cnnmodel_sgdmomentum.add(MaxPooling2D(pool_size=2))
cnnmodel_sgdmomentum.add(Flatten())
cnnmodel_sgdmomentum.add(Dense(50,activation="relu"))
cnnmodel_sgdmomentum.add(Dense(2,activation="softmax"))#2 represent output layer neurons 
#cnnmodel_rmsprop.summary()
cnnmodel_sgdmomentum.compile(optimizer=SGD(momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
#hist_rmsprop = cnnmodel_rmsprop.fit(X_train, Y_train, batch_size=batch_size*2,  nb_epoch=0, validation_data=(X_test,Y_test), callbacks=[reduce_lr])
hist_sgdmomentum = cnnmodel_sgdmomentum.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=1, callbacks=[reduce_lr])


# ## Optimizer 7: SGD + Nesterov with momentum=0.9

# In[ ]:


cnnmodel_sgdnestmomentum = Sequential()
cnnmodel_sgdnestmomentum.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
cnnmodel_sgdnestmomentum.add(MaxPooling2D(pool_size=2))
cnnmodel_sgdnestmomentum.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
cnnmodel_sgdnestmomentum.add(MaxPooling2D(pool_size=2))
cnnmodel_sgdnestmomentum.add(Flatten())
cnnmodel_sgdnestmomentum.add(Dense(50,activation="relu"))
cnnmodel_sgdnestmomentum.add(Dense(2,activation="softmax"))#2 represent output layer neurons 
#cnnmodel_rmsprop.summary()
cnnmodel_sgdnestmomentum.compile(optimizer=SGD(momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])
#hist_rmsprop = cnnmodel_rmsprop.fit(X_train, Y_train, batch_size=batch_size*2,  nb_epoch=0, validation_data=(X_test,Y_test), callbacks=[reduce_lr])
hist_sgdnestmomentum = cnnmodel_sgdnestmomentum.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=1, callbacks=[reduce_lr])


# ## Optimizer 8:Adagrad

# In[ ]:


cnnmodel_adagrad = Sequential()
cnnmodel_adagrad.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
cnnmodel_adagrad.add(MaxPooling2D(pool_size=2))
cnnmodel_adagrad.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
cnnmodel_adagrad.add(MaxPooling2D(pool_size=2))
cnnmodel_adagrad.add(Flatten())
cnnmodel_adagrad.add(Dense(50,activation="relu"))
cnnmodel_adagrad.add(Dense(2,activation="softmax"))#2 represent output layer neurons 
#cnnmodel_rmsprop.summary()
cnnmodel_adagrad.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])
#hist_rmsprop = cnnmodel_rmsprop.fit(X_train, Y_train, batch_size=batch_size*2,  nb_epoch=0, validation_data=(X_test,Y_test), callbacks=[reduce_lr])
hist_adagrad = cnnmodel_adagrad.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=1, callbacks=[reduce_lr])


# ## Optimizer 9: Adadelta

# In[ ]:


cnnmodel_adadelta = Sequential()
cnnmodel_adadelta.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
cnnmodel_adadelta.add(MaxPooling2D(pool_size=2))
cnnmodel_adadelta.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
cnnmodel_adadelta.add(MaxPooling2D(pool_size=2))
cnnmodel_adadelta.add(Flatten())
cnnmodel_adadelta.add(Dense(50,activation="relu"))
cnnmodel_adadelta.add(Dense(2,activation="softmax"))#2 represent output layer neurons 
#cnnmodel_rmsprop.summary()
cnnmodel_adadelta.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
#hist_rmsprop = cnnmodel_rmsprop.fit(X_train, Y_train, batch_size=batch_size*2,  nb_epoch=0, validation_data=(X_test,Y_test), callbacks=[reduce_lr])
hist_adadelta = cnnmodel_adadelta.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=1, callbacks=[reduce_lr])


# ## Optimizer 10: Adamax

# In[ ]:


cnnmodel_adamax = Sequential()
cnnmodel_adamax.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
cnnmodel_adamax.add(MaxPooling2D(pool_size=2))
cnnmodel_adamax.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
cnnmodel_adamax.add(MaxPooling2D(pool_size=2))
cnnmodel_adamax.add(Flatten())
cnnmodel_adamax.add(Dense(50,activation="relu"))
cnnmodel_adamax.add(Dense(2,activation="softmax"))#2 represent output layer neurons 
#cnnmodel_rmsprop.summary()
cnnmodel_adamax.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])
#hist_rmsprop = cnnmodel_rmsprop.fit(X_train, Y_train, batch_size=batch_size*2,  nb_epoch=0, validation_data=(X_test,Y_test), callbacks=[reduce_lr])
hist_adamax = cnnmodel_adamax.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=1, callbacks=[reduce_lr])


# # Plots of the optimizers

# In[ ]:


plt.figure(figsize=(10, 6))  
#plt.axis((-1,14,0.4, 1))

plt.plot(hist_rmsprop.history['val_acc'])
plt.plot(hist_adam.history['val_acc'])
plt.plot(hist_nadam.history['val_acc'])
plt.plot(hist_sgd.history['val_acc'])
plt.plot(hist_sgdnesterov.history['val_acc'])
plt.plot(hist_sgdmomentum.history['val_acc'])
plt.plot(hist_sgdnestmomentum.history['val_acc'])
plt.plot(hist_adagrad.history['val_acc'])
plt.plot(hist_adadelta.history['val_acc'])
plt.plot(hist_adamax.history['val_acc'])
plt.title('val accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['rmsprop', 'adam', 'nadam', 'sgd', 'sgd_w/nesterov', 'sgd_w/momentum', 'sgd_w/nesterov+momentum', 'adagrad', 'adadelta', 'adamax'], loc='lower right',fontsize = 'x-small')  

plt.show()


# In[ ]:


plt.figure(figsize=(10, 6))  
#plt.axis((-1,14,0.4, 1))


plt.plot(hist_rmsprop.history['acc'])  
plt.plot(hist_adam.history['acc'])  
plt.plot(hist_nadam.history['acc']) 
plt.plot(hist_sgd.history['acc']) 
plt.plot(hist_sgdnesterov.history['acc']) 
plt.plot(hist_sgdmomentum.history['acc'])
plt.plot(hist_sgdnestmomentum.history['acc'])
plt.plot(hist_adagrad.history['acc'])
plt.plot(hist_adadelta.history['acc'])
plt.plot(hist_adamax.history['acc'])
plt.title('train accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['rmsprop', 'adam', 'nadam', 'sgd', 'sgd_w/nesterov', 'sgd_w/momentum', 'sgd_w/nesterov+momentum', 'adagrad', 'adadelta', 'adamax'], loc='lower right',fontsize = 'x-small')  

plt.show()


# In[ ]:


plt.figure(figsize=(10, 6))  
#plt.axis((-1,14,0, 1))


plt.plot(hist_rmsprop.history['val_loss'])  
plt.plot(hist_adam.history['val_loss'])  
plt.plot(hist_nadam.history['val_loss']) 
plt.plot(hist_sgd.history['val_loss']) 
plt.plot(hist_sgdnesterov.history['val_loss']) 
plt.plot(hist_sgdmomentum.history['val_loss'])
plt.plot(hist_sgdnestmomentum.history['val_loss'])
plt.plot(hist_adagrad.history['val_loss'])
plt.plot(hist_adadelta.history['val_loss'])
plt.plot(hist_adamax.history['val_loss'])
plt.title('val loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['rmsprop', 'adam', 'nadam', 'sgd', 'sgd_w/nesterov', 'sgd_w/momentum', 'sgd_w/nesterov+momentum', 'adagrad', 'adadelta', 'adamax'], loc='upper right',fontsize = 'x-small')  

plt.show()


# In[ ]:


plt.figure(figsize=(10, 6))  
#plt.axis((-1,14,0, 1))


plt.plot(hist_rmsprop.history['loss'])  
plt.plot(hist_adam.history['loss'])  
plt.plot(hist_nadam.history['loss']) 
plt.plot(hist_sgd.history['loss']) 
plt.plot(hist_sgdnesterov.history['loss']) 
plt.plot(hist_sgdmomentum.history['loss'])
plt.plot(hist_sgdnestmomentum.history['loss'])
plt.plot(hist_adagrad.history['loss'])
plt.plot(hist_adadelta.history['loss'])
plt.plot(hist_adamax.history['loss'])
plt.title('train loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['rmsprop', 'adam', 'nadam', 'sgd', 'sgd_w/nesterov', 'sgd_w/momentum', 'sgd_w/nesterov+momentum', 'adagrad', 'adadelta', 'adamax'], loc='upper right',fontsize = 'x-small')  

plt.show()


# In[ ]:





# # Showing just the best and the worst performing after some observations

# In[ ]:


plt.figure(figsize=(10, 6))  

#plt.axis((-1,14,0.4, 1))


plt.plot(hist_nadam.history['val_acc'])
plt.plot(hist_sgd.history['val_acc'])
plt.plot(hist_sgdmomentum.history['val_acc'])

plt.plot(hist_adamax.history['val_acc'])
plt.title('test accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['nadam', 'sgd', 'sgd_w/momentum', 'adamax'], loc='lower right',fontsize = 'x-small')  

plt.show()


# In[ ]:


plt.figure(figsize=(10, 6))  

#plt.axis((-1,14,0.4, 1))
  
plt.plot(hist_nadam.history['acc']) 
plt.plot(hist_sgd.history['acc']) 
plt.plot(hist_sgdmomentum.history['acc'])

plt.plot(hist_adamax.history['acc'])
plt.title('train accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend([ 'nadam', 'sgd', 'sgd_w/momentum', 'adamax'], loc='lower right',fontsize = 'x-small')  

plt.show()


# In[ ]:


plt.figure(figsize=(10, 6))  


 
plt.plot(hist_nadam.history['val_loss']) 
plt.plot(hist_sgd.history['val_loss']) 
plt.plot(hist_sgdmomentum.history['val_loss'])
plt.plot(hist_adamax.history['val_loss'])
plt.title('test loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend([ 'nadam', 'sgd', 'sgd_w/momentum', 'adamax'], loc='upper right',fontsize = 'x-small')  

plt.show()


# In[ ]:


plt.figure(figsize=(10, 6))  
#plt.axis((-1,14,0, 1))
 
plt.plot(hist_nadam.history['loss']) 
plt.plot(hist_sgd.history['loss']) 
plt.plot(hist_sgdmomentum.history['loss'])
plt.plot(hist_adamax.history['loss'])
plt.title('train loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend([ 'nadam', 'sgd', 'sgd_w/momentum', 'adamax'], loc='upper right',fontsize = 'x-small')  

plt.show()


# In[ ]:





# # Now a deeper model with better hyperparamters

# In[ ]:


from keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)


cnndeepmodel = Sequential()
# first convolution layer
cnndeepmodel.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3))) 
cnndeepmodel.add(BatchNormalization())
cnndeepmodel.add(MaxPooling2D(pool_size=2))
cnndeepmodel.add(Dropout(0.25))

#second convolution  layer
cnndeepmodel.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
cnndeepmodel.add(BatchNormalization())
cnndeepmodel.add(MaxPooling2D(pool_size=2))
cnndeepmodel.add(Dropout(0.5))

#Third convolution  layer
cnndeepmodel.add(Conv2D(64, kernel_size=2 ,padding="same",activation='relu'))
cnndeepmodel.add(BatchNormalization())
cnndeepmodel.add(MaxPooling2D(pool_size=2))
cnndeepmodel.add(Dropout(0.5))

#first Fully connected layer
cnndeepmodel.add(Flatten()) 
cnndeepmodel.add(Dense(256,kernel_regularizer=l2(0.001)))#activity_regularizer=l1(0.001)))
cnndeepmodel.add(BatchNormalization())
cnndeepmodel.add(Activation('relu')) 
cnndeepmodel.add(Dropout(0.5))      

#Final Fully connected layer
cnndeepmodel.add(Dense(2)) #8
cnndeepmodel.add(Activation('softmax')) 

cnndeepmodel.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

cnndeepmodel.summary()

cnndeephistory = cnndeepmodel.fit(X_train, Y_train, epochs=300,verbose=1,validation_data=(X_test, Y_test),
                                  shuffle=True,callbacks=[reduce_lr])

cnnscore = cnndeepmodel.evaluate(X_test, Y_test, verbose=0)

#loss and accuracy
print('Test loss:', cnnscore[0])
print('Test accuracy:', cnnscore[1])

