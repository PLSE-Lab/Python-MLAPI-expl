#!/usr/bin/env python
# coding: utf-8

# Get accuracy >98 on first run.. (recommended not to run on Kaggle Kernal Platform)

# In[2]:



from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.callbacks import *

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# In[3]:



dataFile = "../input/train.csv"
queFile = "../input/test.csv"


# In[4]:



batch_size = 128
nb_classes = 10
nb_epoch = 150 #use 15

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (28,28,1)
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


# In[5]:



print ("^^^INFO: Fix random seed^^^")

seed = 7
np.random.seed(seed)

print ("^^^INFO: Load dataset^^^")
# dataset = numpy.loadtxt("diabetic_data_1000V2.csv", delimiter=",",comments="#")
dataset = np.genfromtxt(dataFile, delimiter=",", comments="#",skip_header=1)

# mask = np.any(np.isnan(dataset), axis=0)
# dataset = dataset[:,~mask]

print ("^^^INFO: Shape of dataset^^^")
print (dataset.shape)

# split into input (X) and output (Y) variables
X = dataset[:, 1:dataset.shape[1]]
Y = dataset[:, 0]

print ("^^^INFO: reShape dataset^^^")

from keras.utils.np_utils import to_categorical
X = X.reshape(dataset.shape[0],28,28,1)
Y = to_categorical(Y)

print ("^^^INFO: Shape of X^^^")
print (X.shape)
print ("^^^INFO: Shape of Y^^^")
print (Y.shape)

print ("^^^INFO: Define Model^^^")


# In[6]:



# create model
model = Sequential()

model.add(Convolution2D(nb_filters,kernel_size[0], kernel_size[1],
                        border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


# In[12]:


from sklearn.model_selection import train_test_split
x_tr,x_val,y_tr,y_val= train_test_split(X,Y, test_size=0.3,stratify=Y)


# In[13]:


es = EarlyStopping(min_delta=0.00001,patience=15,verbose=2)
cp = ModelCheckpoint("bst_model_wts",save_best_only=True)
rlop = ReduceLROnPlateau(patience=5,factor=0.3)


# In[14]:


#this will take around 30 minutes on 2 processors

print ("^^^INFO: Compile Model^^^")
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy', 'mse'])

print ("^^^INFO: Fit Model^^^")
history = model.fit(x_tr, y_tr, epochs=nb_epoch, batch_size=420, verbose=1,validation_data=(x_val, y_val),callbacks=[es,cp,rlop])

print ("^^^INFO: Evaluate Model^^^")
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


# In[11]:



# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# axes = plt.gca()
# axes.set_xlim([0,120])
# axes.set_ylim([90,100])
plt.savefig('acc.png')  # save the figure to file
plt.show()
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')
plt.show()


# In[ ]:


print ("^^^INFO: Load dataset for Prediction^^^")
queset_raw = np.genfromtxt(queFile, delimiter=",", comments="#")
queset_raw=queset_raw[1:,:]
print(queset_raw.shape)

queset = queset_raw.reshape(queset_raw.shape[0],28,28,1)
print(queset.shape)

print ("^^^INFO: Making Prediction^^^")
pred_Y = model.predict_classes(queset)

print ("^^^INFO: Making Prediction Index^^^")
n = list(range(1, pred_Y.shape[0] + 1))

print ("^^^INFO: Concat Prediction & Index^^^")
print(pred_Y.shape)
print(len(n))
result = np.c_[n, pred_Y]

print ("^^^INFO: Add Label Prediction^^^")
print(result.shape)
result = np.r_[[['ImageId', 'Label']], result]

print ("^^^INFO: Prediction to result.csv^^^")
print(result.shape)
# result.tofile('result.csv',sep=',')
np.savetxt('result.csv', result, delimiter=',', fmt='%s')


# In[ ]:




