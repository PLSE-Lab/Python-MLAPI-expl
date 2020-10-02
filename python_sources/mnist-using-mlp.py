# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv("../input/train.csv",encoding='utf-8')
test = pd.read_csv("../input/test.csv",encoding='utf-8')

#Checkout the Data

print('Training data shape : ', train.shape)
print('Testing data shape : ', test.shape)

#convert to array
df_y = train.iloc[:,0].values.astype('int32') 
df_x = train.iloc[:,1:].values.astype('float32') 

test_x = test.values.astype('float32')

# Find the unique numbers from the train labels
classes = np.unique(df_y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

#Reshape only to plot the digits
train_images = df_x.reshape(df_x.shape[0], 28, 28)
test_images = test_x.reshape(test.shape[0], 28, 28)
 
plt.figure(figsize=[10,5])
 
# Display the first image in training data
plt.subplot(121)
plt.imshow(train_images[0,:,:], cmap='gray')
plt.title("True value : {}".format(df_y[0]))
 
# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_images[0,:,:], cmap='gray')

#free the memory
del train_images
del test_images
del test
del train

#Process the data
 
# Scale the data to lie between 0 to 1
df_x = df_x/255
test_x = test_x/255

# Change the labels from integer to categorical data
from keras.utils import to_categorical
df_y_one_hot = to_categorical(df_y)

 
# Display the change for category label using one-hot encoding
print('Original label 0 : ', df_y[0])
print('After conversion to categorical ( one-hot ) : ', df_y_one_hot[0])

del df_y

#split the training dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_x,df_y_one_hot,
                                                    test_size=0.2,random_state=0)

#Create the Network

# 784 (28*28) input node >> 512 hidden >> 512 hidden >> 10 output
from keras.models import Sequential
from keras.layers import Dense

dimData = X_train.shape[1]

'''
#Model without regularization
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(nClasses, activation='softmax'))

#Configure the Network
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model
mlp = model.fit(X_train, y_train, batch_size=256, epochs=20, verbose=1, 
                   validation_data=(X_test, y_test))

#Accuracy on test data.
[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(mlp.history['loss'],'r',linewidth=3.0)
plt.plot(mlp.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(mlp.history['acc'],'r',linewidth=3.0)
plt.plot(mlp.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

# We can see that there is some overfiting in the training . 
# The validation loss initially decrease, but then it starts increasing gradually
# there is a sharp difference between the accuracy of the trainingg and the validation.

'''

# Regularise the model and use advanced activation 
# softmax, relu, sigmoid, LeakyReLU, PReLU

from keras.layers import Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU

# advanced_act = PReLU(init='zero', weights=None)
advanced_act = LeakyReLU(alpha=.003)
model_reg = Sequential()
model_reg.add(Dense(512, activation='linear', input_shape=(dimData,)))
model_reg.add(advanced_act)
model_reg.add(Dropout(0.6))
model_reg.add(Dense(512, activation='linear'))
model_reg.add(advanced_act)
model_reg.add(Dropout(0.6))
model_reg.add(Dense(nClasses, activation='softmax'))


#rmsprop , Adam, SGD
model_reg.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
mlp_reg = model_reg.fit(X_train, y_train, batch_size=256, epochs=20, verbose=2,
                        validation_data=(X_test, y_test))
 
#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(mlp_reg.history['loss'],'r',linewidth=3.0)
plt.plot(mlp_reg.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(mlp_reg.history['acc'],'r',linewidth=3.0)
plt.plot(mlp_reg.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)


# Predict the most likely class
test_y_pred = model_reg.predict_classes(test_x)


#Create the excel for submission
submission = pd.concat([pd.Series(range(1,len(test_y_pred)+1),name = "ImageId"),pd.Series(test_y_pred, name="Label")],axis = 1)
submission.to_csv("mnist_nn.csv",index=False)