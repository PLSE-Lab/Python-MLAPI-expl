#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ### import data, using Pandas

# In[ ]:


dftrain = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")


# ### feature manipulation

# In[ ]:


Xcheck=np.array(dftrain.iloc[:,1:])
ycheck=np.array(dftrain.iloc[:,0])

# scale the X vector, dividing for the maximum value (i.e. 255)
X=Xcheck/255

# convert y in dummy variable. This is necessary as output of the neural network
y=np.array(pd.get_dummies(ycheck))


# ### image visualization

# In[ ]:


# reshape of the vector, to visualize the image

def numimg2(t,X):
    vecreshape=np.reshape(X[t,:],[28,28])
    return vecreshape


# In[ ]:


# visuale a number

print(ycheck[5016])

plt.imshow(numimg2(5016,X),cmap='gray')
plt.colorbar()
plt.show()


# ### Neural Network: building and training

# In[ ]:


# beginning of deep learning

ncols=X.shape[1]       # number of elements for the input layer
numutput=10            # number of elements for output layer


# In[ ]:


# I use train_test_split function to split in 80% training data and 20% test data

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[ ]:


import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from keras.layers.core import Dense, Dropout, Activation


# ### Sequential Neural Network, with 2 hidden layers

# In[ ]:



model = Sequential()

# input and 1th hidden layer with 100 neurons. 
model.add(Dense(500, input_shape=(ncols,)))
model.add(Activation('relu'))                            
model.add(Dropout(0.2))   # setting 20% of neurons to 0 (randomly). This should help in avoiding overfitting

# 2th hidden layer with 200 neurons. 
model.add(Dense(500))
model.add(Activation('relu'))                            
model.add(Dropout(0.2))

# output layer
model.add(Dense(10))
model.add(Activation('softmax'))

# Here I compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# I set 100 epochs as default, with an Early Stop in case of no improvement after 5 epochs
early_stopping_monitor = EarlyStopping(patience=5)

history = model.fit(X_train,y_train,
                    batch_size=128, epochs=100,
                    validation_data=(X_test,y_test),callbacks=[early_stopping_monitor])

print("Loss function: " + model.loss)


# In[ ]:


model.summary()


# ### Neural Network: performance on the validation test

# In[ ]:


# importing the validation dest
dftest = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

X_val_test=np.array(dftest.iloc[:,1:])/255
y_val_test=np.array(dftest.iloc[:,0])

#prediction of the model (probability)
result=np.round(model.predict(X_val_test),1)

#prediction of the model (I select the number with the maximum probability)
prediction=np.argmax(result,axis=1)

# build the confusion matrix
pd.DataFrame(confusion_matrix(y_val_test,prediction))


# In[ ]:


accuracy=round(np.sum(y_val_test==prediction)/len(prediction)*100,1)

print('The accuracy of the model on the validation data is: '+str(accuracy)+'%')

