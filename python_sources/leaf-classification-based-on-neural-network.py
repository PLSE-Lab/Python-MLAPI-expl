#!/usr/bin/env python
# coding: utf-8

# [<center><h1>Kaggle: Leaf Classification</h1></center>](https://www.kaggle.com/c/leaf-classification)

# 
# Public Score :  0.15218<br>
# Private Score : 0.15218<br>
# As it's a regression problem I better be solving this without deep learning model. Whatever, I'm implementing neural network here.

# Get the path of training and testing set

# In[ ]:


import os
for dirname,_,filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname,filename))


# Measure the elapsed time of the whole procedure, which starts with "start" and ends with "end" function.

# In[ ]:


import time
start = time.time()


# In[ ]:


## Importing standard libraries
get_ipython().run_line_magic('pylab', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


## Importing sklearn libraries

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing


# In[ ]:


## Keras Libraries for Neural Networks

from keras.models import Sequential
from keras.layers import merge
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers import Convolution2D, Convolution1D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping


# In[ ]:


## Read data from the CSV file
parent_data = pd.read_csv('/kaggle/input/train.csv.zip')
data = parent_data.copy()   
data.pop('id')


# In[ ]:


## read test file
test = pd.read_csv('/kaggle/input/test.csv.zip')
testId = test.pop('id')


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.describe()


# Separate "species" column to set label, which is why we can't see any column numbers left after calling shape method

# The species are in string format, as such to convert it to one hot encoding format, first I have to labelling it in assistance with labelencoder.

# In[ ]:


## Since the labels are textual, so we encode them categorically
species_label = data.pop('species')
species_label = LabelEncoder().fit(species_label).transform(species_label)
print(species_label.shape)


# After one hot encoding, we have got 99 different species.

# In[ ]:


## We will be working with categorical crossentropy function
## It is required to further convert the labels into "one-hot" representation
# from keras import utils as np_utils
one_hot = to_categorical(species_label)
print(one_hot.shape)


# Most of the learning algorithms are prone to feature scaling <br>
# Standardising the data to give zero mean =)

# In[ ]:


preprocessed_train_data = preprocessing.MinMaxScaler().fit(data).transform(data)
preprocessed_train_data = StandardScaler().fit(data).transform(data)

print(preprocessed_train_data.shape)


# In[ ]:


## we need to perform the same transformations from the training set to the test set
test = preprocessing.MinMaxScaler().fit(test).transform(test)
test = StandardScaler().fit(test).transform(test)


# Use stratifiedshufflesplit, as there exists subspecies in the image set

# In[ ]:


sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2,random_state=12345)
train_index, val_index = next(iter(sss.split(preprocessed_train_data, one_hot)))

x_train, x_val = preprocessed_train_data[train_index], preprocessed_train_data[val_index]
y_train, y_val = one_hot[train_index], one_hot[val_index]

print("x_train dim: ",x_train.shape)
print("x_val dim:   ",x_val.shape)


# column is the input dimension

# 192 is the column size , glorot normal is used for text based cnn. 768 is chosen arbitrarily.

# In[ ]:


model = Sequential()

model.add(Dense(768,input_dim=192,  kernel_initializer='glorot_normal', activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(768, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(99, activation='softmax'))

model.summary()


# In[ ]:


## Adagrad, rmsprop, SGD, Adadelta, Adam, Adamax, Nadam

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics = ["accuracy"])


# In[ ]:


get_ipython().run_cell_magic('time', '', "early_stopping = EarlyStopping(monitor='val_loss', patience=300)\n\nhistory = model.fit(x_train, y_train,batch_size=192,epochs=2500 ,verbose=1,\n                    validation_data=(x_val, y_val),callbacks=[early_stopping])")


# In[ ]:


## we need to consider the loss for final submission to leaderboard
## print(history.history.keys())
print('val_acc: ',max(history.history['val_acc']))
print('val_loss: ',min(history.history['val_loss']))
print('train_acc: ',max(history.history['acc']))
print('train_loss: ',min(history.history['loss']))

print()
print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))


# In[ ]:


## summarize history for loss
## Plotting the loss with the number of iterations
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('model_loss.png')


# In[ ]:


## Plotting the error with the number of iterations
## With each iteration the error reduces smoothly
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('model_accuracy.png')


# In[ ]:


yPred = model.predict_proba(test)


# In[ ]:


## Converting the test predictions in a dataframe as depicted by sample submission
submission = pd.DataFrame(yPred,index=testId,columns=sort(parent_data.species.unique()))


# remove index = False from to_csv

# In[ ]:


submission.to_csv('leafClassificationSubmission.csv')

## print run time
end = time.time()
print(round((end-start),2), "seconds")


# ---------
# 
# Earlier` we used a 4 layer network but the result came out to be overfitting the test set. We dropped the count of neurones in the network and also restricted the number of layers to 3 so as to keep it simple.
# Instead of submitting each test sample as a one hot vector we submitted each samples as a probabilistic distribution over all the possible outcomes. This "may" help reduce the penalty being exercised by the multiclass logloss thus producing low error on the leaderboard! ;)
# Any suggestions are welcome!

# In[ ]:


submission.head()

