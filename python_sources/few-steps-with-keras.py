#!/usr/bin/env python
# coding: utf-8

# # Few steps with keras

# This notebook is divided into below 4 section:
# 1. Introduction
# 2. Data read, preparation and explore
# 3. RandomForest GridSearch
# 4. Model test on validation set
# 5. Result Submission
# 

# # 1. Introduction
# 
# This is the simplest demonstration of keras model using tensorflow as backend. 
# I achieved accuracy of 97% which is not very great but its still descent. Moreover the model is built with default options so it still have scope for improvement. 
# 
# Initially, my accuracy was not that good and it did not improved beyond 85% despite multiple epochs. Then I tried with below options and accuracy improved to current levels.
# 
# 1. Hot-encoding the labels. 
# 
# 2. Scaling X variables. 
# 
# 3. Trying different optimisers
# 
# 4. Increasing epochs
# 
# 5. Adjusting dropout rate
# 

# Data used from kaggle competition link: https://www.kaggle.com/c/digit-recognizer/

# ### Import libraries

# In[ ]:


# Basic libraries
import numpy as np
import pandas as pd

# model libraries
from sklearn.model_selection import train_test_split
import tensorflow
import keras
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import KFold
#from sklearn.model_selection import GridSearchCV

# model accuracy check
from sklearn import metrics

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns


# # 2. Data read, preparation and explore

# ### Read data files

# In[ ]:


# read the data
train_file = '../input/digit-recognizer/train.csv'
test_file = '../input/digit-recognizer/test.csv'

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)


# ### Basic data check

# In[ ]:


df_train.head()


# In[ ]:


df_train.shape


# Each digit image is of 28x28 size which is total of 784 pixels. These pixel values are stored in columns pixel0....pixel785 (independent variable) for a digit. 
# Column label contains the corresponding digit (dependent variable).

# In[ ]:


# columns pixel0....pixel785 are independent variable of a digit
# column label contains the digit (dependent variable)

df_train.columns


# Test data set contains 784 pixel values for a digit. It does not contain the label.
# 
# Need to indentify the digit based on the 784 pixels. 

# In[ ]:


df_test.shape


# In[ ]:


df_test.columns


# ### Explore Train Dataset

# In[ ]:


# no null values in train dataset

df_train.isnull().values.any()


# In[ ]:


df_train.info()


# In[ ]:


# print the frequency of each label

print(df_train['label'].value_counts())
sns.countplot(df_train['label'])


# From train set, display few initial images and corresponding labels:

# In[ ]:


plt.figure(figsize=(12,4))
for i in range(30):  
    plt.subplot(3, 10, i+1)
    plt.imshow(df_train.drop(['label'],axis=1).values[i].reshape(28,28) )
    plt.axis('off')
plt.show()

# print corresponding labels:
print(list(df_train['label'].loc[0:9]))
print(list(df_train['label'].loc[10:19]))
print(list(df_train['label'].loc[20:29]))


# From test, display few initial images. Need to predict the lable for these. 

# In[ ]:


plt.figure(figsize=(12,4))
for i in range(30):  
    plt.subplot(3, 10, i+1)
    plt.imshow(df_test.values[i].reshape(28,28) )
    plt.axis('off')
plt.show()


# # 3. keras model building
# 
# Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. 
# 
# Being able to go from idea to result with the least possible delay is key to doing good research.
# 

# ### Perform train-test split
# 
# Train data set is divided in 80:20 ratio for train/test
# 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df_train.drop(['label'],axis=1),
                                                   df_train['label'],
                                                   test_size = 0.2,
                                                   random_state=13)


# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# ### keras Model
# 
# 

# In[ ]:


# y_train = keras.utils.to_categorical(y_train)
# y_test = keras.utils.to_categorical(y_test)

n_cols = X_train.shape[1]
print("Number of input columns: {0}".format(n_cols))

n_features = len(y_train.unique())
print("Number of output features: {0}".format(n_features))


# In[ ]:


# let's convert labels to categorical variable
y_train = keras.utils.to_categorical(y_train, n_features)
y_test = keras.utils.to_categorical(y_test, n_features)


# In[ ]:


y_train, y_test


# In[ ]:


# Let's do the same with X variables. 
# data has pixels with max values of 255. So will divide values with 255 to scale the data
X_train = X_train / 255
X_test = X_test / 255


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import SGD

m = Sequential()
m.add(Dense(512,activation='relu',input_shape=(n_cols,)))
m.add(Dropout(0.5))
m.add(Dense(256,activation='relu'))
m.add(Dropout(0.5))
m.add(Dense(128,activation='relu'))
m.add(Dropout(0.5))
m.add(Dense(64,activation='relu'))
m.add(Dropout(0.5))
m.add(Dense(n_features,activation='softmax'))

m.summary()


# In[ ]:


m.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#m.compile(optimizer=SGD(),loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


n_batch_size = 256
n_epochs = 200
history = m.fit(X_train, y_train, batch_size=n_batch_size, epochs=n_epochs, validation_data=(X_test,y_test))


# ### Validation accuracy

# In[ ]:


plt.plot(history.history['val_acc'],'b')
plt.plot(history.history['acc'],'r')


# ### Validation loss

# In[ ]:


plt.plot(history.history['val_loss'],'b')
plt.plot(history.history['loss'],'r')


# # 4. Model test on validation set (y_test)
# 
# WIll run it on X_test and result set would be validated against y_test

# In[ ]:


y_pred = np.round(m.predict(X_test)).astype('int64')
y_pred


# In[ ]:


# remove the categories from y_pred
# select the indix with the maximum probability
y_pred1 = np.argmax(y_pred,axis = 1)
y_pred1


# In[ ]:


# do the same for y_test
# select the index with the maximum probability
y_test1 = np.argmax(y_test,axis = 1)
y_test1


# In[ ]:


# print the frequency of each label

y = pd.DataFrame(y_pred1)
y.columns=['label']
print(y['label'].value_counts())

y[y['label'] < 0] = 0
y[y['label'] > 9] = 9

sns.countplot(y['label'])


# In[ ]:


print('Accuracy score for y_test: ', metrics.accuracy_score(y_test1,y_pred1))


# In[ ]:


pd.DataFrame(metrics.confusion_matrix(y_test1,y_pred1))


# In[ ]:


# combine actual and predicted in a single df
X_test['actual'] = y_test1
X_test['pred'] = y_pred1


# In[ ]:


X_test_err = X_test[X_test['actual'] != X_test['pred']]
print(X_test_err.shape[0],"predictions are wrong")


# Many predictions went wrong. Lets draw few of them. 

# In[ ]:


for i in range (10):
    act=X_test_err['actual'].values[i]
    prd=X_test_err['pred'].values[i]
    print("actual {0} ; predicted {1}".format(act,prd))
    plt.figure(figsize=(1,1))
    plt.imshow(X_test_err.drop(['actual','pred'], axis=1).values[i].reshape(28,28))
    plt.axis("off")
    plt.show()


# Few of the images are easy but tricky. 
# 
# Model should be trained further with augmented images to achieve better accuracy

# # 5. Final Submission

# Finally, time to run model on required test set and sumbit the result on Kaggle

# ### Run on test data (df_test) and display few test images

# In[ ]:


# # normalize the input data
df_test = df_test / 255
pred = m.predict(df_test)
pred


# In[ ]:


# select the index with the maximum probability
pred = np.argmax(pred,axis = 1)
pred


# Display few images and their respective predicted labels

# In[ ]:



for i in range(11):  
    print('Prediction {0}'.format(pred[i]))
    plt.figure(figsize=(1,1))
    plt.imshow(df_test.values[i].reshape(28,28) )
    plt.axis('off')
    plt.show()


# Model is doing a good job, but fails for distorted images. 

# In[ ]:


pred = pd.Series(pred,name="Label")


# In[ ]:


submit = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pred],axis = 1)

submit.to_csv("cnn_mnist_fewsteps_keras.csv",index=False)


# Building models on personal computer has its own limitations. Like in this case, training with more epochs and using more custom parameters will eat into computer resources. I plan to run GridSearch with more customized parameters option on AWS cloud solution and let it take time that it needs.

# I end this notebook here with leaving further scope of improvement to keras. Share your improvement ideas in comments. If you found this notebook helpful or you just liked it, some upvotes would be very much appreciated. It will keep me motivated :)
# 
# Thanks for visiting. 

# In[ ]:




