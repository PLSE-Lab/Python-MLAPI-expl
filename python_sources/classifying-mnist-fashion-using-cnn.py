#!/usr/bin/env python
# coding: utf-8

# # Classifying Mnist Fashion Using CNN
# By : Hesham Asem
# 
# ______
# 
# 
# let's build Conv2D Neural Network , to classify tens of thousands of mnist Fashion images . . 
# 
# Data File  :https://www.kaggle.com/zalando-research/fashionmnist
# 
# first to import needed libraries
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten ,Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau


# then read the data 

# In[ ]:


train_data = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")
test_data = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")

print(f'Training Data size is : {train_data.shape}')
print(f'Test Data size is : {test_data.shape}')


# how it looks like ? 

# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# _____
# 
# # Data Processing
# 
# then we can define X & y data for training

# In[ ]:


X = train_data.drop(['label'], axis=1, inplace=False)
y = train_data['label']

print('X shape is ' , X.shape)
print('y shape is ' , y.shape)


# and for testing

# In[ ]:


X_test = test_data.drop(['label'], axis=1, inplace=False)
y_test = test_data['label']

print('X shape is ' , X_test.shape)
print('y shape is ' , y_test.shape)


# let's have a look to a random 20 numbers 

# In[ ]:


plt.figure(figsize=(12,10))
plt.style.use('ggplot')
for i in  range(20)  :
    plt.subplot(4,5,i+1)
    plt.imshow(X.values[ np.random.randint(1,X.shape[0])].reshape(28,28) , cmap='gray')
    


# we also need to be sure that output numbers are kinda equally distributed for training data

# In[ ]:


y.value_counts()


# and for testing data

# In[ ]:


y_test.value_counts()


# Ok great , let's make a pie chart for training output

# In[ ]:


plt.figure(figsize=(12,12))
plt.pie(y.value_counts(),labels=list(y.value_counts().index),autopct ='%1.2f%%' ,
        labeldistance = 1.1,explode = [0.05 for i in range(len(y.value_counts()))] )
plt.show()


# ____
# 
# # Dimension Adjusting
# 
# it;s very important to adjust dimensions for data before building the CNN , let's first normalize both traing & test data . 
# 
# ofcourse y will not be normalized or it will mislead the training

# In[ ]:


X = X / 255.0
X_test = X_test / 255.0


# now how train data dimension looks like

# In[ ]:


X.shape


# and test data

# In[ ]:


X_test.shape


# then we need to reshape them , to be 4 dimensions , so first dimension will be open for all sample size , then 28 x 28 as image size , then 1

# In[ ]:


X = X.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)


# now how X dimension looks like

# In[ ]:


X.shape


# and test data ? 

# In[ ]:


X_test.shape


# also we have to categorize y , to convert single numbers like (7) into One Hot Matrix like [0 0 0 0 0 0 1 0 0 0]

# In[ ]:


ohe  = OneHotEncoder()
y = np.array(y)
y = y.reshape(len(y), 1)
ohe.fit(y)
y = ohe.transform(y).toarray()


# now how y looks like ? 

# In[ ]:


y.shape


# then we'll do it again to test output data

# In[ ]:


ohe  = OneHotEncoder()
y_test = np.array(y_test)
y_test = y_test.reshape(len(y_test), 1)
ohe.fit(y_test)
y_test = ohe.transform(y_test).toarray()


# and check its dimension

# In[ ]:


y_test.shape


# ____
# 
# # Data Splitting .
# 
# we have to split our train data ,  to get cross-validation data , and train data
# 

# In[ ]:


X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.15, random_state=44, shuffle =True)

print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_cv.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_cv.shape)


# ____
# 
# # Conv2D Model
# 
# now we can build our model , which will contain Conv layer then Maxpooling then normalize it 
# 
# then second layer contain Conv then Max then normalize
# 
# then drop it out with 50 %
# 
# then Flatten it 
# 
# then drop it out , then a FC with 64 units , then drop out , then last FC output layer 

# In[ ]:


KerasModel = keras.models.Sequential([
        keras.layers.Conv2D(filters = 32, kernel_size = (3,3),  activation = tf.nn.relu , padding = 'same'),
        keras.layers.MaxPool2D(pool_size=(2,2), strides=None, padding='valid'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=32, kernel_size=(2,2),activation = tf.nn.relu , padding='same'),
        keras.layers.MaxPool2D(),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),        
        keras.layers.Flatten(),    
        keras.layers.Dropout(0.5),        
        keras.layers.Dense(64),    
        keras.layers.Dropout(0.3),            
        keras.layers.Dense(units= 10,activation = tf.nn.softmax ),                

    ])


# complie the model using adam optimizer & loss function : categorical crossentropy , since it's multilassifier

# In[ ]:


KerasModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


# now we can start training for 200 epochs . 

# In[ ]:


epochs_number = 200
hist = KerasModel.fit(X_train,y_train,validation_data=(X_cv, y_cv),epochs=epochs_number,batch_size=64,verbose=1)


# how is the final loss & accuracy

# In[ ]:


score = KerasModel.evaluate(X_test, y_test, verbose=0)
score


# greart . how it looks like now ? 

# In[ ]:


KerasModel.summary()


# how is accuracy for test data , which never seen by the model yet

# In[ ]:


ModelLoss, ModelAccuracy = KerasModel.evaluate(X_test, y_test)

print('Test Loss is {}'.format(ModelLoss))
print('Test Accuracy is {}'.format(ModelAccuracy ))


# a good accuracy , and might increase if we make more epochs
# 
# _____
# 
# now let's have a look to chart of epochs-accuracy , to know if we should do more epochs or we shpuld stop earlier
# 
# first to calculate history accuracy values

# In[ ]:


ModelAcc = hist.history['acc']
ValAcc = hist.history['val_acc']
LossValue = hist.history['loss']
ValLoss = hist.history['val_loss']
epochs = range(len(ModelAcc))


# then draw Training accuracy with epocs

# In[ ]:


plt.plot(range(1,epochs_number+1),ModelAcc, 'ro', label='Accuracy of Training ')
plt.plot(range(1,epochs_number+1), ValAcc, 'r', label='Accuracy of Validation')
plt.title('Training Vs Validation Accuracy')
plt.legend()
plt.figure()


# ok , looks like more epochs might be needed for the model , which will increase its accuracy 
# 
# how about loss value

# In[ ]:


plt.plot(range(1,epochs_number+1), LossValue, 'ro', label='Loss of Training ')
plt.plot(range(1,epochs_number+1), ValLoss, 'r', label='Loss of Validation')
plt.title('Training Vs Validation loss')
plt.legend()
plt.show()


# again , more epochs here will decrease the loss in the model 
# 
# _____
# 
# now to predict X Test

# In[ ]:


y_pred = KerasModel.predict(X_test)

print('Prediction Shape is {}'.format(y_pred.shape))


# let's check random 20 samples , & we need to have a look to any mismatch images , to see why it confused

# In[ ]:


for i in list(np.random.randint(0,len(X_test) ,size= 20)) : 
    print(f'for sample  {i}  the predicted value is   {np.argmax(y_pred[i])}   , while the actual letter is {np.argmax(y_test[i])}')
    if np.argmax(y_pred[i]) != np.argmax(y_test[i]) : 
        print('==============================')
        print('Found mismatch . . ')
        plt.figure(figsize=(5,5))
        plt.style.use('ggplot')
        plt.imshow(X_test[i].reshape(28,28))
        plt.show()
        print('==============================')


# ____
# 
# ok no mismatch found . 
# 
# 
