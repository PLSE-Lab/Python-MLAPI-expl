#!/usr/bin/env python
# coding: utf-8

# # CNN for Digit Recognizer
# By : Hesham Asem
# 
# ______
# 
# 
# let's build Conv2D Neural Network , to classify tens of thousands of numbers which is in hand written format . . 
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
from keras.preprocessing.image import ImageDataGenerator


# then read the data 

# In[ ]:


train_data = pd.read_csv("../input/digit-recognizer/train.csv")
test_data = pd.read_csv("../input/digit-recognizer/test.csv")

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
# then we can define X & y data 

# In[ ]:


X = train_data.drop(['label'], axis=1, inplace=False)
y = train_data['label']

print('X shape is ' , X.shape)
print('y shape is ' , y.shape)


# let's have a look to a random 20 numbers 

# In[ ]:


plt.figure(figsize=(12,10))
plt.style.use('ggplot')
for i in  range(20)  :
    plt.subplot(4,5,i+1)
    plt.imshow(X.values[ np.random.randint(1,X.shape[0])].reshape(28,28))


# we also need to be sure that output numbers are kinda equally distributed

# In[ ]:


y.value_counts()


# Ok great , let's make a pie chart for it

# In[ ]:


plt.figure(figsize=(12,12))
plt.pie(y.value_counts(),labels=list(y.value_counts().index),autopct ='%1.2f%%' ,
        labeldistance = 1.1,explode = [0.05 for i in range(len(y.value_counts()))] )
plt.show()


# ____
# 
# # Dimension Adjusting
# 
# it;s very important to adjust dimensions for data before building the CNN , let;s first normalize both X & test data . 
# 
# ofcourse y will not be normalized or it will mislead the training

# In[ ]:


X = X / 255.0
test_data = test_data / 255.0


# ow how X dimension looks like

# In[ ]:


X.shape


# then we need to reshape them , to be 4 dimensions , so first dimension will be open for all sample size , then 28 x 28 as image size , then 1

# In[ ]:


X = X.values.reshape(-1,28,28,1)
test_data = test_data.values.reshape(-1,28,28,1)


# now how X dimension looks like

# In[ ]:


X.shape


# and test data ? 

# In[ ]:


test_data.shape


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


# ____
# 
# # Data Splitting .
# 
# we have to split our data twice , first to get cross-validation data , then to get test data
# 
# so first we'll get X_part, X_cv, y_part, y_cv , then later we'll divide "part" into training & testing data

# In[ ]:


X_part, X_cv, y_part, y_cv = train_test_split(X, y, test_size=0.15, random_state=44, shuffle =True)

print('X_train shape is ' , X_part.shape)
print('X_test shape is ' , X_cv.shape)
print('y_train shape is ' , y_part.shape)
print('y_test shape is ' , y_cv.shape)


# great , now we'll split part into train & test data , so we can test the accuracy percisely

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_part, y_part, test_size=0.25, random_state=44, shuffle =True)

print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)


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


# now we can start training for 8 epochs , without exceeding to avoid any OF

# In[ ]:


KerasModel.fit(X_train,y_train,validation_data=(X_cv, y_cv),epochs=8,batch_size=64,verbose=1)


# greart . how it looks like now ? 

# In[ ]:


KerasModel.summary()


# how is accuracy for test data , which never seen by the model yet

# In[ ]:


ModelLoss, ModelAccuracy = KerasModel.evaluate(X_test, y_test)

print('Test Loss is {}'.format(ModelLoss))
print('Test Accuracy is {}'.format(ModelAccuracy ))


# 97% , good enough , & might be better if we increase epochs little bit
# 
# now to predict the results

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
# now to submit the results 

# In[ ]:


FinalResults = KerasModel.predict(test_data)
FinalResults = pd.Series(np.argmax(FinalResults,axis = 1) ,name="Label")

FileSubmission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),FinalResults],axis = 1)
FileSubmission.to_csv("sample_submission.csv",index=False)


# 
