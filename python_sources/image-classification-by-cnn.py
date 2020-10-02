#!/usr/bin/env python
# coding: utf-8

# # Image Classification Using CNN
# By : Hesham Asem
# 
# ________
# 
# we'll build a CNN using Keras to use it classifying thousands of pictures in six different categories
# 
# Data link : https://www.kaggle.com/puneet6060/intel-image-classification
# 
# first to import libraries
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="whitegrid")
import os
import glob as gb
import cv2
import tensorflow as tf
import keras


# now to define the path ( to swtich it between jupyter notebook & kaggle kernel)

# In[ ]:


### for Kaggle
trainpath = '../input/intel-image-classification/seg_train/'
testpath = '../input/intel-image-classification/seg_test/'
predpath = '../input/intel-image-classification/seg_pred/'

### for Jupyter
# trainpath = ''
# testpath = ''
# predpath = ''


# # Open Folders
# 
# now let's first check the Train folder to have a look to its content

# In[ ]:


for folder in  os.listdir(trainpath + 'seg_train') : 
    files = gb.glob(pathname= str( trainpath +'seg_train//' + folder + '/*.jpg'))
    print(f'For training data , found {len(files)} in folder {folder}')


# ok , how about the test folder

# In[ ]:


for folder in  os.listdir(testpath +'seg_test') : 
    files = gb.glob(pathname= str( testpath +'seg_test//' + folder + '/*.jpg'))
    print(f'For testing data , found {len(files)} in folder {folder}')


# _____
# now for prediction folder

# In[ ]:


files = gb.glob(pathname= str(predpath +'seg_pred/*.jpg'))
print(f'For Prediction data , found {len(files)}')


# _____
# 
# # Checking Images
# 
# now we need to heck the images sizes , to know ow they looks like
# 
# since we have 6 categories , we first need to create a dictionary with their names & indices , also create a function to get the code back

# In[ ]:


code = {'buildings':0 ,'forest':1,'glacier':2,'mountain':3,'sea':4,'street':5}

def getcode(n) : 
    for x , y in code.items() : 
        if n == y : 
            return x    


# now how about the images sizes in train folder

# In[ ]:


size = []
for folder in  os.listdir(trainpath +'seg_train') : 
    files = gb.glob(pathname= str( trainpath +'seg_train//' + folder + '/*.jpg'))
    for file in files: 
        image = plt.imread(file)
        size.append(image.shape)
pd.Series(size).value_counts()


# ______
# 
# ok , almost all of them are 150,150,3 , how about test images ? 

# In[ ]:


size = []
for folder in  os.listdir(testpath +'seg_test') : 
    files = gb.glob(pathname= str( testpath +'seg_test//' + folder + '/*.jpg'))
    for file in files: 
        image = plt.imread(file)
        size.append(image.shape)
pd.Series(size).value_counts()


# almost same ratios , now to prediction images 

# In[ ]:


size = []
files = gb.glob(pathname= str(predpath +'seg_pred/*.jpg'))
for file in files: 
    image = plt.imread(file)
    size.append(image.shape)
pd.Series(size).value_counts()


# ok , since almost all of pictures are 150,150,3 , we can feel comfort in using all pictures in our model , after resizing it in a specific amount

# # Reading Images
# 
# now it's time to read all images & convert it into arrays
# 
# first we'll create a variable s , which refer to size , so we can change it easily 
# 
# let's use now size = 100 , so it will be suitable amount to contain accuracy without losing so much time in training

# In[ ]:


s = 100


# now to read all pictues in six categories in training folder, ans use OpenCV to resize it , and not to forget to assign the y value , from the predefined function 

# In[ ]:


X_train = []
y_train = []
for folder in  os.listdir(trainpath +'seg_train') : 
    files = gb.glob(pathname= str( trainpath +'seg_train//' + folder + '/*.jpg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (s,s))
        X_train.append(list(image_array))
        y_train.append(code[folder])


# great , now how many items in X_train 

# In[ ]:


print(f'we have {len(X_train)} items in X_train')


# also we have have a look to random pictures in X_train , and to adjust their title using the y value

# In[ ]:


plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_train),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_train[i])   
    plt.axis('off')
    plt.title(getcode(y_train[i]))


# great , now to repeat same steps exactly in test data

# In[ ]:


X_test = []
y_test = []
for folder in  os.listdir(testpath +'seg_test') : 
    files = gb.glob(pathname= str(testpath + 'seg_test//' + folder + '/*.jpg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (s,s))
        X_test.append(list(image_array))
        y_test.append(code[folder])
        


# In[ ]:


print(f'we have {len(X_test)} items in X_test')


# In[ ]:


plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_test),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_test[i])    
    plt.axis('off')
    plt.title(getcode(y_test[i]))


# also with Prediction data , without having title ofcourse

# In[ ]:


X_pred = []
files = gb.glob(pathname= str(predpath + 'seg_pred/*.jpg'))
for file in files: 
    image = cv2.imread(file)
    image_array = cv2.resize(image , (s,s))
    X_pred.append(list(image_array))       


# In[ ]:


print(f'we have {len(X_pred)} items in X_pred')


# In[ ]:


plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_pred),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_pred[i])    
    plt.axis('off')


# ________
# 
# # Building The Model 
# 
# now we need to build the model to train our data
# 
# first to convert the data into arrays using numpy

# In[ ]:


X_train = np.array(X_train)
X_test = np.array(X_test)
X_pred_array = np.array(X_pred)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(f'X_train shape  is {X_train.shape}')
print(f'X_test shape  is {X_test.shape}')
print(f'X_pred shape  is {X_pred_array.shape}')
print(f'y_train shape  is {y_train.shape}')
print(f'y_test shape  is {y_test.shape}')


# now to build the CNN model by Keras , using Conv2D layers , MaxPooling & Denses

# In[ ]:


KerasModel = keras.models.Sequential([
        keras.layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(s,s,3)),
        keras.layers.Conv2D(150,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(4,4),
        keras.layers.Conv2D(120,kernel_size=(3,3),activation='relu'),    
        keras.layers.Conv2D(80,kernel_size=(3,3),activation='relu'),    
        keras.layers.Conv2D(50,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(4,4),
        keras.layers.Flatten() ,    
        keras.layers.Dense(120,activation='relu') ,    
        keras.layers.Dense(100,activation='relu') ,    
        keras.layers.Dense(50,activation='relu') ,        
        keras.layers.Dropout(rate=0.5) ,            
        keras.layers.Dense(6,activation='softmax') ,    
        ])


# now to compile the model , using adam optimizer , & sparse categorical crossentropy loss

# In[ ]:


KerasModel.compile(optimizer ='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# so how the model looks like ? 

# In[ ]:


print('Model Details are : ')
print(KerasModel.summary())


# now to train the model , lets use 50 epochs now

# In[ ]:


epochs = 50
ThisModel = KerasModel.fit(X_train, y_train, epochs=epochs,batch_size=64,verbose=1)


# how is the final loss & accuracy
# 

# In[ ]:


ModelLoss, ModelAccuracy = KerasModel.evaluate(X_test, y_test)

print('Test Loss is {}'.format(ModelLoss))
print('Test Accuracy is {}'.format(ModelAccuracy ))


# ok , only 80% accuracy & can be increased by tuning the hyperparameters
# 

# 
# _______
# 
# now to predict X test

# In[ ]:


y_pred = KerasModel.predict(X_test)

print('Prediction Shape is {}'.format(y_pred.shape))


# great
# 
# now it's time to redict X Predict

# In[ ]:


y_result = KerasModel.predict(X_pred_array)

print('Prediction Shape is {}'.format(y_result.shape))


# and to show random redicted pictures & its predicting category
# 

# In[ ]:


plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_pred),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_pred[i])    
    plt.axis('off')
    plt.title(getcode(np.argmax(y_result[i])))

