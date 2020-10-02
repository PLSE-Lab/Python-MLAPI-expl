#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Digit Recognizer In Computer Vision
# 
# 
# *    ** 1. Data Preprocessing / Feature Engineering**
#     *             a. Load Dataset
#     *             b. Descriptive Statistics
#     *             c. Checking for null and missing values
#     *             d. outliers
#     *             e. Feature Split 
#     *             f. Feature scale (Normalisation)
#     *             g. Reshape
#     *             h. Label Encoding/One Hot Encoder
#     *             i. Evaluation Resample Methods  
# *    ** 2. CNN Algorithm**
#     *             a. Build Model
#     *             b. Building CNN Layers
#     *             c. Data Augmentation
#     *             d. Evaluate Model Confusion Matrix
# *    ** 3. Evaluate Model**
#     *             a. Evaluate Model With Test Set
# *   ** 4. Submission**
#     *             a. Submit Result As .csv File
#     

# # 1. Data Preprocessing / Feature Engineering
# 
# # **a. Load Dataset**

# In[ ]:


# Import the datasets
import pandas as pd
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
test_set = pd.read_csv("../input/digit-recognizer/test.csv")
training_set = pd.read_csv("../input/digit-recognizer/train.csv")


# # **b. Descriptive Statistics**

# In[ ]:


# Displaying the shape and datatype of each attribute

print(training_set.shape)
training_set.dtypes


# We got almost 785 columns and 42000 rows in our dataset.
# 
# All columns having integer as a datatype.

# # Descriptive Visualisation
# 
# Histogram visualisation for output attribute to know what kind of distribution it is.

# In[ ]:


# Histogram Visualisation for Output attribute

import seaborn as sb
sb.distplot(training_set['label'])


# As per above Histogram plot we get to know output attribute is discrete outcome and it is not normal distribution aswell.

# # **c. Checking for null and missing values**

# In[ ]:


# Displaying Null values info in each column

training_set.info()


# In[ ]:


# Displaying the sum count of null or empty values in each count..Due too many columns unable to view
training_set.isna().sum()


# In[ ]:


# Now we going to display only those column which have null value and remaining columns wont display
training_set.isnull().any().describe()


# In[ ]:


test_set.isnull().any().describe()


# There is no missing or empty values in both training and test set, So now we can safely go ahead.

# # **d. outliers**
# 
# As we know we dont get any missing value and we converted image pixel values into array if image so most probabliy it wont have any outliers in each attribute.
# 
# In each row it will contains 0's only some where in that row it have high value due to high pattern detection happen there..

# # e. Feature Split
# 
# Split the input and output attributes.

# In[ ]:


y_train=training_set['label'].values
x_train=training_set.drop(['label'],axis=1)


# # e. Feature scale (Normalisation)
# 
# We are perform a grayscale normalization to reduce the effect of illumination's differences.
# 
# Moreover CNN coverage faster if values lies in between 0 to 1.So by using Normalisation scale we are rescaling all values from [0...255] to [0..1]

# In[ ]:


x_train=x_train/255.0
test_set=test_set/255.0


# # f. Reshape
# 
# Reshape image in 3 dimensions
# 
# Train and test images (28px x 28px) has been stock into pandas.Dataframe as 1D vectors of 784 values. We reshape all data to 28x28x1 3D matrices.
# 
# Keras requires an extra dimension in the end which correspond to channels. MNIST images are gray scaled so it use only one channel. For RGB images, there is 3 channels, we would have reshaped 784px vectors to 28x28x3 3D matrices
# 

# In[ ]:


x_train=x_train.values.reshape(-1,28,28,1)
x_test=test_set.values.reshape(-1,28,28,1)
del test_set
del training_set


# # h. Label Encoding/One Hot Encoder
# 
# Labels are 10 digits numbers from 0 to 9. We need to encode these lables to one hot vectors

# In[ ]:


# Encoding numerics in one hot encoder vector
'''from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(Categorical_features=[0])
y_train=onehotencoder.fit_transform(y_train)

TypeError: __init__() got an unexpected keyword argument 'Categorical_features' some api issue there
'''
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

y_train=to_categorical(y_train,num_classes=10)


# # i. Evaluation Resample Methods
# 
# Now we are going to split again x_train and y_train into training and test set to evaluate performance with training set data.
# 
# Later we will predict label values using test set...

# In[ ]:


# split the training set into train and test set

seed=5
train_size=0.80
test_size=0.20

from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test=train_test_split(x_train,y_train,train_size=train_size,test_size=test_size,random_state=seed)


# # 2. CNN Algorithm
# 
# # a. Build Model
# 
# I used the Keras Sequential API, where you have just to add one layer at a time, starting from the input.

# # b. Building CNN Layers

# In[ ]:


# Building CNN layers

# intialisaing the sequence of layers
from keras.models import Sequential
cnn=Sequential()

# Building fist Convolutional Layer
from keras.layers import Convolution2D
from keras.layers import Dropout
cnn.add(Convolution2D(input_shape=(28,28,1),activation='relu',filters=32,kernel_size=(5,5)))
cnn.add(Dropout(0.2))

# Building first pooling Layer
from keras.layers import MaxPooling2D
cnn. add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.2))

# Building Second Colvolution and pooling layers
cnn.add(Convolution2D(kernel_size=(5,5),filters=32,activation='relu'))
cnn.add(Dropout(0.2))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.2))

# Building Third Convolution and pooling layer
cnn.add(Convolution2D(kernel_size=(3,3),filters=32,activation='relu'))
cnn.add(Dropout(0.2))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.2))

# Building Flatten layer
from keras.layers import Flatten
cnn.add(Flatten())

# Building Fully Connected Layers
from keras.layers import Dense
# First fully connected hidden layer
cnn.add(Dense(256,activation='relu'))
cnn.add(Dropout(0.2))
# Second fully connected hidden layer
cnn.add(Dense(256,activation='relu'))
cnn.add(Dropout(0.2))
# Third Fully connected hidden layer
cnn.add(Dense(128,activation='relu'))
cnn.add(Dropout(0.2))

# Output layer with 10 neurons
cnn.add(Dense(10,activation='softmax'))


# Try changing this line:
# 
# model.add(Dense(output_dim=NUM_CLASSES, activation='softmax'))
# 
# to
# 
# model.add(Dense(NUM_CLASSES, activation='softmax'))
# 
# I could not find a parameter called output_dim on the documentation page for Dense. I think you meant to provide units but labelled it as output_dim

# In[ ]:


# compile the CNN Model
cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# # c. Without Data Augmentation

# In[ ]:


result=cnn.fit(x1_train, y1_train, batch_size = 32, epochs = 10,validation_data = (x1_test,y1_test), verbose = 2)


# For 10 epochs without data augmentation we got...
# 
# training accuracy: 97.54% and loss:0.0892
# test accuracy: 98.79% and loss:0.0525
# 
# 
# Now we are excepting more accuracy in both training and test set by using data augmentation

# # With Data Augmentation

# In[ ]:


# Create Data Augmentation Generator
from keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,# randomly shift images
                          horizontal_flip=False,vertical_flip=False,# randomly flip images
                          rotation_range=10,# randomly rotate images in the range (degrees, 0 to 180)
                          #brightness_range=[0.1,1.0],# randomly brightning images
                          zoom_range=0.1,# Randomly zoom image
                          zca_whitening=False,# apply ZCA whitening
                          featurewise_center=False,  # set input mean to 0 over the dataset
                          samplewise_center=False,  # set each sample mean to 0
                          featurewise_std_normalization=False,  # divide inputs by std of the dataset
                          samplewise_std_normalization=False,  # divide each input by its std
                          )

datagen.fit(x1_train)


# In[ ]:


batch_size=86
# makeing iteration flow
sample=datagen.flow(x1_train,y1_train,batch_size=batch_size)
# fit and generate the outcome
train_predictions=cnn.fit_generator(sample,epochs=10,validation_data=(x1_test,y1_test))


# After Applying image data augmentation to our dataset we got good accuracy
# 
# training set : 97.57% and loss:0.0848
# test set: 99.13% and loss:0.0392

# # d. Evaluate Model Confusion Matrix

# In[ ]:


# predicting the training set test accuracy
import numpy as np
y_trainpred=cnn.predict(x1_test)
# Convert predictions classes to one hot vectors 
y_pred_one=np.argmax(y_trainpred,axis=1)
# Convert validation observations to one hot vectors
y1_test_one=np.argmax(y1_test,axis=1)
from sklearn.metrics import confusion_matrix
accuracy=confusion_matrix(y1_test_one,y_pred_one)


# In[ ]:


print(accuracy)


# # 3. Evaluate Model
# 
# # a. Evaluate Model With Test Set

# In[ ]:


# Predict the result for test set
y_pred=cnn.predict(x_test)


# In[ ]:


# argmax() is used to decode the onehotencoder value to numerical value
result=np.argmax(y_pred,axis=1)
# storing those value as a column name label
result=pd.Series(result,name='label')


# # 4. Submission
# 
# # a. Submit Result As .csv File

# In[ ]:


# Submission 
submission=pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)
submission.to_csv("My_Submission.csv",index=False)
print("Submission Successfully")

