#!/usr/bin/env python
# coding: utf-8

# # CONVOLUTIONAL NEURAL NETWORK

# In this notebook We will build a CNN for classifying the hand written digits

# In[ ]:


# importing the basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## LOADING THE DATA 

# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
print("Traning Set")
print(train.shape)
print("Test set")
print(test.shape)


# In the train data the first column is label and each image is 28*28 pixel by pixel image. Then from 2nd column to last column that is 784 columns represent the pixel values of each image. There are totally 42000 images in the train data set.
# 
# The test data is same but only the label column is not present. the size of test is 28000

# In[ ]:


train.head(10)


# In[ ]:


test.head(10)


#  In the train data and test data there are no missing values. So there is no need to clean the data

# In[ ]:


train.isnull().any().describe()


# In[ ]:


test.isnull().any().describe()


# ## Train and Test Matrices  

# In[ ]:


X=train.iloc[:,1:].values
y=train.iloc[:,0].values

X_test=test.iloc[:,:].values

print("Train data shape : (%d,%d)"% X.shape)
print("Train Labels : (%d,)"% y.shape)
print("Test Data shape : (%d,%d)"% X_test.shape)


# ## VISUALIZATION OF THE DATA

# In[ ]:


def show_image(image, shape, label="", cmp=None):
    img = np.reshape(image,shape)
    plt.imshow(img,cmap=cmp, interpolation='none')
    plt.title(label)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(12,10))

z, x = 5,10
for i in range(0,(z*x)):
    plt.subplot(z, x, i+1)
    k = np.random.randint(0,X.shape[0],1)[0]
    show_image(X[k,:],(28,28), y[k], cmp="gist_gray")
plt.show()


# ## NORMALIZATION 

# We normalize the train data and test data and we do this by dividing the data by 255. This is equal to (max-min) of the pixel values

# In[ ]:


X=X/255
X_test=X_test/255
print("min value :%d"% np.min(X))
print("max value :%d"% np.max(X))


# ## RESHAPING 

# Train and test images (28px x 28px) has been stock into pandas.Dataframe as 1D vectors of 784 values. We reshape all data to 28x28x1 3D matrices.
# 
# Keras requires an extra dimension in the end which correspond to channels. MNIST images are gray scaled so it use only one channel. For RGB images, there is 3 channels, we would have reshaped 784px vectors to 28x28x3 3D matrices.

# In[ ]:


X=X.reshape(-1,28,28,1)
X_test=X_test.reshape(-1,28,28,1)

print("Train data shape : (%d,%d,%d,%d)"% X.shape)
print("Test Data shape : (%d,%d,%d,%d)"% X_test.shape)


# ## ONE HOT ENCODING

# We have actually classes named 0 to 9. But When we are training the data using neural network we have to make the dummies of the classes.It is really useful for classifying.
# 
# For Example:
# 
# y=1 is changed to y=[0 1 0 0 0 0 0 0 0 0]
# 
# y=2 is changed to y=[0 0 1 0 0 0 0 0 0 0]
# 
# and so on.

# In[ ]:


# first we will print y's
print(y[0:10])


# For one hot encoding we will use the onehotencoder from sklearn preprocessing library

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
x=y.reshape(y.size,1)
onehotencoder=OneHotEncoder(categorical_features=[0])
y=onehotencoder.fit_transform(x).toarray().astype(int)
print("SHAPE : (%d,%d)\n" %y.shape)
print(y[0:10,:])


# ## CROSS VALIDATION SET 

# The cross-validation is a techinique used for measure the accuracy and visualizing overfitting.
# 
# Here we will split the data into training set and cross validation set with size of cross-validation as 10% of total data

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_cross,y_train,y_cross=train_test_split(X,y,test_size=0.1,random_state=42)

print("Train Size (%d,%d,%d,%d) \n"% X_train.shape)
print("Validation Size (%d,%d,%d,%d) \n"% X_cross.shape)
print("Train Label Size (%d,%d) \n"% y_train.shape)
print("Validation Label Size (%d,%d) \n"% y_cross.shape)


# We can get a better sense for one of these examples by visualising the image and looking at the label.

# In[ ]:


plt.imshow(X_train[0][:,:,0])


# In[ ]:


plt.imshow(X_test[0][:,:,0])
# print(X_test[0][:,:,0].shape)


# ## MODEL

# For building **Neural network** I am using python keras library (tensorflow backend)

# In[ ]:


# importing the libraries
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
from keras.preprocessing.image  import ImageDataGenerator


# ## Activation Function:
# Exist a lot of types of activations functions, but in this model we'll use the ReLu (Rectifier Linear Function) for hidden layers and Softmax function for output layer
# 
# Softmax function is same as sigmoid function but softmax is for more varaibles that is it is generalization of sigmoid function on a n-dimensional vector
# 
# ## Model
# 
# I have used two convolutional layer with 32 filters each and each filter transforms a part of the image (defined by the kernel size) using the kernel filter.
# 
# The second important layer in CNN is the pooling (MaxPool2D) layer. This layer simply acts as a downsampling filter. It looks at the 2 neighboring pixels and picks the maximal value and i have chosen the pool size as (2,2).
# 
# Before Flattening i made that layer as dropout with 0.25 as keep probability
# 
# In the end i used the features in one fully-connected (Dense) layers which is just artificial an neural networks (ANN). It contained 512,256 neurons in the hidden layer and both are dropout layer with 0.2 as keep probability. 
# 
# 
# I have trained the model in my PC but on kaggle I am commmenting out the code. On my PC It took nearly 5hrs for two epochs. 

# In[ ]:


# Initialising the CNN
model=Sequential()

# Convolution layer
model.add(Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)))

# Max pooling Layer
model.add(MaxPool2D(pool_size=(2,2)))

# adding another Convolution Layer
model.add(Conv2D(64,(3,3),activation='relu'))

# adding Max pooling layer
model.add(MaxPool2D(pool_size=(2,2)))

# model dropout layer
model.add(Dropout(0.25))

# Flattening
model.add(Flatten())

# Full connection 
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

# summary
model.summary()


# In[ ]:


# compiling the CNN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()


# ## DATA AUGMENTATION

# In order to avoid overfitting problem, we need to expand artificially our handwritten digit dataset. We can make your existing dataset even larger. The idea is to alter the training data with small transformations to reproduce the variations occuring when someone is writing a digit.
# 
# For example, the number is not centered The scale is not the same (some who write with big/small numbers) The image is rotated...
# 
# Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as data augmentation techniques. Some popular augmentations people use are grayscales, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more.
# 
# By applying just a couple of these transformations to our training data, we can easily double or triple the number of training examples and create a very robust model.
# 
# 

# In[ ]:


datagen=ImageDataGenerator(rotation_range=10,zoom_range=0.1,width_shift_range=0.1,height_shift_range=0.1)

datagen.fit(X_train)


# ## FITTING THE MODEL

# In[ ]:


epoch=2
batch=100
sp_epoch=X_train.shape[0]


# In[ ]:


# h=model.fit_generator(datagen.flow(X_train,y_train,batch_size=batch),epochs=epoch,validation_data=(X_cross,y_cross),steps_per_epoch=sp_epoch)


# ## PREDICTION

# In[ ]:


# y_pred=model.predict_classes(X_test)
# print(y_pred.shape)


# ## VISUALIZING THE TEST RESULTS

# Here we are visulizing the 20 random test images. The title of each image is the predicted label

# In[ ]:


# %matplotlib inline
# plt.figure(figsize=(12,10))

# z, x = 5,10
# for i in range(0,(z*x)):
#     plt.subplot(z, x, i+1)
#     k = np.random.randint(0,X_test.shape[0],1)[0]
#     show_image(X_test[k,:],(28,28), y_pred[k], cmp="gist_gray")
# plt.show()


# In[ ]:


# imageid=np.linspace(1,28000,28000).astype(int)
# print(imageid,imageid.shape,type(imageid))


# In[ ]:


# ans=pd.DataFrame({
#     "ImageId":imageid,
#     "Label":y_pred
# })
# ans.head(10)


# In[ ]:


# ans.to_csv("CNN.csv",index=False)                
# print("Done")   


# With this I got an accuracy of 99.542%.
# 
# Thanks for reading..!!
# 
# Sugesstion and feedback are welcomed!!
