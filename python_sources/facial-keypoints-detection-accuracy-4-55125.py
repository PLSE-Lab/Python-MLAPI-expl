#!/usr/bin/env python
# coding: utf-8

# [<h1>Facial Keypoints Detection</h1>](https://www.kaggle.com/c/facial-keypoints-detection)
# 
# * I have made this kernel in my own way, kind of newbie friendly. 
# * My public and private score is successively 4.55126 and 4.24206.
# * I have tried to elaborate every code snippets.
# * An upvote is highly appreciated, if you get benifitted. So that it can reach more people.

# At first, Unzip the necessary files.

# In[ ]:


print("Contents of input/facial-keypoints-detection directory: ")
get_ipython().system('ls ../input/facial-keypoints-detection/')

print("\nExtracting .zip dataset files to working directory ...")
get_ipython().system('unzip -u ../input/facial-keypoints-detection/test.zip')
get_ipython().system('unzip -u ../input/facial-keypoints-detection/training.zip')

print("\nCurrent working directory:")
get_ipython().system('pwd')
print("\nContents of working directory:")
get_ipython().system('ls')


# Import Packages

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from time import sleep
import os


# Insert data

# In[ ]:


Train_Dir = 'training.csv'
Test_Dir = 'test.csv'
lookid_dir = '../input/facial-keypoints-detection/IdLookupTable.csv'
train_data = pd.read_csv(Train_Dir)  
test_data = pd.read_csv(Test_Dir)
lookid_data = pd.read_csv(lookid_dir)


# Lets explore our dataset

# In[ ]:


train_data.head().T


# In[ ]:


train_data.shape


# Scrutinize the hightest values to get few insights.

# In[ ]:


train_data.describe()


# Check the data types along with their corresponding values

# In[ ]:


train_data.info()


# Let's check for missing values (based on percentage)

# In[ ]:


all_data_na = (train_data.isnull().sum() / len(train_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)


# In[ ]:


train_data.isnull().any()


# In[ ]:


null_counts = train_data.isnull().sum()
null_counts[null_counts > 0].sort_values(ascending=False)


# Drop data based on threshold

# In[ ]:


# Set the limit
# Drop columns using that limit
# limit = len(train_data) * 0.7
# new=train_data.dropna(axis=1, thresh=limit)
# View columns in the dataset
# new.columns


# So there are missing values exist in 28 columns. We can do two things here. one remove the rows having missing values and another is the fill missing values with something. I used two option as removing rows will reduce our dataset. 
# I filled the missing values with the previous values in that row.

# In[ ]:


type(train_data)


# I have assigned the mean value of ffill and bfill. The result is much more better than using just ffill method.

# In[ ]:


#train_data.fillna(method = 'ffill',inplace = True)
train_data = train_data.fillna(pd.concat([train_data.ffill(), train_data.bfill()]).groupby(level=0).mean())


# Lets check the missing values now

# In[ ]:


train_data.isnull().any()


# In[ ]:


train_data.isnull().any().value_counts()


# As there are no missing values, we can now separate the labels and features.
# The images are our features and other values are different labels of columns that we are gonna predict later.
# As image column values are in string format and there is also some missing values so we have to split the string by space and append it and also handle missing values.

# In[ ]:


train_data[['Image']].describe()


# * here 7049 unique numbers exist in image column

# Preparing training data

# In[ ]:





# In[ ]:


imag = []
for i in range(0,7049):  
    img = train_data['Image'][i].split(' ')
    img = ['0' if x == '' else x for x in img]
    imag.append(img)
    
    


# Preparing test data

# In[ ]:



timag = []
for i in range(0,1783):
    timg = test_data['Image'][i].split(' ')
    timg = ['0' if x == '' else x for x in timg]
    
    timag.append(timg)


# Lets reshape and convert it into floating value.

# In[ ]:


image_list = np.array(imag,dtype = 'float')
X_train = image_list.reshape(-1,96,96,1)


# In[ ]:


timage_list = np.array(timag,dtype = 'float')
X_test = timage_list.reshape(-1,96,96,1) 


# Lets see what is the first image.

# In[ ]:


plt.imshow(X_train[0].reshape(96,96),cmap='gray')
plt.show()


# Now lets separate labels.

# * Convert the values to an image after normalizing so that it can be rendered through the convolutional network
# * The highest and lowest value of float is successively 1 and -1

# In[ ]:


training = train_data.drop('Image',axis = 1)

y_train = []
for i in range(0,7049):
    y = training.iloc[i,:]

    y_train.append(y)
y_train = np.array(y_train,dtype = 'float')


# As our data is ready for training , lets define our model. I am using keras and simple dense layers. For loss function I am using 'mse' ( mean squared error ) as we have to predict new values. Our result evaluted on the basis of 'mae' ( mean absolute error ) . 

# In[ ]:


from keras.layers.advanced_activations import LeakyReLU
# from keras.layers import Activation
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D


# **Based on X_train(image), our model will try to predict their corresponding shapes(y_train). Further we'll exploit the informations after fitting the model on test set**

# In[ ]:


training.shape


# As the shape of the model is 30. We will assert 30 inside the dense layer.

# I had tried with selu and relu as an activation function. relu gave slightly better validation accuracy then selu. Though training accuracy is better in case of selu.
# I would better focus on validation accuracy.

# In[ ]:


'''

model = Sequential()

model.add(Convolution2D(32, (3,3), padding='same', activation = 'selu', use_bias=False, input_shape=(96,96,1)))

model.add(BatchNormalization())

model.add(Convolution2D(32, (3,3), padding='same', activation = 'selu',use_bias=False))

model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3,3), padding='same',activation = 'selu', use_bias=False))

model.add(BatchNormalization())

model.add(Convolution2D(64, (3,3), padding='same',activation = 'selu', use_bias=False))

model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3,3),padding='same', activation = 'selu',use_bias=False))
# model.add(BatchNormalization())

model.add(BatchNormalization())

model.add(Convolution2D(128, (3,3),padding='same', activation = 'selu',use_bias=False))

model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (3,3),padding='same',activation = 'selu',use_bias=False))

model.add(BatchNormalization())

model.add(Convolution2D(256, (3,3),padding='same',activation = 'selu',use_bias=False))

model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(512, (3,3), padding='same', activation = 'selu',use_bias=False))

model.add(BatchNormalization())

model.add(Convolution2D(512, (3,3), padding='same',activation = 'selu', use_bias=False))

model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30))
model.summary()

'''


# In[ ]:


model = Sequential()

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
# model.add(BatchNormalization())
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30))
# I didn't use softmax here, because it's only used in logistic regression, 
# where the outputs are usually categorical. But here our outputs are a bunch of floating
# values, which are not categorical
model.summary()


# **Regression Loss Functions**
# 
#     1. Mean Squared Error Loss
#     2. Mean Squared Logarithmic Error Loss
#     3. Mean Absolute Error Loss
# 
# **Binary Classification Loss Functions**
# 
#     1. Binary Cross-Entropy
#     2. Hinge Loss
#     3. Squared Hinge Loss
# 
# **Multi-Class Classification Loss Functions**
# 
#     1. Multi-Class Cross-Entropy Loss
#     2. Sparse Multiclass Cross-Entropy Loss
#     3. Kullback Leibler Divergence Loss

# * Clearly our output function will be formed as regression value.Hence mean_squared_error is used as a loss function. 
# * Adam seems to be more promising than rmsprop and sgd. It had to be, as Adam can be looked at as a combination of RMSprop and Stochastic Gradient Descent with momentum.

# In[ ]:


import keras
model.compile(optimizer='adam', 
              loss='mean_squared_error',
              metrics=['accuracy'])

'''
# Compile with adam optimizer and cross-entropy loss

model.compile(optimizer='adam', 
              loss='mean_squared_error',
              metrics=['accuracy'])
              
# using adam  
model.compile(optimizer="adam", 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

# Use sgd optimizer
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

# Use RMSprop
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

'''


# Now our model is defined and we will train it by calling fit method. I ran it for 500 iteration keeping batch size and validtion set size as 20% ( 20% of the training data will be kept for validating the model ).

# In[ ]:


log = model.fit(X_train,y_train,epochs = 50,batch_size = 512,validation_split = 0.2)


# Lets predict our results

# In[ ]:


pred = model.predict(X_test)


# In[ ]:


# Plotting loss and accuracy curves for training and verification
fig, ax = plt.subplots(2,1)

# accuracy
ax[0].plot(log.history['accuracy'], color='b', label="Training accuracy")
ax[0].plot(log.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[0].legend(loc='best', shadow=True)

# loss
ax[1].plot(log.history['loss'], color='b', label="Training loss")
ax[1].plot(log.history['val_loss'], color='r', label="validation loss",axes =ax[1])
legend = ax[1].legend(loc='best', shadow=True)
fig.show()
plt.savefig('FacialKeypointsDetection.png')


# Now the last step is to create our submission file keeping in the mind required format.
# There should be two columns :- RowId and Location
# Location column values should be filled according to the provided lookup table( IdLookupTable.csv)
# 

# In[ ]:


# RowID
rowid=list(lookid_data['RowId'])

imageID = list(lookid_data['ImageId']-1)
feature_name = list(lookid_data['FeatureName'])


# In[ ]:


pre_list = list(pred)


# In[ ]:


feature = []
for f in feature_name:
    feature.append(feature_name.index(f))


# From Idlookup table I will fetch the feature using imageId and try to fetch the corresponding value of that particular feature from predictionlist(pre_list)

# In[ ]:


# Location
# predict using image and feature
location = []
for x,y in zip(imageID,feature):
    location.append(pre_list[x][y])


# In[ ]:


rowid = pd.Series(rowid,name = 'RowId')


# In[ ]:


loc = pd.Series(location,name = 'Location')


# According to the statements, size of the image is 96 pixel. If predicted location exceeds 96 pixel threshold, clip method will assert the highest value(96) on that field.

# In[ ]:


loc = loc.clip(0.0,96.0)


# In[ ]:


submission = pd.concat([rowid,loc],axis = 1)


# In[ ]:


submission.to_csv('FacialKeypointsDetection.csv',index = False)

