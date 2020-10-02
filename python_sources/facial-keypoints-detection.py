#!/usr/bin/env python
# coding: utf-8

# This kernel is written for the [Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection) competetion. If you Like the notebook and think that it helped you, <font color="red"><b> please upvote</b></font>.
# 
# ---
# 
# ## Table of Content
# 1. Data Preprocessing
# 2. Modeling
#     * Model Architecture
#     * Model Evaluation
# 3. Prediction & Submission

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from time import sleep
import os

os.listdir('../input')
train_data = pd.read_csv('../input/training/training.csv')  
test_data = pd.read_csv('../input/test/test.csv')
lookid_data = pd.read_csv('../input/IdLookupTable.csv')

train_data.head().T


# # Data Preprocessing

# In[ ]:


# Lets check for missing values
train_data.isnull().any().value_counts()
# Fill the missing values with the previous values in that row
train_data.fillna(method = 'ffill',inplace = True)
# check for missing values again
train_data.isnull().any().value_counts()


# The image is our feature and other values are labes that we have to predict later. As image column values are in string format and there is also some missing values so we have to split the string by space and append it and also handling missing values

# In[ ]:


def split_image_feature(data):
    """Return extracted image feature"""
    imag = []
    for i in range(0, data.shape[0]):
        img = data['Image'][i].split(' ')
        img = ['0' if x == '' else x for x in img]
        imag.append(img)

    # Lets reshape and convert it into float value
    image_list = np.array(imag, dtype = 'float')
    X_train = image_list.reshape(-1,96,96)
    return X_train

X_train = split_image_feature(train_data)
# Now lets separate labels.
training = train_data.drop('Image',axis = 1)
y_train = []
for i in range(0, train_data.shape[0]):
    y = training.iloc[i,:]
    y_train.append(y)
y_train = np.array(y_train,dtype = 'float')

# Lets see what is the first image
plt.imshow(X_train[0],cmap='gray')
plt.show()


# # Modeling

# In[ ]:


from keras.layers import Conv2D,Dropout,Dense,Flatten
from keras.models import Sequential

model = Sequential([Flatten(input_shape=(96,96)),
                         Dense(128, activation="relu"),
                         Dropout(0.1),
                         Dense(64, activation="relu"),
                         Dense(30)
                         ])

model.compile(optimizer='adam', 
              loss='mse',
              metrics=['mae','accuracy'])
model.fit(X_train,y_train,epochs = 500,batch_size = 128,validation_split = 0.2)


# # Prediction & Submission
# Now the last step is the create our submission file keeping in the mind required format with two columns: RowId and Location.
# Location column values should be filled according the lookup table provided (IdLookupTable.csv)

# In[ ]:


X_test = split_image_feature(test_data)
prediction = model.predict(X_test)

lookid_list = list(lookid_data['FeatureName'])
imageID_list = list(lookid_data['ImageId'] - 1)
rowID_list = list(lookid_data['RowId'])
prediction_list = list(prediction)

feature = []
for f in list(lookid_data['FeatureName']):
    feature.append(lookid_list.index(f))
    
location = []
for x,y in zip(imageID_list, feature):
    location.append(prediction_list[x][y])
    
rowid = pd.Series(rowID_list, name = 'RowId')
loc = pd.Series(location, name = 'Location')

submission = pd.concat([rowid,loc], axis = 1)
submission.to_csv('submission.csv',index = False)

