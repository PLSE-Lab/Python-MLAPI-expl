#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Here i used Python 3 environment
# data processing, CSV file 
import pandas as pd
import numpy as np


# Input data files are available in the "../input/" directory.
# The data was stored in the Kaggel website already so i directly check which all data there
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#  output of the files


# In[ ]:


# For Visualisation importing the required libraries
# Importing required packages for the graphics
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Reading the training data , the data is in the json format so we used the pandas read_json. The inclination angle should convert into the numeric.

# In[ ]:


train = pd.read_json('../input/train.json')
train['inc_angle'] = pd.to_numeric(train['inc_angle'],errors='coerce')


# In[ ]:


# Checking the columns and head of the data
train.head()


# In[ ]:


# cheking the data types
train.info()


# In[ ]:


# Conts of the ice bergs and the ships presesnt in training data
iceberg_ship_count = train['is_iceberg'].value_counts()
iceberg_ship_count1=iceberg_ship_count.plot(kind='bar',colormap='gist_rainbow')
plt.xticks(rotation=25)
iceberg_ship_count1.set_xticklabels( ('Ships', 'Iceberg') )
plt.show()


# In[ ]:





# #### If the image  consist of iceberg then it represent as 1 otherwise ship as 0. The iceberg images are shown below and how they appear in image with respect to different color grading here below shown.

# In[ ]:


icebergs = train[train.is_iceberg==1].sample(n=16,random_state=123)

fig = plt.figure(1,figsize=(15,15))
for i in range(16):
    ax = fig.add_subplot(4,4,i+1)
    arr = np.reshape(np.array(icebergs.iloc[i,0]),(75,75))
    ax.imshow(arr,cmap='seismic')
    
plt.show()


# In[ ]:


icebergs = train[train.is_iceberg==1].sample(n=16,random_state=123)

fig = plt.figure(1,figsize=(15,15))
for i in range(16):
    ax = fig.add_subplot(4,4,i+1)
    arr = np.reshape(np.array(icebergs.iloc[i,0]),(75,75))
    ax.imshow(arr,cmap='magma')
    
plt.show()


# #### The images of the ships are shown in different color ratio of images. And these graphs which helpfull for the furthe training of the model. The below codes are very standard and commonly used codes, You can find in Stackoverflow and other Kaggle resources.

# In[ ]:


ships = train[train.is_iceberg==0].sample(n=9,random_state=456)
fig = plt.figure(1,figsize=(15,15))
for i in range(9):
    ax = fig.add_subplot(3,3,i+1)
    arr = np.reshape(np.array(ships.iloc[i,0]),(75,75))
    ax.imshow(arr,cmap='magma')
    
plt.show()


# In[ ]:


ships = train[train.is_iceberg==0].sample(n=16,random_state=456)
fig = plt.figure(1,figsize=(15,15))
for i in range(16):
    ax = fig.add_subplot(4,4,i+1)
    arr = np.reshape(np.array(ships.iloc[i,0]),(75,75))
    ax.imshow(arr,cmap='seismic')
    
plt.show()


# In[ ]:


# Train data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
y_train = np.array(train["is_iceberg"])
print("Xtrain:", X_train.shape)


# #### Here below the normal Keras model used , the below code already refer from [http://www.analyticmunch.com/deep-learning-projects/](http://) . We used normal 20 epochs. The activation function is sigmoid because the output is 0 or 1 (binary output). And the loss function is used binary_crossentropy. The some resaerch papers and publishers recommend to use ADAM optimizer in keras.

# In[ ]:


from keras.models import Sequential
from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense, Dropout
model = Sequential()
model.add(Convolution2D(32, 3, activation="relu", input_shape=(75, 75, 2)))
model.add(Convolution2D(64, 3, activation="relu", input_shape=(75, 75, 2)))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.summary()


# In[ ]:


model.fit(X_train, y_train, validation_split=0.2,epochs = 20)


# #### For Predictions test data has to load. Then the model has to test.
# 

# In[ ]:


test = pd.read_json("../input/test.json")


# In[ ]:


# Test data prediction
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
print("Xtest:", X_test.shape)


# In[ ]:


prediction = model.predict(X_test, verbose=1)


# In[ ]:


submit = pd.DataFrame({'id': test["id"], 'is_iceberg': prediction.flatten()})
submit.to_csv("Submitted_first.csv", index=False)


# In[ ]:


#submit = pd.DataFrame({'id': test["id"], 'is_iceberg': prediction.flatten()})
#submit.to_csv("submitted_2nd.csv", index=False)

