#!/usr/bin/env python
# coding: utf-8

# ### Objective- In this  example we will learn how to reconstruct faces using different regression models. We will compare reconstructed and actual faces side by side. 
# Steps wise explaination-
# 1. First we will load Olivetti images into numpy array of 400 X64 X 64 matrix 3D array. This 3D contains images of 40 different persons and each person has 10 images in this data which have been taken from different angle.
# 2. Then we will reshape this 3D array matrix to 2D array matrix using numpy reshape method. it will reshape into 400 X 4096 matrix arrary
# 3. Then we will divide this array into 300 X 4096(training) and 100 X 4096(test) using olivetti index. So 300 images of first 30 persons will be loaded into training matrix and 100 images of 10 persons will be loaded into test matrix
# 4. We will then randomly pick 10 images randomly from test matrix and will use this as test data
# 5. We will divide columns of training and test data matrix(selected in above step) into 2 equal parts to divide image into upper half and lower part
# 6. Next we will train regression model using lower and upper half from training matrix
# 7. Now we will use this model to predict lower half of randomly selected 10 images by using upper half of test data.

# In[31]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from skimage.io import imshow

# Regresseros
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor


# ### 1.) There are ten different images of each of 40 distinct subjects.That is each subject has 10 images but taken differently.
# 1. )  For some subjects, the images were taken at different times),
# 2. )  Varying the lighting, facial expressions (open / closed eyes),
# 3. )   Smiling / not smiling) and facial details (glasses / no glasses).
# 
# All the images were taken against a dark homogeneous background with the subjects in an upright, frontal position (with tolerance  for some side movement)

# In[32]:


# Input data files consisting of the images
img_ds = np.load("../input/olivetti_faces.npy")
img_ds.shape


# ### 2.) Lets check images of one subject taken from different angle with different facial expression along with glasses and no glasses

# In[33]:


# Sample images of a subject
img_cnt = 10
plt.figure(figsize=(15,15))
for i in range(img_cnt):
    plt.subplot(1,10,i+1)
    x=img_ds[i+30] # 3rd subject
    imshow(x)
plt.show()


# ### 3.) Lets divide dataset into training and test data set.First we will divide data set which is (400,64,64) into 2 dimensional array (400,4096)
# a. )  We will use reshape method of numpy arry to transform it from 3 dimensions to 2 dimensions

# In[34]:


img_ds_flt = img_ds.reshape(img_ds.shape[0],img_ds.shape[1]*img_ds.shape[2])
print(img_ds_flt.shape)


# b. ) We will load images index numpy array to target variable

# In[35]:


targets = np.load("../input/olivetti_faces_target.npy")
print(targets)
# We can see below first 10 index belongs to one image and so on


# c. ) We will now use target indexes to divide the data set into training and test

# In[73]:


training_img_data = img_ds_flt[targets<30] # First 30 types of images out of 40 ie 30 * 10 =300
test_img_data = img_ds_flt[targets>=30] # Test on rest 10 independent people from number 30th to 39th  10 * 10 = 100


# d. ) we will divide the dataset further. We will divide  upper half of the face and lower of the faces for both training and test data

# In[37]:


# Test on a subset of people
#     Generate 10 random integers between 0 and 100
# // is unconditionally "flooring division",
n_faces = test_img_data.shape[0]//10
n_faces


# In[38]:


face_ids = np.random.randint(0 , 100, size =n_faces) # To select some random images from 100 images
face_ids


# In[39]:


test_img_data.shape


# In[40]:


# We will select the random 10 images from test data
test_img_data = test_img_data[face_ids, :] 
test_img_data


# In[41]:


#Total pixels in any image
n_pixels = img_ds_flt.shape[1]
n_pixels


# In[42]:


#Select upper half of the faces as predictors
X_train = training_img_data[:, :(n_pixels + 1) // 2]
X_train


# In[43]:


#Lower half of the faces will be target(s)                 
y_train = training_img_data[:, n_pixels // 2:]
y_train


# In[44]:


# Similarly for test data. Upper and lower half
X_test = test_img_data[:, :(n_pixels + 1) // 2]
y_test = test_img_data[:, n_pixels // 2:]


# ### 4. ) Lets create regression methods to reconstruct faces by taining lower and upper half of the faces then constructing the lower half by input of upper half

# In[45]:


# Prepare a dictionary of estimators after instantiating each one of them
ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32,random_state=0),
    "K-nn": KNeighborsRegressor(),                          # Accept default parameters
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
    "multi_gbm" : MultiOutputRegressor(GradientBoostingRegressor(n_estimators=5))
}


# In[47]:


# Create an empty dictionary to collect prediction values
y_test_predict = dict()
# Fit each model by turn and make predictions
for name, estimator in ESTIMATORS.items():     
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)
#y_test_predict['RandomForestRegressor'].shape


# In[53]:


#Just check    
y_test_predict['Extra trees'].shape


# In[60]:


## Processing output -> Each face should have this dimension
image_shape = (64, 64)


# In[66]:


plt.figure(figsize=(15,15))
j = 0
for i in range(n_faces):
    actual_face = test_img_data[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['Ridge'][i]))# Horizental stack upper actual half and lower predict half
    j = j+1# Image index
    plt.subplot(5,5,j)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(5,5,j)
    y = actual_face.reshape(image_shape)
    imshow(y)
  
plt.show()


# In[72]:


plt.figure(figsize=(10,10))
j = 0
for i in range(n_faces):
    actual_face =    test_img_data[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['Extra trees'][i]))
    j = j+1
    plt.subplot(5,5,j)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(5,5,j)
    y = actual_face.reshape(image_shape)
    imshow(y)
  
plt.show()


# In[71]:


plt.figure(figsize=(15,15))
j = 0
for i in range(n_faces):
    actual_face =    test_img_data[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['Linear regression'][i]))
    j = j+1
    plt.subplot(5,5,j)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(5,5,j)
    y = actual_face
    imshow(y)
  
plt.show()


# In[70]:


plt.figure(figsize=(10,10))
j = 0
for i in range(n_faces):
    actual_face =    test_img_data[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['K-nn'][i]))
    j = j+1
    plt.subplot(5,5,j)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(5,5,j)
    y = actual_face.reshape(image_shape)
    imshow(y)
  
plt.show()


# In[69]:


plt.figure(figsize=(10,10))
j = 0
for i in range(n_faces):
    actual_face =    test_img_data[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['multi_gbm'][i]))
    j = j+1
    plt.subplot(5,5,j)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(5,5,j)
    y = actual_face.reshape(image_shape)
    imshow(y)
  
plt.show()

