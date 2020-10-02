#!/usr/bin/env python
# coding: utf-8

# ** Call Libraries**

# In[ ]:


# 1.0 Call libraries
# For data manipulation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# 1.1 For plotting faces
import matplotlib.pyplot as plt   
from skimage.io import imshow

# 1.2 Our dataset is here
from sklearn.datasets import fetch_olivetti_faces

# 1.3 Regressors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor


# **Read data**

# In[ ]:


data=np.load("../input/olivetti_faces.npy") 
target=np.load("../input/olivetti_faces_target.npy")
 
print(data.shape)
print(target.shape)    


# **See Images **

# In[ ]:


firstImage = data[0]
imshow(firstImage) 


# ** Flatten each image**

# In[ ]:


# Flatten each image
data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])     # 64 X 64 = 4096
# Flattened 64 X 64 array
data.shape `


# ** Partition data set**

# In[ ]:


# Patition datasets into two (fancy indexing)
target < 30                # Output is true/false
train = data[target < 30]  # First 30 types of images out of 40 ie 30 * 10 =300
test = data[target >= 30]  # Test on rest independent people  10 * 10 = 100


# In[ ]:


#Generate 10 random integers between 0 and 100
n_faces = test.shape[0]//10             # // is unconditionally "flooring division",
n_faces
face_ids = np.random.randint(0 , 100, size =n_faces)
face_ids
# 7.1 So we have n_faces random-faces from within 1 to 100
test = test[face_ids, :]  


# In[ ]:


# Total pixels in any image
n_pixels = data.shape[1]

# 8.1 Select upper half of the faces as predictors
X_train = train[:, :(n_pixels + 1) // 2]    # // is unconditionally "flooring division",
                                            #    3.1//1.2 = 2.0
# 8.2 Lower half of the faces will be target(s)                 
y_train = train[:, n_pixels // 2:]


# In[ ]:


# 9.0 Similarly for test data. Upper and lower half
X_test = test[:, :(n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]


# **Set Estimators**

# In[ ]:


# Fit multi-output estimators
#  Experiment with more mulit-output regressors (such as RandomForestRegressor)
#  http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble

# Prepare a dictionary of estimators after instantiating each one of them
ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10,
                                       max_features=32,     # Out of 20000
                                       random_state=0),
    "K-nn": KNeighborsRegressor(),                          # Accept default parameters
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
      "RandomForestRegressor": RandomForestRegressor()  
}


# Create an empty dictionary to collect prediction values
y_test_predict = dict()


# In[ ]:


# Fit each model by turn and make predictions
#     Iterate over dict items. Each item is a tuple: ( name,estimator-object)s
for name, estimator in ESTIMATORS.items():     
    estimator.fit(X_train, y_train)                    # fit() with instantiated object
    y_test_predict[name] = estimator.predict(X_test)   # Make predictions and save it in dict under key: name
                                                       # Note that output of estimator.predict(X_test) is prediction for
                                                       #  all the test images and NOT one (or one-by-one)
# Just check    
y_test_predict

# Just check shape of one of them
y_test_predict['Extra trees'].shape    # 5 X 2048 
print(y_test_predict['K-nn'].shape)
y_test_predict['Linear regression'].shape
y_test_predict['Ridge'].shape
y_test_predict['RandomForestRegressor'].shape


# **Ridge Regression**

# In[ ]:


## Processing output
# 11. Each face should have this dimension
image_shape = (64, 64)

## 11.1 For 'Ridge' regression
##      We will have total images as follows:
#      Per esimator, we will have n_faces * 2
#      So total - n_estimators * n_faces * 2
#      Fig size should be accordingly drawn

# 11.2 Total faces per estimator: 2 * n_faces

plt.figure(figsize=(  n_faces * 1, n_faces))

j = 0

for i in range(n_faces):
    actual_face =    test[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['Ridge'][i]))
    j = j+1
    plt.subplot(n_faces/2,4,j)
    y = actual_face.reshape(image_shape)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(n_faces/2,4,j)
    x = completed_face.reshape(image_shape)
    imshow(y)

plt.show()


# **Extra trees Regression**

# In[ ]:


## 12. For 'Extra trees' regression
plt.figure(figsize=( 2 * n_faces * 2, 5))
j = 0
for i in range(n_faces):
    actual_face =    test[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['Extra trees'][i]))
    j = j+1
    plt.subplot(4,5,j)
    y = actual_face.reshape(image_shape)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(4,5,j)
    x = completed_face.reshape(image_shape)
    imshow(y)
  
plt.show()


# **Linear regression regression**

# In[ ]:


## 13. For 'Linear regression' regression
plt.figure(figsize=( 2 * n_faces * 2, 5))
j = 0
for i in range(n_faces):
    actual_face =    test[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['Linear regression'][i]))
    j = j+1
    plt.subplot(4,5,j)
    y = actual_face.reshape(image_shape)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(4,5,j)
    x = completed_face.reshape(image_shape)
    imshow(y)
  
plt.show()


# **K-nn regression**

# In[ ]:


## For '"K-nn' regression
plt.figure(figsize=(  n_faces * 1, n_faces))
j = 0
for i in range(5):
    actual_face =    test[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['K-nn'][i]))
    j = j+1
    plt.subplot(4,5,j)
    y = actual_face.reshape(image_shape)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(4,5,j)
    x = completed_face.reshape(image_shape)
    imshow(y)
  
plt.show()


# **RandomForestRegressor' regression**

# In[ ]:


# For 'RandomForestRegressor' regression
plt.figure(figsize=(  n_faces * 1, n_faces))
j = 0
for i in range(5):
    actual_face =    test[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['RandomForestRegressor'][i]))
    j = j+1
    plt.subplot(4,5,j)
    y = actual_face.reshape(image_shape)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(4,5,j)
    x = completed_face.reshape(image_shape)
    imshow(y)
  
plt.show()

