#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# For Data manipulation
import numpy as np

#For Plotting
import matplotlib.pyplot as plt   
from skimage.io import imshow

#Regressors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


images = np.load("../input/olivetti_faces.npy")
images.shape


# In[ ]:


targets = np.load("../input/olivetti_faces_target.npy")
targets.shape        


# In[ ]:


firstImage = images[0]
imshow(firstImage) 


# In[ ]:


# Flatten individual image
data = images.reshape(images.shape[0], images.shape[1] * images.shape[2])

# Flattened 64 X 64 array
data.shape


# In[ ]:


# Patition datasets into two (fancy indexing)
targets < 30 


# In[ ]:


# First 30 types of images out of 40 ie 30 * 10 =300
train = data[targets < 30]
train.shape


# In[ ]:


# Test on rest independent people  10 * 10 = 100
test = data[targets >= 30]
test.shape


# In[ ]:


# Test on a subset of people
#     Generate 10 random integers between 0 and 100
# // is unconditionally "flooring division",
n_faces = test.shape[0]//10
n_faces


# In[ ]:


face_ids = np.random.randint(0 , 100, size =n_faces)
face_ids


# In[ ]:


test = test[face_ids, :]   
test.shape


# In[ ]:


#Total pixels in any image
n_pixels = data.shape[1]
n_pixels


# In[ ]:


#Select upper half of the faces as predictors
X_train = train[:, :(n_pixels + 1) // 2]
X_train


# In[ ]:


#Lower half of the faces will be target(s)                 
y_train = train[:, n_pixels // 2:]
y_train


# In[ ]:


# Similarly for test data. Upper and lower half
X_test = test[:, :(n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]

# Fit multi-output estimators
ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10,
                                       max_features=32,
                                       random_state=0),
    "K-nn": KNeighborsRegressor(),
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
    "RandomForestRegressor": RandomForestRegressor()    
}

# Create an empty dictionary to collect prediction values
y_test_predict = dict()

# Fit each model by turn and make predictions
for name, estimator in ESTIMATORS.items():     
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)
y_test_predict['RandomForestRegressor'].shape


# In[ ]:


image_shape = (64, 64)


# In[ ]:


# For 'Ridge' regression
plt.figure(figsize=( 2 * n_faces * 2, 5))
j = 0
for i in range(n_faces):
    actual_face =    test[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['Ridge'][i]))
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


# In[ ]:


# For 'Extra trees' regression
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


# In[ ]:


# For 'Linear regression' regression
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


# In[ ]:


# For '"K-nn' regression
plt.figure(figsize=( 2 * n_faces * 2, 5))
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


# In[ ]:


# For 'RandomForestRegressor' regression
plt.figure(figsize=( 2 * n_faces * 2, 5))
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

