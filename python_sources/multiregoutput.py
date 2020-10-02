#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reset', '-f')
import gc
gc.collect()


# In[ ]:


import numpy as np


# In[ ]:


import matplotlib.pyplot as plt   
from skimage.io import imshow


# In[ ]:


from sklearn.datasets import fetch_olivetti_faces


# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV


# In[ ]:


image_data = np.load("../input/olivetti_faces.npy")
print(image_data.shape)


# In[ ]:


targets = np.load("../input/olivetti_faces_target.npy")
targets.shape      


# In[ ]:


firstImage = image_data[0]
imshow(firstImage) 


# In[ ]:


data = image_data.reshape(image_data.shape[0], image_data.shape[1] * image_data.shape[2])
data.shape


# In[ ]:


targets < 30 


# In[ ]:


train = data[targets < 30]
train.shape


# In[ ]:


test = data[targets >= 30]
test.shape


# In[ ]:


n_faces = test.shape[0]//10
n_faces


# In[ ]:


face_ids = np.random.randint(0 , 100, size =n_faces)
face_ids


# In[ ]:


test = test[face_ids, :] 
test.shape


# In[ ]:


n_pixels = data.shape[1]
n_pixels


# In[ ]:


X_train = train[:, :(n_pixels + 1) // 2]
X_train


# In[ ]:


y_train = train[:, n_pixels // 2:]
y_train


# In[ ]:


X_test = test[:, :(n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10,
                                       max_features=32,
                                       random_state=0),
    "K-nn": KNeighborsRegressor(),
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
    "RandomForestRegressor": RandomForestRegressor()    
}
y_test_predict = dict()
for name, estimator in ESTIMATORS.items():     
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)
y_test_predict['RandomForestRegressor'].shape


# In[ ]:


image_shape = (64, 64)


# In[ ]:


plt.figure(figsize=( 2 * n_faces * 2, 5))
j = 0
for i in range(n_faces):
    actual_face =test[i].reshape(image_shape)
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


For 'RandomForestRegressor' regression
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

