#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
faces_image = np.load('../input/olivetti-faces/olivetti_faces.npy')
faces_target = np.load('../input/olivetti-faces/olivetti_faces_target.npy')


# In[ ]:


#print(faces_image)
faces_image


# In[ ]:


n_row = 64
n_col = 64
faces_image.shape


# In[ ]:


faces_data = faces_image.reshape(faces_image.shape[0], faces_image.shape[1] * faces_image.shape[2])
faces_data.shape


# In[ ]:


print(faces_target)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
from skimage.io import imshow
loadImage = faces_image[20]
imshow(loadImage) 


# In[ ]:


loadImage.shape


# **Faces recognition using eigenfaces and SVM**

# In[ ]:


from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# In[ ]:


n_samples = faces_image.shape[0]
# for machine learning we use the 2 data directly
X = faces_data
n_features = faces_data.shape[1]
# the label to predict is the id of the person
y = faces_target
n_classes = faces_target.shape[0]


# In[ ]:


print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# In[ ]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
print("Xtrain",Xtrain)
print("Length of Xtrain:",len(Xtrain))
print("Xtest",Xtest)
print("Length of Xtest:",len(Xtest))
print("ytrain",ytrain)
print("Length of ytrain:",len(ytrain))
print("ytest",ytest)
print("Length of ytest:",len(ytest))


# In[ ]:


# Compute a PCA (eigenfaces) on the olivetti dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, Xtrain.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(Xtrain)
print("done in %0.3fs" % (time() - t0))


# In[ ]:


eigenfaces = pca.components_.reshape((n_components, n_row, n_col))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
Xtrain_pca = pca.transform(Xtrain)
Xtest_pca = pca.transform(Xtest)
print("done in %0.3fs" % (time() - t0))


# In[ ]:


# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(Xtrain_pca, ytrain)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


# In[ ]:


# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(Xtest_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(ytest, y_pred))


# In[ ]:


print(confusion_matrix(ytest, y_pred, labels=range(n_classes)))


# In[ ]:


#Displaying Eigenfaces
fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(64, 64), cmap='bone')


# In[ ]:


loadeigen = eigenfaces[20]
imshow(loadeigen) 


# In[ ]:


# plot the single eigen face of the most significative eigenfaces
import cv2
img = cv2.imread('loadeigen',0)
plt.imshow(loadeigen,cmap = 'gray', interpolation = 'bicubic')
plt.show()


# In[ ]:


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# We see that these 150 components account for just over 90% of the variance. That would lead us to believe that using these 150 components, we would recover most of the essential characteristics of the data. To make this more concrete, we can compare the input images with the images reconstructed from these 150 components.

# Please help with suggestions to improve this article
