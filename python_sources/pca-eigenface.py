#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn import svm
import os


# In[ ]:


print(os.listdir('../input'))


# In[ ]:


import pickle 
# Load from file
pkl_filename = '../input/lfw_data_save.pkl'
with open(pkl_filename, 'rb') as file:
    lfw_people = pickle.load(file)
print(type(lfw_people))


# In[ ]:


n_samples, h, w = lfw_people.images.shape
print(n_samples)


# In[ ]:


for name in lfw_people.target_names:
    print(name)
print("image_colum_vector:", lfw_people.data.shape)
print("image size:", lfw_people.images.shape)
print("label:", lfw_people.target[:20])
print("name_label",lfw_people.target_names[lfw_people.target[:20]])


# In[ ]:


X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# In[ ]:


#plot a few images
# plt.imshow(X[0,:].reshape((h, w)), cmap=plt.cm.gray)
# plt.imshow(X[1,:].reshape((h, w)), cmap=plt.cm.gray)
for i in range(4 * 2):
        plt.subplot(4, 2, i + 1)
        plt.imshow(X[i,:].reshape((h, w)), cmap=plt.cm.gray)
#         plt.title(y[i], size=12)
        plt.title(target_names[y[i]]+"_label_"+str(y[i]),size= 10)
        plt.xticks(())
        plt.yticks(())


# In[ ]:


# #############################################################################
# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=41)
print("X_train",X_train.shape)
print("X_test",X_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)


# In[ ]:


# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 100 # 100

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='auto',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

model = LogisticRegression(C = 5, penalty='l2')
# print(model)
model.fit(X_train_pca, y_train)
parameters = model.coef_
predicted_classes = model.predict(X_test_pca)
accuracy = accuracy_score(predicted_classes,y_test)
print('The accuracy score using scikit-learn is {}'.format(accuracy))
print("The model parameters using scikit learn")
# print(parameters)
print("confusion_matrix")
print(confusion_matrix(predicted_classes,y_test,labels=range(n_classes)))
# print(classification_report(y_test, predicted_classes, target_names=target_names))

# Recall considered by columns
# Precision by considered by rows


# In[ ]:


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(predicted_classes, y_test, target_names, i)
                     for i in range(predicted_classes.shape[0])]

# print(prediction_titles)
plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()


# In[ ]:


# SVM classifier
clf = svm.SVC(C=1000.0, class_weight='balanced', gamma=0.005, kernel='rbf')
# clf = svm.SVR(C=1000.0, gamma=0.005, kernel='rbf')
clf = clf.fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)
accuracy = accuracy_score(y_pred,y_test)
print('The accuracy score using scikit-learn is {}'.format(accuracy))
print("The model parameters using scikit learn")
# print(parameters)
print("confusion_matrix")
print(confusion_matrix(y_pred,y_test,labels=range(n_classes)))
print(classification_report(y_test, y_pred, target_names=target_names)) 
# Recall considered by columns
# Precision by considered by rows  
# Check carefully: transposed  of confusion matrix


# In[ ]:


import numpy as np
# Reconstruction 
print("shape of eigenface",eigenfaces.shape)
print("eigenvector_shape",pca.components_.shape)
U = pca.components_.T
print("projection matrix shape",U.shape)
xx = X_train_pca[0]
print("first sample in training data_vector_shape",xx.shape)
A = (X_train[0].reshape(X_train[0].shape[0],1))
print("image_shape",A.shape)
yy = (A.T).dot(U)
print("projected face_shape",yy.shape)

#reconstruct
org = X_train[0].reshape((h,w))
plt.imshow(org,cmap=plt.cm.gray)
x_tilde = U.dot(yy.T)
print("reconstruction_shape",x_tilde.shape)
imgg = x_tilde.reshape((h,w))
plt.figure()
plt.imshow(imgg,cmap=plt.cm.gray)

