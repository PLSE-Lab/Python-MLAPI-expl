#!/usr/bin/env python
# coding: utf-8

# # MNITS Naive Bayes Classifier

# In[ ]:


import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from skimage.transform import pyramid_gaussian
from sklearn.model_selection import train_test_split

print(glob.glob("../input/*"))

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('precision', '2')


# In[ ]:


def draw_confusionmatrix(ytest, yhat):
    plt.figure(figsize=(10,7))
    cm = confusion_matrix(ytest, yhat)
    ax = sns.heatmap(cm, annot=True, fmt="d")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    acc = accuracy_score(ytest, yhat)
    print(f"Sum Axis-1 as Classification accuracy: {acc}")


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

X = train_df.iloc[:,1:]
y = train_df.label
label = sorted(y.unique())


# In[ ]:


plt.figure(figsize=(9,9))
for i in label:
    plt.subplot(1,10, i+1)
    img = np.array( X[y==i][1:2] ).reshape(28,28)
    plt.imshow(img)


# ## Draw Average Point of every class

# In[ ]:


m = y.unique().shape[0]
n = X.shape[1]

mu = np.zeros((m,n))
si = np.zeros((m,n))
for i in label:
    mu[i] = X[y==i].mean()
    si[i] = X[y==i].std()


# In[ ]:


plt.figure(figsize=(9,9))
for i in label:
    plt.subplot(3,4, i+1)
    img = np.array( mu[i] ).reshape(28,28)
    plt.imshow(img)


# ## Naive Bayse Classifier

# In[ ]:


model = GaussianNB()


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
model.fit(X_train, y_train)
yhat = model.predict(X_valid)

draw_confusionmatrix(y_valid, yhat)


# ## Naive Bayes Classification on Histogram as Feature

# In[ ]:


Xtmp = np.array(X).reshape(X.shape[0], 28, 28)
Xaxis1 = Xtmp.sum(axis=1)
Xaxis2 = Xtmp.sum(axis=2)


# ### 1. Axis 1 As Feature

# In[ ]:


Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xaxis1, y, random_state=0)
model.fit(Xtrain, ytrain)
yhat = model.predict(Xvalid)

draw_confusionmatrix(yvalid, yhat)


# ### 2. Axis 2 As Feature

# In[ ]:


Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xaxis2, y)
model.fit(Xtrain, ytrain)
yhat = model.predict(Xvalid)

draw_confusionmatrix(yvalid, yhat)


# ### 3. Axis 1 and 2 As Feature

# In[ ]:


Xaxises = np.hstack((Xaxis1, Xaxis2))


# In[ ]:


Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xaxises, y)
model.fit(Xtrain, ytrain)
yhat = model.predict(Xvalid)

draw_confusionmatrix(yvalid, yhat)


# ## Image Filters
# 
# ### 1. Sobel Filter

# In[ ]:


from scipy import ndimage
Xtmp = np.array(X).reshape(X.shape[0], 28, 28)
result = np.zeros_like(Xtmp)
for i in range(Xtmp.shape[0]):
    result[i] = ndimage.sobel(Xtmp[i])
    
result = result.reshape(result.shape[0], 28*28)


# In[ ]:


Xtrain, Xvalid, ytrain, yvalid = train_test_split(result, y)


# In[ ]:


model = GaussianNB()
model.fit(Xtrain, ytrain)
yhat = model.predict(Xvalid)
draw_confusionmatrix(yvalid, yhat)


# ### 2. Convolution

# In[ ]:


k = np.array([[1,1,1], [1,5,1], [1,1,1]])
images = np.zeros_like(Xtmp)
for i in range(Xtmp.shape[0]):
    images[i] = ndimage.convolve(Xtmp[i], k, mode='constant')

images = images.reshape(images.shape[0], 28*28)


# In[ ]:


Xtrain, Xvalid, ytrain, yvalid = train_test_split(result, y)

model = GaussianNB()
model.fit(Xtrain, ytrain)
yhat = model.predict(Xvalid)
draw_confusionmatrix(yvalid, yhat)


# In[ ]:




