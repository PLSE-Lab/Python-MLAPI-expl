#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #for plotting
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split


# **Read the data**

# In[ ]:


train = pd.read_csv("../input/train_sample.csv")
test = pd.read_csv("../input/test_sample.csv")


# In[ ]:


print(train.shape)
train.head(5)


# In[ ]:


print(test.shape) #checking for test data head
test.head(5)


# ### split the x and y from train and test

# In[ ]:


X_train = train.iloc[:,1:]
y_train = train.iloc[:,0]
print(X_train.shape)
print(y_train.shape)


#  **.loc :-** here you've to give the label of the indicies and which eitherbe a index or your columns
#  
#  **.iloc :-** you've to pass a nuerical indicies, the postive of your indicies

# In[ ]:


X_test = test.iloc[:,1:]
y_test = test.iloc[:,0]

print(X_test.shape)
print(y_test.shape)


# In[ ]:


import seaborn as sns


# In[ ]:


## Viasualize number of digits classes
plt.figure(figsize = (15,8))
g = sns.countplot(y_train, palette = "icefire")
plt.title("Number of digit classes")
y_train.value_counts()
        


# In[ ]:


#plot some samples
img = X_train.iloc[0].as_matrix()
img = img.reshape(28,28)
plt.imshow(img, cmap = 'gray')
plt.title(train.iloc[0,0])
plt.axis("off")
plt.show()


# In[ ]:


# plot some samples
img = X_train.iloc[3].as_matrix()
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train.iloc[3,0])
plt.axis("off")
plt.show()


# ## Try to visually look at the data by writing one of the records to csv

# In[ ]:


x1 = X_train.iloc[0,:].values.reshape(28,28)
x1[x1 > 0] = 1
x1 = pd.DataFrame(x1)
x1.to_csv("one.csv")


# ## Look at 5 sample records from train and test

# In[ ]:


train_sample = np.random.choice(range(0,X_train.shape[0]),replace=False,size=5)
test_sample = np.random.choice(range(0,X_test.shape[0]),replace=False,size=5)


# In[ ]:


train_sample


# In[ ]:


test_sample


# In[ ]:


plt.figure(figsize=(10,5))
for i,j in enumerate(train_sample):
    plt.subplot(2,5,i+1)
    plt.imshow(X_train.iloc[j,:].values.reshape(28,28))
    plt.title("Digit:"+str(y_train[j]))
    plt.gray()


# In[ ]:


plt.figure(figsize=(10,5))
for i,j in enumerate(test_sample):
    plt.subplot(2,5,i+1)
    plt.imshow(X_test.iloc[j,:].values.reshape(28,28))
    plt.title("Digit:"+str(y_test[j]))
    plt.gray()


# ## Build knn model with k = 3. Check the model performance

# In[ ]:


knn_classifier = KNeighborsClassifier(n_neighbors=3,weights="distance",algorithm="brute")
knn_classifier.fit(X_train, y_train)


# In[ ]:


pred_train = knn_classifier.predict(X_train) 
pred_test = knn_classifier.predict(X_test)


# In[ ]:


#Build confusion matrix and find the accuracy of the model
cm_test = confusion_matrix(y_pred=pred_test, y_true=y_test)

print(cm_test)


# In[ ]:


# Accuracy: 

sum(np.diag(cm_test))/np.sum(cm_test)


# In[ ]:


print("Accuracy on train is:",accuracy_score(y_train,pred_train))
print("Accuracy on test is:",accuracy_score(y_test,pred_test))


# In[ ]:


# Look at the some misclassified points
misclassified = y_test[pred_test != y_test] 


# In[ ]:


## First 5 misclassified points
misclassified.index[:5]


# In[ ]:


plt.figure(figsize=(10,5))
for i,j in enumerate(misclassified.index[:5]):
    plt.subplot(2,5,i+1)
    plt.imshow(X_test.iloc[j,:].values.reshape(28,28))
    plt.title("Digit:"+str(y_test[j])+" "+"Pred:"+str(pred_test[j]))
    plt.gray()


# In[ ]:




