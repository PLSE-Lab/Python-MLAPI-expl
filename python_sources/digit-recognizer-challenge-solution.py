#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import os
print("libraries installed successfully !")


# In[ ]:


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
print("data imported sucessfully !")


# In[ ]:


print("No. of images: %d" %len(train_data))
train_data.head()


# In[ ]:


train_data.tail()


# In[ ]:


print("No. of images %d" %len(test_data))
test_data.head()


# In[ ]:


test_data.tail()


# In[ ]:


image1 = train_data.loc[0, train_data.columns != "label"]
plt.imshow(np.array(image1).reshape((28,28)), cmap="gray")
plt.show()


# In[ ]:


image2 = test_data.loc[0, test_data.columns != "label"]
plt.imshow(np.array(image2).reshape((28,28)),cmap = "gray")
plt.show()


# In[ ]:


plt.hist(image1)
plt.xlabel("px intensity")
plt.ylabel("count")
plt.show()


# In[ ]:


plt.hist(image2)
plt.xlabel("pixel intensity")
plt.ylabel("count")
plt.show()


# In[ ]:


train_image=train_data.loc[:, train_data.columns !="label"]/255
train_labels = train_data.loc[:, :]/255
test_image = test_data.loc[:, test_data.columns !="labels"]/255
test_labels = test_data.loc[:, :]/255
test_data = test_data.loc[:, :]/255
train_data = train_data.loc[:, :]/255
x_train, x_test, y_train, y_test = train_test_split(test_image, test_labels, test_size = 0.25, random_state =1)
x_train = x_train.values.ravel()
y_train = y_train.values.ravel()
x_test = x_test.values.ravel()
y_test = y_test.values.ravel()
print("data cleaned and split successfully !")


# In[ ]:


sample_size = len(x_train)
x_train_sample = x_train[0:sample_size]
y_train_sample = y_train[0:sample_size]
x_test_sample = x_test[0:sample_size]
y_test_sample = y_test[0:sample_size]

print("Data samples created")


# In[ ]:


x_train_use = x_train_sample.reshape(1, -1)
y_train_use = y_train_sample.reshape(1, -1)
x_test_use = x_test_sample.reshape(1, -1)
y_test_use = y_test_sample.reshape(1, -1)
print("samples made usable successfully !")


# In[ ]:


x_train.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# In[ ]:


x_test.shape


# In[ ]:


model = SVC()
model.fit(x_train_sample, y_train_sample)
print("Model trained")


# In[ ]:


#training metrics
train_predicts = model.predict(x_train_sample)
train_acc = round(accuracy_score(y_train_sample, train_predicts) * 100)
print("Training Accuracy: %d%%" %train_acc)

#test metric
test_predicts = model.predict(x_test_sample)
test_acc = round(accuracy_score(y_test_sample, test_predicts) * 100)
print("Training Accuracy: %d%%" %test_acc)


# In[ ]:




