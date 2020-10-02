#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import random
import datetime

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
train = pd.read_csv("../input/digit-recognizer/train.csv")


# We are gonna detect numbers with this dataset with KNN algorithm and see what the result will be ? 
# <br>First of all we look at our dataset and prepare it for learning algorithm.</br>
# <br>Previously we analyse this dataset with logisticRegression algorithm and the result was successful</br>

# In[ ]:


print(train.shape)
train.head(10)


# So we've got 42000 images in our dataset with size $\sqrt{785} \times \sqrt{785} \approx 28 \times 28$
# <br>But we should reshape this dataset for showing images or learning algorith</br>

# Lets see some of these images, randomly![](http://)

# In[ ]:


X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test.values.astype('float32')


# In[ ]:


plt.figure(figsize=(25,10))
X_train = X_train.reshape(X_train.shape[0], 28, 28)
for i in range(10):
    rand = random.randrange(len(X_train))
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[rand], cmap=plt.get_cmap('gray'))
    plt.title(y_train[rand]);
X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values


# In[ ]:


classifier = KNeighborsClassifier(n_neighbors=11)
classifier.fit(X_train, y_train)


# Wow, So lets check the result of model

# At the first, we check the model on train dataset and then if the result be good, we will check it on test dataset

# In[ ]:


rand = random.randrange(len(X_train))
plt.imshow(X_train[rand].reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.title("real: %s , predicted : %s" % (y_train[rand], classifier.predict(np.reshape(X_train[rand], (1, -1)))));


# The result was successfull!! , So lets check it on test dataset

# In[ ]:


plt.figure(figsize=(18,15))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    rand = random.randrange(len(X_train))
    plt.imshow(X_train[rand].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.title("predicted : %s" % (classifier.predict(np.reshape(X_train[rand], (1, -1)))));


# Result are so successful, let take output from them

# In[ ]:


now = datetime.datetime.now()
labels = classifier.predict(X_test[:100]) # probably its gonna take too much time
print((datetime.datetime.now() - now).microseconds)
labels

