#!/usr/bin/env python
# coding: utf-8

# ## Hi, all.
# I'm a business analyst in Coupang which is the fastest growing e-commerce company in South Korea.   
# I use Python as data analytics language fluently and have some experience with machine learning basics. But it is the first time to computer vision.   
# I'll simply use **Scikit learn** Random Forest Classifier in this kernal.  

# In[137]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# In[138]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ## Look through the data set

# In[166]:


X_train = train.iloc[:, 1:]
Y_train = train.iloc[:, 0]
X_test = test


# This is a second image in training set.  
# I think linear model is not suit for this. That is why I choose random forest classifier. 

# In[140]:


plt.imshow(train.iloc[1, 1:].values.reshape(28, 28), cmap='gray')
plt.title('label: {}'.format(train.iloc[1, 0]))


# Labels are well distributed.

# In[141]:


train['label'].hist(bins=20)


# Examining the pixel value.   
# These images are not actually black and white. They are gray scale.  

# In[142]:


# examining the pixel values
# These images are not actually black and white. They are gray scale (0-255).
plt.hist(train.ix[:, 1:].iloc[1])


# ## Training our model
# * First, I use the scikit learn random forest classifier.

# In[164]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score


# In[144]:


train_images, vali_images, train_labels, vali_labels = train_test_split(X_train, Y_train, train_size=0.8, random_state=0)


# In[145]:


print("shape of train images: {}".format(train_images.shape))
print("shape of validation images: {}".format(vali_images.shape))


# In[162]:


forest = RandomForestClassifier(n_estimators=100, random_state=5)
forest.fit(train_images, train_labels)


# In[167]:


print('accuracy of training set: {}'.format(forest.score(train_images, train_labels)))
print('accuracy of validation set: {}'.format(forest.score(vali_images, vali_labels)))


# In[165]:


cross_val_score(forest, X_train, Y_train)


# ## Submission

# In[154]:


submission = pd.DataFrame()
submission['Label'] = forest.predict(X_test)
submission.index += 1
submission.index.name = 'ImageId'


# In[155]:


plt.imshow(test.iloc[0, :].values.reshape(28, 28), cmap='gray')
plt.title('label: {}'.format(train.iloc[0, 0]))


# In[156]:


submission.head(2)


# In[ ]:


submission.to_csv('./submission.csv')


# ## References
# * Charlie H., A Beginner's Approach to Classification, https://www.kaggle.com/archaeocharlie/a-beginner-s-approach-to-classification
# 
