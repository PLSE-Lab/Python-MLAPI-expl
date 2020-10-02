#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Here we'll try to use many classification models and see how much accuracy they can provide.

# ## Importing dataset

# In[2]:


dogs_train_files = os.listdir('../input/dog vs cat/dataset/training_set/dogs')
dogs_train_files_size = len(dogs_train_files)
print(dogs_train_files_size)
limit = 1000


# In[3]:


dogs_training_data = [None] * limit


# In[4]:


j = 0
for i in dogs_train_files:
    if j < limit:
        dogs_training_data[j] = cv2.imread('../input/dog vs cat/dataset/training_set/dogs/' + i, cv2.IMREAD_GRAYSCALE)
        j += 1
    else:
        break


# In[5]:


dogs_training_data = np.array(dogs_training_data)


# In[6]:


dogs_training_data[0].shape


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


plt.imshow(dogs_training_data[0])
plt.plot()


# In[9]:


cats_train_files = os.listdir('../input/dog vs cat/dataset/training_set/cats')
cats_train_files_size = len(cats_train_files)
print(cats_train_files_size)


# In[10]:


cats_training_data = [None] * limit


# In[11]:


j = 0
for i in cats_train_files:
    if j < limit:
        cats_training_data[j] = cv2.imread('../input/dog vs cat/dataset/training_set/cats/' + i, cv2.IMREAD_GRAYSCALE)
        j += 1
    else:
        break


# In[12]:


cats_training_data = np.array(cats_training_data)


# In[13]:


plt.imshow(cats_training_data[123])
plt.show()


# ## EDA

# In[14]:


s00 = 0
s01 = 0
s10 = float('inf')
s11 = float('inf')
for i in range(limit):
    s00 = max(s00, dogs_training_data[i].shape[0])
    s01 = max(s01, dogs_training_data[i].shape[1])
    s10 = min(s10, dogs_training_data[i].shape[0])
    s11 = min(s11, dogs_training_data[i].shape[1])
print(s00, s01)
print(s10, s11)


# In[15]:


s00 = 0
s01 = 0
s10 = float('inf')
s11 = float('inf')
for i in range(limit):
    s00 = max(s00, cats_training_data[i].shape[0])
    s01 = max(s01, cats_training_data[i].shape[1])
    s10 = min(s10, cats_training_data[i].shape[0])
    s11 = min(s11, cats_training_data[i].shape[1])
print(s00, s01)
print(s10, s11)


# ## Image cleansing

# In[16]:


img_size = (256, 256)


# In[17]:


i = dogs_training_data[53]


# In[18]:


plt.imshow(i)
plt.show()


# In[19]:


l = img_size[0] - i.shape[0]
w = img_size[1] - i.shape[1]


# In[20]:


up = l // 2
lo = l - up
wdl = w // 2
wdr = w - wdl


# In[21]:


image = cv2.resize(i, img_size)


# In[22]:


print(image.shape)
plt.imshow(image)
plt.plot()


# In[24]:


m1, m2 = img_size
for j in range(limit):
    i = dogs_training_data[j]
    dogs_training_data[j] = cv2.resize( i, img_size)
    


# In[25]:


s = (0,0)
s2 = (702, 1050)
for i in range(limit):
    s = max(s, dogs_training_data[i].shape)
    s2 = min(s2, dogs_training_data[i].shape)
print(s)
print(s2)


# In[26]:


m1, m2 = img_size
for j in range(limit):
    i = cats_training_data[j]
    cats_training_data[j] = cv2.resize( i, img_size)
    


# In[27]:


s = (0,0)
s2 = (702, 1050)
for i in range(limit):
    s = max(s, cats_training_data[i].shape)
    s2 = min(s2, cats_training_data[i].shape)
print(s)
print(s2)


# In[28]:


j = 0
sum = 0
for i in cats_training_data:
    if i.shape != img_size:
        print(j)
        sum += 1
    j += 1


# In[29]:


j = 0
sum = 0
for i in dogs_training_data:
    if i.shape != img_size:
        print(j)
        sum += 1
    j += 1


# In[ ]:





# ## Vectorize the dataset

# In[30]:


flatten_size = img_size[0] * img_size[1]
m = len(dogs_training_data)


# In[31]:


for i in range(m):
    dogs_training_data[i] = np.ndarray.flatten(dogs_training_data[i]).reshape(flatten_size, 1)


# In[ ]:





# In[32]:


dogs_training_data = np.dstack(dogs_training_data)


# In[33]:


dogs_training_data.shape


# In[34]:


dogs_training_data = np.rollaxis(dogs_training_data, axis=2, start=0)


# In[35]:


dogs_training_data.shape


# In[36]:


m = len(cats_training_data)
for i in range(m):
    cats_training_data[i] = np.ndarray.flatten(cats_training_data[i]).reshape(flatten_size, 1)


# In[37]:


cats_training_data = np.dstack(cats_training_data)


# In[38]:


cats_training_data.shape


# In[39]:


cats_training_data = np.rollaxis(cats_training_data, axis=2, start=0)


# In[40]:


cats_training_data.shape


# In[41]:


dogs_training_data.shape


# In[42]:


dogs_training_data = dogs_training_data.reshape(m, flatten_size)


# In[43]:


cats_training_data = cats_training_data.reshape(m, flatten_size)


# In[44]:


dogs_training_data = pd.DataFrame(dogs_training_data)


# In[45]:


cats_training_data = pd.DataFrame(cats_training_data)


# In[46]:


dogs_training_data['is_cat'] = pd.Series(np.zeros(m), dtype=int)


# In[47]:


dogs_training_data.head()


# In[48]:


cats_training_data['is_cat'] = pd.Series(np.ones(m), dtype=int)


# In[49]:


cats_training_data.head()


# In[50]:


df = pd.concat([dogs_training_data, cats_training_data])


# In[51]:


df.head()


# In[ ]:





# ## Splitting data

# In[52]:


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# In[53]:


df = shuffle(df).reset_index()


# In[54]:


df.head()


# In[55]:


df = df.drop(['index'], axis = 1)
df.head()


# In[ ]:





# In[56]:


df.info()


# In[57]:


df_train, df_test = train_test_split(df, test_size = 0.15)


# In[58]:


df_train, df_validation = train_test_split(df_train, test_size = 0.176)


# In[59]:


df_train.info()


# In[60]:


df_validation.info()


# # Run basic algorithm
# Just to check its accuracy so that we can understand importance of CNN

# In[61]:


from sklearn.linear_model import LogisticRegression as Lr
from sklearn.ensemble import RandomForestClassifier as Rfc
from sklearn.ensemble import GradientBoostingClassifier as Gbc
from xgboost import XGBClassifier as Xgb


# ## Linear Regression

# In[62]:


df_train_y = df_train['is_cat']


# In[63]:


df_train_x = df_train.drop(['is_cat'], axis = 1)


# In[64]:


df_train_x.head()


# In[65]:


lr = Lr()


# In[66]:


lr.fit(df_train_x, df_train_y)


# In[67]:


lr.score(df_train_x, df_train_y)


# In[68]:


df_validation_y = df_validation['is_cat']


# In[69]:


df_validation_x = df_validation.drop(['is_cat'], axis = 1) 


# In[70]:


lr.score(df_validation_x, df_validation_y)


# ## Random forest classifier

# In[71]:


rfc = Rfc()


# In[72]:


rfc.fit(df_train_x, df_train_y)


# In[73]:


rfc.score(df_train_x, df_train_y)


# In[74]:


plt.plot(rfc.feature_importances_)
plt.show


# In[75]:


rfc.score(df_validation_x, df_validation_y)


# ## XGB

# In[76]:


xgb = Xgb()


# In[77]:


xgb.fit(df_train_x, df_train_y)


# In[78]:


xgb.score(df_train_x, df_train_y)


# In[79]:


plt.plot(xgb.feature_importances_)
plt.show


# In[80]:


xgb.score(df_validation_x, df_validation_y)


# ## Max Vote
# we'll take max vote from  all the above algo as  answer

# In[81]:


df_train_x = df_train_x.reset_index()
df_train_y = df_train_y.reset_index()
df_train_x = df_train_x.drop(['index'], axis = 1)
df_train_y = df_train_y.drop(['index'], axis = 1)
df_train_x.head()


# In[82]:


def max_vote(x):
    s1 = lr.predict(x)
    s2 = rfc.predict(x)
    s3 = xgb.predict(x)
    result = pd.DataFrame(s1 + s2 + s3, columns=['is_cat'])
    result['is_cat'] = pd.Series(np.where(result['is_cat'] > 1, 1, 0))
    return result


# In[83]:


result_maxvote = max_vote(df_train_x)


# In[84]:


result_maxvote.head()


# In[85]:


def check_accuracy(result, output):
    total = result.shape[0]
    true = (result.is_cat == output.is_cat).sum()
    print(true)
    print(true / total)
    return (true / total)


# In[86]:


check_accuracy(result_maxvote, df_train_y)


# In[87]:


result_validation =  max_vote(df_validation_x)


# In[88]:


check_accuracy(result_validation, df_validation_y)


# ## Logistic regression with PCA 

# In[89]:


from sklearn.decomposition import PCA


# In[100]:


pca = PCA(n_components=512)


# In[101]:


pca.fit(df_train_x)


# In[102]:


plt.plot(pca.explained_variance_ratio_)
plt.show()


# In[103]:


plt.plot(pca.singular_values_)
plt.show()


# In[104]:


print(np.sum(pca.explained_variance_ratio_))


# In[105]:


df_train_pca_x = pd.DataFrame(pca.transform(df_train_x))
df_validation_pca_x = pd.DataFrame(pca.transform(df_validation_x))
df_train_pca_x.head()


# In[106]:


lr_pca = Lr()


# In[107]:


lr_pca.fit(df_train_pca_x, df_train_y, )


# In[108]:


lr_pca.score(df_train_pca_x, df_train_y)


# In[109]:


lr_pca.score(df_validation_pca_x, df_validation_y)


# ## SVM

# In[111]:


from sklearn.svm import LinearSVC as SVC


# In[112]:


svc = SVC()


# In[113]:


svc.fit(df_train_pca_x, df_train_y)


# In[114]:


svc.score(df_train_pca_x, df_train_y)


# In[115]:


svc.score(df_validation_pca_x, df_validation_y)


# ### As we can see all the algorithms that are used for classification problem are no more better than random guess.In next kernel we'll learn state of the art image classfication algorithm CNN.

# In[ ]:




