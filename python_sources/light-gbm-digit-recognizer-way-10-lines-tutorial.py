#!/usr/bin/env python
# coding: utf-8

# # **PLEASE UPVOTE !!!!!**
# # **If you find this notebook helpful !!! Happy Learning**

# **Light GBM+Digit recognizer - Tutorial - QUICK Solution - 15-25 Lines of Code **
# 
# 
# 
# # *Handwritten Digit Recognition (MNIST Dataset)*
# 
# History of Handwritten Digit dataset
# Modified National Institute of Standards and Technology database (MNIST dataset) is a large dataset of handwritten digits which is widely used in image processing and machine learning. The set of images in the MNIST database is a combination of two of NIST's databases: Special Database 1 and Special Database 3. Special Database 1 and Special Database 3 consist of digits written by high school students and employees of the United States Census Bureau, respectively.
# 
# # *Task*
# Classify the images in 10 class, i.e., [0-9], inclusively.
# 
# Please post comment/feedback for the same. Thank you 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


# In[ ]:


#read the data set
digits_train = pd.read_csv("../input/train.csv")
digits_test = pd.read_csv("../input/test.csv")
sample = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


#head
digits_train.head()


# In[ ]:


digits_test.head()


# In[ ]:


four = digits_train.iloc[3,1:]
four.shape


# In[ ]:



four= four.values.reshape(28,28)
plt.imshow(four,cmap='gray')


# In[ ]:


#visuallise the array
print(four[5:-5,5:-5])


# In[ ]:


#avearage values/distributions of features
description = digits_train.describe()
description


# In[ ]:


num_class = len(digits_train.iloc[:,0].unique())


# In[ ]:


x_train= digits_train.iloc[:,1:]
y_train=digits_train.iloc[:,0]

x_test = digits_test.values
y_test=digits_test.iloc[:,0]

#rescaling the feature
from sklearn.preprocessing import scale
x_train = scale(x_train)
x_test=scale(x_test)

#print
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


import lightgbm as lgb
print ('Training lightgbm')

lgtrain = lgb.Dataset(x_train, y_train)
lgval = lgb.Dataset(x_test, y_test)

# params multiclass
params = {
          "objective" : "multiclass",           
          "max_depth": -1,
           "num_class":num_class,
          "learning_rate" : 0.01,                 
          "verbosity" : -1 }

model = lgb.train(params, lgtrain, 500, valid_sets=[lgtrain, lgval], early_stopping_rounds=750, verbose_eval=200)


# In[ ]:


# predict results
results = model.predict(x_test)

# select the index's with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)


# In[ ]:


submission.head()

