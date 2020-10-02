#!/usr/bin/env python
# coding: utf-8

# **About PYCARET library**

# PyCaret is an open source machine learning library in Python to train and deploy supervised and unsupervised machine learning models in a low-code environment. It is very useful package if you are completely new for machine learning and everything is self explanatory if you are well known with machine learning. 

# In[ ]:


get_ipython().system('pip install pycaret')


# **Packages required**

# In[ ]:


import pandas as pd
from pycaret.classification import create_model,setup
from pycaret import classification


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
train.head()


# In[ ]:


test = pd.read_csv('../input/titanic/test.csv')
test.head()


# **Set up the model**
# 
# 1) Mention training data, target feature
# 
# 2) missing value imputation: default:mean, median is the another option
# 
# 3) normalization: default: z-score. minmax,maxabs,robust is optional
# 
# 4) outlier treatment: linear dimensionality reduction
# 

# In[ ]:


classification_setup = setup(data= train, target='Survived',remove_outliers=True,normalize=True,normalize_method='minmax',
                            ignore_features= ['Name'])


# **Comparing the model and choosing best model for prediction**

# In[ ]:


classification.compare_models()


# In[ ]:


rc = create_model('ridge')


# In[ ]:


pred = classification.predict_model(rc, data = test)


# In[ ]:


output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred.Label})
output.to_csv('submission.csv', index=False)


# In[ ]:


output.head()


# **AMAZING**
# 
# thats it!!!!!
# 
# try this. I hope you like this :)

# In[ ]:




