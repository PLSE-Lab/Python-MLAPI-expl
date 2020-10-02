#!/usr/bin/env python
# coding: utf-8

# # Auto-PyTorch Implementation
# 
# - For more documentation see: https://github.com/automl/Auto-PyTorch and https://www.automl.org/automl/autopytorch/
# - Please upvote this kerel if you found it helpful! :)

# In[ ]:


#!git clone https://github.com/automl/Auto-PyTorch.git


# In[ ]:


#cd Auto-PyTorch


# In[ ]:


#!cat requirements.txt | xargs -n 1 -L 1 pip install


# In[ ]:


#!python setup.py install


# ## Import Required Libraries

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from autoPyTorch import AutoNetClassification

# data and metric imports
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics


# ## Loading The Data

# In[ ]:


#cd ..


# In[ ]:


train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")


# In[ ]:


Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1) 


# ### Check for Null or Missing Values

# In[ ]:


X_train.isnull().any().describe()


# In[ ]:


test.isnull().any().describe()


# ### Normalization, Reshaping, Label Encoding

# In[ ]:


X_train = X_train/255.0
test = test/255.0

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# ## Examining the Training Data

# In[ ]:


# plotting the first five training images
fig = plt.figure(figsize=(20,20))
for i in range(5):
    ax = fig.add_subplot(1, 5, i+1, xticks=[], yticks=[])
    ax.imshow(X_train[i].reshape(28,28), cmap='gray')
    ax.set_title(str(Y_train[i]))


# ## Applying One-hot Encoding to Labels

# In[ ]:


Y_train = np_utils.to_categorical(Y_train, num_classes=10)

# print the first five encoded training labels
print('One-hot Encoded labels:')
print(Y_train[:10])


# ### Splitting the Data (For built-in)

# In[ ]:


X, y = sklearn.datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test =         sklearn.model_selection.train_test_split(X, y, random_state=69)


# # Auto-PyTorch Model

# In[ ]:


autoPyTorch = AutoNetClassification("tiny_cs",  # config preset
                                    log_level='info',
                                    max_runtime=300,
                                    min_budget=30,
                                    max_budget=90)


# In[ ]:


autoPyTorch.fit(X_train, y_train, validation_split=0.3)


# In[ ]:


Y_test = autoPyTorch.predict(X_test)


# In[ ]:


Y_test_true = autoPyTorch.predict(test)


# In[ ]:


submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
submission['Label'] = Y_test
submission.to_csv('submission.csv',index=False)


# In[ ]:


submission_true = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
submission_true['Label'] = Y_test_true
submission_true.to_csv('submission_true.csv',index=False)

