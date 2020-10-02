#!/usr/bin/env python
# coding: utf-8

# # Images to arrays

# In[ ]:


import numpy as np
import pandas as pd 
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_data_dir = '../input/train/images/'
train_label_dir = '../input/train/masks/'
test_data_dir = '../input/test/images/'


# In[ ]:


ls_tr = os.listdir(train_data_dir)
ls_te = os.listdir(test_data_dir)


# In[ ]:


for i in ls_tr:
    x = plt.imread(os.path.join(train_data_dir,i)).reshape(1,101,101,3)
    y = plt.imread(os.path.join(train_label_dir,i)).reshape(1,101*101,1)
    if i == ls_tr[0]:
        train_X = x
        train_Y = y
    else:
        train_X = np.append(train_X, x, axis=0)
        train_Y = np.append(train_Y, y, axis=0)
print('Shape of training datas is {}, shape of training labels is {}.'.format(train_X.shape, train_Y.shape))


# In[ ]:


for i in ls_te:
    x = plt.imread(os.path.join(test_data_dir,i)).reshape(1,101,101,3)
    if i == ls_te[0]:
        test_X = x
    else:
        test_X = np.append(test_X, x, axis=0)
print('Shape of test datas is {}.'.format(test_X.shape))


# ## Save Datas to .npy files for later use

# In[ ]:


np.save('X_train.npy', train_X)
np.save('Y_train.npy', train_Y)
np.save('X_test.npy', test_X)


# ## If you do like this kernel or find this kernel useful, please UPVOTE. Thanks
