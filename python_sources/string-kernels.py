#!/usr/bin/env python
# coding: utf-8

# Due to some restrictions in sklearn, the use of string kernels is not as straightforward as it should be. Luckily, I have written some code explaining the problem and a workaround. This code can easily be plugged into your custom notebooks.
# 
# The use of string kernels is a very common approach when creating models involving omics data. For this dataset it is recommended to consider the use of these when using support vector machines. The string kernel used in the notebook is a very simple (read: bad) example. Various amounts of string kernels have been created specifically for DNA interactions, which can be recovered from literature. The implementation of some more advanced kernels might be challenging, and might not necessarily result in good performances. Therefore I remind everybody that a good reasoning behind chosing different approaches (even when you find out they didn't work well) is more important than the actual performances. 

# In[ ]:


from sklearn import neighbors
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# 
# Input data files are available in the "../input/" directory.
# 
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# 

# In[ ]:



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# In[ ]:


#Dataset
data = pd.read_csv('../input/train_data.csv')
#X= np.array([['AAAAAAA'],['AAATTAA'], ['TTTTTTT'],['TTTAAAA'],['TTTTAAA'],['TATATAT']]) 
## class 0 if #A>#T, class 1 if #A<#T
#y= [0,0,1,0,1,1]
data.head()


# When using support vector machines, a list of default kernel functions is available (e.g. rbf,...). These functions use the kernel trick to compute the distance between different points in a higher dimensional space. With these distances, a support vector machine can be calculated to best differentiate the classes.
# When using string kernels, no kernel trick is used as the idea is to measure a distance between two strings using a custom function. This function determines the similarity of two strings (given a criteria) and are at the basis of how well the support vector machine will work.

# In[ ]:


# Example of string kernel
# compare poisition-wise 2 sequences(X & Y) and return similarity score.
def equal_elements(s1,s2):
    score = 0
    for i in range(len(s1)):
        score += (s1[i] == s2[i])*1 # This is an unoptimized way to do this. 
    return score

equal_elements("STRING","KERNEL")


# 
#  In sklearn, a problem exists where the inputs of the custom function given the support vector machine class object has to be of the type integer/float:

# In[ ]:


clf = SVC(kernel=equal_elements)
clf.fit(data['DNA_sequence'],data['RPOD']) # this producecs an error


# As a workaround, we use the following procedure:
#     1. create precomputed distance kernel
#     2. use function which obtains a distance given the index of the string kernel
#     3. obtain indices of training and testing set
#     
#  This is actually more advantaguous because some functions that calculate the string kernels require a lot of processing power. By precomputing these distances they only have to be computed once as the resulting string kernel can be saved locally.
#  

# In[ ]:


data.iloc[1,0]


# 

# ## Step 1

# In[ ]:


# We use only a small fraction of the data for demonstration purposes,
# be aware that the equal_elements function is unoptimized and will 
# take some time if executed on the whole dataset

size = 12
not_so_good_string_kernel = np.zeros((size, size))
for row in range(size):
    for column in range(size):
        not_so_good_string_kernel[row,column] = equal_elements(data.iloc[row, 0],data.iloc[column, 0])
not_so_good_string_kernel


# ## Step 2

# In[ ]:


def compose_kernel(row_idxs, col_idxs):
    row_idxs = np.array(row_idxs).astype(np.int)
    col_idxs = np.array(col_idxs).astype(np.int)
    select_kernel = np.zeros((len(row_idxs),len(col_idxs)))
    for i, row_idx in enumerate(row_idxs):
        for j, col_idx in enumerate(col_idxs):
            select_kernel[i,j] = not_so_good_string_kernel[row_idx,col_idx]  # Change to custom distance kernel
    
    return select_kernel

compose_kernel([5,2,3,1],[5,2,3,1]) # random example


# > ## Step 3

# In[ ]:


y = data['RPOD'].values
X_train_idx, X_test_idx, y_train, y_test = train_test_split(np.arange(size),y[:size], test_size=4) # OR USE KFoldStratified()
X_train_idx, X_test_idx, y_train, y_test


# In[ ]:


# KERNEL used for training
compose_kernel(X_train_idx, X_train_idx) # Distances between the training sequences


# In[ ]:


# KERNEL used for predictions
compose_kernel(X_train_idx, X_test_idx) # Distances between the training sequences and the testing sequences


# For training, distances between training samples and their corresponding  are used to fit a support vector machine

# In[ ]:


clf= SVC(kernel=compose_kernel)
clf.fit(X_train_idx.reshape(-1,1), y_train) # reshape X_train_idx to be 2D


# In[ ]:


pred = clf.predict(X_test_idx.reshape(-1,1))
print(pred, y_test)

