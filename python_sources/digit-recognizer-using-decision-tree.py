#!/usr/bin/env python
# coding: utf-8

# In this notebook I  have  performed digit  classification using a  decision tree

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
#constants

IMG_HEIGHT=28
IMG_WIDTH=28

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Loading the data**  
# using pandas.read_csv to load in the training data. 

# In[ ]:


loaded_images=pd.read_csv('../input/train.csv')
loaded_images.head()


# Split into images and labels.  

# In[ ]:


images=loaded_images.iloc[:,1:]
labels=loaded_images.iloc[:,:1]   # for the labels to be a dataframe . iloc[:,0] returns a Series  iloc[:,:1] returns a Dataframe
labels.head()


# further split into training and test sets

# In[ ]:


train_images,test_images,train_labels,test_labels=train_test_split(images,labels,test_size=0.2,random_state=13)


# In[ ]:


train_images.describe()


# Since out of all the features that are used to split a node, the one that maximizes the Information Gain is the one that the node is split  on.
# 
# *IG is defined in terms of the impurity measure "I" which could be defined in terms of either entropy or the gini index. Each of these indices describes how pure a node is . So entropy of 1 implies a very impure node and an entropy of 0 implies that all members of the node belong to the same class i.e a pure node. similar with the gini index. Scaling is not an issue here since the impurity functions are defined in terms of probabilities which are values between 0 and 1*  
# 
# Therefore features need not be scaled when dealing with DTs.  
#   
# **Classification using a Decision Tree**  
# using the [DecisionTreeClassifier in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

# In[ ]:


tree=DecisionTreeClassifier(criterion='gini',random_state=1)
tree.fit(train_images,train_labels)


# get both the train and the test scores.

# In[ ]:


tree.score(train_images,train_labels.values.ravel())


# possible overfitting on the training data.

# In[ ]:


tree.score(test_images,test_labels.values.ravel())


# An 85% accuracy on the test data.
# Spot check to see if the predictions are correct . Plotting the predictions as labels.

# In[ ]:


figr,axes=plt.subplots(figsize=(10,10),ncols=3,nrows=3)
axes=axes.flatten()
for i in range(0,9):
    jj=np.random.randint(0,test_images.shape[0])          #pick a random image
    axes[i].imshow(test_images.iloc[[jj]].values.reshape(IMG_HEIGHT,IMG_WIDTH))
    axes[i].set_title('predicted: '+str(tree.predict(test_images.iloc[[jj]])[0]))



# 

# **Submission **  
# 
# load the data in test.csv  
# predict using the model

# In[ ]:


new_data=pd.read_csv('../input/test.csv')
new_data.head(n=3)


# In[ ]:


y_pred=tree.predict(new_data)


# In[ ]:


y_pred.shape


# create a dataframe which will then be exported as a csv file for submissions.

# In[ ]:


submissions=pd.DataFrame({"ImageId":list(range(1,len(y_pred)+1)), "Label":y_pred})
submissions.head()


# In[ ]:


submissions.to_csv("mnist_decision_tree_submit.csv",index=False,header=True)


# In[ ]:


get_ipython().system('ls')


# In[ ]:




