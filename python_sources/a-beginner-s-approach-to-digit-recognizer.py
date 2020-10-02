#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# ###This is my attempt to classify digits using conventional SVM approac. I am not using neural network  or any other deep learning methods because this is meant for beginners.This notebook is meant to be for someone who might not know where to start. Suggestions are welcomed.

# In[69]:


import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing


# ## Loading the data
# - We use panda's [read_csv][1]  to read train.csv into a [dataframe][2].
# - Then we separate our images and labels for supervised learning. 
# - We also do a [train_test_split][3] to break our data into two sets, one for training and one for testing. This let's us measure how well our model was trained by later inputting some known test data
# 
#   [1]: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
#   [2]: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html#pandas.DataFrame
#   [3]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#   
#   I am standardizing the pixel values for a better result. For this I have imported preprocessing module from Scikit-learn

# In[70]:


labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,:1]
images = preprocessing.scale(images)
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)


# ## Training our model
# - First, we use the [sklearn.svm][1] module to create a [vector classifier][2]. 
# - Next, we pass our training images and labels to the classifier's [fit][3] method, which trains our model. 
# - Finally, the test images and labels are passed to the [score][4] method to see how well we trained our model. Fit will return a float between 0-1 indicating our accuracy on the test data set
# 
# ### Try playing with the parameters of svm.SVC to see how the results change. 
# 
# 
#   [1]: http://scikit-learn.org/stable/modules/svm.html
#   [2]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#   [3]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.fit
#   [4]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.score
#   [5]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.score

# In[71]:


clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)


# ## How did our model do?
# ### You should have gotten around 0.91, or 91% accuracy.  Let's just simplify our images by making them true black and white.
# 
# - To make this easy, any pixel with a value simply becomes 1 and everything else remains 0.
# - We'll plot the same image again to see how it looks now that it's black and white. Look at the histogram now.

# In[72]:


test_images[test_images>0]=1
train_images[train_images>0]=1


# ## Retraining our model
# ### We follow the same procedure as before, but now our training and test sets are black and white instead of gray-scale. Our score still isn't great, but it's a huge improvement.

# In[73]:


clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)


# ## Labelling the test data
# ### Now for those making competition submissions, we can load and predict the unlabeled data from test.csv.  We then output this data to a results.csv for competition submission.

# In[76]:


test_data=pd.read_csv('../input/test.csv')
test_data= preprocessing.scale(test_data)
test_data[test_data>0]=1


# In[75]:


results = clf.predict(test_data)


# In[83]:


import numpy as np
results = pd.DataFrame(results)
results.index =results.index+1
results.columns =['Label']
results.to_csv('result.csv', index_label='ImageId')


# In[ ]:




