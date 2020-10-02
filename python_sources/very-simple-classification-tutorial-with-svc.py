#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# For this tutorial we use **Support Vector Classification.**.
# 
# First of all, we'd like to say this is not the best aproach to solve digit recognition problem. But we choose that because it is simple and very useful for beginner machine learner.
# 
# Despite the fact that SVC is not the best way to solve this problem, we could get a good results. You can see details about SVC at [sklearn webpage](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

# ### Python imports

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm

import warnings

# Trick for plotting inline in Jupter Notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Ignoring warnings
warnings.filterwarnings("ignore")


# ### Utilities method

# In[ ]:


def show_some_sample_images(dataset, k=5):
    '''
        Shows k random image samples from dataset.
        
        In the train dataset, there are 728 columns that represent the image.
        We need to reshape this 728 x 1 array to 28 x 28, in order to plot the image correctly.
        You can see it at line: "img.reshape((28, 28))"
        
        :param dataset: Pandas DataFrame
        :param k: Number of images to be shown
    '''
    sample = dataset.sample(n=k)
    for index in range(k):
        img = sample.iloc[index].as_matrix()
        img = img.reshape((28, 28))
        plt.figure(figsize = (20,2))
        plt.grid(False)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
        plt.show()


# ### Loading the data

# In[ ]:


data = pd.read_csv('../input/train.csv')
data.head()


# ### Separating the dataset into images and labels
# 
# As we can see label is the first column of csv and the others columns are the pixels of digit image.
# 
# Note that we are using only a few datapoints in order to save running time.

# In[ ]:


labels = data.iloc[0:10000, :1]
images = data.iloc[0:10000, 1:]


# ### Showing some images from dataset

# In[ ]:


show_some_sample_images(images)


# ### Splitting the data into testing and training data points

# In[ ]:


train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)


# ### Using SVC with sklearn
# 
# [sklearn](http://scikit-learn.org) is a very simple tool as you can see bellow.
# 
# In the next cell we will just create a classifier and train with our training data points.

# In[ ]:


clf = svm.SVC(kernel='linear')
clf = clf.fit(train_images, train_labels.values.ravel())


# ### Validating the classifier
# 
# Now we use the testing data points to validate and get a score for our aproach. The score we got is not too bad for a SVC classifier!

# In[ ]:


print(clf.score(test_images, test_labels))


# ### Submission
# 
# For making competition submission we need to load and make predictions to unlabeled images from **test.csv**. 

# In[ ]:


test_data=pd.read_csv('../input/test.csv')
results=clf.predict(test_data)

test_data['Label'] = pd.Series(results)
test_data['ImageId'] = test_data.index +1
sub = test_data[['ImageId','Label']]

sub.to_csv('submission.csv', index=False)

