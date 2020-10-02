#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Credit where credit is due: I got started by following parts of this notebook: https://www.kaggle.com/pulkitmehtawork1985/simple-model-to-get-into-top-20/output

# The firs step is to load that data from the files on the Kaggle server:

# In[ ]:


train = pd.read_csv("../input/learn-together/train.csv")
test = pd.read_csv("../input/learn-together/test.csv")
sub = pd.read_csv("../input/learn-together/sample_submission.csv")


# Next we isolate the parts of the training data we intend to use. `y` is for the "answers," that is, the category of tree. This is what we will ultimately be trying to predict, but for now we already have the answers. `X` is the type of information we will use to make a prediction. It includes elevation, slope, soil type, etc...  
# Note that X is capital and 'y' is lower case. I'm not sure of the reason for this convention, but it seems to be stardard, so I'm using it.

# In[ ]:


y = train["Cover_Type"] # cover type is the prediction we will want to make
X = train.drop(["Id", "Cover_Type"], axis = 1) # Id and Cover_Type are the two things we won't use
# to make a prediction


# Here's a look at that data:

# In[ ]:


X.head()


# In[ ]:


y.head()


# We need to import the tree module from sklearn because the houses the decision tree classifier we will be using.

# In[ ]:


from sklearn import tree


# This code creates the decision tree classifier.

# In[ ]:


clf = tree.DecisionTreeClassifier()


# And this bit fits it to our train data.

# In[ ]:


clf = clf.fit(X, y)


# In[ ]:





# In[ ]:





# Now let's look at the test data. (I dropped "Id" because it's not relevant to the predicion, and also because I dropped it above and it's import that the train and test data sets have the same shape and format.

# In[ ]:


Z = test.drop(["Id"], axis = 1)


# We can see that this data is similar to X.

# In[ ]:


Z.head()


# I'm looking at the shape to confirm that X and Z are compatible

# In[ ]:


X.shape


# In[ ]:


Z.shape


# They are because they have the same number of columns.

# And next is to generate the predictions from the test dataset using the decision tree classifier we created and trained above.

# In[ ]:


predicted = clf.predict(Z)


# In[ ]:


predicted


# As an easy way to ensure my submission matches the sample submission in format, I read the sample submission into a pandas data frame, then write the predicted values to the `"Cover_Type"` column. Note that this will overwrite the dummy data in that column from the sample submission document, but leave the column of Ids in place

# In[ ]:


sub = pd.read_csv("../input/learn-together/sample_submission.csv")


# In[ ]:


sub["Cover_Type"] = predicted


# Let's take a look at that submission:

# In[ ]:


sub.head()


# And finally, we just need to write that submission data to a csv so we can submit it.

# In[ ]:


sub.to_csv('DecTrePrediction.csv',index = False)


# In[ ]:




