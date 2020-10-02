#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Let's make a submission by taking the average finishing placement and saying that everybody places there.
# To start, we need to load in the training set, and find that average.

# In[ ]:


train = pd.read_csv('../input/train_V2.csv') #loading in the training set
print(train.head()) #examining the first few rows of the training set


# In[ ]:


training_average_placement = train['winPlacePerc'].mean() #this calculates the average value in the "winPlacePerc" columns
print(training_average_placement)


# Now, let's create a submission file. I am going to make sure that my submission file matches the expected format by modifying the sample submission.

# In[ ]:


submission = pd.read_csv('../input/sample_submission_V2.csv')
submission['winPlacePerc'] = training_average_placement
print(submission.head())


#  Notice that by loading in the sample submission this way (without assigning an index column), we get this extra index column tagging along. We can fix this by assigning Id as the index, or by not saving that column when saving our submission. I will do the latter.

# In[ ]:


submission.to_csv("Everyone_Averaged.csv", index=False) #no bad characters in the csv, or you will get an error!


# I can then submit my solution by pressing "Commit" on my kernel, and going to the output tab. For more info, go to the "Kernels FAQ" tab of the competition overview page. 
