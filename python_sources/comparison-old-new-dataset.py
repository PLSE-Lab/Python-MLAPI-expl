#!/usr/bin/env python
# coding: utf-8

# # Just comparison Old & New Dataset for understanding

# In[ ]:


import numpy as np 
import pandas as pd 

#=== New Dataset
train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
#=== Old Dataset
train_old = pd.read_csv('../input/toxic-old-dataset/train.csv')
test_old = pd.read_csv('../input/toxic-old-dataset/test.csv')

#=== print Length of dataset
print("Length\ntrain_new:\t{}\ntest_new:\t{}\ntrain_old:\t{}\ntest_old:\t{}\n".format(len(train), len(test), len(train_old), len(test_old)))


# In[ ]:


#=== look the new dataset
train.head()


# # How many datas in train and test comes from old dataset?

# In[ ]:


train2train = train['comment_text'].isin(train_old['comment_text']).sum()
test2train = train['comment_text'].isin(test_old['comment_text']).sum()

train2test = test['comment_text'].isin(train_old['comment_text']).sum()
test2test = test['comment_text'].isin(test_old['comment_text']).sum()

print("old train to new train:\t{} \nold test to new train:\t{} \nold train to new test:\t{} \nold test to new test:\t{}\n"
      .format(train2train, test2train, train2test, test2test))


# We can see that new train dats is consist of old train & old test data <br>And almost the new test data comes newly
# <br>
# However, there is a little quantity gap  between** old train data** and ***new train data from old train *** 

# In[ ]:


len(train_old) - train2train 


# # What is it?

# In[ ]:


# missing old train data
train_old[train_old['comment_text']. isin(train['comment_text'])==False]


# it is surprsing that # of missing train data is same as # of staying test data which is 114. (but, nothing relevant)
# # Additionally look new test data from old test data<br>

# In[ ]:


# test data from old test data
test_old[test_old['comment_text']. isin(test['comment_text'])]


# ## So we found 114 labeled data missing. However, I don't know that we can use them for competition.<br> I will ask it to the hosts through the discussion.
# 
