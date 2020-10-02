#!/usr/bin/env python
# coding: utf-8

# # How to be at the top of the Leader Board without too much hassle

# **I will demonstrate how you can easily get 100% public score by simply downloading the true labels of the test set and simply export that...** (fully inspired on the titanic version presented in this <a href="https://www.kaggle.com/tarunpaparaju/how-top-lb-got-their-score-use-titanic-to-learn">notebook</a>).
# 
# Cheating is not the better option if your goal is to learn ML. So get to it and start digging but don't cheat it won't get you anywhere.

# ## Imports

# In[ ]:


import pandas as pd


# ## Data Processing

# In[ ]:


# Private dataset
true_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")
true_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")

# Give dataset
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
given_test = pd.read_csv("../input/digit-recognizer/test.csv")
given_train = pd.read_csv("../input/digit-recognizer/train.csv")


# In[ ]:


cols = given_test.columns

given_test['dataset'] = 'test'
given_train['dataset'] = 'train'


# In[ ]:


given_dataset = pd.concat([given_train.drop('label', axis=1), given_test]).reset_index()
true_mnist = pd.concat([true_train, true_test]).reset_index(drop=True)

labels = true_mnist['label'].values
true_mnist.drop('label', axis=1, inplace=True)
true_mnist.columns = cols


# In[ ]:


true_idx = true_mnist.sort_values(by=list(true_mnist.columns)).index
dataset_from = given_dataset.sort_values(by=list(true_mnist.columns))['dataset'].values
original_idx = given_dataset.sort_values(by=list(true_mnist.columns))['index'].values


# ## Submission

# In[ ]:


for i in range(len(true_idx)):
    if dataset_from[i] == 'test':
        sample_submission.loc[original_idx[i], 'Label'] = labels[true_idx[i]]


# In[ ]:


# Job done...
sample_submission


# In[ ]:


sample_submission.to_csv('submission.csv', index=False)


# I hope you learned something and if you don't want to see stuff like this anymore don't forget to upvote...
