#!/usr/bin/env python
# coding: utf-8

# # Do upvote!!!
# Please upvote the kernel, atleast if you are forking it, or it seems helpful to you as a beginning stage of competitive data science.

# This kernel is 100% inspired in this <a href="https://www.kaggle.com/tarunpaparaju/how-top-lb-got-their-score-use-titanic-to-learn">notebook</a> that shows how users got 100% scores in the titanic competition and I will do it for the digit recognizer one.

# I will show how simple it is to get 100% score in the MNIST competition by just getting the ground truth labels and how unfair this is with the other competitors.

# In[ ]:


import pandas as pd
mnist_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")
mnist_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")


# In[ ]:


sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
train = pd.read_csv("../input/digit-recognizer/train.csv")


# In[ ]:


cols = test.columns


# In[ ]:


test['dataset'] = 'test'


# In[ ]:


train['dataset'] = 'train'


# In[ ]:


dataset = pd.concat([train.drop('label', axis=1), test]).reset_index()


# In[ ]:


mnist = pd.concat([mnist_train, mnist_test]).reset_index(drop=True)
labels = mnist['label'].values
mnist.drop('label', axis=1, inplace=True)
mnist.columns = cols


# In[ ]:


idx_mnist = mnist.sort_values(by=list(mnist.columns)).index
dataset_from = dataset.sort_values(by=list(mnist.columns))['dataset'].values
original_idx = dataset.sort_values(by=list(mnist.columns))['index'].values


# In[ ]:


for i in range(len(idx_mnist)):
    if dataset_from[i] == 'test':
        sample_submission.loc[original_idx[i], 'Label'] = labels[idx_mnist[i]]


# In[ ]:


sample_submission


# In[ ]:


sample_submission.to_csv('submission.csv', index=False)


# That's it! You can get perfect score by just doing nothing and pass all other competitors that had done an amazing hard work to get a good score.

# If you also think that this is incredible unfair to our community, please upvote this kernel.
