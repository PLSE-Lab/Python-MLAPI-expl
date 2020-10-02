#!/usr/bin/env python
# coding: utf-8

# # An issue to Submission CSV Not Found
# As @Giba wrote in https://www.kaggle.com/c/deepfake-detection-challenge/discussion/121484, the "Submission CSV Not Found"  message comes when there is a bug in your notebook. But that bug could not have been seen on the train dataset or on the first test dataset (the one with only 400 videos). But the bug comes only on the big test dataset (the one with 4000 videos).<br>
# **In this notebook, I will explain a way to find where is your bug, without put exceptions handling every where.**<br>
# First, **let's suppose you have a very long notebook like the next cell**, which is doing a lot of things until your submission file to competition :

# In[ ]:


import numpy as np
import pandas as pd

# First part of your notebook
 
# Part2

# Part 3

# Part 4
def prob(): # Your very complex model to do predictions
    return .5

# Part 5
sf = pd.read_csv('../input/deepfake-detection-challenge/sample_submission.csv', index_col='filename')
sf["label"] = prob()
sf["label"].fillna(0.5, inplace = True)
sf.reset_index(inplace=True)
sf[['filename','label']].to_csv('submission.csv', index=False)

# Part 6
print("Shape {}\nmin {}\nmax {}\nNA {}".format(sf.shape
                , sf["label"].min(), sf["label"].max(), sf["label"].isna().sum()))
sf.head()


# Let's suppose you have worked the best as it is possible to do, let's suppose you don't have any bug with this notebook, neither on the train dataset nor the first test dataset. **Let's suppose you're ready to click on "Submit to competition"**.<br>
# And a few hours after submitting, you read the score of your submission : **"Submission CSV not found"**.<br>
# ARRRRRGR !!! What could have happen ?
# 
# This message told us that the fifth part of the code in the previous cell had not been executed. Why it did not ? Because the notebook had stopped before of course !<br>
# But where ?  In part 1, part 2 part 3 ? Let's use public score to find the place in your code where the bug is.
# 
# We know that there is 4000 videos in the big data set and that there 2000 are fake. So :

# In[ ]:


from sklearn.metrics import log_loss

y = np.concatenate([np.ones(2000), np.zeros(2000)], axis=0)

# If you submit a submission file with those values
for i in [0, 0.05, 0.1, .15, .2, .25, .3, .35, .4, .45, .5, 1]:
    y_pred = np.full(4000, i)
    print("{:.2f} : {:.5f}".format(i, log_loss(y, y_pred)))
    
# Then you will have those public log loss :


# You don't believe me ? Look at the end of public leaderboard. And look with some small differences : 

# In[ ]:


# with small differences 
y = np.concatenate([np.ones(2010), np.zeros(1990)], axis=0)

# the log loss is not the same (except for 0.5)
for i in [0, 0.05, 0.1, .15, .2, .25, .3, .35, .4, .45, .5, 1]:
    y_pred = np.full(4000, i)
    print("{:.2f} : {:.5f}".format(i, log_loss(y, y_pred)))


# **Hence a way to find where is your bug, is to submit many files** in your code, and **next, read your public score to guess where the code had stopped**. Let's modify your work from first cell by submit many files with different constants as prediction :

# In[ ]:


import numpy as np
import pandas as pd

# read the sample submission file
_temp = pd.read_csv('../input/deepfake-detection-challenge/sample_submission.csv', index_col='filename')
_cte = len(_temp)

# Create a submssion file with i as a constant prediction
def submission_to_find_bug(i, verbose=False):
    ts = _temp.copy()
    y_pred = np.full(_cte, i)
    ts["label"] = y_pred
    ts.reset_index(inplace=True)
    ts[['filename','label']].to_csv('submission.csv', index=False)
    if verbose:
        print("Debug with value {}".format(i))
        print(ts.head(3))
        print(ts.tail(3))

# First part of the notebook
# ...
# Create the submission file with 0 as predcition for all videos 
submission_to_find_bug(0)

# Part2
# ...
# Create the submission file with 0.05 as predcition for all videos 
submission_to_find_bug(0.05)

# Part 3
# ...
submission_to_find_bug(0.1)

# Part 4
def prob(): # Imagine your very complex model making your predictions
    submission_to_find_bug(0.15)
    return .5
submission_to_find_bug(0.2, verbose=True)

# Part 5
sf = pd.read_csv('../input/deepfake-detection-challenge/sample_submission.csv', index_col='filename')
submission_to_find_bug(0.25)
sf["label"] = prob()
sf["label"].fillna(0.5, inplace = True)
sf.reset_index(inplace=True)
sf[['filename','label']].to_csv('submission.csv', index=False)

# Part 6
print("Shape {}\nmin {}\nmax {}\nNA {}".format(sf.shape
                , sf["label"].min(), sf["label"].max(), sf["label"].isna().sum()))
sf.head()


# ## Run your notebook and submit to competition !

# Now, you have a score (not a brillant one) on the public leaderboard.<br>
# If your score is 17.26939, then you have a bug in the part 2 of your notebook.<br>
# If your score is 1.20397, then you have a bug in part 4 of your notebook.<br>
# If your score is 0.15, you will probably be the winner and you did not need to read my notebook (but thank you) !

# In[ ]:




