#!/usr/bin/env python
# coding: utf-8

# ## About this Competition
# 
# ![](https://media.giphy.com/media/H4DjXQXamtTiIuCcRU/giphy.gif)
# 
# > #### In this competition, you will be predicting the probability [0, 1] of a binary target column.
# 
# The data contains binary features (bin_*), nominal features (nom_*), ordinal features (ord_*) as well as (potentially cyclical) day (of the week) and month features. The string ordinal features ord_{3-5} are lexically ordered according to string.ascii_letters.
# Since the purpose of this competition is to explore various encoding strategies, the data has been simplified in that (1) there are no missing values, and (2) the test set does not contain any unseen feature values (See this). (Of course, in real-world settings both of these factors are often important to consider!)
# 
# #### Files
# - train.csv - the training set
# - test.csv - the test set; you must make predictions against this data
# - sample_submission.csv - a sample submission file in the correct format
# 
# ## From Kernel> https://www.kaggle.com/caesarlupum/catcomp-simple-target-encoding (, if you find this kernel useful or interesting, please don't forget to upvote the kernel =) )

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# # Leaderboard is calculated with approximately 80% of the test data.
# ** The final results will be based on the other 20%, so the final standings may be different.

# In[ ]:


submission = pd.read_csv('../input/catcomp/sub_2019-10-18_11-51-06.csv')
possible_public = int(len(submission)*0.80)
submission.iloc[possible_public:,1] = 0


# # Submission

# In[ ]:


from datetime import datetime
submission.to_csv(
    'sub_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.csv', 
    index=False)


# - LB before - 0.80446
# 
# - LB after -> 0.50985 -> seems it's random split or 20% first lines of data is private (need one more test).
# 
# - LB after -> 0.69524 -> seems it's random split or 80% first lines of data is private (need one more test).
# 
# > ### Be Careful to generalize data
# 

# In[ ]:


aristocats  = pd.read_csv('../input/aristocat-data/submission (11) (1).csv')
possible_public_ = int(len(aristocats)*0.80)
aristocats.iloc[possible_public_:,1] = 0


# In[ ]:


from datetime import datetime
aristocats.to_csv(
    'aristocats_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.csv', 
    index=False)


# 
# <html>
# <body>
# 
# <p><font size="5" color="Red">If you like my kernel please consider upvoting it</font></p>
# <p><font size="4" color="Green">Don't hesitate to give your suggestions in the comment section</font></p>
# 
# </body>
# </html>

# # Final
