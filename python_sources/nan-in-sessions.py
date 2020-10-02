#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
sessions = pd.read_csv("../input/sessions.csv")
print(sessions.apply(lambda x: x.nunique(),axis=0))


# I also have a question here. Is the sessions.csv in the Kaggle notebook the same as the one we downloaded?
# Because on my local computer I got:
# 
#     user_id          135483
#     action              359
#     action_type          10
#     action_detail       155
#     device_type          14
#     secs_elapsed     337661
#     dtype: int64
#     
# which is a larger set than the one printed on the Kaggle notebook.

# In[ ]:


print ("NaN percentage in action:", np.sum(sessions.action.isnull()) / len(sessions.action))


# In[ ]:


sessions[sessions.action.isnull()].action_type.value_counts()


# All the NaN in action have action_type message_post

# In[ ]:


print ("NaN percentage in action_type:", np.sum(sessions.action_type.isnull()) / len(sessions.action_type))


# In[ ]:




