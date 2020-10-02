#!/usr/bin/env python
# coding: utf-8

# # 252 views and 3 Upvotes :((

# This kernel introduces a simple benchmark.<br>
# In this competition, game information and information on events in the game are given. And what we ultimately want is to determine how many times the owner of a device can clear the game.
# 
# # import

# In[ ]:


import numpy as np
import pandas as pd 
import gc
import json
import matplotlib.pyplot as plt


# In[ ]:


train_df = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv")
# specs_df = pd.read_csv("/kaggle/input/data-science-bowl-2019/specs.csv")
test_df = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")
train_label_df = pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv")


# In[ ]:


train_df.head()


# ## eda

# In[ ]:


plt.hist(list(train_df["game_session"].value_counts()))


# Games with many events have more than 3000 events per play, while most games have fewer than 500 events.

# In[ ]:


train_df["game_time"].value_counts()


# If you look at the game_time, you can see that events occur in many games at least at 0.<br>
# 
# Now consider the data and results of the owner of the device with installation_id is 0006a69f.

# In[ ]:


gc.collect()


# In[ ]:


train_label_df[train_label_df.installation_id=="0006a69f"]


# He is playing Mushroom Sorter and Bird Measurer. This time, let's focus on the event when the game ends.
# 
# ### Mushroom Sorter (Assessment) by 0006a69f

# In[ ]:


train_df[(train_df.installation_id=="0006a69f") & (train_df.title=="Mushroom Sorter (Assessment)") & (train_df.event_code==4100)]


# If event_date contains correct: true, the game has been cleared, and if false, it has failed.<br>
# This result is consistent with that of the first train_label_df.

# ### Mushroom Sorter (Assessment) by 0006a69f

# In[ ]:


train_df[(train_df.installation_id=="0006a69f") & (train_df.title=="Bird Measurer (Assessment)") & (train_df.event_code==4110)]


# This result is also consistent with the first train_label_df result.<br>
# 
# If you calculate the probability that the player will clear at the end event, it will help the final classification.
# 
# ## make simple benchmark
# 
# This time, it is classified by game clear probability.

# In[ ]:


train_df_clear = train_df[((train_df.event_code==4100)|(train_df.event_code==4110))
                          &(train_df.event_data.str.contains("true"))]
train_df_fail = train_df[((train_df.event_code==4100)|(train_df.event_code==4110))
                         &(train_df.event_data.str.contains("false"))]


# In[ ]:


train_df_clear_g = train_df_clear.groupby(["installation_id"]).count()["event_id"]
train_df_fail_g = train_df_fail.groupby(["installation_id"]).count()["event_id"]


# In[ ]:


train_df_clear_g


# In[ ]:


sample_submission = pd.read_csv("/kaggle/input/data-science-bowl-2019/sample_submission.csv")


# In[ ]:


test_df


# In[ ]:


test_df_clear = test_df[((test_df.event_code==4100)|(test_df.event_code==4110))
                          &(test_df.event_data.str.contains("true"))]
test_df_fail = test_df[((test_df.event_code==4100)|(test_df.event_code==4110))
                         &(test_df.event_data.str.contains("false"))]

test_df_clear_g = test_df_clear.groupby(["installation_id"]).count()["event_id"]
test_df_fail_g = test_df_fail.groupby(["installation_id"]).count()["event_id"]


# In[ ]:


test_clear_dic=dict(zip(test_df_clear_g.index,list(test_df_clear_g)))
test_fail_dic=dict(zip(test_df_fail_g.index,list(test_df_fail_g)))


# In[ ]:


for i in range(len(sample_submission)):
    id = sample_submission["installation_id"][i]
    fail = test_fail_dic[id] if id in test_fail_dic else 0
    clear = test_clear_dic[id] if id in test_clear_dic else 0
    if fail+clear!=0:
        score = clear/(fail+clear)
        if score>0.95:
            sample_submission["accuracy_group"][i]=3
        elif score>0.5:
            sample_submission["accuracy_group"][i]=2
        elif score>0.3:
            sample_submission["accuracy_group"][i]=1
        else:
            sample_submission["accuracy_group"][i]=0
    else:
        sample_submission["accuracy_group"][i]=1


# In[ ]:


sample_submission


# In[ ]:


sample_submission.to_csv("submission.csv",index=False)


# Since the clearing tendency changes depending on the game, it is difficult to classify only the players.
