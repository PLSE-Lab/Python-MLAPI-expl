#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train=pd.read_csv('../input/data-science-bowl-2019/train.csv')
train.timestamp=pd.to_datetime(train['timestamp'])
train=train.sort_values(by=['installation_id','timestamp'])


# In[ ]:


test=pd.read_csv('../input/data-science-bowl-2019/test.csv')
test.timestamp=pd.to_datetime(test['timestamp'])
test=test.sort_values(by=['installation_id','timestamp'])


# In[ ]:


train_labels=pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")
specs=pd.read_csv("../input/data-science-bowl-2019/specs.csv")


# # What are we missing?
# 
# There are 3614 unique installation in the train labels. However, it has 17690 rows meaning that a installation_id (or a player) can take many assessments. Are the players with record in train_labels all we need?. Let's dig a little deeper into the players that has **no record** on the train_labels data. 

# In[ ]:


len(train_labels)


# In[ ]:


#Extract installation_id without assessment record
no_record=train[~train.installation_id.isin(train_labels.installation_id)]

#Id of players who have no record but accually took the assessment
id_no_record_with_assessment=no_record[no_record.type=='Assessment'].installation_id.unique()

#The unique event code of these players
event_code_no_record=no_record[no_record.type=='Assessment'].event_code.unique()


# In[ ]:


print("Number of players with no record:", no_record.installation_id.nunique())
print("Number of players 'took' the assessment but has no record:", len(id_no_record_with_assessment))
print("Event code of the players above:\n", event_code_no_record)


# We can safely ignore the (13386-628=12758) people who have never taken the assessment for now since every player in the test set has taken the assessment **at least** once. A simpler explaination is that if we take these people and categorize them as group 0, then it has no beneficial for our training set since their game_time on the assessment is always zero and their is no event_code recorded. 
# 
# However, we still are still not sure if we can use the group of 628 players to categorize them as 0. Let's investigate their time spent on the assessment.

# In[ ]:


no_record_with_assessment=no_record[no_record.installation_id.isin(id_no_record_with_assessment)]
no_record_with_assessment=no_record_with_assessment[no_record_with_assessment.type=='Assessment']
#Note that I have sort the timestamp and game_time is accumulated, thus 
#the following code tell me the max game time for each sesstion
by_session_no_record=no_record_with_assessment.groupby("game_session").last()


# In[ ]:


print("Number of assessment session", len(by_session_no_record))
print("Number of assessment that has time less than 5s:",(by_session_no_record.game_time.values <5000).sum())


# Observe that more than half players finish the game less than 5 seconds. This may be a good idicator that they don't even try (I actually played the game and 5 seconds is super fast for a kid.). Less look at the train set to see if we have that much percentage of players finish a game within 5 seconds.

# In[ ]:


record=train[(train.installation_id.isin(train_labels.installation_id)) & (train.type=='Assessment')]
by_session_record=record.groupby("game_session").last()
print("Number of assessment session", len(by_session_record))
print("Number of assessment that has time longer than 5s:",(by_session_record.game_time.values<5000).sum())


# Less take a step further to see which group these kids fall into

# In[ ]:


index_lessthan_5s=by_session_record[by_session_record.game_time.values<5000].index


# In[ ]:


train_labels[train_labels.game_session.isin(index_lessthan_5s)].accuracy_group.value_counts()


# So most of them have fall in group 0. Also, observe that we have 1066 different session on players that have record in train_labels. However, the output is only 27 in total. This shows that not all players assessment is recorded, even with those who have their record on train labels.

# # Conclusion
# 
# Since the last assessment of each players in test set is truncated, we would not know if they will complete or forfeit the game. The extra information of 628 players we get above may be useful!
# 
# The code below show that 424/1000 players has never finished an assessment before.

# In[ ]:


test_assessment_count=test.groupby('installation_id').apply(lambda x: x[x.type=='Assessment'].game_session.nunique())


# In[ ]:


test_assessment_count.value_counts()


# In[ ]:




