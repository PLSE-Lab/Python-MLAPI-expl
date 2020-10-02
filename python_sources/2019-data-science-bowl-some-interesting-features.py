#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering

# In this competition, I have found a few useful features by using **event_data** feature. These features are generated in the below section.

# In[ ]:


import pandas as pd
pd.set_option('max_colwidth', 999)

train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')


# # Feature 1

# Let's explore the **event_code == 2030** for **Game** session type.

# In[ ]:


train[(train.type=="Game") & (train.event_code==2030)]


# Here, there exist a **misses** parameter in **event_data** column. Let's see what this **misses** means?
# 
# We will check in **specs.csv** file for first **event_id=08fd73f3**

# In[ ]:


specs[specs.event_id=="08fd73f3"]


# **misses** parameter indicates: {"name":"misses","type":"int","info":"the number of incorrect size objects clicked"}

# This might be helpful to predict the assessment. Let's see how this can be calculate for particulare game_session of the installation_id.

# In[ ]:


def cnt_miss(df):
    cnt = 0
    for e in range(len(df)):
        x = df['event_data'].iloc[e]
        y = json.loads(x)['misses']
        cnt += y
    return cnt

if session_type=="Game":
    misses_cnt = cnt_miss(g_session[g_session.event_code == 2030] )
    type_dict['accumulated_game_miss'] += misses_cnt


# # Feature 2

# The **Game** session type event_data column include **level**, **round** and **duration** fields. Let's see what these fields provides the information.

# {"name":"level","type":"int","info":"number of the current level when the event takes place or 0 if no level"},
# 
# {"name":"duration","type":"int","info":"the duration of the round in milliseconds. Time between start round and beat round."}
# 
# {"name":"round","type":"int","info":"the number of the round that has just been completed"}

# So, we can consider the last record of the particular game_session which provides the information about howmany game level and round has been completed by user.

# Let's see the event records of the game_session=="f11eb823348bfa23" of installation_id=="0001e90f"

# In[ ]:


train[(train.installation_id=="0001e90f") & (train.game_session=="f11eb823348bfa23")]


# The above game_session of the installation_id has completed 15 level of game. This will help for assessment.  Let's see how this calculate.

# In[ ]:


# For particular game_session
try:
    game_level = json.loads(g_session['event_data'].iloc[-1])["level"]
    type_dict['mean_game_level'] = (type_dict['mean_game_level'] + game_level) / 2.0
except:
    pass

try:
    game_round = json.loads(g_session['event_data'].iloc[-1])["round"]
    type_dict['mean_game_round'] = (type_dict['mean_game_round'] + game_round) / 2.0
except:
    pass

try:
    game_duration = json.loads(g_session['event_data'].iloc[-1])["duration"]
    type_dict['mean_game_duration'] = (type_dict['mean_game_duration'] + game_duration ) / 2.0
except:
    pass 


# # Feature 3

# The **event_code == 4020** provides helpful information for assessment session type. 
# Let's explore it.

# In[ ]:


train[(train.type=="Assessment") & (train.event_code==4020) ]


# For the above event data, let's see what the first event_id tells in specs.csv file.

# In[ ]:


specs[specs.event_id=="5f0eb72c"]


# This event is used to calculate accuracy and to diagnose player strategies and understanding.
# 
# {"name":"correct","type":"boolean","info":"is this the correct stump for this mushroom?"}

# Here, we can count the true and false attemps for the above event. Let's see how to find it.

# In[ ]:


def get_4020_acc(df):
     
    counter_dict = {'Cauldron Filler (Assessment)_4020_accuracy':0,
                    'Mushroom Sorter (Assessment)_4020_accuracy':0,
                    'Bird Measurer (Assessment)_4020_accuracy':0,
                    'Chest Sorter (Assessment)_4020_accuracy':0 }
        
    for e in ['Cauldron Filler (Assessment)','Bird Measurer (Assessment)','Mushroom Sorter (Assessment)','Chest Sorter (Assessment)']:
        
        Assess_4020 = df[(df.event_code == 4020) & (df.title==activities_map[e])]   
        true_attempts_ = Assess_4020['event_data'].str.contains('true').sum()
        false_attempts_ = Assess_4020['event_data'].str.contains('false').sum()

        measure_assess_accuracy_ = true_attempts_/(true_attempts_+false_attempts_) if (true_attempts_+false_attempts_) != 0 else 0
        counter_dict[e+"_4020_accuracy"] += (counter_dict[e+"_4020_accuracy"] + measure_assess_accuracy_) / 2.0
    
    return counter_dict


# # Feature 4

# There is a interesing feature found for **event_code==4025** for **Cauldron Filler (Assessment)**.
# Let's discover it.

# In[ ]:


train[ (train.event_code==4025) & (train.title == 'Cauldron Filler (Assessment)')]


# Let's explore event_id 91561152 

# In[ ]:


specs[specs.event_id=="91561152"]


# This event is used to calculate accuracy and to diagnose player strategies and understanding.
# 
# {"name":"correct","type":"boolean","info":"is this the correct bucket?"}
# 
# 
# The **correct** parameter in the event_data column provides the information about the event attempt is True or False. Using this we can calcuate the accuracy for that event by inding the number of correct attempts and number of incorrect attempts. 

# In[ ]:


def calculate_accuracy(session):
    Assess_4025 = session[(session.event_code == 4025) & (session.title=='Cauldron Filler (Assessment)')]   
    true_attempts_ = Assess_4025['event_data'].str.contains('true').sum()
    false_attempts_ = Assess_4025['event_data'].str.contains('false').sum()

    accuracy_ = true_attempts_/(true_attempts_+false_attempts_) if (true_attempts_+false_attempts_) != 0 else 0


# If you found these fetures are helpful to you, please do upvote. 
# 
# ![](http://)Thanks for reading

# In[ ]:




