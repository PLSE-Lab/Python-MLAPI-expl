#!/usr/bin/env python
# coding: utf-8

# # Some logs are gone!

# Assessment is important in this competition. We classify `Accruacy_group` to use the Assessment log.
# 
# When I look up the Assessment log, I found something strange. 
# 
# ### Please comment if this EDA is not correct!
# ### Please upvote to inform the issue if you agree.
# 
# Discussion is https://www.kaggle.com/c/data-science-bowl-2019/discussion/124779.

# In[ ]:


import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', 100)  # or 1000
pd.set_option('display.max_colwidth', 100)  # or 199


# In[ ]:


train = pd.read_csv("../input/data-science-bowl-2019/train.csv")
train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")


# ## 1. Session : 47e17c338e0ee335

# In train_labels data, '47e17c338e0ee335' accuracy_group is 0

# In[ ]:


display(train_labels[train_labels['game_session'] == '47e17c338e0ee335'])


# '47e17c338e0ee335' is Bird measure. If you play the PBS app, you can easily find that Bird Measure has 2 phases. 
# 
# First is caterpillars and second is hats. Also, first's evaluation code is 4110, the second's evaluation code is 4100 in normal case. That code has `contains "correct":true or false.` 
# 
# We can see the `"description":"Nice work` in `3318102` data. The description raises when the player completes the first phase in Bird measure. 
# 
# __But there is not 4110 event_code including  `"correct":true.`in prevous log.__ The log have just one 4110 event code `"correct":false.`in `3318086` data.
# 
# Though this player completes the assessment, accuracy_group is 0. I think it is more suitable this player's accuracy group is 2.

# In[ ]:


temp = train[train['game_session'] == '47e17c338e0ee335']


# In[ ]:


display(temp)


# ## 2. Session : a9c6f88dee27142c

# This session title is Mushroom Sorter (Assessment). Let's see a normal case.
# 
# In normal case, there are descriptions `"description":"That's one!"`, `"description":"two..."`. `"description":"and three!` in 2192, 2195 and 2198 data.
# 
# In the logs, `4025` code means player pull up the mushroom and `4030` code means player pick up the mushroom that height is `"height":(number)`.
# 
# There are three mushrooms in the assessment. we can find a specific mushroom in event data `"height":(number)`.

# In[ ]:


temp = train[train['game_session'] =='901acc108f55a5a1']


# In[ ]:


display(temp)


# '47e17c338e0ee335' session log is something strange. These logs don't have the description `"description":"and three!`. Also, there are no logs about height 2 and 4 mushrooms moving.

# In[ ]:


temp = train[train['game_session'] =='a9c6f88dee27142c']


# In[ ]:


display(temp)


# In train_labels data, '47e17c338e0ee335' accuracy_group is 3, but if other evaluation code was gone, it can be change.

# In[ ]:


display(train_labels[train_labels['game_session'] == 'a9c6f88dee27142c'])


# If founding missing data is part of the competition, it's ok. But this is not intended, it will be a problem.
