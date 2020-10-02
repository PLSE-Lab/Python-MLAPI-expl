#!/usr/bin/env python
# coding: utf-8

# This changes the sqlite database into one ~1.5MB .txt file of all 45 episodes worth of scripts. This is more useful to me for text generation tasks and the like. Here is the code I used.

# In[1]:


import numpy as np
import pandas as pd
import sqlite3

import os
print(os.listdir("../input"))


# In[3]:


conn = sqlite3.connect('../input/database.sqlite')
df = pd.read_sql(con=conn, sql='select * from scripts')

df.character = df.character.astype(str)
df.actor = df.actor.astype(str)
df[:10]


# In[4]:


get_ipython().run_cell_magic('time', '', 'All_MP_Scripts = \'\'\nlast_type = \'\'\n\nfor index, line in df.iterrows():\n    type_of_line = line[4]\n    actor = line[5]\n    character = line[6]\n    detail = line[7]\n    Script = \'\'\n    if type_of_line == \'Direction\':\n        if last_type == \'Direction\':\n            Script += \' \'+detail\n        else:\n            Script += \'<Direction: \'+ detail+\'\'\n    else:\n        if last_type == \'Direction\':\n            Script += "> \\n\\n"\n        Script += character+\'(\'+actor+\'): \'+ detail+\' \\n\\n\'\n    last_type = type_of_line\n    All_MP_Scripts += Script')


# In[6]:


print(All_MP_Scripts[:1000])


# In[7]:


text_file = open("All_MP_Scripts.txt", "w")
text_file.write(All_MP_Scripts)
text_file.close()


# In[ ]:




