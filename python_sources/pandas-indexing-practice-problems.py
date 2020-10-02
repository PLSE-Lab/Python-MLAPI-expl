#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Indexing is one of the most important skills for Pandas users. You will practice your indexing skills in these exercises. If you get stuck take a look at [this reference kernel](https://www.kaggle.com/sohier/tutorial-accessing-data-with-pandas/).
# 
# All exercises use the dataframe loaded below. Note that the park codes are the index column; this will be important later.
# 
# The dataset is small enough that you can print the entire dataframe to help with debugging.
# 
# **Check your answers or get hints with the `check_question` functions. For example, to check if 3 is the answer for question 6, you would run `check_question_6(3)`.**

# In[ ]:


import pandas as pd
df = pd.read_csv('../input/park-biodiversity/parks.csv', index_col=['Park Code'])
df.columns = [col.replace(' ', '_').lower() for col in df.columns]


# In[ ]:


import sys
sys.path.append('../input/pandas-indexing-challenges-validation')
from pandas_indexing_challenges_validation import *


# # Basic Exercises
# **1) Print the first five rows of the dataframe.**

# In[ ]:





# ** 2) Print rows 15 through 20 of the dataframe.**

# In[ ]:





# **3) What is the average acreage of parks in rows 15 through 20?**

# In[ ]:





# **4) What is the name of the park that has the code `GRSA`?**

# In[ ]:





# **5) What is the total acreage of park codes `GRSA`, `YELL`, and `BADL`?**

# In[ ]:





# **6) How many parks are smaller than 10,000 acres?**

# In[ ]:





# **7) How many more parks are there west of the Mississippi river than east? For simplicity, you can assume the Mississippi is at longitude `-90`. West means more negative longitudes, like `-95`.**

# In[ ]:





# # Advanced exercises
# These exercises may require methods that aren't described in [the reference kernel](https://www.kaggle.com/sohier/tutorial-accessing-data-with-pandas/). You can get ideas for more tools to try by looking at [the Pandas documentation](pandas.pydata.org/pandas-docs/stable/) or running `dir()` on a dataframe.
# 
# 
# **8) What is the name of the park closest in size to but larger than Yellowstone?**

# In[ ]:





# **9) What is the name of the largest park outside of Alaska?**

# In[ ]:





# **10) How many parks cross state borders? See park ID `YELL` for an example.**

# In[ ]:





# **11) What is the total acreage of all parks in the states that border Yellowstone (`ID`, `WY`, `MT`)?**

# In[ ]:




