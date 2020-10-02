#!/usr/bin/env python
# coding: utf-8

# Importing Modules

# In[ ]:


import numpy as np
import pandas as pd
import collections
import seaborn as sns


# Loading Data

# In[ ]:


df = pd.read_csv("../input/steam-200k.csv", header=None, index_col=None, names=['UserID', 'Game', 'Action', 'Hours', 'Other'])
df = df[df['Hours'] != 1]
df.head()


# Data description shows that the 'other' column is blank. So we have hours and games to work with.

# In[ ]:


df.describe().round()


# Creating Hours Interval Column

# In[ ]:


interval = df['Hours'].max()/df['Hours'].mean()
print(interval)
df['HoursIntervals'] = pd.cut(df['Hours'],interval)
df.head()


# **Part I: User behavior based on playing time**  
# *A) How Many Games does User Play in Defined Time-Intervals*

# In[ ]:


df_pivot = df.pivot_table(index = 'UserID', columns = 'HoursIntervals', values = 'Game', aggfunc = np.count_nonzero)
df_pivot.head()


# *B) Histogram to see the distribution*

# In[ ]:


sns.countplot(x="HoursIntervals", data=df, palette="Greens_d");


# **(A) Pivot Table** and **(B) Histogram**  clearly show that the playing hours by users are majorly skewed towards left i.e. towards lesser numerical value of playing hours.  
# Apologies for the formatting.

# **Part II:**

# In[ ]:


df_gpu = df.groupby('UserID').agg({'Game': np.count_nonzero, 'Hours': np.mean}).round()
df_gpu['Heavy Gamer'] = df_gpu['Hours']>400 #400 being Mean + 1 Std. Devition
Heavy_Gamer_Count = collections.Counter(df_gpu['Heavy Gamer'])
print('Only', Heavy_Gamer_Count[1], 'out of', Heavy_Gamer_Count[0]+Heavy_Gamer_Count[0], ' are serious gamers')


# In[ ]:


df_gpu.describe().round()


# **Part III:**  
# Most common Playing Time 

# In[ ]:


counts_hours = collections.Counter(df['HoursIntervals'])
counts_hours_most = counts_hours.most_common(5)
counts_hours_most = pd.DataFrame(np.array(counts_hours_most).reshape(5,2), columns = ['HoursIntervals', 'No_of_Users'])
counts_hours_most


# **Below are the takeaways:**    
# **Part I:**  
# Very few users are engaged with games for high number of hours. It could be attributed to game characteristic or user behavior.   
# **Part II:**  
# Only 531 (Approx.) serious gamers out of almost 20000 gamers. Both they categories can furhter be studies.  
# **Part III:**  
# Maximum users spend 0 -50 hours per game.  
# *Reason could be length of the game, checking out the game or may be that is the average time games can keep users engaged and interested. this gives a hint of further exploration into the data.*  

# In[ ]:




