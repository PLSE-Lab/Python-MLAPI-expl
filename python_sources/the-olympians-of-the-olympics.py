#!/usr/bin/env python
# coding: utf-8

# # The Olympians of the Olympics!

# I wasn't born yet to see the Olympic games that were held in Athens in 1896 but I sure can analyze it now thanks to this dataset. 

# ## I) Importing the data

# In[ ]:


import numpy as np
import pandas as pd 


# athlete_events.csv contains the information about the different athletes that participated in the Olympic games.

# In[ ]:


df = pd.read_csv('../input/athlete_events.csv')


# In[ ]:


df.columns


# The noc_regions.csv contains the full form of the different regions in the National Olympics Committee and doesn't provide much information except the region names. Importing it just for future reference if needed,

# In[ ]:


regions = pd.read_csv('../input/noc_regions.csv')
regions.head(2)


# ## II) Understanding the data

# The data has 271116 rows and 15 columns in it.

# In[ ]:


df.shape


# And the column names are,

# In[ ]:


df.columns


# So, we have all kind of information about different athletes. Let's see if there are any repeated number of athletes or not.

# In[ ]:


df.ID.value_counts().head(2)


# > So there is a lot of data for the same athlete. This may help us in analyzing a single athlete in-depth besides from the total number of athletes. Thus, making our EDA much more exciting.

# For a single athlete analysis, I'll be taking the athlete having ID number 77710 as he has been repeated the most in the dataset. Let's find out why!

# In[ ]:


robert = df[df.ID==77710].reset_index(drop=True)


# In[ ]:


robert.head(10)


# > Now that's a dissapointment, the number of rows pertaining to a single athlete is nothing but the same data repeated over and over again. Let us carry on with the overall analysis.

# Let us remove the duplicates inorder to make the data a bit clean.

# In[ ]:


df = df.drop_duplicates(['ID','NOC','Age','City','Season','Event','Team'])


# We still have a substantial data left.

# In[ ]:


df.shape


# I'm going to  fill in the empty data for the medals with a value so that we can easily analyze the data.

# In[ ]:


df['Medal'].unique()


# In[ ]:


df['Medal'].fillna(0,inplace=True)


# Let's see who have received the highest number of different medals and who has been in the olympics for the highest number of olympics without any medals.

# In[ ]:


df.groupby(['Medal','ID','Name'])['Team'].count().reset_index().rename(columns={'Team':'Number of medals'}).sort_values('Number of medals',ascending=False).drop_duplicates('Medal')


# > Hmm, so the dataset seems to have some kind of missing data as Michael Fred Phelps, II has actually won 28 gold medals and not just 23 by 2016.

# **I'll be updating the Kernel with more analysis regularly. Upvote and keep me encouraged.**

# In[ ]:




