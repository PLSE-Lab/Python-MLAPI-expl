#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df=pd.read_csv("../input/brics.csv",index_col=0)


# In[ ]:


df


# # Accessing a Row :

# ### a. as a Seires

# In[ ]:


df.iloc[2]


# In[ ]:


df.loc["IN"]


# ### b. as a DataFrame

# In[ ]:


df.iloc[[2]]


# In[ ]:


df.loc[["IN"]]


# # Accessing multiple Rows :

# In[ ]:


df.iloc[[0,2,4]]


# In[ ]:


df.loc[["BR","IN","SA"]]


# # Accessing specified Rows & Columns :

# In[ ]:


df.iloc[2:4,1:3]                            # DataFrame_name.iloc[ start_row : end_row+1 , start_column : end_column+1 ]


# In[ ]:


df.loc["IN":"SA","capital":"population"]    # DataFrame_name.loc[ start_row : end_row , start_column : end_column ]


# ## Note :
# ### In ***iloc***, the first element of the range is included and the last one excluded. 
# ### Meanwhile, in ***loc*** both indexes are inclusively.

# # Another way of selection :

# In[ ]:


df.iloc[[0,2,4],[1,3]]


# In[ ]:


df.loc[["BR","IN","SA"],["capital","population"]]


# # Accesssing all rows / columns :

# In[ ]:


df.iloc[:,1:3]


# In[ ]:


df.loc[:,"capital":"area"]


# In[ ]:


df.iloc[1:3,:]


# In[ ]:


df.loc["RU":"IN",:]

