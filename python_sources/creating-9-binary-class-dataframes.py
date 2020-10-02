#!/usr/bin/env python
# coding: utf-8

# # [Good for beginners]
# # I'll show you how I created 9 dataframes with binary Classes.

# #### I'm using this method to furtherly apply some feature engineering in each one of the dataframes and a correspondent method, ensembling this 9 methods at the end. In this notebook, I'll just show you how to separate the *training* dataset into 9. It's not going to solve the complete challenge, but it's an important early stage that can make you ring a bell on the solving. It's obviously just one way of the inumerous possiblities to do this. Hope this can help you :)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Loading the train datasets:
trainvar = pd.read_csv("../input/training_variants")
traintxt = pd.read_csv("../input/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID","Text"])

train = pd.merge(trainvar, traintxt, how='left', on='ID').fillna('')


# In[ ]:


train.head()


# Now creating the 9 dataframes:

# In[ ]:


for i in range(1,10):
    globals()[str("train")+str(i)] = train.copy()
    globals()[str("train")+str(i)].loc[globals()[str("train")+str(i)]["Class"]!=i,"Class"] = 0


# So, we've just created our *train1*, *train2*, *train3*, *train4*, *train5*, *train6*, *train7*, *train8* and *train9* dataframes. 
# 
# Checking training Class 1 dataframe:

# In[ ]:


train1.head()


# The binary classification is now ready to all the classes. 
# 
# Checking our training Class 2 dataframe:

# In[ ]:


train2.head()


# Feedback or new ideas are always welcome. Happy Kaggling!
