#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv("../input/train.csv")
# Any results you write to the current directory are saved as output.


# In[ ]:


#taking a look at the data
df.info()


# In[ ]:


df.columns


# # What does this mean?
# 
# We see that the frame has a first column "ID" and 131 columns with still unidentified variables.
# 
#  Let us see what kind of information has been given to us:
# 
# >*You are provided with an anonymized dataset containing both categorical and numeric variables available when the claims were received by BNP Paribas Cardif. All string type variables are categorical. There are no ordinal variables.*
# 
# If a claimant is given a '0' they fail. A '1' is a pass. First, let us look at "raw" probabilities for the whole distribution.

# In[ ]:


df['target'].value_counts


# In[ ]:


df['target']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




