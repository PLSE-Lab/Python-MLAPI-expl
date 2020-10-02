#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/tgslb/tgs.csv")


# In[ ]:


df=df.sort_values(ascending=False,by=['Score'])


# In[ ]:


df[:100]


# In[ ]:


df.loc[df['TeamName']=='SeuTao@CHAN&Venn&Kele'].sort_values(ascending=False,by=['SubmissionDate'])


# In[ ]:


df.loc[df['TeamName']=='Giba&Heng'].sort_values(ascending=False,by=['SubmissionDate'])


# In[ ]:


df.loc[df['TeamName']=='Learning the Future'].sort_values(ascending=False,by=['SubmissionDate'])


# In[ ]:


df.loc[df['TeamName']=='b.e.s. & phalanx'].sort_values(ascending=False,by=['SubmissionDate'])


# In[ ]:


df.loc[df['TeamName']=='DISK'].sort_values(ascending=False,by=['SubmissionDate'])


# In[ ]:


df.loc[df['TeamName']=='earhian'].sort_values(ascending=False,by=['SubmissionDate'])

