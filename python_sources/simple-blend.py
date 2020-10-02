#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


df1 = pd.read_csv('../input/lanl-blend-csv/blend_ver1.csv')
df2 = pd.read_csv('../input/lanl-blend-csv/blend_ver2.csv')
df3 = pd.read_csv('../input/lanl-blend-csv/blend_ver3.csv')


# In[ ]:


blend = df1['time_to_failure'] *0.33333333334 + df2['time_to_failure'] * 0.33333333333 + df3['time_to_failure'] * 0.33333333333 
sample = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')


# In[ ]:


sample['time_to_failure'] = blend
sample.to_csv('blend_ver4.csv',index=False)


# In[ ]:


sample.head()


# In[ ]:


df4 = sample


# In[ ]:


blend1 = df1['time_to_failure'] *0.33333333333 + df2['time_to_failure'] * 0.33333333333 + df4['time_to_failure'] * 0.33333333334 
sample1 = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')


# In[ ]:


sample1['time_to_failure'] = blend1
sample1.to_csv('blend_ver5.csv',index=False)


# In[ ]:


sample1.head()


# In[ ]:


df5 = sample1


# In[ ]:


blend2 = df1['time_to_failure'] *0.2 + df2['time_to_failure'] * 0.2 + df3['time_to_failure'] * 0.2 + df4['time_to_failure'] * 0.2 + df5['time_to_failure'] * 0.2
sample2 = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')


# In[ ]:


sample2['time_to_failure'] = blend2
sample2.to_csv('blend_ver6.csv',index=False)


# In[ ]:


sample2.head()


# In[ ]:


df6 = sample2


# In[ ]:


blend3 = df1['time_to_failure'] *0.16666666666 + df2['time_to_failure'] * 0.16666666666 + df3['time_to_failure'] * 0.16666666666 + df4['time_to_failure'] * 0.16666666666 + df5['time_to_failure'] * 0.16666666666 + df6['time_to_failure'] * 0.16666666666
sample3 = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')


# In[ ]:


sample3['time_to_failure'] = blend3
sample3.to_csv('blend_ver7.csv',index=False)


# In[ ]:


sample3.head()


# In[ ]:





# In[ ]:





# In[ ]:




