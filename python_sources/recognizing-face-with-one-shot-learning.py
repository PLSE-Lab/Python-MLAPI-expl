#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


print(check_output(["ls", "../input/faceinthewild"]).decode("utf8"))


# In[ ]:


print(check_output(["ls", "../input/facesinrheworld"]).decode("utf8"))


# In[ ]:


sub1 = pd.read_csv('../input/faceinthewild/vgg_face_88.csv')
sub2 = pd.read_csv('../input/faceinthewild/vgg_face_90.csv')
sub3 = pd.read_csv('../input/faceinthewild/vgg_face_891.csv')

sub4 = pd.read_csv('../input/facesinrheworld/vgg_face_85.csv')
sub5 = pd.read_csv('../input/facesinrheworld/vgg_face_v122.csv')
sub6 = pd.read_csv('../input/facesinrheworld/vgg_face_v14.csv')
sub7 = pd.read_csv('../input/facesinrheworld/vgg_face_v19.csv')
temp=  pd.read_csv('../input/faceinthewild/vgg_face_891.csv')


# In[ ]:


sns.set(rc={'figure.figsize':(18,6.5)})
sns.kdeplot(sub1['is_related'],label="sub1",shade=True,bw=.1)
sns.kdeplot(sub2['is_related'], label="sub2",shade=True,bw=.1)
sns.kdeplot(sub3['is_related'], label="sub3",shade=True,bw=.1)

sns.kdeplot(sub4['is_related'],label="sub4",shade=True,bw=.1)
sns.kdeplot(sub5['is_related'], label="sub5",shade=True,bw=.1)
sns.kdeplot(sub6['is_related'], label="sub6",shade=True,bw=.1)
sns.kdeplot(sub7['is_related'], label="sub7",shade=True,bw=.1)


# In[ ]:


temp['is_related'] = 0.5*sub1['is_related'] + 0.125*sub2['is_related'] + 0.125*sub3['is_related'] + 0.15*sub4['is_related'] + 0.1*sub5['is_related'] + 0.1*sub6['is_related'] + 0.1*sub7['is_related'] 
temp.to_csv('submission4.csv', index=False )

