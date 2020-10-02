#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os


# In[ ]:


os.listdir('../input/simple-blend-upvote-0-697-accuracy')


# In[ ]:


import pandas as pd
a = pd.read_csv('../input/nffm-baseline-0-690-on-lb/nffm_submission.csv')
b = pd.read_csv('../input/ensemble-lb-score-0-697/sub.csv')
#c = pd.read_csv('../input/upd-lightgbm-baseline-model-using-sparse-matrix/lgb_submission.csv')
d = pd.read_csv('../input/ms-malware-starter/ms_malware.csv')
e = pd.read_csv('../input/simple-blend-upvote-0-697-accuracy/sub.csv')
f = pd.read_csv('../input/why-no-blend/super_blend.csv')


# In[ ]:


a.head()


# In[ ]:


submission = a[['MachineIdentifier']]
submission['HasDetections'] = (a['HasDetections']+b['HasDetections']+d['HasDetections']+e['HasDetections']+f['HasDetections'])/6
submission.to_csv('submission.csv', index=False)
submission.head()


# In[ ]:




