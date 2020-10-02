#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.listdir('../input')


# In[ ]:


import pandas as pd
model_9872_baseline_submission = pd.read_csv("../input/model-9872/model_9872_baseline_submission.csv")


# In[ ]:


import pandas as pd
oof_model_9872_baseline_submission = pd.read_csv("../input/oof-model-9872/oof-model_9872_baseline_submission.csv")


# In[ ]:


model_9872_baseline_submission.to_csv('model_9872_baseline_submission.csv',index=False)


# In[ ]:


oof_model_9872_baseline_submission.to_csv('oof-model_9872_baseline_submission.csv', index=False)


# In[ ]:




