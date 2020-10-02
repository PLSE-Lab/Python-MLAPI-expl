#!/usr/bin/env python
# coding: utf-8

# Here I show how you can submit from a CSV file and save a lot of time in a Kernel-only competition format.
# 
# This is only for you to quickly check hypotheses.
# 
# **BE CAREFUL!** This will score 0 on the private part because the private part is actually ignored here. 

# In[ ]:


import pandas as pd


# In[ ]:


sub = pd.read_csv("../input/google-quest-qa-subm-files/final_submission.csv")
sample_submission = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")


# In[ ]:


id_in_sub = set(sub.qa_id)
id_in_sample_submission = set(sample_submission.qa_id)
diff = id_in_sample_submission - id_in_sub

sample_submission = pd.concat([
    sub,
    sample_submission[sample_submission.qa_id.isin(diff)]
]).reset_index(drop=True)


# In[ ]:


sample_submission.head()


# In[ ]:


sample_submission.to_csv("submission.csv", index=False)

