#!/usr/bin/env python
# coding: utf-8

# ## CSV Direct Submit
# 
# In this competition I am training remotely using the TPU quota provided by GCP. Inference is done on GPU in a Kaggle kernel. As we reach the last 10 days of the comp, even the generous allocation of 30 hours of free GPU time from Kaggle looks like it might be a bit tight.
# 
# Consequently, I have started doing inference offline as well, and submitting csv predictions directly through this kernel. This has the added advantage of provideng fast feedback on the leaderboard with no kaggle GPU time consumed. Obviously this kernel cannot be used for a final submission, as it will score zero on the private dataset. In the `first50.csv` sample provided, there are only 50 predictions to keep the LB score low. Replace this with your own dataset and csv predictions.
# 
# **PLEASE NOTE: This kernel will score ZERO on the final leaderboard. It is shared only as a time-saving utility to get quick LB feedback from remote prediction**

# In[ ]:


import pandas as pd
pred_path = "../input/nq-sample-csv/first50.csv" #replace this with your own dataset.


# This creates a dummy submission using the `examle_ids` from `simplified-nq-test.jsonl`. This will work for private and public test sets, so that commit and submit both run correctly.

# In[ ]:


df = pd.read_json('../input/tensorflow2-question-answering/simplified-nq-test.jsonl', lines = True, dtype={'example_id':'Object'})
submission = pd.DataFrame(index=pd.concat([df.example_id + '_short', df.example_id + '_long']), columns=['PredictionString']).sort_index()


# This updates the submission from your csv, based on an intersection of `example_ids`. For the public LB, this will submit your csv. For the private test set, there will be no intersection, so all predictions are left blank and the score will be zero.

# In[ ]:


updates = pd.read_csv(pred_path, na_filter=False).set_index('example_id').sort_index()
submission.loc[updates.index.intersection(submission.index),'PredictionString'] = updates['PredictionString']


# In[ ]:


submission.head(50)


# In[ ]:


submission.to_csv('submission.csv')


# In[ ]:




