#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 


# In[ ]:


sub_image = pd.read_csv('/kaggle/input/melanoma-submissions2/submissionImage.csv')
sub_tabular = pd.read_csv('/kaggle/input/melanoma-submissions2/submissionTabular.csv')
sub_tabular78 = pd.read_csv('/kaggle/input/tabular-prediction/submission_tabular_78.csv')
sub_multiple = pd.read_csv('/kaggle/input/melanoma-submissions2/submission_multiple_data_source.csv')
sub_public_merge = pd.read_csv('/kaggle/input/submission-9/submission_935.csv')
sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
sub_score946 = pd.read_csv('../input/submission-9/submission_946.csv')
sub_score940 = pd.read_csv('../input/submission-9/submission_940.csv')


sub_mean = pd.read_csv('/kaggle/input/siim-isic-multiple-model-training-stacking-923/submission_mean.csv')


# In[ ]:


# sub.target = sub_multiple.target *0.0 + sub_mean.target *0.55 + sub_tabular.target *0.45
# sub.target = sub_mean.target *0.2 + sub_public_merge.target *0.5 + sub_tabular78.target *0.3
sub.target = sub_score946.target *0.97 + sub_score940.target *0.0 + sub_tabular78.target *0.03


# In[ ]:


sub.to_csv('submission.csv', index = False)


# In[ ]:




