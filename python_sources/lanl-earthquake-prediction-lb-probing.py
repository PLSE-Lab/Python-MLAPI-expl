#!/usr/bin/env python
# coding: utf-8

# Reference
# - [kaggle - LANL-Earthquake-Prediction/discussion/91583#527970](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/91583#527970)

# In[ ]:


import pandas as pd
from pathlib import Path


# In[ ]:


get_ipython().run_line_magic('ls', '../input/LANL-Earthquake-Prediction/')


# In[ ]:


dataset_path = Path("../input/LANL-Earthquake-Prediction/")
submit_path = dataset_path / "sample_submission.csv"


# In[ ]:


# read sample_submission.csv
sub = pd.read_csv(submit_path)
sub.head()


# Try this !
# > Mean (as already been discussed, submission with 0 constant) - 4.017

# In[ ]:


target_col = "time_to_failure"
all_zeros = sub.copy()
all_zeros[target_col] = 0
# save submit
all_zeros.to_csv("baseline_probe_0.0.csv", index=False)


# ![lanl_all_0.PNG](attachment:lanl_all_0.PNG)

# > For 11, 10, 9: 6.982, 5.982, 5.017. Maximum value is less than 10. We could get better estimate by reducing step, but since it is unlikely that public set is randomly sampled (given the mean and the maximum value), finding the best fit in train set could be an option.

# In[ ]:


for score in [10, 20, 30]:
    sub = sub.copy()
    sub[target_col] = score
    # save submit
    sub.to_csv("baseline_probe_{}.csv".format(score), index=False)


# ### submit all 10
# ![submit_all10.PNG](attachment:submit_all10.PNG)
# ### submit all 20
# ![submit_all20.PNG](attachment:submit_all20.PNG)
# ### submit all 30
# ![lanl_all_30.PNG](attachment:lanl_all_30.PNG)
