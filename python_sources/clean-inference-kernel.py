#!/usr/bin/env python
# coding: utf-8

# If you like this kernel please consider upvoting it and the associated datasets:
# - https://www.kaggle.com/abhishek/mlframework
# - https://www.kaggle.com/abhishek/catfeats-model/

# In[ ]:


get_ipython().system('pip install -U scikit-learn==0.22 > /dev/null')


# In[ ]:


import sys
sys.path.insert(0, "../input/")
from mlframework.predict import predict


# In[ ]:


sub = predict(test_data_path="../input/cat-in-the-dat-ii/test.csv",
              model_type="randomforest",
              model_path="../input/catfeats-model/")


# In[ ]:


sub.loc[:, "id"] = sub.loc[:, "id"].astype(int)
sub.to_csv("submission.csv", index=False)

