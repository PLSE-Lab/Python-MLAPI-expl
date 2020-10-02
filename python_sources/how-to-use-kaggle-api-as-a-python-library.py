#!/usr/bin/env python
# coding: utf-8

# # Install Kaggle API
# 
# https://github.com/Kaggle/kaggle-api

# In[ ]:


get_ipython().system('pip install --quiet kaggle')


# # Set environment variables
# 
# Follow the instruction below and register your secrets.
# 
# https://www.kaggle.com/product-feedback/114053

# In[ ]:


import os
from kaggle_secrets import UserSecretsClient

# "KAGGLE_USERNAME" and "KAGGLE_KEY" are required to use Kaggle API.
for label in ["KAGGLE_USERNAME", "KAGGLE_KEY"]:
    os.environ[label] = UserSecretsClient().get_secret(label)


# # Submit

# In[ ]:


from kaggle import api as kaggle_api

competition_slug = "m5-forecasting-accuracy"
sbm_path = f"/kaggle/input/{competition_slug}/sample_submission.csv"
message = "test submission"

kaggle_api.competition_submit_cli(sbm_path, message, competition_slug)


# Open the link below and make sure the submission is successful.
# 
# https://www.kaggle.com/c/m5-forecasting-accuracy/submissions

# In[ ]:




