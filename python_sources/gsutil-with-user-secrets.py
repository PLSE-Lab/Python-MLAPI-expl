#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
gcs_service_account = user_secrets.get_secret("gcs_service_account")
print(gcs_service_account, file=open("/tmp/key.json", "w"))


# In[ ]:


get_ipython().system('gcloud auth activate-service-account --key-file /tmp/key.json')


# In[ ]:


get_ipython().system('gsutil ls gs://foo/bar')


# In[ ]:




