#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from kaggle_secrets import UserSecretsClient
secret_label = "hello"
secret_value = UserSecretsClient().get_secret(secret_label)


# In[ ]:


print(secret_value)


# In[ ]:


get_ipython().system('echo "${secret_value}"')


# In[ ]:


get_ipython().system('echo $secret_value')


# In[ ]:


get_ipython().system('echo "$secret_value"')


# In[ ]:


get_ipython().system('echo {secret_value}')


# In[ ]:


get_ipython().system('echo "{secret_value}"')

