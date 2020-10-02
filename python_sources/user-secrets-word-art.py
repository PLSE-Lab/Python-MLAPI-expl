#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pyfiglet')


# In[ ]:


from kaggle_secrets import UserSecretsClient

secret_label = "my_cool_word_art"
secret_value = UserSecretsClient().get_secret(secret_label)


import pyfiglet

ascii_banner = pyfiglet.figlet_format(secret_value)
print(ascii_banner)


# In[ ]:




