#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import string
import random


# In[ ]:


l_alphabets = list(string.ascii_lowercase)
u_alphabets = list(string.ascii_uppercase)
digits = list(string.digits)
punct = ['!','@','$','#','^','&','*']


# In[ ]:


lst = []
lst.extend(u_alphabets)
lst.extend(l_alphabets)
lst.extend(digits)
lst.extend(punct)


# In[ ]:


pwd = ''
for i in range(12):
    idx = random.randint(0,len(lst)-1)
    pwd += (lst[idx])


# In[ ]:


print(pwd)

