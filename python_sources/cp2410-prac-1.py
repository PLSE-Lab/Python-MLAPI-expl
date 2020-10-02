#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def is_multiple(n, m):
    return (n % m) == 0


# In[ ]:


p = [pow(2, n) for n in range(9)]


# In[ ]:


def is_distinct(l):
    exists = []
    for element in l:
        if l not in exist:
            exists.append(element)
        else:
            return false
    return true


# In[ ]:


def harmonic_list(n):
    result = []
    h = 0
    for i in range(1, n + 1):
        h += 1 / i
    return h

