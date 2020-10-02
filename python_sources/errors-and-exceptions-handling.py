#!/usr/bin/env python
# coding: utf-8

# # Errors

# In[ ]:


1 + "1"


# In[ ]:


1/0


# In[ ]:


spglobal(8)


# # Exceptions Handling

# In[ ]:


def simpleSum(a, b):
    try:
        c = a + b
        return c
    except:
        print("there was an error")

a = 1
b = "1"
simpleSum(a,b)


# In[ ]:


def simpleSum(a, b):
    try:
        c = a + b
    except TypeError:
        c = float(a) + float(b)
    return c

a = 1
b = "1"
print(simpleSum(a,b))

