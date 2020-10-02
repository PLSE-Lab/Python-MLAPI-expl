#!/usr/bin/env python
# coding: utf-8

# In[ ]:


f = open("test.txt","w+")
f.write("hello world")
f.close()


# In[ ]:


get_ipython().run_line_magic('cat', 'test.txt')

