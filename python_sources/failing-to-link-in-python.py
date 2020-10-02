#!/usr/bin/env python
# coding: utf-8

# Trying to get pycode style to work...

# In[ ]:


get_ipython().system('pip install pycodestyle pycodestyle_magic')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'pycodestyle_magic')


# In[ ]:


get_ipython().run_cell_magic('pycodestyle', '', 'def square_of_number(\n     num1, num2, num3, \n     num4):\n    return num1**2, num2**2, num3**2, num4**2')

