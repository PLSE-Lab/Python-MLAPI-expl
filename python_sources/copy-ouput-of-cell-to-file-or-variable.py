#!/usr/bin/env python
# coding: utf-8

# # Manipulate Cell Output
# Sometimes it is desirable to capture the output from a cell either to a variable or to a file. This is simple to achieve with either cell magic or redirection of `stdout`.

# ## Capture cell output to variable
# Here we use cell magic `%%capture` to redirect cell output to a variable.

# In[ ]:


get_ipython().run_cell_magic('capture', 'cap_out --no-stderr', '\nfor i in range (10):\n    print("Capture me line {}".format(i))')


# In[ ]:


txt = cap_out.stdout
print(txt)


# ## Redirect cell output to file
# Capturing to file can be achieved by using the `sys` library. This action affects all cells following too, unless we restore `stdout` at the end.

# In[ ]:


import sys

old_stdout = sys.stdout # keep reference to existing stdout
sys.stdout = open('logfile.txt', 'w')

for i in range (10):
    print("Log me line {}".format(i))

sys.stdout = old_stdout # restore stdout


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('cat logfile.txt')


# 
