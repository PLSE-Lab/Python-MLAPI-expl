#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('apt install python3-tk ghostscript -y')


# In[ ]:


get_ipython().system('pip install camelot-py[cv]')


# In[ ]:


get_ipython().system('pip install camelot-py[plot]')


# In[ ]:


import camelot


# In[ ]:


get_ipython().system('wget -O example.pdf http://apm.amegroups.com/article/download/38244/29000')


# In[ ]:


tables = camelot.read_pdf('example.pdf', flavor='stream', pages='4', table_regions=["0,792,306,0"], row_tol=10, column_tol=0)
tables[0].df

