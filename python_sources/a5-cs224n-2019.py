#!/usr/bin/env python
# coding: utf-8

# # Instructions to train CS224N 2019 Assignment in Kaggle kernel
# 
# There're 4 steps:
# - In setting tab (on the right), select **GPU on** and **Internet connected**
# - Clone github project
# - Remove .git and pdf directories or you'll get errors when commit
# - Commit and wait
# 
# A5 takes 6 hours for training. you'll get your model.bin on output tab in 

# In[ ]:


get_ipython().system('git clone https://github.com/Luvata/CS224N-2019.git')


# In[ ]:


get_ipython().run_line_magic('cd', 'CS224N-2019/')


# In[ ]:


# remove this folder or you'll get error when commit
get_ipython().system('rm -rf .git/')
get_ipython().system('rm -rf note/')


# In[ ]:


get_ipython().run_line_magic('cd', 'Assignment/')


# In[ ]:


get_ipython().system('rm -rf a1 a2 a3 a4 # remove redundant directory')


# In[ ]:


get_ipython().run_line_magic('cd', 'a5-v1.2/')


# In[ ]:


get_ipython().system('sh run.sh vocab # make vocab file first')


# In[ ]:


get_ipython().system('sh run.sh train # train and wait :D')


# In[ ]:


get_ipython().system('sh run.sh test # will show BLEU score')


# In[ ]:




