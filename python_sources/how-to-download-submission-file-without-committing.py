#!/usr/bin/env python
# coding: utf-8

# Let's say you've struggled for quite a while, trained a network in a Kaggle Kernel and would like to save the submission file locally. One of solutions is described by Abhishek [here](https://www.kaggle.com/general/77352). But there exists a much simpler way!

# Let's assume this is the file that you'd like to save locally:

# In[ ]:


get_ipython().system('echo "qid,prediction" > submission.csv')


# In[ ]:


from IPython.display import FileLink
FileLink('submission.csv')


# Voula! Happy blending!
# 
# <img src='https://habrastorage.org/webt/ut/zk/ip/utzkip2sdihrzu0xd9iatqhbvii.jpeg' width=70%>
# *Image from the [Discussion](https://www.kaggle.com/general/76963) "Nerd Laughing Loud"*
