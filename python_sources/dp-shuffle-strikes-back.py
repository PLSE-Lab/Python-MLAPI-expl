#!/usr/bin/env python
# coding: utf-8

# We can squeeze a little extra from https://www.kaggle.com/golubev/reversing-and-shifting using same code that was published before:

# In[ ]:


get_ipython().system('cp -f ../input/dp-shuffle/dp ./ && chmod a+rx dp')


# In[ ]:


get_ipython().system('./dp 14 <../input/reversing-and-shifting/submission.csv >submission.csv')

