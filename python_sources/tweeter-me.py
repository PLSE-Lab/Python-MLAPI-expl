#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pandas import pandas
pn = pandas.read_csv("../input/tweeterclean/submission (2).csv")


# In[ ]:


print(pn)


# In[ ]:


pn.to_csv("/kaggle/working/submission.csv",index=False)

