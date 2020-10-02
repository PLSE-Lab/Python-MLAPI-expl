#!/usr/bin/env python
# coding: utf-8

# This is where I ask question and try to answer them!!

# In[ ]:


import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


# In[ ]:


data_tkd=pd.read_csv('../input/taekwondo-techniques-classification/Taekwondo_Technique_Classification_Stats.csv')


# In[ ]:


data_tkd.head(20)


# In[ ]:


data_tkd.drop([0, 1])


# In[ ]:


data_tkd.info()
data_tkd.describe()


# In[ ]:


from IPython.display import HTML,IFrame
IFrame('https://i.imgur.com/8QM16yb.png',height=500,width=1200)


# 

# [](https://i.imgur.com/8QM16yb.png)

# In[ ]:



IFrame('https://i.imgur.com/TqfmgUE.png',height=700,width=1200)


# In[ ]:




