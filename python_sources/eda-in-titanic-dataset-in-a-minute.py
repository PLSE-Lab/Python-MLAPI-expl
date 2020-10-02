#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install sweetviz')


# In[ ]:


import pandas as pd
import sweetviz


# In[ ]:


train=pd.read_csv("../input/titanic/train (1).csv")
test=pd.read_csv("../input/titanic/test (1).csv")


# In[ ]:


train.head()


# In[ ]:


final_report=sweetviz.analyze([train,"TRAIN"],target_feat="Survived")


# In[ ]:


final_report.show_html("report.html")


# #      **END of the notebook**
# 
# ###   CLICK ON file.html TO SEE THE OUTPUT
# 
# #            **OR**
# 
# ###   CLICK ON THE LINK PRESENT IN COMMENT BOX

# In[ ]:




