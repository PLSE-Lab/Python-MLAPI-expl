#!/usr/bin/env python
# coding: utf-8

# ### SHAP Features
# 
# Created additional feature using SHAP values. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# ### Resources
# 
# * [Python Business Analytics](https://github.com/firmai/python-business-analytics) - Python solutions to solve practical business problems. 
# * [Industry Machine Learning](https://github.com/firmai/industry-machine-learning) - A curated list of applied machine learning and data science notebooks and libraries across different industries.
# * [Business Machine Learning](https://github.com/firmai/business-machine-learning) - A curated list of practical business machine learning (BML) and business data science (BDS) applications.
# * [Financial Machine Learning](https://github.com/firmai/financial-machine-learning) - A curated list of practical financial machine learning (FinML) tools and applications in Python.
# * [Machine Learning Asset Management](https://github.com/firmai/machine-learning-asset-management) - Machine learning trading and portfolio optimisation models and techniques.
# * [Newsletter](https://mailchi.mp/a0e3989a5dc4/firmaikaggle) -  Linkletter of all the recently uncovered projects in the open source industry machine learning space.

# ## File

# https://colab.research.google.com/drive/1Pi6boPA-YHSKNwoB26LRIKwY-kXSVYw1

# In[ ]:


sample_submission = pd.read_csv('../input/submission-transactions/submission.csv')
sample_submission.to_csv('submission.csv', index=False)


# In[ ]:




