#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


SurveySchema = pd.read_csv("../input/SurveySchema.csv")
freeFormResponses = pd.read_csv("../input/freeFormResponses.csv")
multipleChoiceResponses = pd.read_csv("../input/multipleChoiceResponses.csv")


# In[ ]:


SurveySchema.head()


# In[ ]:


freeFormResponses.head()


# In[ ]:


multipleChoiceResponses.head()


# In[ ]:


SurveySchema.info()


# In[ ]:


# freeFormResponses.info()


# In[ ]:


# multipleChoiceResponses.info()


# In[ ]:


SurveySchema.shape


# In[ ]:


freeFormResponses.shape


# In[ ]:


multipleChoiceResponses.shape


# In[ ]:


SurveySchema.columns = SurveySchema.iloc[0]


# In[ ]:


col = SurveySchema.columns
col


# In[ ]:


remove_words = ["what", "options","your","select"]

