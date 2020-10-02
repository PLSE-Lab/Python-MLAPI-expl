#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# In[7]:


BERT_FP = '../input/torch-bert-weights/bert-base-uncased/bert-base-uncased/'


# In[ ]:


from shutil import copy


# In[8]:


import os
#os.environ['USER'] = 'root'
os.system('pip install --no-index --find-links="../input/pytorchpretrainedbert/" pytorch_pretrained_bert')

#import xlearn as xl


# In[9]:


from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel


# In[10]:


bert = BertModel.from_pretrained(BERT_FP).cuda()


# In[11]:


bert.eval()


# In[ ]:




