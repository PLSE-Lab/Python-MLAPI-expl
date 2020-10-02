#!/usr/bin/env python
# coding: utf-8

# Text summarization is an approach that shortens long pieces of information into a shorter version. From this notebook, you will find how easy it is to generate a summarized text with just a couple lines of code. This is a subtask of my [original](https://www.kaggle.com/latong/text-summarization-ner-exploration) work. Note: The data is imported from this kernel([paringData](https://www.kaggle.com/latong/parsedata/)). When doing summarization tasks, please do not remove punctuations from the texts. For comparison, I am going to apply the following methods:
# 
# * [Bert-extractive-summarizer](https://pypi.org/project/bert-extractive-summarizer/)
# * GPT2 text summarizer
# * XL text summarizer
# * [Bart text summarizer](https://github.com/pytorch/fairseq/tree/master/examples/bart)
# 
# The length of generated texts is set to min_length=50 and max_length=100.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import glob
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from  collections import OrderedDict


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# In[ ]:


df=pd.read_csv("/kaggle/input/corowp/coroWP.csv")
df.head()


# In[ ]:


get_ipython().system('pip install bert-extractive-summarizer')


# In[ ]:


body=df['text_body'][0]


# **Bert Text Summarization**

# In[ ]:


from summarizer import Summarizer
model = Summarizer()
result = model(body, min_length=50,max_length=100)
full0 = ''.join(result)


# In[ ]:


print(full0)


# **GPT2 Text Summarization**

# In[ ]:


#GPT2
body=df['text_body'][0]
from summarizer import Summarizer,TransformerSummarizer
GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
full = ''.join(GPT2_model(body, min_length=50, max_length=100))


# In[ ]:


print(full)


# **XLNet Text Summarization**

# In[ ]:


model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
full2 = ''.join(model(body, min_length=60,max_length=100))


# In[ ]:


print(full2)


# **Bart Text Summarization**

# In[ ]:


# load BART summarizer
import transformers
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
from transformers import pipeline
summarizer = pipeline(task="summarization")


# In[ ]:


summary = summarizer(body, min_length=60, max_length=100)
print (summary)


# **Original Text**

# In[ ]:


print(df['summary'][0])

