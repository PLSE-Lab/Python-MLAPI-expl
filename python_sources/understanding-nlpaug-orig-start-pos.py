#!/usr/bin/env python
# coding: utf-8

# ### Install nlpaug 0.0.14
# Must uninstall librosa or run into a error: https://github.com/makcedward/nlpaug/issues/127

# In[ ]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install ../input/nlpaug-0-0-14/nlpaug-master')
get_ipython().system('pip uninstall librosa --y')


# In[ ]:


import numpy as np
import pandas as pd

import torch
import transformers

import nlpaug.augmenter.word as naw


# ### Augment text using ContextualWordEmbsAug
# 
# Source code https://github.com/makcedward/nlpaug/blob/master/nlpaug/augmenter/word/context_word_embs.py

# In[ ]:


TOPK=20 
ACT = 'substitute' #'insert'
aug_bert = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased',
    action=ACT, top_k=TOPK,include_detail=True,aug_p=0.6)


# ### Issue:
# 
# The **'orig_start_pos'** doesn't align with **'orig_token'**.
# 
# For some tokens **'orig_start_pos'** is greater than the length of the **original_text**.

# In[ ]:


original_text = "testttt 123 http://curious.org"

output = aug_bert.augment(original_text)
new_text = output[0]
swaps = output[1]

print("Original:", original_text)
print("Length:", len(original_text))
print(" ")
print("New:",new_text)
print("Length:", len(new_text))
print(" ")

for s in swaps:
    print(s)


# A little hard to read but looking at positions of **original_text** vs. **new_text**

# In[ ]:


a = original_text
b = new_text
for i in range(max(len(a),len(b))):
    try:
        print(i,a[i],b[i])
    except:
        if len(b) > len(a):
            print(i,"~",b[i])
        else:
            print(i,a[i],"~")

