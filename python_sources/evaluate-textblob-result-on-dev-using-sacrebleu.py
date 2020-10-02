#!/usr/bin/env python
# coding: utf-8

# This notebook suposed to be kind of "baseline" on metrics score we are going to reach.
# 
# Since by today the submission still got an error, I will try it on dev dataset which already got paired
# 
# 

# # UPDATED V2: 
# * include score with lower case only (from 18.79 to 39.42)
# * include score with removing special character and unwanted space (from 39.42 to 43.21)

# **On this Notebook we will:**
# 1. Translate Dev dataset using Textblob package
# 2. Evaluate using Sacrebleu
# 3. And See some "benchmark" on research paper

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np 
import pandas as pd
from textblob import TextBlob
from tqdm import tqdm
from urllib.error import HTTPError
import textblob
import time
import re


# In[ ]:


dev_tcn = pd.read_csv('/kaggle/input/shopee-product-title-translation-open/dev_tcn.csv')
dev_en  = pd.read_csv('/kaggle/input/shopee-product-title-translation-open/dev_en.csv' )
dev = pd.concat([dev_en, dev_tcn], axis=1)


# In[ ]:


dev.head()


# # Translate

# In[ ]:


translation = []
for k in tqdm(range(len(dev))):

    try:
        one_translation = TextBlob(dev['text'][k]).translate(to="en")
        translation.append(one_translation)
        time.sleep(0.4)
        
    except (textblob.exceptions.NotTranslated, HTTPError) as e:
        print(k, e)
        if isinstance(e, textblob.exceptions.NotTranslated):
            translation.append("")
        else:
            break
            


# In[ ]:


textblob_translation = [x.string if isinstance(x, textblob.blob.TextBlob) else x for x in translation]
dev['textblob_translation'] = textblob_translation
dev.head()


# # Evaluate using Sacrebleu

# Source:
# * https://github.com/mjpost/sacrebleu
# * https://blog.machinetranslation.io/compute-bleu-score/
# 

# In[ ]:


get_ipython().system('pip install sacrebleu')


# In[ ]:


import sacrebleu
import matplotlib.pyplot as plt


# ## Score for translation without modification = 18.79

# In[ ]:


refs = [list(dev['translation_output'])]
preds = list(dev['textblob_translation'])
bleu = sacrebleu.corpus_bleu(preds, refs)
print(bleu.score)


# ## Score for translation with lowercase = 39.42

# thanks to @EnJun who remind me at comment section

# In[ ]:


refs = [list(dev['translation_output'].str.lower())]
preds = list(dev['textblob_translation'].str.lower())
bleu = sacrebleu.corpus_bleu(preds, refs)
print(bleu.score)


# ## Score for translation removed special char and strange space = 43.21

# inspired from this thread https://www.kaggle.com/c/shopee-product-title-translation-open/discussion/166251 findings 

# In[ ]:


def cleaning_string(my_string):
    my_string = re.sub(r"[^a-z0-9 ]+", ' ', my_string.lower()) # lowercase then change special char to '' 
    my_string = " ".join(my_string.split()) # remove white space

    return my_string


# In[ ]:


refs = [list(dev['translation_output'].map(cleaning_string))]
preds = list(dev['textblob_translation'].map(cleaning_string))
bleu = sacrebleu.corpus_bleu(preds, refs)
print(bleu.score)


# ## Score for identical inputs = 100 (obviously, just check :))

# In[ ]:


refs = [list(dev['translation_output'])]
preds = list(dev['translation_output'])
bleu = sacrebleu.corpus_bleu(preds, refs)
print(bleu.score)


# By this tranlation API, we get score 43.21 / 100 
# 
# But I am still confused how to calculated it actually and how good we should reach.

# # How good is 43.21?

# By this [paper](https://research.fb.com/wp-content/uploads/2016/11/bilingual_methods_for_adaptive_training_data_selection_for_machine_translation.pdf) 
# on Table 4, appeared their score on zh2en is around `24.6-25.6`
# 
# But it is BLEU score, I do not sure if it can be compared

# # Still Curious on the scoring?
# 
# See the score one by one

# In[ ]:


list_score_one_sample = []
for i in range(len(dev)):
    refs = [[list(dev['translation_output'].str.lower())[i]]]
    preds = list(dev['textblob_translation'].str.lower())[i]
    bleu = sacrebleu.corpus_bleu(preds, refs)
    list_score_one_sample.append(bleu.score)


# In[ ]:


plt.plot(list_score_one_sample);


# several translation hit perfect score

# Hope this useful,
# 
# if so, please upvote this notebook :D 
# 
# Thanks!!!

# In[ ]:




