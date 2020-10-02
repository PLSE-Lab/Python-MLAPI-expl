#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm
tqdm.pandas()


# In[ ]:


train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')


# In[ ]:


def process(text, selected_text, ib_space):

    added_extra_space = False
    splitted = text.split(selected_text)
    if splitted[0][-1] == ' ':
        added_extra_space = True
        splitted = text.split(" " + selected_text)
    
    sub = len(splitted[0]) - len(" ".join(splitted[0].split()))
    if sub == 1 and text[0] == ' ':
        splitted = text.split(selected_text)
        add_space = True if splitted[0] and splitted[0][-1] == ' ' else False

        add_space = False
        if splitted[0] and splitted[0][-1] == ' ':
            add_space = True
        if add_space == False:
            start = text.find(selected_text)
        else:        
            start = text.find(selected_text) - 1
    elif sub == 2:
        start = text.find(selected_text)
    else:
        start = text.find(selected_text)
        

    splitted = text.split(selected_text)

    add_space = False
    if splitted[0] and splitted[0][-1] == ' ':
        add_space = True

    text_pr =  " ".join(splitted[0].split())
    if add_space:
        text_pr += " "

    text_pr = text_pr + selected_text

    if len(splitted) > 1:
        text_pr = text_pr + splitted[1]

    if sub > 1 :
        if text[0] == ' ': 
            end = start + len(selected_text) - 1 + ib_space            
        else:
            end = start + len(selected_text)
    else:
        end = start + len(selected_text)
        
    new_st = text_pr[start : end]
    return new_st


# In[ ]:


def process_selected_text(text, selected_text):
    splitted = text.split(selected_text)
    add_space = True if splitted[0] and splitted[0][-1] == ' ' else False    
    sub = len(splitted[0]) - len( " ".join(splitted[0].split()) )
    in_between_space = len(selected_text) - len( " ".join(selected_text.split()) )
        
    new_selected_text = selected_text
    if sub > 0 and text.strip() != selected_text.strip() and add_space == False and text.find(selected_text) != 0:
        if in_between_space == 0:
            new_selected_text = process(text, selected_text, in_between_space)
    return new_selected_text


# In[ ]:


train['new_selected_text'] = train.selected_text
train = train[train.textID != '12f21c8f19']
pn_df = train[train.sentiment != 'neutral']


# In[ ]:


pn_df['new_selected_text'] = pn_df.progress_apply(lambda x: process_selected_text(x.text, x.selected_text), axis=1)


# In[ ]:


conflicted_df = pn_df[pn_df.selected_text != pn_df.new_selected_text]


# In[ ]:


conflicted_df.shape


# In[ ]:


conflicted_df.head(20)


# In[ ]:


conflicted_df.to_csv('conflicted_df.csv', index=False)


# In[ ]:




