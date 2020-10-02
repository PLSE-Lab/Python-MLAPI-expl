#!/usr/bin/env python
# coding: utf-8

# ## Text Segment Visualization using HTML
# - Colorize the answer's text span
# - Better readability!!
# - Debugging

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

INPUT_BASE = '/kaggle/input/tweet-sentiment-extraction/'

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import time
import random
import datetime


# In[ ]:


train_data = pd.read_csv('{}train.csv'.format(INPUT_BASE))


# In[ ]:


train_data = train_data.dropna()
train_data.reset_index(inplace=True)
train_data.head()


# In[ ]:


train_data['text'] = train_data.apply(lambda row: str(row.text).strip(), axis=1)


# In[ ]:


starts = []
ends = []
for idx, _ in enumerate(train_data.text):
    text = train_data.text[idx]
    selected = train_data.selected_text[idx]
    
    words = text.split()
    selected_words = selected.split()
    
    for idx, word in enumerate(words):
        if selected_words[0] in word:
            start = idx
            break
    for idx, word in enumerate(words):
        if selected_words[-1] in word:
            end = idx
            break
    starts.append(start)
    ends.append(end)

train_data['start_idx'] = starts
train_data['end_idx'] = ends


# ## Function for visualizing GT & Predicted Span of Text
# - Input
#     - **words** ; list of words
#     - **gt** ; (start_index, end_index) tuple
#     - **pred** ; (start_index, end_index) tuple
#     - **bg_color** ; bg color for GT
#     - **text_color** ; text color for GT
#     - **pred_bg_color** ; bg color for prediction
#     - **pred_text_color** ; text color for prediction

# In[ ]:


from IPython.core.display import display, HTML

def visualize_text_segment(words, gt, pred=None, bg_color='rgba(255,255,0,0.5)', text_color='black', pred_bg_color='rgba(0,0,255,0.3)', pred_text_color='red'):
    start, end = gt
    if pred is not None:
        start_, end_ = pred
    
    """both index are inclusive"""
    html = ''
    for idx, word in enumerate(words):
        if idx == start:
            html += "<span style='background:{};color:{}'>".format(bg_color, text_color)
        if pred is not None and idx == start_:
                html += "<span style='background:{};color:{}'>".format(pred_bg_color, pred_text_color)
            
        html += word + ' '
        
        if pred is not None and idx == end_:
                html += '</span>'
        if idx == end:
            html += '</span>'
        
    display(HTML(html))


# # Example 1 (Comparison of two text segments)
# - [!] start,end idx can be wrong in this examples
# - yellow ; Ground Truth
# - skyblue ; Prediction (Random Guess)

# In[ ]:


import random

for idx, _ in enumerate(train_data.text):
    if idx > 30:
        break
    words = train_data.text[idx].split()
    start = train_data.start_idx[idx]
    end = train_data.end_idx[idx]
    
    gt = (start, end)
    if start > end:
        gt = (end, start)
    
    pred = (random.randint(gt[0], gt[1]), random.randint(gt[0], gt[1])) # random guess
    
    if pred[0] > pred[1]:
        pred = (pred[1], pred[0])
    
    visualize_text_segment(words, gt, pred)


# # Example 2 (Single text Segment)

# In[ ]:


import random

for idx, _ in enumerate(train_data.text):
    if idx > 30:
        break
    words = train_data.text[idx].split()
    start = train_data.start_idx[idx]
    end = train_data.end_idx[idx]
    
    gt = (start, end)
    if start > end:
        gt = (end, start)
    
    visualize_text_segment(words, gt)


# In[ ]:




