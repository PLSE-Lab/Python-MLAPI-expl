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


import csv


# In[ ]:


path = '../input/yt8mvidlevelpreds/yt8m_train_video_level_logistic_model_predictions.csv'


# In[ ]:


preds = pd.read_csv(path)
preds.head()


# In[ ]:


top_preds = {}
vids = preds['VideoId'].values
pred_label_lists = preds['LabelConfidencePairs'].values


# In[ ]:


sample = '../input/youtube8m-2019/sample_submission.csv'
sample_pd = pd.read_csv(sample)
    
num = 0

for predclass in sample_pd['Class'].values:
    top_preds[predclass] = []
    
for vid,pred_label_list in zip(vids,pred_label_lists):
    vid_class = pred_label_list.split()[0]
    confidence = pred_label_list.split()[1]
    if int(vid_class) in sample_pd['Class'].values:
        vid_seg_id = vid + ":55" #just use segment starting frame 55 for all vids
        buff_list = top_preds[int(vid_class)]
        buff_list.append((vid_seg_id,confidence))
        top_preds[int(vid_class)] = buff_list


# In[ ]:


segmentos = []
idx = 1
for vid_class in top_preds.keys():
    segment_id_and_confs = sorted(top_preds[vid_class], key=lambda x: x[1])
    segmento_ids = [i[0] for i in segment_id_and_confs]
    segmento = segmento_ids[:5]
    segmentos.append(' '.join(segmento))
    idx += 1
    if idx == 5:
        print(segmentos)
        


# In[ ]:


submission = pd.DataFrame()
submission['Class'] = sample_pd['Class']
submission['Segments'] = segmentos
submission.reset_index(drop=True, inplace=True)
submission.to_csv('submission.csv', index=False)

