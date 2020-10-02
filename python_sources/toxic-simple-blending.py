#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Now this gives 0.9824 ROC score. ALL CREDITS GOES TO ORIGINAL KERNEL AUTHORS. I, as a newbie, just created a blend.


# In[ ]:


import pandas as pd


# In[ ]:


glove = pd.read_csv("../input/nb-svm-strong-linear-baseline/submission.csv")
subb = pd.read_csv('../input/fasttext-like-baseline-with-keras-lb-0-053/submission_bn_fasttext.csv')
ave = pd.read_csv('../input/toxic-avenger/submission.csv')
lstm = pd.read_csv('../input/toxicfiles/baselinelstm0069.csv')
svm = pd.read_csv("../input/toxicfiles/lstmglove0072ge.csv")


# In[ ]:


ble = svm.copy()
col = svm.columns


# In[ ]:


col = col.tolist()
col.remove('id')


# In[ ]:


for i in col:
    ble[i] = (subb[i] + lstm[i] + glove[i] + svm[i] + ave[i]) / 5


# In[ ]:


ble.to_csv('blend_sub.csv', index = False)

